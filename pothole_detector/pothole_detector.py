import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import cv2
import numpy as np
import tensorflow as tf
from yolov3.utils import Load_Yolo_model, image_preprocess, postprocess_boxes, nms, draw_bbox, read_class_names
from yolov3.configs import *
import time
from PIL import Image

from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from deep_sort import generate_detections as gdet

class PotholeDetector(object):
    def __init__(self) -> None:
        self.NUM_CLASS = read_class_names(YOLO_COCO_CLASSES)
        self.key_list = list(self.NUM_CLASS.keys()) 
        self.val_list = list(self.NUM_CLASS.values())
        self.input_size = 416
        self.yolo = Load_Yolo_model()
        self.track_class_filter = []
        self.iou_threshold= 0.1
        self.score_threshold= 0.3
        self.show = True
        self.print = False

    def detect_and_track(self, video_file: cv2.VideoCapture, output_file: str) -> list:
        """Detect and track video file for unique potholes.

        Args:
            video_file (bytes): Input video file.

        Returns:
            list: list of dictionaries of the pothole image and timestamp.
        """

        tracker, encoder = self.initialize_model()
        
        times, times_2 = [], []

        # by default VideoCapture returns float instead of int
        width = int(video_file.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video_file.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(video_file.get(cv2.CAP_PROP_FPS))
        codec = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(output_file, codec, fps, (width, height)) # output_file must be .mp4

        while True:
            # Read the next frame from the video
            _, frame = video_file.read()

            try:
                original_frame, image_data = self.preprocess(frame)
            except: break

            t1 = time.time()
            if YOLO_FRAMEWORK == "tf":
                pred_bbox = self.yolo.predict(image_data)
            
            t2 = time.time()

            detections = self.detect(encoder, original_frame, pred_bbox)

            tracked_bboxes = self.track(tracker, detections)

            # Compute time
            self.calculate_time(t1, t2, times, times_2)

            # draw detection on frame
            image = self.visualize(original_frame, tracked_bboxes, fps)

            if output_file != '': out.write(image)
            if self.show:
                cv2.imshow('output', image)
                
                if cv2.waitKey(25) & 0xFF == ord("q"):
                    cv2.destroyAllWindows()
                    break
                
        cv2.destroyAllWindows()

    def calculate_time(self, t1, t2, times, times_2):

        t3 = time.time()
        times.append(t2-t1)
        times_2.append(t3-t1)
        
        times = times[-20:]
        times_2 = times_2[-20:]

        ms = sum(times)/len(times)*1000
        fps = 1000 / ms
        fps2 = 1000 / (sum(times_2)/len(times_2)*1000)

        if self.print:
            print("Time: {:.2f}ms, Detection FPS: {:.1f}, total FPS: {:.1f}".format(ms, fps, fps2))

        return fps, fps2
 
    def visualize(self, original_frame, tracked_bboxes, fps) -> Image:
        image = draw_bbox(original_frame, tracked_bboxes, CLASSES=YOLO_COCO_CLASSES, tracking=True)
        
        image = cv2.putText(image, "Time: {:.1f} FPS".format(fps), (0, 30), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 2)

        return image

    def track(self, tracker, detections):
        # Pass detections to the deepsort object and obtain the track information.
        tracker.predict()
        tracker.update(detections)

        # Obtain info from the tracks
        tracked_bboxes = []
        for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 5:
                continue 
            bbox = track.to_tlbr() # Get the corrected/predicted bounding box
            class_name = track.get_class() #Get the class name of particular object
            tracking_id = track.track_id # Get the ID for the particular track
            index = self.key_list[self.val_list.index(class_name)] # Get predicted object index by object name
            tracked_bboxes.append(bbox.tolist() + [tracking_id, index]) # Structure data, that we could use it with our draw_bbox function
            print(track)
        return tracked_bboxes

    def detect(self, encoder, original_frame, pred_bbox):
        pred_bbox = [tf.reshape(x, (-1, tf.shape(x)[-1])) for x in pred_bbox]
        pred_bbox = tf.concat(pred_bbox, axis=0)

        bboxes = postprocess_boxes(pred_bbox, original_frame, self.input_size, self.score_threshold)
        bboxes = nms(bboxes, self.iou_threshold, method='nms')

        # extract bboxes to boxes (x, y, width, height), scores and names
        boxes, scores, names = [], [], []
        for bbox in bboxes:
            if len(self.track_class_filter) !=0 and self.NUM_CLASS[int(bbox[5])] in self.track_class_filter or len(self.track_class_filter) == 0:
                boxes.append([bbox[0].astype(int), bbox[1].astype(int), bbox[2].astype(int)-bbox[0].astype(int), bbox[3].astype(int)-bbox[1].astype(int)])
                scores.append(bbox[4])
                names.append(self.NUM_CLASS[int(bbox[5])])

        # Obtain all the detections for the given frame.
        boxes = np.array(boxes) 
        names = np.array(names)
        scores = np.array(scores)
        features = np.array(encoder(original_frame, boxes))
        detections = [Detection(bbox, score, class_name, feature) for bbox, score, class_name, feature in zip(boxes, scores, names, features)]

        return detections

    def initialize_model(self) -> Tracker: 
        # Definition of the parameters
        max_cosine_distance = 0.7
        nn_budget = None
        
        #initialize deep sort object
        model_filename = 'model_data/mars-small128.pb'
        encoder = gdet.create_box_encoder(model_filename, batch_size=1)
        metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
        tracker = Tracker(metric)
        return tracker, encoder

    def preprocess(self, frame) -> np.ndarray:
        original_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        original_frame = cv2.cvtColor(original_frame, cv2.COLOR_BGR2RGB)
        image_data = image_preprocess(np.copy(original_frame), [self.input_size, self.input_size])
        image_data = image_data[np.newaxis, ...].astype(np.float32)
        return original_frame, image_data
