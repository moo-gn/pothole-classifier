import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import cv2
import numpy as np
import tensorflow as tf
from yolov3.utils import image_preprocess, postprocess_boxes, nms, draw_bbox, read_class_names
from yolov3.configs import *
import time
from PIL import Image

from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from deep_sort import generate_detections as gdet

import tensorflow as tf
import numpy as np
import pandas as pd

class PotholeDetector(object):
    def __init__(self) -> None:
        self.NUM_CLASS = read_class_names("model_data/pothole.names")
        self.key_list = list(self.NUM_CLASS.keys()) 
        self.val_list = list(self.NUM_CLASS.values())
        self.input_size = 100
        self.track_class_filter = []
        self.iou_threshold= 0.1
        self.score_threshold= 0.3
        self.show = True
        self.print = True
        self.unique_tracks = {}

    def detect_and_track(self, video_file: cv2.VideoCapture, output_file: str) -> list:
        """Detect and track video file for unique potholes.

        Args:
            video_file (v2.VideoCapture): Input video file object.
            output_file (str): path to the output file.

        Returns:
            list: list of dictionaries of the pothole image and their timestamp in milliseconds.

            Format of the dictionaries: 
            
            tracking_id:
            {
                "image": PIL.Image
                "timestamp": float (in milliseconds)
                "size": int in pixels (calculated from width x height)
            }
        """

        tracker, encoder, model = self.initialize_model()

        # To keep track of timestamps
        frame_no = 0

        # by default VideoCapture returns float instead of int
        width = int(video_file.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video_file.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(video_file.get(cv2.CAP_PROP_FPS))
        codec = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(output_file, codec, fps, (width, height)) # output_file must be .mp4
        with model.as_default():
            with tf.compat.v1.Session(graph=model) as session:
                while True:
                    # Read the next frame from the video
                    _, frame = video_file.read()

                    try:
                        detections = self.detect(frame, encoder, model, session, width, height)
                    except: break

                    # Compute time
                    # timestamp = self.calculate_time(frame_no)
                    timestamp = video_file.get(cv2.CAP_PROP_POS_MSEC)
                    frame_no += 1

                    tracked_bboxes = self.track(tracker, detections, frame, timestamp)

                    # Draw detection on frame
                    image = self.visualize(frame, tracked_bboxes, timestamp)

                    if output_file != '': out.write(image)
                    if self.show:
                        cv2.imshow('output', image)
                    
                        if cv2.waitKey(25) & 0xFF == ord("q"):
                            cv2.destroyAllWindows()
                            break

                cv2.destroyAllWindows()

        for id, track in self.unique_tracks.items():
            print("image size", track["size"])
            # track["image"].save(f"result_images/pothole-{id}.png")

        return self.unique_tracks

    def visualize(self, original_frame, tracked_bboxes, timestamp) -> Image:
        image = draw_bbox(original_frame, tracked_bboxes, CLASSES=YOLO_COCO_CLASSES, tracking=True)
        
        image = cv2.putText(image, "Time: {:.1f} milliseconds".format(timestamp), (0, 30), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 2)

        return image

    def track(self, tracker, detections, frame, timestamp):
        # Pass detections to the deepsort object and obtain the track information.
        tracker.predict()
        tracker.update(detections)

        # Obtain info from the tracks
        tracked_bboxes = []
        for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 5:
                continue 
            bbox = track.to_tlbr() # Get the corrected/predicted bounding box `(min x, min y, max x, max y)
            class_name = "Pothole" #Get the class name of particular object
            tracking_id = track.track_id # Get the ID for the particular track
            index = self.key_list[self.val_list.index(class_name)] # Get predicted object index by object name
            tracked_bboxes.append(bbox.tolist() + [tracking_id, index]) # Structure data, that we could use it with our draw_bbox function

            curr_size = self.get_image_size(bbox)
            abs_size = abs(750*750 - curr_size)

            if tracking_id not in self.unique_tracks.keys():
                self.unique_tracks[tracking_id] = {"timestamp":timestamp, "image": self.crop_image_from_bbox(bbox, frame), "size": abs_size}
            else:
                # If smaller size, we take it 
                if self.unique_tracks[tracking_id]["size"] > abs_size:
                    self.unique_tracks[tracking_id]["size"] = abs_size
                    self.unique_tracks[tracking_id]["image"] = self.crop_image_from_bbox(bbox, frame)
                    
        return tracked_bboxes

    def detect(self, frame: np.ndarray, encoder, model, session: tf.compat.v1.Session, width, height):

        image_data = np.expand_dims(frame, axis=0)
        image_tensor = model.get_tensor_by_name('image_tensor:0')
        boxes = model.get_tensor_by_name('detection_boxes:0')
        scores = model.get_tensor_by_name('detection_scores:0')

        (boxes, scores) = session.run(
            [boxes, scores],
            feed_dict={image_tensor: image_data})

        # Boxes formatted for deepsorted
        pred_boxes = []
        for box, score in zip(boxes[0], scores[0]):
            # Score thershold
            if score < self.score_threshold:
                continue
            x1, y1, w, h = int(box[1] * width), int(box[0] * height), int(box[3] * width - box[1] * width), int(box[2] * height - box[0] * height)
            bbox = [x1, y1, w, h]

            pred_boxes.append(bbox)

        processed_frame, _ = self.preprocess(frame)

        features = np.array(encoder(processed_frame, pred_boxes))

        detections = [Detection(bbox, score, class_name, feature) for bbox, score, class_name, feature in zip(pred_boxes, scores[0], ["Pothole" * len(features)], features)]

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

        # Open model
        with tf.compat.v1.gfile.GFile("model/saved_model.pb", 'rb') as fid:
            graph_def = tf.compat.v1.GraphDef()
            graph_def.ParseFromString(fid.read())

        with tf.Graph().as_default() as model:
            tf.import_graph_def(graph_def, name="")

        return tracker, encoder, model

    def preprocess(self, frame) -> np.ndarray:
        original_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        original_frame = cv2.cvtColor(original_frame, cv2.COLOR_BGR2RGB)
        image_data = image_preprocess(np.copy(original_frame), [self.input_size, self.input_size])
        image_data = image_data[np.newaxis, ...].astype(np.float32)
        return original_frame, image_data

    def get_image_size(self, bbox):
        bbox_list = bbox.tolist()
        x1, y1, x2, y2 = [int(b) for b in bbox_list]
        return (x2 - x1) * (y2 - y1)

    def crop_image_from_bbox(self, bbox, frame):
        x1, y1, x2, y2 = bbox.tolist()
        image = Image.fromarray(frame)
        cropped = image.crop((x1, y1, x2, y2))
        return cropped