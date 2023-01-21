import argparse
from depth_estimator.depth_estimator import DepthEstimator
from pothole_detector.pothole_detector import PotholeDetector
import pandas as pd
import numpy as np
import cv2
from common import convert_milliseconds_to_timestamp

DEFAULT_OUTPUT = "output"
pothole_detector = PotholeDetector()
depth_estimator = DepthEstimator()

def import_video(input_file: str) -> cv2.VideoCapture:
    """Import video file from string path.

    Args:
        input_file (str): string path to the file.

    Returns:
        cv2.VideoCapture: the video file.
    """
    video_file = cv2.VideoCapture(input_file)
    return video_file

def segment_video_to_images(video_file: bytes) -> dict:
    # TODO: do not prioritize
    pass

def create_report(predictions: dict, output_file: str) -> None:
    """Creates and saves a report of the potholes.
    The report should include the severity of each pothole detected in the video and their timestamp.

    Args:
        predictions (dict): a dictionary which contains pothole images with a severity score and a timestamp.
        output_file (str): the path to which to save the report .csv file to.
    """
    # TODO
    output_data = pd.DataFrame(predictions, columns=["id", "timestamp", "severity", "image_path"])

    output_data.to_csv(f"{output_file}.csv")

def make_predictions(video_file: cv2.VideoCapture, output_file: str):
    """Retrieve list of dictionaries of pothole images and their timestamp
    Example of dictionary unique_potholes = 
    [
        id:
        {
            "image": PIL.Image
            "timestamp": float
            "size": int
        }, 
        id:
        {
            "image": PIL.Image 
            "timestamp": float
            "size": int (pixels width x height)
        }
    ]
    """
    unique_potholes = pothole_detector.detect_and_track(video_file, output_file + ".mp4")

    # Just to check output, remove

    predictions = {
        "id": [],
        "timestamp": [],
        "severity": [],
        "image_path": [] 
    }

    for track_id, pothole in unique_potholes.items():
        # Classify severity of a cropped plothole image and add it to predictions list
        severity = depth_estimator.classify(pothole["image"])

        predictions["id"].append(track_id)
        predictions["timestamp"].append(convert_milliseconds_to_timestamp(pothole["timestamp"]))
        predictions["severity"].append(severity)
        predictions["image_path"].append(f"result_images/pothole_image_{track_id}.png")

        pothole["image"].save(predictions["image_path"][-1])

    # Normalize severity
    predictions["severity"] = [float(i)/sum(predictions["severity"]) for i in predictions["severity"]]

    return predictions

def run_program(input_file: str, output_file: str):
    """Main function for handling the args and running the program

    Args:
        input_file (str): Path to input MP4 video file.
        output_file (str): Name of the output files. (output.csv & output.mp4)
    """

    # Retrieve video file
    video_file = import_video(input_file)

    # Handle prediction logic
    predictions = make_predictions(video_file, output_file)

    # Join the data to create and save a report
    pothole_report = create_report(predictions, output_file)

def main():
    """Main function for Arg parsing"""
    argParser = argparse.ArgumentParser()
    argParser.add_argument("-i", "--input", required=True, help="Name of the input MP4 video file.")
    argParser.add_argument("-o", "--output", required=False, help="Name of the output files. Do not put a suffix like .mp4 or .csv.")

    args = argParser.parse_args()

    run_program(args.input, args.output if args.output else DEFAULT_OUTPUT)
    

if __name__ == "__main__":
    main()