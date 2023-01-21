import argparse
import depth_estimator
from pothole_detector.pothole_detector import PotholeDetector
import pandas as pd
import cv2

DEFAULT_OUTPUT = "output"
pothole_detector = PotholeDetector()

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
    pass

def make_predictions(video_file: cv2.VideoCapture, output_file: str):
    """Retrieve list of dictionaries of pothole images and their timestamp
    Example of dictionary unique_potholes = 
    [
        {
            "image": PIL.Image,
            "timestamp": "00:00"
        }, 
        {
            "image": PIL.Image, 
            "timestamp": "00:01"
        }
    ]

    Questions to address [IMPORTANT]:
    1. What format should the timestamp be?

    2. What should the interval be? 1 second? 1 frame? Or is it determined by tracking method?
    """
    unique_potholes = pothole_detector.detect_and_track(video_file, output_file)

    # Just to check output, remove
    print(unique_potholes)

    predictions = []

    for pothole in unique_potholes:
        # Classify severity of a cropped plothole image and add it to predictions list
        severity = depth_estimator.classify(pothole)
        predictions.append({
            "image": pothole["image"],
            "severity": severity,
            "timestamp": pothole["timestamp"]
        })

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
    argParser.add_argument("-o", "--output", required=False, help="Name of the output files.")

    args = argParser.parse_args()

    run_program(args.input, args.output if args.output else DEFAULT_OUTPUT)
    

if __name__ == "__main__":
    main()