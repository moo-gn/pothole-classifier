# pothole-classifier
A pothole detection and classification system. Input a .mp4 video and get a .csv report of all potholes in the video with their respective timestamps and severity classification. The program automates the creation of pothole reports to ease the process of addressing road conditions.


<img src="https://cdn.discordapp.com/attachments/1010369978714837005/1066303874543140944/retro_japanese_70s_style_with_text_moo-gn-4.png" align="left" width="192px" height="192px"/>
<img align="left" width="0" height="192px" hspace="10"/>

[![Under Development](https://img.shields.io/badge/under-development-orange.svg)](https://github.com/cezaraugusto/github-template-guidelines) [![Public Domain](https://img.shields.io/badge/public-domain-lightgrey.svg)](https://creativecommons.org/publicdomain/zero/1.0/) [![Travis](https://img.shields.io/travis/cezaraugusto/github-template-guidelines.svg)](http://github.com/cezaraugusto/github-template-guidelines)

This project was made for the SDAIA smartathon, and we had much fun putting it together as a team.

## Installing

Clone this project and name it accordingly:

``git clone https://github.com/moo-gn/pothole-classifier``

NOTE: In order to run the program with no dependency conflicts, make sure to create a virtualenv to install the requirements

``$ pip3 install -r requirements``


# Getting Started

## Usage

```$ python3 main.py -i <input_video.mp4_file> -o <output_path>```

Output should be a name without a suffix. For example if output_path is set as "file", the outputs will be file.mp4 and file.csv in the working directory.
```
pothole-classifier/
  - file.mp4
  - file.csv
  - result_images/
    - pothole_image_1.png
    - ...
    - pothole_image_n.png
```


Full usage details
```
usage: main.py [-h] -i INPUT [-o OUTPUT] [-q]

options:
  -h, --help            show this help message and exit
  -i INPUT, --input INPUT
                        Name of the input MP4 video file.
  -o OUTPUT, --output OUTPUT
                        Name of the output files. Do not put a suffix like .mp4 or .csv.
  -q, --quiet           Hides the real-time detection screen.
  ```

## The Architecture

The architecture of our Plothole Classifier Pipeline
![Architecture Diagram](https://user-images.githubusercontent.com/48159946/226072480-0769bceb-fccd-4252-bacf-c9e622914a3d.png)



## Features
Example of the output.csv file which contains information regarding all potholes in the video and their severity classification.
<p align="center">
  <img width="537" alt="image" src="https://cdn.discordapp.com/attachments/893029017186361365/1066332698840404049/Screenshot_2023-01-21_at_7.25.20_AM.png">
</p>


Notice that the image_path column points to the cropped pothole images that were saved to the result_images/ directory. This is so that you could review the images that were classified in the final report.

The arclength was calculated with the assumption of 0.125cm/px for archlength.

## References

> DeepSort
* https://arxiv.org/abs/1703.07402

> Single Shot Detection model
* https://github.com/Qberto/arcgis-tf-roaddamagedetector/tree/master/trained_models

> Detection infra

* https://github.com/anushkadhiman/ObjectTracking-DeepSORT-YOLOv3-TF2

> DIODE  depth dataset

* https://diode-dataset.org/

> PotHole-600 depth dataset

* https://sites.google.com/view/pothole-600/dataset
