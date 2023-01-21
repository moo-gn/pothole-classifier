# pothole-classifier
A pothole detection and classification system. Input a .mp4 video and get a .csv report of all potholes in the video with their respective timestamps and severity classification. The program automates the creation of pothole reports to ease the process of addressing road conditions.


Note: In order to run the program with no issue, make sure to create a virtualenv and install the requirements with 

`$ pip3 install -r requirements`

To run the program:

`$ python3 main.py -i <input_video.mp4_file> -o <output_path>`

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

The architecture of our Plothole Classifier Pipeline
![Architecture Diagram](https://user-images.githubusercontent.com/48159946/213861748-0570330b-428e-4d0a-9c8e-6d44639102d2.png)
