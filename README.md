# pothole-classifier
A pothole detection and classification system. Input a .mp4 video and get a .csv report of all potholes in the video with their respective timestamps and severity classification. The program automates the creation of pothole reports to ease the process of addressing road conditions.


Note: In order to run the program with no issue, make sure to create a virtualenv and install the requirements with 

`$ pip3 install -r requirements`

To run the program:

`$ python3 main.py -i <input_video.mp4_file> -o <output_path>`

Output should be a name without a suffix (like mp4 or csv, since the output will be two files)




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
