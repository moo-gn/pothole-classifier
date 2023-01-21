


To run the program:




The architecture of our Plothole Classifier Pipeline
![Architecture Diagram](https://user-images.githubusercontent.com/48159946/213861748-0570330b-428e-4d0a-9c8e-6d44639102d2.png)

Example of the output.csv file which contains information regarding all potholes in the video and their severity classification.
![image](https://user-images.githubusercontent.com/48159946/213862004-b659335f-4726-4665-83bd-c78c23b264f9.png)

Notice that the image_path column points to the cropped pothole images that were saved to the result_images/ directory. This is so that you could review the images that were classified in the final report.


# pothole-classifier
A pothole detection and classification system. Input a .mp4 video and get a .csv report of all potholes in the video with their respective timestamps and severity classification. The program automates the creation of pothole reports to ease the process of addressing road conditions.


<img src="project-logo.png" align="left" width="192px" height="192px"/>
<img align="left" width="0" height="192px" hspace="10"/>

> Template files for writing maintanable GitHub projects. Make your repo pretty!

[![Under Development](https://img.shields.io/badge/under-development-orange.svg)](https://github.com/cezaraugusto/github-template-guidelines) [![Public Domain](https://img.shields.io/badge/public-domain-lightgrey.svg)](https://creativecommons.org/publicdomain/zero/1.0/) [![Travis](https://img.shields.io/travis/cezaraugusto/github-template-guidelines.svg)](http://github.com/cezaraugusto/github-template-guidelines)

So you had an idea and developed the next world's industry-changing application. You decided to open-source it because you're way cool. Now you need to educate people about your project and need some docs to get started. You got it. :alien:

<br>
<p align="center">
<strong>Templates included:</strong>
<a href="/.github/README.md">README</a> • <a href="/.github/CONTRIBUTING.md">CONTRIBUTING </a> • <a href="/.github/PULL_REQUEST_TEMPLATE.md">PULL REQUEST</a> • <a href="/.github/ISSUE_TEMPLATE.md">ISSUE TEMPLATE</a> • <a href="/.github/CONTRIBUTORS.md">CONTRIBUTORS</a>
</p>
<br>

## Installing

Clone this project and name it accordingly:

``git clone https://github.com/moo-gn/pothole-classifier``

NOTE: In order to run the program with no dependency conflicts, make sure to create a virtualenv to install the requirements

``$ pip3 install -r requirements``



# Getting Started

This project is a collection of [boilerplate](http://whatis.techtarget.com/definition/boilerplate) (template) files with resumed guidelines for `README`, `CONTRIBUTING` and `CONTRIBUTORS` documentation. It also includes a basic `ISSUE_TEMPLATE` and `PULL_REQUEST_TEMPLATE` which are now [allowed by GitHub](https://github.com/blog/2111-issue-and-pull-request-templates). All templates are filled under `.github/` folder. This `README` itself is a fork of the `README` [template](.github/README.md).

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

## Useful Resources :thumbsup:

> References for starting a Project

* [Helping people contribute to your Project](https://help.github.com/articles/helping-people-contribute-to-your-project/)
* [Am I Ready to Open Source it?](https://gist.github.com/PurpleBooth/6f1ba788bf70fb501439#file-am-i-ready-to-open-source-this-md)

> `README` References

* [How To Write A Readme](http://jfhbrook.github.io/2011/11/09/readmes.html)
* [How to Write a Great Readme](https://robots.thoughtbot.com/how-to-write-a-great-readme)
* [Eugene Yokota - StackOverflow Answer](http://stackoverflow.com/a/2304870)

> `CONTRIBUTING` References

* [Setting Guidelines for Repository Contributors](https://help.github.com/articles/setting-guidelines-for-repository-contributors/)
* [Contributor Covenant](http://contributor-covenant.org/)

> `CHANGELOG` References

> This boilerplate intentionally did not provide any `CHANGELOG` file as example, since [this tool](https://github.com/skywinder/github-changelog-generator) could make it automatically, fulfilling the file's objective. If you still want to keep it handwritten, to keep you (and your project) sane, I'd recommend you to follow the references below:

* [Semantic Versioning 2.0.0](http://semver.org/)
* [Keep a Changelog](http://keepachangelog.com/)

> `ISSUE_TEMPLATE` and `PULL_REQUEST_TEMPLATE` References

* [Creating an Issue Template for your repository](https://help.github.com/articles/creating-an-issue-template-for-your-repository/)
* [Creating a Pull Request Template for your repository](https://help.github.com/articles/creating-a-pull-request-template-for-your-repository/)
* [Awesome GitHub Templates](https://github.com/devspace/awesome-github-templates)

> `CONTRIBUTORS` References

* [All Contributors](https://github.com/kentcdodds/all-contributors/)
* [All Contributors (CLI)](https://github.com/jfmengels/all-contributors-cli)

## Contributors

<!-- Contributors START
Cezar_Augusto cezaraugusto http://cezaraugusto.net doc example prReview
Nathalia_Bruno nathaliabruno http://nathaliabruno.com doc prReview
Billie_Thompson PurpleBooth http://purplebooth.co.uk example
Contributors END -->

<!-- Contributors table START -->
| [![Cezar Augusto](https://avatars.githubusercontent.com/cezaraugusto?s=100)<br /><sub>Cezar Augusto</sub>](http://cezaraugusto.net)<br />[📖](git@github.com:cezaraugusto/You-Dont-Know-JS/commits?author=cezaraugusto) 💡 👀 | [![Nathalia Bruno](https://avatars.githubusercontent.com/nathaliabruno?s=100)<br /><sub>Nathalia Bruno</sub>](http://nathaliabruno.com)<br />[📖](git@github.com:cezaraugusto/You-Dont-Know-JS/commits?author=nathaliabruno) 👀 | [![Billie Thompson](https://avatars.githubusercontent.com/PurpleBooth?s=100)<br /><sub>Billie Thompson</sub>](http://purplebooth.co.uk)<br />💡 |
| :---: | :---: | :---: |
<!-- Contributors table END -->

This project follows the [all-contributors](https://github.com/kentcdodds/all-contributors) specification.
Contributions of any kind welcome!

## License
[![CC0](https://i.creativecommons.org/p/zero/1.0/88x31.png)](https://creativecommons.org/publicdomain/zero/1.0/)

To the extent possible under law, [Cezar Augusto](http://cezaraugusto.net) has waived all copyright and related or neighboring rights to this work.

