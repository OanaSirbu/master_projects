# Computer Vision - Project 2 / Visual surveillance of parking spaces on the street

### Author: Sirbu Oana-Adriana

## Overview

This project uses computer vision techniques to detect and track objects on the street, focusing on parking spot occupancy and vehicle tracking. It includes implementations for four main tasks: detecting parking spot occupancy from images (Task 1), analyzing final video frames for occupancy (Task 2), real-time object tracking in videos (Task 3) using the YOLOv8m model and the CSRT tracker, and detecting stationary vehicles within a specified region of interest in video sequences, equivalent to counting cars stopped at a traffic light (Task 4).

## Dependencies

To run this project, ensure you have the following libraries installed with the specified versions:

- Python 3.10
- TensorFlow 2.16.1
- NumPy 1.26.4
- Torch 2.3.1
- Pandas 2.2.2
- Ultralytics 8.2.35
- OpenCV-Contrib-Python 4.10.0.84

## How to Run

All solutions are implemented within a single Python file. After saving the project and installing all the dependencies listed above, make sure to update the paths in the code (located at the beginning of the file - lines 8 to 37), as indicated by the comments. There, you can specify the input files and also configure the location where the predicted results files will be saved. Save the modified python file. 

Then, simply run the python file. It will automatically perform all the operations. 


