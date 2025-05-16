# Time-of-Flight Pose Landmarker Estimation

A pose landmark estimation model trained for grayscale images captured by ToF (Time-of-Flight) cameras.

## Project Overview

### Data Capture

For this project, we would use a ToF camera and an RGB camera at the same time to capture both RGB frames and depth frames.

In the room, there would be a person standing in front of the camera with different poses, and an ArUco marker would be placed on the wall behind the person (in the camera frame all the time). The ArUco marker would be used to calibrate the ToF camera and the RGB camera. 
To use the ArUco marker for calibration, parts of the ArUco marker would be covered with retroreflective tape to reflect the infrared light emitted by the ToF camera, and this would ideally make parts of the marker brighter in the ToF frame.

The data from the RGB camera would be passed into the Mediapipe Pose Landmarker model to get the annotations for the RGB frames. 
The annotations would be applied to the depth frames to get the training datasets for the time-of-flight pose landmarker estimation model.

<p align="center"><img src="https://github.com/user-attachments/assets/498fa63f-af16-4202-9867-65414ad38ab2" width="300"/><br/><span style="text-align: center; opacity: 80;"><i>ArUco 6x6 (ID: #777)</i></span></p>

### Data Preparation

TODO.

### Model Training

TODO.

### Model Evaluation

TODO.

