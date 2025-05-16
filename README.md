# Time-of-Flight Pose Landmarker Estimation

A pose landmark estimation model trained for grayscale images captured by ToF (Time-of-Flight) cameras.

## Project Overview

### Data Capture

For this project, we would use a ToF camera and a RGB camera at the same time to capture both RGB frames and depth frames.

In the room, there would be a person standing in front of the camera, and an ArUco marker would be placed on the wall behind the person. The ArUco marker would be used to calibrate the ToF camera and the RGB camera. 
Parts of the ArUco marker would be covered with retroreflective tape to reflect the infrared light emitted by the ToF camera.

The data from the RGB camera would be passed into Mediapipe Pose Landmarker model to get the annotations for the RGB frames. 
The annotations would be applied to the depth frames to get the annotations for the depth frames.

### Data Preparation

TODO.

### Model Training

TODO.

### Model Evaluation

TODO.

