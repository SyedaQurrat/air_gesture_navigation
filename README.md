# Hand Gesture Control Apps 

This repository contains two Python applications using OpenCV and MediaPipe for hand gesture control:

1. **Drawing App** – Draw on the screen using your index finger like a virtual pen. Select tools and colors by pinching gestures.
2. **Website Navigation App** – Hover your hand over buttons to open websites without using a mouse or keyboard.

##  Requirements
- Python 3.8+
- OpenCV
- MediaPipe
- NumPy

Install dependencies:
```bash
pip install opencv-python mediapipe numpy
```

## Camera Setup
Use a webcam or connect your mobile camera via apps like DroidCam/IP Webcam. Replace MOBILE_CAM_URL with the correct stream link.

## Run
python app.py   # for drawing
python main.py   # for website navigation

## Powered By
MediaPipe Hand Tracking & OpenCV