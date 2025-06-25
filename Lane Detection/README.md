# Lane Detector 🛣️

This project implements a lane detection system using OpenCV. The system detects the **left and right lane lines** in a video feed using edge detection, ROI masking, and Hough Transform.

## 🧠 Project Description

- Input: Road video from dashcam or live video from your cam
- Output: Same video with red lines overlaid on the **immediate left and right lanes**
- Handles both straight and curved roads with dynamic ROI and smoothing

## 📦 Setup Instructions

**Clone or download** this repository.

Run the lane detection on a video:
```bash
python WebCamSave.py -f input.mp4 -o output.mp4
```

### 📷 Optional: Use Webcam

If no `-f` is provided, the program will capture **live video** from your webcam:

```bash
python WebCamSave.py -o output.mp4
```

Use a Specific Webcam (e.g., USB Camera)
Use the -c flag to set a specific camera index:
```bash
python WebCamSave.py -c 1
```
Output

The processed video is saved automatically:

Named webcam_output.mp4 if using webcam with no -o specified.

Named <inputname>_out.mp4 if using -f input.mp4 with no -o.

Frame rate is capped at 60 FPS during recording. (adjustable)


## ▶️ Usage Guide

Key	Functionality

p	Pause/resume playback

s	Step forward 1 frame

r	Toggle ROI (green overlay)

q	Quit

The output video is saved to the path specified by -o.


## 🧠 Lane Detection Process
1. Preprocessing
   
Convert the frame to grayscale.

Threshold to isolate bright lane markings.

Apply Gaussian blur and Canny edge detection.

Use morphological closing to connect gaps in edges.

2. Region of Interest (ROI)
   
A trapezoidal mask filters out irrelevant areas (sky, signs, road edges).

Keeps focus on road surface in front of the vehicle.

3. Line Detection
   
cv2.HoughLinesP detects line segments in ROI.

Lines are filtered based on slope:

Left lane: negative slope, left half of image.

Right lane: positive slope, right half of image.

4. Filtering
   
Vertical-ish segments (e.g., poles) are removed if nearly vertical (abs(x2 - x1) < 30).

5. Weighted Averaging
   
Lines are weighted by length.

Produces a stable, smoothed lane line.

6. Drawing
   
Final lines are extended downward to bottom of ROI.

Lane lines drawn in red, ROI outline in green (if enabled).

FPS displayed on top-left corner.

## ✅ Features

Handles dashed lines and slight curves.

Filters out poles, shadows, and irrelevant edges.

Frame pausing and stepping controls.

ROI toggle for debugging.

FPS counter overlay.

