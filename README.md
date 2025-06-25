# YOLOv5 + Lucas-Kanade Optical Flow Tracking

This project integrates **YOLOv5 object detection** with **Lucas-Kanade optical flow tracking using pyramids**.

---

## 🚀 Features

- ✅ **YOLOv5** detection (supports 80 COCO classes)
- ✅ **Lucas-Kanade Optical Flow** with pyramids for robust tracking
- ✅ Press `t` to start tracking selected corners
- ✅ Press `r` to return to YOLO detection mode
- ✅ Press `q` to quit
- ✅ Corner-level trajectory visualization (colored lines and dots)
- ✅ Support for webcam and video input

---

## 🎮 Keyboard Controls

| Key | Action                      |
|-----|-----------------------------|
| `t` | Enter tracking mode         |
| `r` | Return to detection mode    |
| `q` | Exit the program            |

---

## ▶️ How to Run

### ✅ Real-time Detection with Webcam

```bash
python WebCamSave.py 
```

- `--source 0` uses the default webcam.
- Try `--source 1` for an external camera if needed.

### ✅ Detection + Tracking on Video File (test video are provided in "/testvideo")

```bash
python WebCamSave.py  -f testvideo/video1.mp4 
```
---

## 🧠 How It Works

1. **Detection Phase**
   - Objects are detected using YOLOv5 with COCO classes (80 classes).

2. **Tracking Phase**
   - When `t` is pressed, corner points (`cv2.goodFeaturesToTrack`) are extracted from bounding box centers.
   - Optical flow with image pyramids (`cv2.calcOpticalFlowPyrLK`) is used to track these points across frames.
   - Colored motion trails (lines and dots) are drawn to visualize the movement.

3. **Switching Modes**
   - `r` returns to detection mode, refreshing object detection.
   - `q` closes the video stream and exits the application.
---

## 📸 Output Preview

- **Detection Mode**: bounding boxes with COCO class labels.
- **Tracking Mode**: colored optical flow lines and tracked feature points.

---

## 📘 Notes

- The script is based on a lightly modified version of `detect.py` from YOLOv5.
- LK tracking uses `cv2.calcOpticalFlowPyrLK` with pyramidal support for small and large motions.
- Supports CPU and GPU inference (`--device cpu` or `--device 0`).
- Works well for both webcam streams and offline video files.
---

## 🗂️ Project Structure

```
mini-project11/
├── data/                 # Dataset config files (e.g., coco.yaml, custom.yaml)
├── models/               # (Optional) model definitions or custom heads
├── runs/                 # Output directory for detection/tracking results
├── testvideo/            # Test videos for offline detection/tracking
├── utils/                # Utility functions and scripts
├── WebCamSave.py         # ✅ Main script: YOLOv5 + LK Optical Flow tracking
├── yolov5s.pt            # Pretrained YOLOv5 weights (COCO 80 classes)
```
---

Implementation: Feiyan Zhou, Jinzhe Chao
Testing: Chieh-Han Chen, Jing Hui Ng
Course: CS5330 - Computer Vision, Northeastern University
