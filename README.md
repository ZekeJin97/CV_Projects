# 🚘 CV_Projects: Modular Components for Autonomous Driving Perception

This repository contains a collection of computer vision modules developed as part of an advanced perception system for autonomous driving. Each project is independently functional but designed to integrate into a larger pipeline — enabling real-time understanding of roads, lanes, signs, and objects.

> 💡 Think of it like your own mini Tesla Autopilot stack — built from scratch, component by component.

---

## 🔧 Modules

### 🛣️ [Lane Detection](./Lane%20Detection)
- Classical CV pipeline using **Canny edge detection**, **ROI masking**, and **Hough Transform**.
- User-adjustable filters for fast prototyping and region-of-interest debugging.
- Output: binary mask + overlay on raw frame.

### 🚦 [YOLO Detection with Motion Tracking](./YoloDetection_with_MotionTracking)
- Real-time object detection with **YOLOv5**, trained to recognize **stop signs** and **traffic signals**.
- Integrated with **Lucas-Kanade optical flow** to track detected objects across frames.
- Output: annotated frame + persistent object IDs.

### 🔍 [RCNN](./RCNN)
- Custom **R-CNN** implementation with support for **VGG16** and **MobileNet** backbones.
- Region proposal via selective search and fine-grained classification.
- Evaluated on TV + remote dataset, but adaptable for traffic-specific classes.

---

## 🧠 Supporting Components

### 🧪 [CNN Live Classifier](./CNN%20Live%20Classifier)
- Simple webcam-based live classifier for binary or multiclass detection.
- Can be adapted as a real-time inference viewer for any classifier model.

### 🧱 [Custom U-Net](./Custom%20U-Net)
- Semantic segmentation module using a modified U-Net.
- Tested on aerial imagery but ready to segment road features (e.g. drivable area, lane types).
- Features **focal loss**, **batchnorm**, and **color remapping** for small dataset generalization.

---

## 🔐 [Spoofing Detection System](./Spoofing)
- While not strictly a driving task, this demonstrates multi-modal feature fusion and MLP classification.
- Can inspire **driver identity verification**, **face presence detection**, or **in-cabin monitoring** systems.

---

## 🧩 Integration Idea

These modules can be combined into a full perception stack:
- **U-Net** → road segmentation.
- **Lane Detection** → refine lane boundaries within road mask.
- **YOLO + Optical Flow** → detect and track relevant signs/objects.
- **RCNN** → secondary detector for small/high-resolution targets.
- **Live Classifier** → diagnostics / model switchboard.
- **Spoof Detection** → optional for biometric driver monitoring.

---


## 🧠 Author
**Zhechao Jin**  
M.S. Computer Science — Northeastern University  
GitHub: [@ZekeJin97](https://github.com/ZekeJin97)  
LinkedIn: [ZcJin](https://www.linkedin.com/in/zcjin/)

---

## ⚠️ Disclaimer
This repo was developed for academic and demo purposes — not intended for production-grade AV systems (yet 😏).


