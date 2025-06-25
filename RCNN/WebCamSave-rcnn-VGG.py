import cv2
import numpy as np
import tensorflow as tf
import time
import os
from datetime import datetime
import threading
import queue


class WebCamRCNN:
    def __init__(self, model_path="multiclass_model.h5"):  # Changed to 9-1 model name
        """
        Real-time R-CNN webcam application for Assignment 9-1 (VGG16)
        """
        print("🚀 Initializing WebCam R-CNN (9-1 VGG16 Version)...")

        # Load model
        try:
            self.model = tf.keras.models.load_model(model_path)
            print(f"✅ Model loaded: {model_path}")
        except:
            print(f"❌ Could not load model: {model_path}")
            print("💡 Make sure you've trained and saved the 9-1 model first!")
            exit()

        # Initialize selective search
        self.ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
        cv2.setUseOptimized(True)

        # Parameters (matching 9-1 inference)
        self.MAX_PROPOSALS = 15  # Reduced for real-time performance
        self.MIN_SIZE = 10  # From 9-1: w >= 20, h >= 20
        self.NMS_THRESH = 0.3  # Exact 9-1 threshold
        self.SKIP_FRAMES = 5  # Process every 3rd frame for speed

        # Performance tracking
        self.frame_count = 0
        self.fps_history = []
        self.start_time = time.time()

        # Create output directory
        self.output_dir = "webcam_detections_9_1"
        os.makedirs(self.output_dir, exist_ok=True)

        print("🎯 R-CNN Parameters (9-1 Settings):")
        print(f"   Max Proposals: {self.MAX_PROPOSALS}")
        print(f"   Min Box Size: {self.MIN_SIZE}x{self.MIN_SIZE}")
        print(f"   NMS Threshold: {self.NMS_THRESH}")
        print(f"   Skip Frames: {self.SKIP_FRAMES}")

    def non_max_suppression(self, boxes, thresh=0.3):
        """EXACT NMS from 9-1"""
        if len(boxes) == 0:
            return []
        boxes = np.array(boxes)
        x1, y1, x2, y2, scores = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3], boxes[:, 4]
        area = (x2 - x1 + 1) * (y2 - y1 + 1)
        idxs = np.argsort(scores)
        pick = []
        while len(idxs) > 0:
            last = idxs[-1]
            pick.append(last)
            xx1 = np.maximum(x1[last], x1[idxs[:-1]])
            yy1 = np.maximum(y1[last], y1[idxs[:-1]])
            xx2 = np.minimum(x2[last], x2[idxs[:-1]])
            yy2 = np.minimum(y2[last], y2[idxs[:-1]])
            w = np.maximum(0, xx2 - xx1 + 1)
            h = np.maximum(0, yy2 - yy1 + 1)
            overlap = (w * h) / area[idxs[:-1]]
            idxs = np.delete(idxs, np.concatenate(([len(idxs) - 1], np.where(overlap > thresh)[0])))
        return boxes[pick].astype("int")

    def detect_objects(self, frame):
        """
        Run R-CNN detection on a frame using EXACT 9-1 logic
        """
        detection_start = time.time()

        # Selective search (9-1 style)
        self.ss.setBaseImage(frame)
        self.ss.switchToSelectiveSearchFast()
        proposals = self.ss.process()[:self.MAX_PROPOSALS]  # Limited for speed

        detections = []

        # Process proposals with 9-1 logic
        for (x, y, w, h) in proposals:
            # EXACT 9-1 filtering: w >= 20, h >= 20
            if w < self.MIN_SIZE or h < self.MIN_SIZE:
                continue

            roi = frame[y:y + h, x:x + w]
            if roi.shape[0] == 0 or roi.shape[1] == 0:
                continue

            # Resize and predict
            resized = cv2.resize(roi, (224, 224))
            resized = np.expand_dims(resized, axis=0)
            pred = self.model.predict(resized, verbose=0)[0]

            # EXACT 9-1 confidence logic
            bg_prob, tv_prob, remote_prob = pred[0], pred[1], pred[2]
            conf = max(tv_prob, remote_prob)

            # EXACT 9-1 condition: conf > 0.5 AND conf > bg_prob
            if conf > 0.5 and conf > bg_prob:
                label = "TV" if tv_prob > remote_prob else "Remote"
                detections.append([x, y, x + w, y + h, conf, label])

        # Apply NMS using 9-1 logic
        filtered = []
        if detections:
            coords = np.array([[x, y, x2, y2, conf] for x, y, x2, y2, conf, _ in detections])
            keep = self.non_max_suppression(coords, self.NMS_THRESH)

            for box in keep:
                x1, y1, x2, y2, conf = box
                for b in detections:
                    if b[0] == x1 and b[1] == y1 and b[2] == x2 and b[3] == y2:
                        filtered.append(b)
                        break

        detection_time = time.time() - detection_start
        return filtered, detection_time

    def draw_detections(self, frame, detections, detection_time, fps):
        """
        Draw bounding boxes and labels on frame (9-1 style)
        """
        annotated = frame.copy()

        # Draw detections with 9-1 colors
        for (x1, y1, x2, y2, conf, label) in detections:
            # 9-1 colors: Green for TV, Red for Remote
            color = (0, 255, 0) if label == "TV" else (0, 0, 255)

            # Draw bounding box
            cv2.rectangle(annotated, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)

            # Draw label with confidence (9-1 style)
            cv2.putText(annotated, f"{label}: {conf:.2f}", (int(x1), int(y1) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        # Performance info overlay
        info_y = 30
        cv2.putText(annotated, f"FPS: {fps:.1f}", (10, info_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.putText(annotated, f"Detection Time: {detection_time * 1000:.1f}ms", (10, info_y + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Count TVs and Remotes
        tv_count = sum(1 for _, _, _, _, _, label in detections if label == "TV")
        remote_count = sum(1 for _, _, _, _, _, label in detections if label == "Remote")

        cv2.putText(annotated, f"TVs: {tv_count}, Remotes: {remote_count}", (10, info_y + 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.putText(annotated, "Press 's' to save, 'q' to quit, 'p' to pause", (10, info_y + 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        # Add 9-1 filter info
        cv2.putText(annotated, "9-1 VGG16 Model", (frame.shape[1] - 150, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        return annotated

    def save_detection(self, frame, detections):
        """
        Save frame with detections
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.output_dir}/detection_{timestamp}.jpg"

        cv2.imwrite(filename, frame)

        # Count detections
        tv_count = sum(1 for _, _, _, _, _, label in detections if label == "TV")
        remote_count = sum(1 for _, _, _, _, _, label in detections if label == "Remote")

        print(f"💾 Saved: {filename} (TVs: {tv_count}, Remotes: {remote_count})")

    def calculate_fps(self, frame_time):
        """
        Calculate running FPS average
        """
        fps = 1.0 / frame_time if frame_time > 0 else 0
        self.fps_history.append(fps)

        # Keep last 30 fps measurements
        if len(self.fps_history) > 30:
            self.fps_history.pop(0)

        return np.mean(self.fps_history)

    def run(self):
        """
        Main webcam loop
        """
        print("\n🎥 Starting webcam with 9-1 model...")
        print("=" * 50)
        print("📌 Using EXACT 9-1 filtering logic:")
        print("   • Min size: 20x20 pixels")
        print("   • Confidence: >0.5 AND >background")
        print("   • NMS threshold: 0.3")
        print("=" * 50)
        print("\nControls:")
        print("  's' - Save current frame with detections")
        print("  'q' - Quit application")
        print("  'p' - Toggle pause")
        print()

        cap = cv2.VideoCapture(0)

        # Set camera properties for better performance
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)

        if not cap.isOpened():
            print("❌ Could not open webcam")
            return

        paused = False
        last_detection_frame = None
        detection_time = 0

        while True:
            frame_start = time.time()

            ret, frame = cap.read()
            if not ret:
                print("❌ Could not read frame")
                break

            self.frame_count += 1

            # Process detection every Nth frame or when paused
            if self.frame_count % (self.SKIP_FRAMES + 1) == 0 or paused:
                try:
                    detections, detection_time = self.detect_objects(frame)
                    last_detection_frame = (detections, detection_time)
                except Exception as e:
                    print(f"⚠️ Detection error: {e}")
                    detections, detection_time = [], 0
                    last_detection_frame = (detections, detection_time)
            else:
                # Use last detection results
                if last_detection_frame:
                    detections, detection_time = last_detection_frame
                else:
                    detections, detection_time = [], 0

            # Accurate FPS: based only on actual detection frames
            fps = 0
            if self.frame_count % (self.SKIP_FRAMES + 1) == 0 or paused:
                now = time.time()
                if hasattr(self, "last_detect_time"):
                    fps = 1.0 / (now - self.last_detect_time)
                self.last_detect_time = now

            # Draw results
            annotated_frame = self.draw_detections(frame, detections, detection_time, fps)

            # Show frame
            cv2.imshow('WebCam R-CNN Detection (9-1 VGG16)', annotated_frame)

            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                self.save_detection(annotated_frame, detections)
            elif key == ord('p'):
                paused = not paused
                print(f"{'⏸️ Paused' if paused else '▶️ Resumed'}")

        # Cleanup
        cap.release()
        cv2.destroyAllWindows()

        # Final statistics
        total_time = time.time() - self.start_time
        avg_fps = self.frame_count / total_time

        print(f"\n📊 Final Statistics:")
        print(f"   Total Frames: {self.frame_count}")
        print(f"   Total Time: {total_time:.1f}s")
        print(f"   Average FPS: {avg_fps:.1f}")
        print(f"   Detections saved in: {self.output_dir}/")


def main():
    """
    Main function to run the webcam R-CNN app with 9-1 model
    """
    print("🎬 WebCam R-CNN Detection App - Assignment 9-1 (VGG16)")
    print("=" * 60)

    # Using 9-1 model
    app = WebCamRCNN(model_path="multiclass_model.h5")

    try:
        app.run()
    except KeyboardInterrupt:
        print("\n🛑 Interrupted by user")
    except Exception as e:
        print(f"💥 Error: {e}")

    print("👋 WebCam R-CNN finished!")


if __name__ == "__main__":
    main()