import cv2
import numpy as np
import tensorflow as tf
import time
from datetime import datetime
import os

class WebCamRCNNLight:
    def __init__(self, model_path="assignment_9_2_model.h5"):
        print("🚀 Initializing WebCam R-CNN (9-2 MobileNet Version)...")

        self.model = tf.keras.models.load_model(model_path)
        print(f"✅ Model loaded: {model_path}")

        self.ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
        cv2.setUseOptimized(True)

        # Parameters tuned for MobileNet real-time
        self.MAX_PROPOSALS = 15
        self.MIN_SIZE = 20
        self.NMS_THRESH = 0.3
        self.SKIP_FRAMES = 4

        self.frame_count = 0
        self.output_dir = "webcam_detections_9_2"
        os.makedirs(self.output_dir, exist_ok=True)

    def non_max_suppression(self, boxes, thresh=0.3):
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

    def detect(self, frame):
        self.ss.setBaseImage(frame)
        self.ss.switchToSelectiveSearchFast()
        proposals = self.ss.process()[:self.MAX_PROPOSALS]

        detections = []
        for (x, y, w, h) in proposals:
            if w < self.MIN_SIZE or h < self.MIN_SIZE:
                continue
            roi = frame[y:y+h, x:x+w]
            if roi.size == 0:
                continue
            resized = cv2.resize(roi, (224, 224))
            resized = np.expand_dims(resized, axis=0)
            pred = self.model.predict(resized, verbose=0)[0]
            bg, tv, remote = pred[0], pred[1], pred[2]
            conf = max(tv, remote)
            if conf > 0.5 and conf > bg:
                label = "TV" if tv > remote else "Remote"
                detections.append([x, y, x+w, y+h, conf, label])

        final = []
        if detections:
            coords = np.array([[x, y, x2, y2, conf] for x, y, x2, y2, conf, _ in detections])
            keep = self.non_max_suppression(coords, self.NMS_THRESH)
            for box in keep:
                x1, y1, x2, y2, conf = box
                for d in detections:
                    if d[0] == x1 and d[1] == y1 and d[2] == x2 and d[3] == y2:
                        final.append(d)
                        break
        return final

    def draw(self, frame, detections, detection_time, fps):
        output = frame.copy()
        for (x1, y1, x2, y2, conf, label) in detections:
            color = (0, 255, 0) if label == "TV" else (0, 0, 255)
            cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)
            cv2.putText(output, f"{label}: {conf:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        cv2.putText(output, f"FPS: {fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(output, f"Detection Time: {detection_time:.1f} ms", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        return output

    def run(self):
        print("🎥 Starting webcam with MobileNet model...")
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)

        if not cap.isOpened():
            print("❌ Cannot open webcam")
            return

        paused = False
        last_time = time.time()
        detections = []
        detection_time = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            self.frame_count += 1
            start = time.time()

            if self.frame_count % (self.SKIP_FRAMES + 1) == 0 or paused:
                det_start = time.time()
                detections = self.detect(frame)
                detection_time = (time.time() - det_start) * 1000
                last_time = time.time()
            fps = 1 / (time.time() - last_time + 1e-8)

            annotated = self.draw(frame, detections, detection_time, fps)
            cv2.imshow("WebCam R-CNN (9-2 MobileNet)", annotated)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('p'):
                paused = not paused
            elif key == ord('s'):
                fname = f"{self.output_dir}/capture_{int(time.time())}.jpg"
                cv2.imwrite(fname, annotated)
                print(f"💾 Saved to: {fname}")

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    app = WebCamRCNNLight("assignment_9_2_model.h5")
    app.run()
