# USAGE: python WebCamSave.py -f video_file_name -o out_video.avi
import pickle

import cv2
import numpy as np
import time
import argparse
import tensorflow as tf
from tensorflow.keras import layers, models
from Constants import IMG_SIZE, RESOLUTION

# Constants
with open("label_binarizer.pkl", "rb") as f:
    lb = pickle.load(f)
class_names = lb.classes_

# Set up argument parser
parser = argparse.ArgumentParser(description="Video file path or camera input")
parser.add_argument("-f", "--file", type=str, help="Path to the video file")
parser.add_argument("-o", "--out", type=str, help="Output video file name")
args = parser.parse_args()

# Load model
model = tf.keras.models.load_model("cnn_model.keras")

# Capture source
vs = cv2.VideoCapture(args.file if args.file else 0)
time.sleep(2.0)

# Get video properties
width = int(vs.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(vs.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Setup output writer
out = None
if args.out:
    out_filename = args.out
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(out_filename, fourcc, 20.0, (width, height), True)

start_time = time.time()
frame_count = 0

try:
    while True:
        ret, frame = vs.read()
        if not ret:
            print("Failed to grab frame")
            break

        # Preprocess frame
        input_frame = cv2.resize(frame, IMG_SIZE)
        input_frame = cv2.cvtColor(input_frame, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
        input_frame = input_frame.astype(np.float32) / 255.0
        input_frame = np.expand_dims(input_frame, axis=0)

        # Update frame count
        frame_count += 1
        if frame_count % 3 != 0:
            continue

        # Predict
        predictions = model.predict(input_frame, verbose=0)[0]  # Shape: (num_classes,)
        predicted_index = np.argmax(predictions)
        predicted_class = class_names[predicted_index]
        confidence = predictions[predicted_index]

        elapsed_time = time.time() - start_time
        fps = frame_count / elapsed_time if elapsed_time > 0 else 0

        # Draw all class confidences
        start_y = 60  # Starting y-coordinate for class list
        for i, (cls, prob) in enumerate(zip(class_names, predictions)):
            color = (0, 255, 0) if i == predicted_index else (255, 255, 255)
            fontScale = 1 if i == predicted_index else 0.8
            label = f"{cls}: {prob:.2f}"
            cv2.putText(frame, label, (10, start_y + i * 30), cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale, color, 2)

        fps_label = f"FPS: {fps:.2f}"

        cv2.putText(frame, fps_label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 255, 255), 2)

        # Show frame
        cv2.imshow("Live Classification", frame)

        if out is not None:
            out.write(frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("Interrupted by user.")

finally:
    vs.release()

    if out is not None:
        out.release()

    cv2.destroyAllWindows()

