import cv2
import numpy as np

# Load video from webcam
cap = cv2.VideoCapture(0)  # 0 is the default webcam; use 1, 2, etc., for additional cameras

# Initialize variables
roi_selected = False
old_points = None
mask = None

# Loop to capture frames and display the video
while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if roi_selected:
        # Calculate optical flow
        new_points, status, error = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, old_points, None, **lk_params)

        # Check if new_points is None
        if new_points is None:
            print("Tracking lost.")
            break

        # Select good points
        good_new = new_points[status == 1]
        good_old = old_points[status == 1]

        # Draw the tracks
        for i, (new, old) in enumerate(zip(good_new, good_old)):
            a, b = new.ravel()
            c, d = old.ravel()
            # Convert coordinates to integers
            a, b, c, d = int(a), int(b), int(c), int(d)
            mask = cv2.line(mask, (a, b), (c, d), color=(0, 255, 0), thickness=2)
            frame = cv2.circle(frame, (a, b), 5, color=(0, 0, 255), thickness=-1)

        img = cv2.add(frame, mask)

        # Show the frame with the tracked points
        cv2.imshow('Object Tracking', img)

        # Update the previous frame and previous points
        old_gray = frame_gray.copy()
        old_points = good_new.reshape(-1, 1, 2)
    else:
        # Show the frame without tracking
        cv2.imshow('Object Tracking', frame)

    # Wait for 's' key to select ROI
    key = cv2.waitKey(30) & 0xFF
    if key == ord('s') and not roi_selected:
        # Select ROI manually
        roi = cv2.selectROI("Select ROI", frame, fromCenter=False, showCrosshair=True)
        x, y, w, h = roi
        old_points = np.array([[x + w // 2, y + h // 2]], dtype=np.float32).reshape(-1, 1, 2)

        cv2.destroyWindow("Select ROI")

        # Parameters for Lucas-Kanade optical flow
        lk_params = dict(winSize=(15, 15),
                         maxLevel=2,
                         criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

        # Initialize mask for drawing
        mask = np.zeros_like(frame)

        # Save the initial grayscale frame
        old_gray = frame_gray.copy()

        # Set flag to indicate ROI has been selected
        roi_selected = True

    # Break the loop on 'q' key press
    if key == ord('q'):
        break

# Release the capture and close windows
cap.release()
cv2.destroyAllWindows()
