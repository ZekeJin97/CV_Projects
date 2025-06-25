import cv2
import numpy as np
import time
import argparse

# Parse CLI arguments
parser = argparse.ArgumentParser(description="Lane Detection")
parser.add_argument("-f", "--file", type=str, help="Path to the video file")
parser.add_argument("-o", "--out", type=str, help="Output video file name")
parser.add_argument("-c", "--camera", type=int, default=0, help="Camera index if using webcam")

args = parser.parse_args()

# Auto-name output if user didn’t specify -o
if not args.out:
    if args.file:
        base = args.file.rsplit('.', 1)[0]
        args.out = f"{base}_out.mp4"
    else:
        args.out = "webcam_output.mp4"

# Video setup
vs = cv2.VideoCapture(args.file if args.file else args.camera)

time.sleep(2.0)
width, height = int(vs.get(3)), int(vs.get(4))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(args.out, fourcc, 60.0, (width, height))


# Initial state
prev_left, prev_right = None, None
frame_counter = 0
UPDATE_EVERY_N_FRAMES = 30 # reduce jitter

# Controls
paused = False
step_mode = False
show_roi = False  # ROI starts off

# ROI mask
def region_of_interest(image, display_on=None):
    height, width = image.shape[:2]
    scale_x = width / 960
    scale_y = height / 540

    bottom_left = (int(50 * scale_x), int(540 * scale_y))
    top_left = (int(30 * scale_x), int(220 * scale_y))
    top_right = (int(700 * scale_x), int(240 * scale_y))
    bottom_right = (int(800 * scale_x), int(540 * scale_y))
    polygon = np.array([[bottom_left, top_left, top_right, bottom_right]], dtype=np.int32)

    mask = np.zeros_like(image)
    if len(mask.shape) == 2:
        temp = np.zeros_like(mask)
        cv2.fillPoly(temp, [polygon], 255)
        result = cv2.bitwise_and(image, temp)
    else:
        cv2.fillPoly(mask, [polygon], 255)
        result = cv2.bitwise_and(image, mask)

    if display_on is not None and show_roi and display_on.shape[2] == 3:
        cv2.polylines(display_on, [polygon], True, (0, 255, 0), 2)

    return result

# Weighted average of lines
def average_weighted_line(lines):
    if not lines:
        return None
    total_weight = sum(length for _, length in lines)
    x1 = sum(line[0] * length for line, length in lines) / total_weight
    y1 = sum(line[1] * length for line, length in lines) / total_weight
    x2 = sum(line[2] * length for line, length in lines) / total_weight
    y2 = sum(line[3] * length for line, length in lines) / total_weight
    return int(x1), int(y1), int(x2), int(y2)

# Extend line downward
def extend_line_downward(x1, y1, x2, y2, target_y):
    if y1 > y2:
        x1, y1, x2, y2 = x2, y2, x1, y1
    dx, dy = x2 - x1, y2 - y1
    if dy == 0:
        return x1, y1, x2, y2
    slope = dx / dy
    new_x2 = int(x1 + (target_y - y1) * slope)
    return x1, y1, new_x2, target_y

# Calculate slope
def calculate_slope(line):
    x1, y1, x2, y2 = line
    return (y2 - y1) / (x2 - x1 + 1e-6)

# Main loop
prev_time = time.time()

while True:
    if paused and not step_mode:
        key = cv2.waitKey(10) & 0xFF
        if key == ord("p"):
            paused = not paused
        elif key == ord("s"):
            step_mode = True
            paused = True
        elif key == ord("r"):
            show_roi = not show_roi
        elif key == ord("q"):
            break
        continue

    ret, frame = vs.read()
    if not ret:
        break
    frame_counter += 1

    # FPS
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time)
    prev_time = curr_time

    # Preprocessing
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, lane_mask = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)
    masked = cv2.bitwise_and(gray, gray, mask=lane_mask)
    blur = cv2.GaussianBlur(masked, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))

    # ROI
    masked_edges = region_of_interest(edges, display_on=frame)

    # Detect lines every N frames
    if frame_counter % UPDATE_EVERY_N_FRAMES == 0:
        lines = cv2.HoughLinesP(masked_edges, 1, np.pi / 180, threshold=50,
                                minLineLength=100, maxLineGap=40)
        left_lines, right_lines = [], []

        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]

                if abs(x2 - x1) < 30:
                    continue   # Skip nearly vertical lines (poles, edge clutter)

                slope = calculate_slope((x1, y1, x2, y2))
                length = np.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2)

                if slope < -0.5 and x1 < width // 2 and x2 < width // 2:
                    left_lines.append(((x1, y1, x2, y2), length))
                elif slope > 0.5 and x1 > width // 2 and x2 > width // 2:
                    right_lines.append(((x1, y1, x2, y2), length))

        left_avg = average_weighted_line(left_lines)
        right_avg = average_weighted_line(right_lines)

        if left_avg is not None:
            prev_left = left_avg
        if right_avg is not None:
            prev_right = right_avg

    # Draw lanes
    ROI_BOTTOM = height
    if prev_left:
        x1, y1, x2, y2 = extend_line_downward(*prev_left, target_y=ROI_BOTTOM)
        cv2.line(frame, (x1, y1), (x2, y2), (0, 0, 255), 6)
    if prev_right:
        x1, y1, x2, y2 = extend_line_downward(*prev_right, target_y=ROI_BOTTOM)
        cv2.line(frame, (x1, y1), (x2, y2), (0, 0, 255), 6)

    # Display FPS
    cv2.putText(frame, f"FPS: {fps:.2f}", (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    out.write(frame)
    cv2.imshow("Lane Detection", frame)

    # Key handling
    wait_time = 0 if paused else 1
    key = cv2.waitKey(wait_time) & 0xFF

    if key == ord("q"):
        break
    elif key == ord("p"):
        paused = not paused
    elif key == ord("s"):
        step_mode = True
        paused = True
    elif key == ord("r"):
        show_roi = not show_roi

    if step_mode:
        step_mode = False
        paused = True

# Cleanup
vs.release()
out.release()
cv2.destroyAllWindows()
