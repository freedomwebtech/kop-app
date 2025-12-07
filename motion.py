import cv2
import numpy as np
from ultralytics import YOLO

# Load YOLO
model = YOLO("model.pt")

# Video input
cap = cv2.VideoCapture("video.mp4")

# Background subtractor (MOTION)
fgbg = cv2.createBackgroundSubtractorMOG2(history=300, varThreshold=40)

# Line position
line_y = 300

# Track memory
track_history = {}
track_id = 0
up_count = 0
down_count = 0

def get_center(x1, y1, x2, y2):
    return int((x1 + x2) / 2), int((y1 + y2) / 2)

# Ignore small objects
MIN_AREA = 1200

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape

    # ----------------------------
    # ✅ YOLO DETECTION
    # ----------------------------
    yolo_boxes = []
    results = model(frame, conf=0.4)[0]

    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        area = (x2 - x1) * (y2 - y1)

        if area > MIN_AREA:
            yolo_boxes.append((x1, y1, x2, y2))

    # ----------------------------
    # ✅ MOTION DETECTION
    # ----------------------------
    fgmask = fgbg.apply(frame)
    _, fgmask = cv2.threshold(fgmask, 200, 255, cv2.THRESH_BINARY)

    kernel = np.ones((5, 5), np.uint8)
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    motion_boxes = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > MIN_AREA:
            x, y, w1, h1 = cv2.boundingRect(cnt)
            motion_boxes.append((x, y, x + w1, y + h1))

    # ----------------------------
    # ✅ COMBINE YOLO + MOTION
    # ----------------------------
    all_boxes = yolo_boxes + motion_boxes

    for box in all_boxes:
        x1, y1, x2, y2 = box
        cx, cy = get_center(x1, y1, x2, y2)

        assigned = False

        for tid in list(track_history.keys()):
            prev = track_history[tid]
            px, py = prev

            if abs(px - cx) < 40 and abs(py - cy) < 40:
                track_history[tid] = (cx, cy)

                # ✅ UP / DOWN LOGIC
                if py < line_y and cy > line_y:
                    down_count += 1
                if py > line_y and cy < line_y:
                    up_count += 1

                assigned = True
                label_id = tid
                break

        # New object
        if not assigned:
            track_history[track_id] = (cx, cy)
            label_id = track_id
            track_id += 1

        # ✅ Draw box
        cv2.rectangle(frame, (x1,y1), (x2,y2), (255,255,0), 2)
        cv2.putText(frame, f"ID {label_id}", (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,0), 2)

        # Draw center
        cv2.circle(frame, (cx, cy), 4, (0,0,255), -1)

    # ✅ Draw counting line
    cv2.line(frame, (0, line_y), (w, line_y), (0,255,0), 3)

    # ✅ Show counters
    cv2.putText(frame, f"UP: {up_count}", (30,40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)

    cv2.putText(frame, f"DOWN: {down_count}", (30,80),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)

    cv2.imshow("YOLO + Motion Tracking", frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
