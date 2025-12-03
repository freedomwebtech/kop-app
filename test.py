import cv2
from ultralytics import YOLO
import cvzone
import json
import os
import numpy as np
from imutils.video import VideoStream
import time

# ==========================================================
#                HSV COLOR DETECTION (Brown + White)
# ==========================================================

BROWN_LOWER = np.array([5, 80, 60])
BROWN_UPPER = np.array([20, 255, 255])

WHITE_LOWER = np.array([0, 0, 200])
WHITE_UPPER = np.array([180, 40, 255])

def detect_box_color(frame, box):
    x1, y1, x2, y2 = box
    roi = frame[y1:y2, x1:x2]

    if roi.size == 0:
        return "Unknown"

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    brown_mask = cv2.inRange(hsv, BROWN_LOWER, BROWN_UPPER)
    brown_intensity = brown_mask.mean()

    white_mask = cv2.inRange(hsv, WHITE_LOWER, WHITE_UPPER)
    white_intensity = white_mask.mean()

    if brown_intensity > 20:
        return "Brown"
    elif white_intensity > 20:
        return "White"

    return "Unknown"


# ==========================================================
#                     OBJECT COUNTER CLASS
# ==========================================================

class ObjectCounter:
    def __init__(self, source, model="best_float32.tflite",
                 classes_to_count=[0], show=True,
                 json_file="line_coords.json"):

        self.source = source
        self.model = YOLO(model)
        self.names = self.model.names
        self.classes = classes_to_count
        self.show = show

        # -------- RTSP or File --------
        if isinstance(source, str) and source.startswith("rtsp://"):
            self.cap = VideoStream(source).start()
            time.sleep(2.0)
            self.is_rtsp = True
        else:
            self.cap = cv2.VideoCapture(source)
            self.is_rtsp = False

        # -------- Tracking Data --------
        self.hist = {}
        self.last_seen = {}
        self.crossed_ids = set()
        self.counted = set()

        # -------- Counters --------
        self.in_count = 0
        self.out_count = 0

        self.color_in_count = {}
        self.color_out_count = {}

        # ✅ MISSED LOGIC ADDED
        self.missed_in = set()
        self.missed_out = set()
        self.missed_cross = set()
        self.max_missing_frames = 40

        # -------- Line --------
        self.line_p1 = None
        self.line_p2 = None
        self.temp_points = []
        self.json_file = json_file
        self.load_line()

        self.frame_count = 0

        cv2.namedWindow("ObjectCounter")
        cv2.setMouseCallback("ObjectCounter", self.mouse_event)

    # ---------------- Mouse ----------------
    def mouse_event(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.temp_points.append((x, y))
            if len(self.temp_points) == 2:
                self.line_p1, self.line_p2 = self.temp_points
                self.temp_points = []
                self.save_line()

    # ---------------- Save / Load Line ----------------
    def save_line(self):
        with open(self.json_file, "w") as f:
            json.dump({"line_p1": self.line_p1, "line_p2": self.line_p2}, f)

    def load_line(self):
        if os.path.exists(self.json_file):
            with open(self.json_file) as f:
                data = json.load(f)
                self.line_p1 = tuple(data["line_p1"])
                self.line_p2 = tuple(data["line_p2"])

    # ---------------- Utility ----------------
    def side(self, px, py, x1, y1, x2, y2):
        return (x2 - x1) * (py - y1) - (y2 - y1) * (px - x1)

    # ================= MISSED TRACK HANDLER (FROM OLD CODE) =================
    def check_lost_ids(self):
        current = self.frame_count
        lost = []

        for tid, last in self.last_seen.items():
            if current - last > self.max_missing_frames:
                lost.append(tid)

        for tid in lost:
            if tid in self.crossed_ids and tid not in self.counted:
                self.missed_cross.add(tid)

            elif tid not in self.counted and tid in self.hist:
                cx, cy = self.hist[tid]
                s = self.side(cx, cy, *self.line_p1, *self.line_p2)

                if s > 0:
                    self.missed_in.add(tid)
                else:
                    self.missed_out.add(tid)

            self.hist.pop(tid, None)
            self.last_seen.pop(tid, None)

    # ---------------- Main Loop ----------------
    def run(self):
        print("RUNNING... Press R to Reset | ESC to Exit")

        while True:
            if self.is_rtsp:
                frame = self.cap.read()
            else:
                ret, frame = self.cap.read()
                if not ret:
                    break

            self.frame_count += 1
            if self.frame_count % 3 != 0:
                continue

            frame = cv2.resize(frame, (1020, 600))

            for pt in self.temp_points:
                cv2.circle(frame, pt, 5, (0, 0, 255), -1)

            if self.line_p1:
                cv2.line(frame, self.line_p1, self.line_p2,
                         (255, 255, 255), 2)

            results = self.model.track(
                frame, persist=True, classes=self.classes, conf=0.7)

            if results[0].boxes.id is not None and self.line_p1:
                ids = results[0].boxes.id.cpu().numpy().astype(int)
                boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)

                for tid, box in zip(ids, boxes):
                    x1, y1, x2, y2 = box
                    cx = int((x1 + x2) / 2)
                    cy = int((y1 + y2) / 2)

                    self.last_seen[tid] = self.frame_count
                    color_name = detect_box_color(frame, box)

                    if tid in self.hist:
                        px, py = self.hist[tid]
                        s1 = self.side(px, py, *self.line_p1, *self.line_p2)
                        s2 = self.side(cx, cy, *self.line_p1, *self.line_p2)

                        if s1 * s2 < 0:
                            self.crossed_ids.add(tid)

                            if tid not in self.counted:
                                if s2 > 0:
                                    self.in_count += 1
                                    self.color_in_count[color_name] = self.color_in_count.get(color_name, 0) + 1
                                else:
                                    self.out_count += 1
                                    self.color_out_count[color_name] = self.color_out_count.get(color_name, 0) + 1

                                self.counted.add(tid)

                    self.hist[tid] = (cx, cy)

                    cv2.rectangle(frame, (x1, y1),
                                  (x2, y2), (0, 255, 0), 2)

                    cv2.putText(frame, color_name, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.6, (255, 200, 0), 2)

            # ✅ MISSED CHECK ACTIVE
            if self.line_p1:
                self.check_lost_ids()

            # ================= MODERN PROFESSIONAL DISPLAY PANEL =================

            # Create dark semi-transparent overlay
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (1020, 140), (20, 20, 20), -1)
            frame = cv2.addWeighted(overlay, 0.75, frame, 0.25, 0)

            # --------- TITLE BAR ---------
            cv2.rectangle(frame, (10, 10), (1010, 50), (40, 40, 40), -1)
            cv2.rectangle(frame, (10, 10), (1010, 50), (100, 200, 255), 2)
            
            cv2.putText(frame, "OBJECT TRACKING SYSTEM", (30, 38),
                        cv2.FONT_HERSHEY_DUPLEX, 0.9, (255, 255, 255), 2)
            
            # Status indicator
            cv2.circle(frame, (980, 30), 8, (0, 255, 0), -1)
            cv2.circle(frame, (980, 30), 8, (255, 255, 255), 1)

            # --------- MAIN COUNTERS ROW ---------
            y_pos = 80
            
            # IN Counter Box
            cv2.rectangle(frame, (15, 60), (180, 130), (0, 100, 0), -1)
            cv2.rectangle(frame, (15, 60), (180, 130), (0, 255, 0), 2)
            cv2.putText(frame, "IN", (70, 85),
                        cv2.FONT_HERSHEY_DUPLEX, 0.7, (200, 255, 200), 2)
            cv2.putText(frame, str(self.in_count), (70, 115),
                        cv2.FONT_HERSHEY_DUPLEX, 1.2, (255, 255, 255), 3)

            # OUT Counter Box
            cv2.rectangle(frame, (195, 60), (360, 130), (0, 0, 100), -1)
            cv2.rectangle(frame, (195, 60), (360, 130), (0, 100, 255), 2)
            cv2.putText(frame, "OUT", (255, 85),
                        cv2.FONT_HERSHEY_DUPLEX, 0.7, (200, 200, 255), 2)
            cv2.putText(frame, str(self.out_count), (250, 115),
                        cv2.FONT_HERSHEY_DUPLEX, 1.2, (255, 255, 255), 3)

            # --------- MISSED TRACKING SECTION ---------
            # Missed IN
            cv2.rectangle(frame, (375, 60), (510, 95), (60, 40, 0), -1)
            cv2.rectangle(frame, (375, 60), (510, 95), (255, 140, 0), 1)
            cv2.putText(frame, f"Miss IN: {len(self.missed_in)}", (385, 83),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 180, 100), 2)

            # Missed OUT
            cv2.rectangle(frame, (375, 100), (510, 130), (60, 0, 60), -1)
            cv2.rectangle(frame, (375, 100), (510, 130), (200, 0, 200), 1)
            cv2.putText(frame, f"Miss OUT: {len(self.missed_out)}", (385, 118),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 150, 255), 2)

            # Missed CROSS
            cv2.rectangle(frame, (520, 60), (655, 95), (60, 0, 0), -1)
            cv2.rectangle(frame, (520, 60), (655, 95), (255, 0, 0), 1)
            cv2.putText(frame, f"Miss Cross: {len(self.missed_cross)}", (530, 83),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 100, 100), 2)

            # --------- COLOR BREAKDOWN SECTION ---------
            x_start = 520
            y_color = 105
            
            # Color IN breakdown
            for idx, (color, cnt) in enumerate(self.color_in_count.items()):
                x_pos = x_start + (idx * 145)
                cv2.rectangle(frame, (x_pos, y_color), (x_pos + 140, y_color + 25), (0, 80, 0), -1)
                cv2.rectangle(frame, (x_pos, y_color), (x_pos + 140, y_color + 25), (0, 200, 0), 1)
                cv2.putText(frame, f"{color} IN: {cnt}", (x_pos + 8, y_color + 18),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 255, 150), 1)

            # Color OUT breakdown (below IN)
            y_color_out = 105
            if self.color_out_count:
                y_color_out = 105 if not self.color_in_count else 133
                
            for idx, (color, cnt) in enumerate(self.color_out_count.items()):
                x_pos = 670 + (idx * 145)
                y_use = 105 if not self.color_in_count else y_color_out
                cv2.rectangle(frame, (x_pos, y_use), (x_pos + 140, y_use + 25), (0, 0, 80), -1)
                cv2.rectangle(frame, (x_pos, y_use), (x_pos + 140, y_use + 25), (100, 100, 255), 1)
                cv2.putText(frame, f"{color} OUT: {cnt}", (x_pos + 5, y_use + 18),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 255), 1)

            if self.show:
                cv2.imshow("ObjectCounter", frame)
                key = cv2.waitKey(1) & 0xFF

                if key == ord('r'):
                    self.hist.clear()
                    self.last_seen.clear()
                    self.crossed_ids.clear()
                    self.counted.clear()
                    self.color_in_count.clear()
                    self.color_out_count.clear()
                    self.missed_in.clear()
                    self.missed_out.clear()
                    self.missed_cross.clear()
                    self.in_count = 0
                    self.out_count = 0
                    print("✅ RESET DONE")

                elif key == 27:
                    break

        if self.is_rtsp:
            self.cap.stop()
        else:
            self.cap.release()

        cv2.destroyAllWindows()
