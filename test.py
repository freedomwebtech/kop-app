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

            # ================= MINIMAL COMPACT TOP DISPLAY =================

            # Ultra light transparent overlay
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (1020, 65), (10, 10, 10), -1)
            frame = cv2.addWeighted(overlay, 0.35, frame, 0.65, 0)

            # --------- TITLE BAR (Compact) ---------
            cv2.rectangle(frame, (5, 5), (250, 28), (25, 25, 25), -1)
            cv2.rectangle(frame, (5, 5), (250, 28), (70, 160, 255), 1)
            cv2.putText(frame, "TRACKING SYSTEM", (15, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.circle(frame, (240, 16), 4, (0, 255, 0), -1)

            # --------- COUNTERS ROW (Minimal) ---------
            y_top = 35
            y_bottom = 60
            box_width = 85
            gap = 8
            
            # TOTAL IN
            x1 = 5
            cv2.rectangle(frame, (x1, y_top), (x1 + box_width, y_bottom), (0, 70, 0), -1)
            cv2.rectangle(frame, (x1, y_top), (x1 + box_width, y_bottom), (0, 255, 0), 1)
            cv2.putText(frame, "IN", (x1 + 8, y_top + 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 255, 180), 1)
            cv2.putText(frame, str(self.in_count), (x1 + 30, y_top + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # TOTAL OUT
            x2 = x1 + box_width + gap
            cv2.rectangle(frame, (x2, y_top), (x2 + box_width, y_bottom), (0, 0, 70), -1)
            cv2.rectangle(frame, (x2, y_top), (x2 + box_width, y_bottom), (0, 120, 255), 1)
            cv2.putText(frame, "OUT", (x2 + 8, y_top + 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 200, 255), 1)
            cv2.putText(frame, str(self.out_count), (x2 + 30, y_top + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # MISSED IN
            x3 = x2 + box_width + gap
            cv2.rectangle(frame, (x3, y_top), (x3 + box_width, y_bottom), (45, 30, 0), -1)
            cv2.rectangle(frame, (x3, y_top), (x3 + box_width, y_bottom), (255, 150, 0), 1)
            cv2.putText(frame, "Miss IN", (x3 + 8, y_top + 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 200, 120), 1)
            cv2.putText(frame, str(len(self.missed_in)), (x3 + 30, y_top + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # MISSED OUT
            x4 = x3 + box_width + gap
            cv2.rectangle(frame, (x4, y_top), (x4 + box_width, y_bottom), (45, 0, 45), -1)
            cv2.rectangle(frame, (x4, y_top), (x4 + box_width, y_bottom), (220, 0, 220), 1)
            cv2.putText(frame, "Miss OUT", (x4 + 5, y_top + 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 180, 255), 1)
            cv2.putText(frame, str(len(self.missed_out)), (x4 + 30, y_top + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # MISS CROSS
            x5 = x4 + box_width + gap
            cv2.rectangle(frame, (x5, y_top), (x5 + box_width, y_bottom), (45, 0, 0), -1)
            cv2.rectangle(frame, (x5, y_top), (x5 + box_width, y_bottom), (255, 0, 0), 1)
            cv2.putText(frame, "Cross", (x5 + 8, y_top + 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 120, 120), 1)
            cv2.putText(frame, str(len(self.missed_cross)), (x5 + 30, y_top + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # --------- COLOR BREAKDOWN (Ultra Compact) ---------
            x_color = x5 + box_width + gap + 3
            
            for idx, (color, cnt) in enumerate(self.color_in_count.items()):
                x_pos = x_color + (idx * 75)
                cv2.rectangle(frame, (x_pos, y_top), (x_pos + 72, y_top + 11), (0, 60, 0), -1)
                cv2.rectangle(frame, (x_pos, y_top), (x_pos + 72, y_top + 11), (0, 200, 0), 1)
                cv2.putText(frame, f"{color} IN:{cnt}", (x_pos + 3, y_top + 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.3, (180, 255, 180), 1)

            for idx, (color, cnt) in enumerate(self.color_out_count.items()):
                x_pos = x_color + (idx * 75)
                cv2.rectangle(frame, (x_pos, y_top + 14), (x_pos + 72, y_bottom), (0, 0, 60), -1)
                cv2.rectangle(frame, (x_pos, y_top + 14), (x_pos + 72, y_bottom), (120, 120, 255), 1)
                cv2.putText(frame, f"{color} OUT:{cnt}", (x_pos + 3, y_top + 22),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.3, (200, 200, 255), 1)

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
