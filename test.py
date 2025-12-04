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

            # ================= BEAUTIFUL TRANSPARENT DISPLAY =================

            # Transparent dark overlay for readability
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (1020, 80), (0, 0, 0), -1)
            frame = cv2.addWeighted(overlay, 0.25, frame, 0.75, 0)

            # --------- TITLE BAR ---------
            cv2.putText(frame, "TRACKING SYSTEM", (15, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 200, 255), 2)
            cv2.circle(frame, (210, 18), 5, (0, 255, 0), -1)

            # --------- MAIN COUNTERS ROW ---------
            y_pos = 55
            x_start = 15
            gap = 95
            font_size = 0.55
            
            # TOTAL IN
            cv2.putText(frame, "IN:", (x_start, y_pos),
                        cv2.FONT_HERSHEY_SIMPLEX, font_size, (0, 255, 150), 2)
            cv2.putText(frame, str(self.in_count), (x_start + 35, y_pos),
                        cv2.FONT_HERSHEY_SIMPLEX, font_size, (255, 255, 255), 2)

            # TOTAL OUT
            x_out = x_start + gap
            cv2.putText(frame, "OUT:", (x_out, y_pos),
                        cv2.FONT_HERSHEY_SIMPLEX, font_size, (100, 180, 255), 2)
            cv2.putText(frame, str(self.out_count), (x_out + 50, y_pos),
                        cv2.FONT_HERSHEY_SIMPLEX, font_size, (255, 255, 255), 2)

            # MISSED IN
            x_miss_in = x_out + gap + 10
            cv2.putText(frame, "MISS IN:", (x_miss_in, y_pos),
                        cv2.FONT_HERSHEY_SIMPLEX, font_size, (100, 255, 255), 2)
            cv2.putText(frame, str(len(self.missed_in)), (x_miss_in + 90, y_pos),
                        cv2.FONT_HERSHEY_SIMPLEX, font_size, (255, 255, 255), 2)

            # MISSED OUT
            x_miss_out = x_miss_in + gap + 25
            cv2.putText(frame, "MISS OUT:", (x_miss_out, y_pos),
                        cv2.FONT_HERSHEY_SIMPLEX, font_size, (255, 100, 255), 2)
            cv2.putText(frame, str(len(self.missed_out)), (x_miss_out + 105, y_pos),
                        cv2.FONT_HERSHEY_SIMPLEX, font_size, (255, 255, 255), 2)

            # MISS CROSS
            x_cross = x_miss_out + gap + 35
            cv2.putText(frame, "CROSS:", (x_cross, y_pos),
                        cv2.FONT_HERSHEY_SIMPLEX, font_size, (100, 100, 255), 2)
            cv2.putText(frame, str(len(self.missed_cross)), (x_cross + 80, y_pos),
                        cv2.FONT_HERSHEY_SIMPLEX, font_size, (255, 255, 255), 2)

            # --------- COLOR BREAKDOWN ---------
            x_color_start = 15
            y_color = 75
            color_gap = 120
            
            for idx, (color, cnt) in enumerate(self.color_in_count.items()):
                x_pos = x_color_start + (idx * color_gap)
                cv2.putText(frame, f"{color} IN: {cnt}", (x_pos, y_color),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 255, 150), 1)

            for idx, (color, cnt) in enumerate(self.color_out_count.items()):
                x_pos = x_color_start + (idx * color_gap) + 250
                cv2.putText(frame, f"{color} OUT: {cnt}", (x_pos, y_color),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 255), 1)

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


# ==========================================================
#                        MAIN EXECUTION
# ==========================================================

if __name__ == "__main__":
    # Example usage:
    counter = ObjectCounter(
        source="your_video.mp4",  # or 0 for webcam, or "rtsp://..."
        model="best_float32.tflite",
        classes_to_count=[0],
        show=True
    )
    counter.run()
