import cv2
from ultralytics import YOLO
import json
import os
import numpy as np
from imutils.video import VideoStream
import time
from datetime import datetime

# ==========================================================
#                     OBJECT COUNTER CLASS
# ==========================================================

class ObjectCounter:
    def __init__(self, source, model="best_float32.tflite",
                 classes_to_count=[0], show=True,
                 json_file="line_coords_back.json"):

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

        # -------- Session Data --------
        self.session_start_time = datetime.now()
        self.current_session_data = None
        self.start_new_session()

        # -------- Tracking Data --------
        self.hist = {}
        self.last_seen = {}
        self.crossed_ids = set()
        self.counted = set()

        # -------- Counters --------
        self.in_count = 0
        self.out_count = 0

        # -------- MISSED LOGIC --------
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

    # ---------------- Session Management ----------------
    def start_new_session(self):
        self.current_session_data = {
            'day': datetime.now().strftime('%A'),
            'date': datetime.now().strftime('%Y-%m-%d'),
            'start_time': datetime.now().strftime('%H:%M:%S'),
            'end_time': None,
            'in_count': 0,
            'out_count': 0,
            'missed_in': 0,
            'missed_out': 0,
            'missed_cross': 0
        }

    def end_current_session(self):
        if self.current_session_data:
            self.current_session_data['end_time'] = datetime.now().strftime('%H:%M:%S')
            self.current_session_data['in_count'] = self.in_count
            self.current_session_data['out_count'] = self.out_count
            self.current_session_data['missed_in'] = len(self.missed_in)
            self.current_session_data['missed_out'] = len(self.missed_out)
            self.current_session_data['missed_cross'] = len(self.missed_cross)

    def print_session_summary(self):
        print("\n" + "=" * 80)
        print("SESSION SUMMARY")
        print("=" * 80)
        for k, v in self.current_session_data.items():
            print(f"{k.replace('_',' ').title():15}: {v}")
        print("=" * 80 + "\n")

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

    # ---------------- UTILS ----------------
    def side(self, px, py, x1, y1, x2, y2):
        return (x2 - x1) * (py - y1) - (y2 - y1) * (px - x1)

    # ✅ -------- APPLY MASK (ONLY LINE REGION) --------
    def apply_line_mask(self, frame, thickness=160):
        if self.line_p1 is None or self.line_p2 is None:
            return frame

        mask = np.zeros(frame.shape[:2], dtype=np.uint8)

        cv2.line(mask, self.line_p1, self.line_p2, 255, thickness)

        masked = cv2.bitwise_and(frame, frame, mask=mask)

        return masked

    # ---------------- MISSED HANDLER ----------------
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
                if s < 0:
                    self.missed_in.add(tid)
                else:
                    self.missed_out.add(tid)

            self.hist.pop(tid, None)
            self.last_seen.pop(tid, None)

    # ---------------- RESET ----------------
    def reset_all_data(self):
        self.end_current_session()
        self.print_session_summary()

        self.hist.clear()
        self.last_seen.clear()
        self.crossed_ids.clear()
        self.counted.clear()
        self.missed_in.clear()
        self.missed_out.clear()
        self.missed_cross.clear()
        self.in_count = 0
        self.out_count = 0

        self.start_new_session()
        print("✅ RESET DONE - New session started")

    # ---------------- MAIN LOOP ----------------
    def run(self):
        print("Press O = Reset | ESC = Exit")

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
                cv2.circle(frame, pt, 6, (0, 0, 255), -1)

            if self.line_p1:
                cv2.line(frame, self.line_p1, self.line_p2, (0,255,255), 3)

            # ✅ MASKED FRAME FOR DETECTION ONLY NEAR LINE
            detection_frame = self.apply_line_mask(frame)

            results = self.model.track(
                detection_frame, persist=True, classes=self.classes, conf=0.80
            )

            if results[0].boxes.id is not None and self.line_p1:
                ids = results[0].boxes.id.cpu().numpy().astype(int)
                boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)

                for tid, box in zip(ids, boxes):
                    x1, y1, x2, y2 = box
                    cx = int((x1 + x2) / 2)
                    cy = int((y1 + y2) / 2)

                    self.last_seen[tid] = self.frame_count

                    if tid in self.hist:
                        px, py = self.hist[tid]
                        s1 = self.side(px, py, *self.line_p1, *self.line_p2)
                        s2 = self.side(cx, cy, *self.line_p1, *self.line_p2)

                        if s1 * s2 < 0:
                            self.crossed_ids.add(tid)

                            if tid not in self.counted:
                                if s2 > 0:
                                    self.in_count += 1
                                    print(f"✅ IN ID {tid}")
                                else:
                                    self.out_count += 1
                                    print(f"✅ OUT ID {tid}")
                                self.counted.add(tid)

                    self.hist[tid] = (cx, cy)

                    # Draw box only for valid detections
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)

            if self.line_p1:
                self.check_lost_ids()

            # ------------ DASHBOARD DISPLAY ---------------
            overlay = frame.copy()
            cv2.rectangle(overlay, (0,0), (1020,90), (0,0,0), -1)
            frame = cv2.addWeighted(overlay, 0.5, frame, 0.5, 0)

            cv2.putText(frame, "LINE BASED OBJECT COUNTING", (20,30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,200,0), 2)

            cv2.putText(frame, f"IN: {self.in_count}", (20,70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

            cv2.putText(frame, f"OUT: {self.out_count}", (150,70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,200,255), 2)

            cv2.putText(frame, f"MISSED IN: {len(self.missed_in)}", (320,70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2)

            cv2.putText(frame, f"MISSED OUT: {len(self.missed_out)}", (520,70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,100,255), 2)

            cv2.putText(frame, f"CROSS LOST: {len(self.missed_cross)}", (760,70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100,100,255), 2)

            if self.show:
                cv2.imshow("ObjectCounter", frame)
               # cv2.imshow("Detection Mask", detection_frame)

                key = cv2.waitKey(1) & 0xFF

                if key == ord('o') or key == ord('O'):
                    self.reset_all_data()

                elif key == 27:
                    break

        self.end_current_session()
        self.print_session_summary()

        if self.is_rtsp:
            self.cap.stop()
        else:
            self.cap.release()

        cv2.destroyAllWindows()


# ==========================================================
#                        MAIN
# ==========================================================
if __name__ == "__main__":

    counter = ObjectCounter(
        source="your_video.mp4",   # or 0 or "rtsp://..."
        model="best_float32.tflite",
        classes_to_count=[0],
        show=True
    )

    counter.run()
