import cv2
from ultralytics import YOLO
import json
import os
import numpy as np
from imutils.video import VideoStream
import time
from datetime import datetime

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
    white_mask = cv2.inRange(hsv, WHITE_LOWER, WHITE_UPPER)

    if brown_mask.mean() > 20:
        return "Brown"
    if white_mask.mean() > 20:
        return "White"
    return "Unknown"


# ==========================================================
#             STEP-3 TIME MEMORY CONFIG
# ==========================================================

CROSS_TOLERANCE = 18
MEMORY_FRAMES = 6


# ==========================================================
#                     OBJECT COUNTER CLASS
# ==========================================================

class ObjectCounter:
    def __init__(self, source, model="best_float32.tflite",
                 classes_to_count=[0], show=True,
                 json_file="line_coords_front.json"):

        self.source = source
        self.model = YOLO(model)
        self.names = self.model.names
        self.classes = classes_to_count
        self.show = show

        if isinstance(source, str) and source.startswith("rtsp://"):
            self.cap = VideoStream(source).start()
            time.sleep(2.0)
            self.is_rtsp = True
        else:
            self.cap = cv2.VideoCapture(source)
            self.is_rtsp = False

        # -------- SESSION --------
        self.session_start_time = datetime.now()
        self.current_session_data = None
        self.start_new_session()

        # -------- TRACKING --------
        self.hist = {}
        self.last_seen = {}
        self.crossed_ids = set()
        self.counted = set()
        self.color_at_crossing = {}
        self.origin_side = {}

        # ----- MEMORY SYSTEM -----
        self.near_line_memory = {}
        self.missed_in = set()
        self.missed_out = set()

        # -------- COUNTERS --------
        self.in_count = 0
        self.out_count = 0
        self.color_in_count = {}
        self.color_out_count = {}

        self.missed_cross = set()
        self.max_missing_frames = 40

        # -------- LINE --------
        self.line_p1 = None
        self.line_p2 = None
        self.temp_points = []
        self.json_file = json_file
        self.load_line()

        self.frame_count = 0

        cv2.namedWindow("ObjectCounter")
        cv2.setMouseCallback("ObjectCounter", self.mouse_event)

    # ---------------- Session ----------------
    def start_new_session(self):
        self.current_session_data = {
            'day': datetime.now().strftime('%A'),
            'date': datetime.now().strftime('%Y-%m-%d'),
            'start_time': datetime.now().strftime('%H:%M:%S'),
            'end_time': None,
            'in_count': 0,
            'out_count': 0,
            'missed_cross': 0,
            'missed_in': 0,
            'missed_out': 0,
            'color_in': {},
            'color_out': {}
        }

    def end_current_session(self):
        self.current_session_data['end_time'] = datetime.now().strftime('%H:%M:%S')
        self.current_session_data['in_count'] = self.in_count
        self.current_session_data['out_count'] = self.out_count
        self.current_session_data['missed_cross'] = len(self.missed_cross)
        self.current_session_data['missed_in'] = len(self.missed_in)
        self.current_session_data['missed_out'] = len(self.missed_out)
        self.current_session_data['color_in'] = dict(self.color_in_count)
        self.current_session_data['color_out'] = dict(self.color_out_count)

    def print_session_summary(self):
        print("\n" + "=" * 80)
        print("                    SESSION SUMMARY")
        print("=" * 80)
        for k, v in self.current_session_data.items():
            print(f"{k:15}: {v}")
        print("=" * 80 + "\n")

    # ---------------- Mouse ----------------
    def mouse_event(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.temp_points.append((x, y))
            if len(self.temp_points) == 2:
                self.line_p1, self.line_p2 = self.temp_points
                self.temp_points = []
                self.save_line()

    # ---------------- Save / Load ----------------
    def save_line(self):
        with open(self.json_file, "w") as f:
            json.dump({"line_p1": self.line_p1, "line_p2": self.line_p2}, f)

    def load_line(self):
        if os.path.exists(self.json_file):
            with open(self.json_file) as f:
                data = json.load(f)
                self.line_p1 = tuple(data["line_p1"])
                self.line_p2 = tuple(data["line_p2"])

    # ---------------- Geometry ----------------
    def side(self, px, py, x1, y1, x2, y2):
        return (x2 - x1) * (py - y1) - (y2 - y1) * (px - x1)

    def point_line_distance(self, px, py, x1, y1, x2, y2):
        num = abs((y2-y1)*px - (x2-x1)*py + x2*y1 - y2*x1)
        den = np.hypot(y2-y1, x2-x1)
        return num / den if den else 999

    # ---------------- LOST CHECK ----------------
    def check_lost_ids(self):
        for tid, last in list(self.last_seen.items()):
            if self.frame_count - last > self.max_missing_frames:
                if tid not in self.counted and tid not in self.crossed_ids:
                    self.missed_cross.add(tid)
                    print(f"⚠️ MISSED CROSS - ID:{tid}")

                self.hist.pop(tid, None)
                self.last_seen.pop(tid, None)
                self.color_at_crossing.pop(tid, None)
                self.origin_side.pop(tid, None)
                self.near_line_memory.pop(tid, None)

    # ---------------- RESET ----------------
    def reset_all_data(self):
        self.end_current_session()
        self.print_session_summary()

        self.hist.clear()
        self.last_seen.clear()
        self.crossed_ids.clear()
        self.counted.clear()
        self.color_at_crossing.clear()
        self.origin_side.clear()
        self.color_in_count.clear()
        self.color_out_count.clear()
        self.missed_cross.clear()
        self.missed_in.clear()
        self.missed_out.clear()
        self.near_line_memory.clear()

        self.in_count = 0
        self.out_count = 0

        self.start_new_session()
        print("✅ RESET DONE")

    # ---------------- MAIN LOOP ----------------
    def run(self):
        print("RUNNING... Press O to Reset | ESC to Exit")

        while True:
            frame = self.cap.read() if self.is_rtsp else self.cap.read()[1]
            if frame is None:
                break

            self.frame_count += 1
            if self.frame_count % 3 != 0:
                continue

            frame = cv2.resize(frame, (1020, 600))

            if self.line_p1:
                cv2.line(frame, self.line_p1, self.line_p2, (255, 255, 255), 2)

            results = self.model.track(frame, persist=True, classes=self.classes, conf=0.80)

            if results[0].boxes.id is not None and self.line_p1:
                ids = results[0].boxes.id.cpu().numpy().astype(int)
                boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)

                for tid, box in zip(ids, boxes):
                    x1, y1, x2, y2 = box
                    cx = (x1+x2)//2
                    cy = (y1+y2)//2

                    self.last_seen[tid] = self.frame_count

                    if tid not in self.hist:
                        side_init = self.side(cx, cy, *self.line_p1, *self.line_p2)
                        self.origin_side[tid] = "IN" if side_init < 0 else "OUT"

                    px, py = self.hist.get(tid, (cx, cy))
                    s1 = self.side(px, py, *self.line_p1, *self.line_p2)
                    s2 = self.side(cx, cy, *self.line_p1, *self.line_p2)
                    dist = self.point_line_distance(cx, cy, *self.line_p1, *self.line_p2)

                    normal = s1 * s2 < 0
                    soft = abs(s2) < CROSS_TOLERANCE and dist < CROSS_TOLERANCE

                    if soft:
                        self.near_line_memory[tid] = (s2, self.frame_count)

                    failed = False
                    if tid in self.near_line_memory:
                        old_side, old_frame = self.near_line_memory[tid]
                        if self.frame_count - old_frame <= MEMORY_FRAMES:
                            if old_side * s2 < 0:
                                failed = True
                                del self.near_line_memory[tid]

                    if normal or failed:
                        if tid not in self.counted:
                            self.crossed_ids.add(tid)
                            direction = "IN" if s2 > 0 else "OUT"

                            if normal:
                                color = detect_box_color(frame, box)
                                if direction == "IN":
                                    self.in_count += 1
                                    self.color_in_count[color] = self.color_in_count.get(color, 0) + 1
                                else:
                                    self.out_count += 1
                                    self.color_out_count[color] = self.color_out_count.get(color, 0) + 1
                                print(f"✅ {direction} - ID:{tid}")

                            else:
                                if direction == "IN":
                                    self.missed_in.add(tid)
                                    print(f"⚠️ MISSED IN - ID:{tid}")
                                else:
                                    self.missed_out.add(tid)
                                    print(f"⚠️ MISSED OUT - ID:{tid}")

                            self.counted.add(tid)

                    if s2 < 0:
                        self.color_at_crossing[tid] = detect_box_color(frame, box)

                    self.hist[tid] = (cx, cy)

                    label = self.color_at_crossing.get(tid, detect_box_color(frame, box))
                    cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
                    cv2.putText(frame,label,(x1,y1-5),
                                cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,255),2)

            self.check_lost_ids()
            self.draw_dashboard(frame)

            if self.show:
                cv2.imshow("ObjectCounter", frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('o'):
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

    # ---------------- DASHBOARD ----------------
    def draw_dashboard(self, frame):
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (1020, 180), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.35, frame, 0.65, 0)

        cv2.putText(frame, f"IN: {self.in_count}", (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)
        cv2.putText(frame, f"OUT: {self.out_count}", (150, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,100,255), 2)

        cv2.putText(frame, f"MISSED IN: {len(self.missed_in)}", (350, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
        cv2.putText(frame, f"MISSED OUT: {len(self.missed_out)}", (550, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100,255,255), 2)

        cv2.putText(frame, f"MISSED CROSS: {len(self.missed_cross)}", (780, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)


# ==========================================================
#                        MAIN
# ==========================================================
if __name__ == "__main__":
    counter = ObjectCounter(
        source="your_video.mp4",
        model="best_float32.tflite",
        classes_to_count=[0],
        show=True
    )
    counter.run()
