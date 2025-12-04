import cv2
from ultralytics import YOLO
import json
import os
import numpy as np
from imutils.video import VideoStream
import time
from datetime import datetime

# ==========================================================
#                HSV COLOR DETECTION
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
    elif white_mask.mean() > 20:
        return "White"
    return "Unknown"


# ==========================================================
#                     OBJECT COUNTER
# ==========================================================

class ObjectCounter:
    def __init__(self, source, model="best_float32.tflite", classes_to_count=[0], show=True, json_file="line.json"):
        self.source = source
        self.model = YOLO(model)
        self.classes = classes_to_count
        self.show = show

        if isinstance(source, str) and source.startswith("rtsp"):
            self.cap = VideoStream(source).start()
            time.sleep(2)
            self.is_rtsp = True
        else:
            self.cap = cv2.VideoCapture(source)
            self.is_rtsp = False

        self.hist = {}
        self.counted = set()
        self.last_seen = {}
        self.frame_count = 0

        self.in_count = 0
        self.out_count = 0
        self.color_in = {}
        self.color_out = {}

        # Missed tracking
        self.max_missing = 40
        self.missed_in = set()
        self.missed_out = set()

        # Line storage
        self.line_p1 = None
        self.line_p2 = None
        self.temp = []
        self.json_file = json_file
        self.load_line()

        cv2.namedWindow("Counter")
        cv2.setMouseCallback("Counter", self.mouse_event)


    # ---------------- Mouse ----------------
    def mouse_event(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.temp.append((x, y))
            if len(self.temp) == 2:
                self.line_p1, self.line_p2 = self.temp
                self.temp = []
                self.save_line()


    # ---------------- Save / Load ----------------
    def save_line(self):
        with open(self.json_file, "w") as f:
            json.dump({"p1": self.line_p1, "p2": self.line_p2}, f)

    def load_line(self):
        if os.path.exists(self.json_file):
            with open(self.json_file) as f:
                d = json.load(f)
                self.line_p1 = tuple(d["p1"])
                self.line_p2 = tuple(d["p2"])


    # ---------------- Side ----------------
    def side(self, px, py, x1, y1, x2, y2):
        return (x2-x1)*(py-y1) - (y2-y1)*(px-x1)


    # ---------------- Lost IDs ----------------
    def check_lost(self):
        lost = []
        for tid, last in self.last_seen.items():
            if self.frame_count - last > self.max_missing:
                lost.append(tid)

        for tid in lost:
            if tid not in self.counted:
                px, py = self.hist.get(tid, (0, 0))
                s = self.side(px, py, *self.line_p1)
                if s > 0:
                    self.missed_in.add(tid)
                else:
                    self.missed_out.add(tid)

            self.hist.pop(tid, None)
            self.last_seen.pop(tid, None)


    # ---------------- Reset ----------------
    def reset_all(self):
        self.hist.clear()
        self.last_seen.clear()
        self.counted.clear()

        self.in_count = 0
        self.out_count = 0
        self.color_in.clear()
        self.color_out.clear()

        self.missed_in.clear()
        self.missed_out.clear()

        print("✅ RESET DONE")


    # ---------------- Main ----------------
    def run(self):
        print("▶ Press O to Reset | ESC to Exit")

        while True:
            frame = self.cap.read() if self.is_rtsp else self.cap.read()[1]
            if frame is None:
                break

            frame = cv2.resize(frame, (1020, 600))
            self.frame_count += 1

            if self.line_p1:
                cv2.line(frame, self.line_p1, self.line_p2, (255,255,255), 2)

            results = self.model.track(frame, persist=True, classes=self.classes, conf=0.8)

            if results[0].boxes.id is not None and self.line_p1:
                ids = results[0].boxes.id.cpu().numpy().astype(int)
                boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)

                for tid, box in zip(ids, boxes):
                    x1,y1,x2,y2 = box
                    cx,cy = (x1+x2)//2, (y1+y2)//2

                    self.last_seen[tid] = self.frame_count

                    if tid in self.hist:
                        px,py = self.hist[tid]
                        s1 = self.side(px,py,*self.line_p1)
                        s2 = self.side(cx,cy,*self.line_p1)

                        if s1 * s2 < 0 and tid not in self.counted:
                            color = detect_box_color(frame, box)
                            if s2 > 0:
                                self.in_count += 1
                                self.color_in[color] = self.color_in.get(color,0)+1
                                print(f"IN {tid} {color}")
                            else:
                                self.out_count += 1
                                self.color_out[color] = self.color_out.get(color,0)+1
                                print(f"OUT {tid} {color}")
                            self.counted.add(tid)

                    self.hist[tid] = (cx,cy)

                    label = detect_box_color(frame, box)
                    cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
                    cv2.putText(frame,label,(x1,y1-10),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,255),2)

            if self.line_p1:
                self.check_lost()

            # --- UI Overlay ---
            cv2.rectangle(frame,(0,0),(400,80),(0,0,0),-1)
            cv2.putText(frame,f"IN: {self.in_count}",(10,25),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,255,0),2)
            cv2.putText(frame,f"OUT: {self.out_count}",(10,55),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,150,255),2)

            cv2.imshow("Counter", frame)
            key = cv2.waitKey(1) & 0xFF

            if key in [ord('o'), ord('O')]:
                self.reset_all()
            elif key == 27:
                break

        self.cap.stop() if self.is_rtsp else self.cap.release()
        cv2.destroyAllWindows()


# ==========================================================
#                        RUN
# ==========================================================
if __name__ == "__main__":
    counter = ObjectCounter(
        source=0,  # webcam | video.mp4 | rtsp://...
        model="best_float32.tflite",
        classes_to_count=[0],
        show=True
    )
    counter.run()
