import cv2
from ultralytics import YOLO
import json
import os
import numpy as np
from imutils.video import VideoStream
import time

# ==========================================================
#                HSV COLOR DETECTION
# ==========================================================
BROWN_LOWER = np.array([5, 80, 60])
BROWN_UPPER = np.array([20, 255, 255])
WHITE_LOWER = np.array([0, 0, 200])
WHITE_UPPER = np.array([180, 40, 255])

def detect_box_color(frame, box):
    x1, y1, x2, y2 = map(int, box)
    roi = frame[y1:y2, x1:x2]
    if roi.size == 0:
        return "Unknown"
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    if cv2.inRange(hsv, BROWN_LOWER, BROWN_UPPER).mean() > 20:
        return "Brown"
    elif cv2.inRange(hsv, WHITE_LOWER, WHITE_UPPER).mean() > 20:
        return "White"
    return "Unknown"


# ==========================================================
#                     OBJECT COUNTER
# ==========================================================
class ObjectCounter:
    def __init__(self, source, model="best_float32.tflite", classes_to_count=[0], json_file="line.json"):
        self.model = YOLO(model)
        self.classes = classes_to_count

        if isinstance(source, str) and source.startswith("rtsp"):
            self.cap = VideoStream(source).start()
            time.sleep(2)
            self.is_rtsp = True
        else:
            self.cap = cv2.VideoCapture(source)
            self.is_rtsp = False

        self.hist = {}
        self.last_seen = {}
        self.counted = set()

        self.in_count = 0
        self.out_count = 0
        self.color_in = {}
        self.color_out = {}

        self.missed_in = set()
        self.missed_out = set()
        self.max_missing = 40
        self.frame_count = 0

        self.line_p1 = None
        self.line_p2 = None
        self.json_file = json_file
        self.temp = []
        self.load_line()

        cv2.namedWindow("Counter")
        cv2.setMouseCallback("Counter", self.mouse_event)

    # ---------------- MOUSE ----------------
    def mouse_event(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.temp.append((x, y))
            if len(self.temp) == 2:
                self.line_p1, self.line_p2 = self.temp
                self.temp.clear()
                self.save_line()

    def save_line(self):
        with open(self.json_file, "w") as f:
            json.dump({"p1": self.line_p1, "p2": self.line_p2}, f)

    def load_line(self):
        if os.path.exists(self.json_file):
            with open(self.json_file) as f:
                d = json.load(f)
                self.line_p1 = tuple(d["p1"])
                self.line_p2 = tuple(d["p2"])

    def side(self, px, py, x1, y1, x2, y2):
        return (x2-x1)*(py-y1) - (y2-y1)*(px-x1)

    # ---------------- LOST ----------------
    def check_lost(self):
        for tid in list(self.last_seen):
            if self.frame_count - self.last_seen[tid] > self.max_missing:
                if tid not in self.counted:
                    px, py = self.hist.get(tid, (0,0))
                    if self.side(px, py, *self.line_p1) > 0:
                        self.missed_in.add(tid)
                    else:
                        self.missed_out.add(tid)

                self.last_seen.pop(tid, None)
                self.hist.pop(tid, None)

    # ---------------- RESET ----------------
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

    # ---------------- MAIN ----------------
    def run(self):
        print("▶ DRAW LINE WITH MOUSE | O=RESET | ESC=EXIT")

        while True:
            frame = self.cap.read() if self.is_rtsp else self.cap.read()[1]
            if frame is None:
                break

            frame = cv2.resize(frame, (1020, 600))
            self.frame_count += 1

            # Draw line
            if self.line_p1:
                cv2.line(frame, self.line_p1, self.line_p2, (0,255,255), 3)

            results = self.model.track(frame, persist=True, classes=self.classes, conf=0.80)

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
                            else:
                                self.out_count += 1
                                self.color_out[color] = self.color_out.get(color,0)+1
                            self.counted.add(tid)

                    self.hist[tid] = (cx, cy)

                    label = detect_box_color(frame, box)
                    cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
                    cv2.putText(frame,label,(x1,y1-10),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,255),2)

            if self.line_p1:
                self.check_lost()

            # ################ DASHBOARD ################
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (1020, 140), (0, 0, 0), -1)
            frame = cv2.addWeighted(overlay, 0.4, frame, 0.6, 0)

            # Title
            cv2.putText(frame, "OBJECT TRACKING DASHBOARD", (20, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,255), 3)

            # Total
            cv2.putText(frame, f"IN: {self.in_count}", (20, 65),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0,255,0), 3)
            cv2.putText(frame, f"OUT: {self.out_count}", (150, 65),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0,150,255), 3)

            # Missed
            cv2.putText(frame, f"MISS IN: {len(self.missed_in)}", (300, 65),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100,255,255), 2)
            cv2.putText(frame, f"MISS OUT: {len(self.missed_out)}", (460, 65),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,100,255), 2)

            # Colors
            y = 105
            for i,(c,v) in enumerate(self.color_in.items()):
                cv2.putText(frame,f"{c} IN: {v}",(20+i*200,y),
                            cv2.FONT_HERSHEY_SIMPLEX,0.6,(200,255,200),2)
            for i,(c,v) in enumerate(self.color_out.items()):
                cv2.putText(frame,f"{c} OUT: {v}",(520+i*200,y),
                            cv2.FONT_HERSHEY_SIMPLEX,0.6,(200,200,255),2)

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
        source=0,    # webcam | video.mp4 | rtsp://...
        model="best_float32.tflite",
        classes_to_count=[0]
    )
    counter.run()
