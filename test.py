import cv2
from ultralytics import YOLO
import cvzone
import json
import os
from imutils.video import VideoStream
import time

class ObjectCounter:
    def __init__(self, source, model="yolo12n.pt", classes_to_count=[0], show=True, json_file="line_coords.json"):
        self.source = source
        self.model = YOLO(model)
        self.names = self.model.names

        self.classes = classes_to_count
        self.show = show

        # --- RTSP support ---
        if isinstance(source, str) and source.startswith("rtsp://"):
            self.cap = VideoStream(source).start()
            time.sleep(2.0)
            self.is_rtsp = True
        else:
            self.cap = cv2.VideoCapture(source)
            self.is_rtsp = False

        # --- Tracking Data ---
        self.hist = {}
        self.last_seen = {}
        self.counted = set()
        self.crossed_ids = set()

        # --- Counters ---
        self.in_count = 0
        self.out_count = 0
        self.missed_in = set()
        self.missed_out = set()
        self.missed_cross = set()

        # --- Line ---
        self.line_p1 = None
        self.line_p2 = None
        self.temp_points = []
        self.json_file = json_file
        self.load_line()

        self.frame_count = 0
        self.max_missing_frames = 40

        cv2.namedWindow("ObjectCounter")
        cv2.setMouseCallback("ObjectCounter", self.mouse_event)

    # ---------- Mouse ----------
    def mouse_event(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.temp_points.append((x, y))
            if len(self.temp_points) == 2:
                self.line_p1, self.line_p2 = self.temp_points
                self.temp_points = []
                self.save_line()

    # ---------- Save / Load Line ----------
    def save_line(self):
        with open(self.json_file, "w") as f:
            json.dump({"line_p1": self.line_p1, "line_p2": self.line_p2}, f)

    def load_line(self):
        if os.path.exists(self.json_file):
            with open(self.json_file) as f:
                data = json.load(f)
                self.line_p1 = tuple(data["line_p1"])
                self.line_p2 = tuple(data["line_p2"])

    # ---------- Utility ----------
    def side(self, px, py, x1, y1, x2, y2):
        return (x2 - x1)*(py - y1) - (y2 - y1)*(px - x1)

    # ---------- Lost Object Handler ----------
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
                side = self.side(cx, cy, *self.line_p1, *self.line_p2)

                if side > 0:
                    self.missed_in.add(tid)
                else:
                    self.missed_out.add(tid)

            self.hist.pop(tid, None)
            self.last_seen.pop(tid, None)

    # ---------- Main Loop ----------
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
                cv2.circle(frame, pt, 5, (0,0,255), -1)

            if self.line_p1:
                cv2.line(frame, self.line_p1, self.line_p2, (255,255,255), 2)

            # --- YOLO ---
            results = self.model.track(frame, persist=True, classes=self.classes, conf=0.8)

            if results[0].boxes.id is not None and self.line_p1:
                ids = results[0].boxes.id.cpu().numpy().astype(int)
                boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)

                for tid, box in zip(ids, boxes):
                    x1, y1, x2, y2 = box
                    cx, cy = int((x1+x2)/2), int((y1+y2)/2)

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
                                else:
                                    self.out_count += 1
                                self.counted.add(tid)

                    self.hist[tid] = (cx, cy)

                    cv2.circle(frame, (cx, cy), 4, (0,0,255), -1)
                    cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
                    cvzone.putTextRect(frame, f"ID:{tid}", (x1,y1), 1, 1)

            if self.line_p1:
                self.check_lost_ids()

            # ---------- DISPLAY ----------
            cvzone.putTextRect(frame, f"IN: {self.in_count}", (30,30), 2,2, colorR=(0,255,0))
            cvzone.putTextRect(frame, f"OUT: {self.out_count}", (30,80), 2,2, colorR=(0,0,255))

            cvzone.putTextRect(frame, f"MISSED IN: {len(self.missed_in)}", (30,140), 2,2, colorR=(255,165,0))
            cvzone.putTextRect(frame, f"MISSED OUT: {len(self.missed_out)}", (30,190), 2,2, colorR=(200,0,200))
            cvzone.putTextRect(frame, f"MISSED CROSS: {len(self.missed_cross)}", (30,240), 2,2, colorR=(255,0,0))

            y = 290
            for tid in list(self.missed_in)[:10]:
                cvzone.putTextRect(frame, f"MI-{tid}", (30,y), 1,1)
                y += 25

            for tid in list(self.missed_out)[:10]:
                cvzone.putTextRect(frame, f"MO-{tid}", (130,y), 1,1)
                y += 25

            for tid in list(self.missed_cross)[:10]:
                cvzone.putTextRect(frame, f"MC-{tid}", (230,y), 1,1)
                y += 25

            if self.show:
                cv2.imshow("ObjectCounter", frame)
                key = cv2.waitKey(1) & 0xFF

                if key == ord('r'):
                    print("RESET")
                    self.hist.clear()
                    self.last_seen.clear()
                    self.counted.clear()
                    self.crossed_ids.clear()
                    self.missed_in.clear()
                    self.missed_out.clear()
                    self.missed_cross.clear()
                    self.in_count = 0
                    self.out_count = 0

                elif key == 27:
                    break

        if self.is_rtsp:
            self.cap.stop()
        else:
            self.cap.release()
        cv2.destroyAllWindows()


# -------------- RUN --------------------
if __name__ == "__main__":
    source = "rtsp://username:password@ip:554/stream"
    app = ObjectCounter(source, classes_to_count=[0])
    app.run()
