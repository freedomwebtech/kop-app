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

        # ---- RTSP ----
        if isinstance(source, str) and source.startswith("rtsp://"):
            self.cap = VideoStream(source).start()
            time.sleep(2.0)
            self.is_rtsp = True
        else:
            self.cap = cv2.VideoCapture(source)
            self.is_rtsp = False

        # ---- counters ----
        self.hist = {}
        self.counted = set()
        self.track_info = {}

        self.in_count = 0
        self.out_count = 0
        self.missed_in = 0
        self.missed_out = 0

        self.max_missing_frames = 30
        self.frame_count = 0

        # ---- line storage ----
        self.line_p1 = None
        self.line_p2 = None
        self.temp_points = []
        self.json_file = json_file
        self.load_line()

        cv2.namedWindow("ObjectCounter")
        cv2.setMouseCallback("ObjectCounter", self.mouse_event)

    # ---------------- Mouse ----------------
    def mouse_event(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.temp_points.append((x, y))
            print(f"Point selected: {x},{y}")
            if len(self.temp_points) == 2:
                self.line_p1, self.line_p2 = self.temp_points
                self.temp_points = []
                self.save_line()
                print(f"Line set: {self.line_p1},{self.line_p2}")

    # ---------------- Save/Load ----------------
    def save_line(self):
        with open(self.json_file, "w") as f:
            json.dump({"line_p1": self.line_p1, "line_p2": self.line_p2}, f)

    def load_line(self):
        if os.path.exists(self.json_file):
            with open(self.json_file, "r") as f:
                data = json.load(f)
                self.line_p1 = tuple(data["line_p1"])
                self.line_p2 = tuple(data["line_p2"])
                print("Loaded saved line:", self.line_p1, self.line_p2)

    # ---------------- Side Check ----------------
    def side(self, px, py, x1, y1, x2, y2):
        return (x2 - x1)*(py - y1) - (y2 - y1)*(px - x1)

    # ---------------- Main Loop ----------------
    def run(self):
        print("Starting Object Counter...")
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

            # temp points
            for pt in self.temp_points:
                cv2.circle(frame, pt, 5, (0,0,255), -1)

            if self.line_p1 and self.line_p2:
                cv2.line(frame, self.line_p1, self.line_p2, (255,255,255), 2)

            results = self.model.track(frame, persist=True, classes=self.classes, conf=0.8)

            visible_ids = set()

            if results[0].boxes.id is not None and self.line_p1 and self.line_p2:

                ids = results[0].boxes.id.cpu().numpy().astype(int)
                boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)

                for track_id, box in zip(ids, boxes):
                    visible_ids.add(track_id)

                    x1, y1, x2, y2 = box
                    cx = int((x1 + x2) / 2)
                    cy = int((y1 + y2) / 2)

                    curr_side = self.side(cx, cy, *self.line_p1, *self.line_p2)

                    # create track record
                    if track_id not in self.track_info:
                        self.track_info[track_id] = {
                            "first_side": curr_side,
                            "last_side": curr_side,
                            "last_seen": self.frame_count,
                            "counted": False
                        }

                    # crossing logic
                    if track_id in self.hist:
                        prev_cx, prev_cy = self.hist[track_id]
                        prev_side = self.side(prev_cx, prev_cy, *self.line_p1, *self.line_p2)

                        if prev_side * curr_side < 0 and not self.track_info[track_id]["counted"]:
                            if curr_side > 0:
                                self.in_count += 1
                                direction = "IN"
                            else:
                                self.out_count += 1
                                direction = "OUT"

                            self.track_info[track_id]["counted"] = True
                            self.counted.add(track_id)
                            print(f"[COUNT] {track_id} -> {direction}")

                    self.track_info[track_id]["last_side"] = curr_side
                    self.track_info[track_id]["last_seen"] = self.frame_count
                    self.hist[track_id] = (cx, cy)

                    # draw
                    cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
                    cv2.circle(frame,(cx,cy),4,(255,0,0),-1)

                    status = "COUNTED" if self.track_info[track_id]["counted"] else "ACTIVE"
                    cvzone.putTextRect(frame, f"ID:{track_id} {status}", (x1,y1), 1, 1)

            # -------- MISSED CHECK --------
            for tid in list(self.track_info.keys()):
                info = self.track_info[tid]

                if tid not in visible_ids and (self.frame_count - info["last_seen"] > self.max_missing_frames):

                    if not info["counted"]:
                        start = info["first_side"]
                        end = info["last_side"]

                        if start < 0 and end < 0:
                            self.missed_out += 1
                            missed = "MISSED OUT"
                        elif start > 0 and end > 0:
                            self.missed_in += 1
                            missed = "MISSED IN"
                        else:
                            missed = "UNKNOWN"

                        print(f"[MISS] {tid} -> {missed}")

                    del self.track_info[tid]

            # -------- DISPLAY --------
            cvzone.putTextRect(frame,f"IN: {self.in_count}",(50,30),2,2,colorR=(0,255,0))
            cvzone.putTextRect(frame,f"OUT: {self.out_count}",(50,80),2,2,colorR=(0,0,255))

            cvzone.putTextRect(frame,f"MISSED IN: {self.missed_in}",(50,130),2,2,colorR=(255,255,0))
            cvzone.putTextRect(frame,f"MISSED OUT: {self.missed_out}",(50,180),2,2,colorR=(255,50,50))

            if self.show:
                cv2.imshow("ObjectCounter", frame)

            key = cv2.waitKey(1) & 0xFF

            if key == ord('r'):
                print("RESET LINE ONLY")
                self.line_p1 = None
                self.line_p2 = None
                self.temp_points = []
                if os.path.exists(self.json_file):
                    os.remove(self.json_file)

            elif key == ord('o'):
                print("RESET IN & OUT COUNTERS")
                self.in_count = 0
                self.out_count = 0

            elif key == 27:
                break

        # -------- CLEANUP --------
        if self.is_rtsp:
            self.cap.stop()
        else:
            self.cap.release()
        cv2.destroyAllWindows()
        print("Stopped.")

# ---------------- RUN ----------------
if __name__ == "__main__":
    src = 0   # 0 for webcam or "rtsp://...." or video file path
    counter = ObjectCounter(source=src, model="yolo12n.pt", classes_to_count=[0], show=True)
    counter.run()
