import cv2
import json
import os
from ultralytics import YOLO
from collections import defaultdict
import time

class ObjectCounter:
    def __init__(self, source, model="yolov8n.pt",
                 classes_to_count=[0], show=True,
                 json_file="line_coords.json"):

        self.source = source
        self.model = YOLO(model)
        self.classes = classes_to_count
        self.show = show

        self.json_file = json_file

        # video / rtsp
        if isinstance(source, str) and source.startswith("rtsp://"):
            from imutils.video import VideoStream
            self.cap = VideoStream(source).start()
            time.sleep(2.0)
            self.is_rtsp = True
        else:
            self.cap = cv2.VideoCapture(source)
            self.is_rtsp = False

        # line points
        self.line_p1 = None
        self.line_p2 = None
        self.temp_points = []

        self.load_line()

        # tracking
        self.hist = {}
        self.origin_side = {}
        self.crossed_ids = set()
        self.counted = set()
        self.last_seen = defaultdict(int)
        self.frame_count = 0

        self.in_count = 0
        self.out_count = 0

        cv2.namedWindow("ObjectCounter")
        cv2.setMouseCallback("ObjectCounter", self.mouse_event)

    # -----------------------------------------
    # Mouse click to select 2 points for line
    # -----------------------------------------
    def mouse_event(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.temp_points.append((x, y))
            if len(self.temp_points) == 2:
                self.line_p1, self.line_p2 = self.temp_points
                self.temp_points = []
                self.save_line()
                print(f"Line Set: {self.line_p1} -> {self.line_p2}")

    def save_line(self):
        with open(self.json_file, "w") as f:
            json.dump({"p1": self.line_p1, "p2": self.line_p2}, f)

    def load_line(self):
        if os.path.exists(self.json_file):
            with open(self.json_file) as f:
                d = json.load(f)
                self.line_p1 = tuple(d["p1"])
                self.line_p2 = tuple(d["p2"])

    # -----------------------------------------
    # Side detection
    # -----------------------------------------
    def side(self, px, py, x1, y1, x2, y2):
        return (px - x1) * (y2 - y1) - (py - y1) * (x2 - x1)

    # -----------------------------------------
    # Remove lost IDs
    # -----------------------------------------
    def check_lost_ids(self, max_lost=40):
        lost = []
        for tid in list(self.last_seen.keys()):
            if self.frame_count - self.last_seen[tid] > max_lost:
                lost.append(tid)

        for tid in lost:
            self.last_seen.pop(tid, None)
            self.hist.pop(tid, None)
            self.origin_side.pop(tid, None)
            if tid in self.counted:
                self.counted.remove(tid)

    # -----------------------------------------
    # Reset all
    # -----------------------------------------
    def reset_all_data(self):
        print("\n--- RESET DONE ---\n")
        self.hist.clear()
        self.origin_side.clear()
        self.counted.clear()
        self.crossed_ids.clear()
        self.last_seen.clear()
        self.in_count = 0
        self.out_count = 0

    # -----------------------------------------
    # MAIN LOOP
    # -----------------------------------------
    def run(self):
        print("Press O = RESET | ESC = EXIT")

        while True:

            # read frame
            if self.is_rtsp:
                frame = self.cap.read()
            else:
                ret, frame = self.cap.read()
                if not ret:
                    break

            self.frame_count += 1

            # fixed resize
            frame = cv2.resize(frame, (640, 360))

            # display IN / OUT
            cv2.putText(frame, f"IN  : {self.in_count}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            cv2.putText(frame, f"OUT : {self.out_count}", (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

            # temp points while selecting
            for pt in self.temp_points:
                cv2.circle(frame, pt, 5, (0, 0, 255), -1)

            # draw final line
            if self.line_p1:
                cv2.line(frame, self.line_p1, self.line_p2, (255, 255, 255), 2)

            # run YOLO tracking
            results = self.model.track(frame, persist=True, classes=self.classes, conf=0.70)

            if results and results[0].boxes.id is not None and self.line_p1:

                ids = results[0].boxes.id.cpu().numpy().astype(int)
                boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)

                for tid, box in zip(ids, boxes):
                    x1, y1, x2, y2 = box
                    cx = (x1 + x2) // 2
                    cy = (y1 + y2) // 2

                    self.last_seen[tid] = self.frame_count

                    # first time
                    if tid not in self.hist:
                        s_init = self.side(cx, cy, *self.line_p1, *self.line_p2)
                        self.origin_side[tid] = "IN" if s_init < 0 else "OUT"

                    # check crossing
                    if tid in self.hist:
                        px, py = self.hist[tid]
                        s1 = self.side(px, py, *self.line_p1, *self.line_p2)
                        s2 = self.side(cx, cy, *self.line_p1, *self.line_p2)

                        if s1 * s2 < 0 and tid not in self.counted:
                            if s2 > 0:
                                self.in_count += 1
                                print(f"IN +1 (ID:{tid})")
                            else:
                                self.out_count += 1
                                print(f"OUT +1 (ID:{tid})")

                            self.counted.add(tid)

                    # save last center
                    self.hist[tid] = (cx, cy)

                    # draw box
                    cv2.rectangle(frame, (x1, y1), (x2, y2),
                                  (0, 255, 0), 2)
                    cv2.putText(frame, f"ID:{tid}",
                                (x1, y1 - 5),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.6, (255, 200, 0), 2)

            # cleanup
            self.check_lost_ids()

            # show window
            if self.show:
                cv2.imshow("ObjectCounter", frame)
                key = cv2.waitKey(1) & 0xFF

                if key == ord('o'):
                    self.reset_all_data()

                if key == 27:
                    break

        if self.is_rtsp:
            self.cap.stop()
        else:
            self.cap.release()

        cv2.destroyAllWindows()
