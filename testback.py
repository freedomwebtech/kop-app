import cv2
from ultralytics import YOLO
import json
import os
import time
from datetime import datetime
from imutils.video import VideoStream
import numpy as np
from shapely.geometry import LineString


class ObjectCounter:
    def __init__(self, source, model="best_float32.tflite",
                 classes_to_count=[0], show=True,
                 json_file="region_coords.json"):

        self.source = source
        self.model = YOLO(model)
        self.names = getattr(self.model, "names", {})
        self.classes = classes_to_count
        self.show = show

        # -------- Video Source --------
        if isinstance(source, str) and source.startswith("rtsp://"):
            self.cap = VideoStream(source).start()
            time.sleep(2.0)
            self.is_rtsp = True
        else:
            self.cap = cv2.VideoCapture(source)
            self.is_rtsp = False

        # -------- Session --------
        self.start_new_session()

        # -------- Tracking Data --------
        self.tracks = {}
        self.counted_ids = set()

        # -------- Counters --------
        self.in_count = 0
        self.out_count = 0

        # -------- Line Data --------
        self.region = []
        self.region_initialized = False
        self.json_file = json_file
        self.r_s = None
        self.line_orientation = None

        self.load_region()
        if len(self.region) == 2:
            self.initialize_region()

        # -------- GUI --------
        cv2.namedWindow("ObjectCounter")
        cv2.setMouseCallback("ObjectCounter", self.mouse_event)
        print("✅ Click mouse twice to draw line. Click again to redraw.")

    # ================= SESSION =================
    def start_new_session(self):
        self.session_start_time = datetime.now()
        self.current_session_data = {
            "start_time": self.session_start_time.strftime("%H:%M:%S"),
            "end_time": None,
            "in": 0,
            "out": 0
        }

    def end_session(self):
        self.current_session_data["end_time"] = datetime.now().strftime("%H:%M:%S")
        self.current_session_data["in"] = self.in_count
        self.current_session_data["out"] = self.out_count

    def print_summary(self):
        print("\n==== SESSION SUMMARY ====")
        print(self.current_session_data)

    # ================= MOUSE LINE DRAW =================
    def mouse_event(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:

            # Clear old line if exists
            if len(self.region) == 2:
                self.region = []
                self.region_initialized = False
                self.r_s = None
                self.line_orientation = None
                print("♻️ Line reset")

            # Add new point
            self.region.append((x, y))
            print(f"✅ Point {len(self.region)}: {x,y}")

            # Save when finished
            if len(self.region) == 2:
                self.save_region()
                self.initialize_region()
                print("✅ Line ready!")

    # ================= LINE FILE =================
    def save_region(self):
        with open(self.json_file, "w") as f:
            json.dump({"region": self.region}, f)

    def load_region(self):
        if os.path.exists(self.json_file):
            with open(self.json_file) as f:
                self.region = json.load(f).get("region", [])

    def delete_region(self):
        self.region = []
        self.region_initialized = False
        self.r_s = None
        self.line_orientation = None
        if os.path.exists(self.json_file):
            os.remove(self.json_file)
        print("🗑️ Line deleted")

    def initialize_region(self):
        self.r_s = LineString(self.region)

        dx = abs(self.region[0][0] - self.region[1][0])
        dy = abs(self.region[0][1] - self.region[1][1])

        if dx < dy:
            self.line_orientation = "vertical"
            print("➡️ Vertical line: Right=IN, Left=OUT")
        else:
            self.line_orientation = "horizontal"
            print("⬇️ Horizontal line: Down=IN, Up=OUT")

        self.region_initialized = True

    # ================= COUNT =================
    def count_objects(self, prev, curr, tid):
        if not prev or tid in self.counted_ids or not self.region_initialized:
            return

        if not self.r_s.intersects(LineString([prev, curr])):
            return

        if self.line_orientation == "vertical":
            if curr[0] > prev[0]:
                self.in_count += 1
                print("✅ IN:", tid)
            else:
                self.out_count += 1
                print("✅ OUT:", tid)
        else:
            if curr[1] > prev[1]:
                self.in_count += 1
                print("✅ IN:", tid)
            else:
                self.out_count += 1
                print("✅ OUT:", tid)

        self.counted_ids.add(tid)

    # ================= LOOP =================
    def run(self):
        while True:
            if self.is_rtsp:
                frame = self.cap.read()
            else:
                ok, frame = self.cap.read()
                if not ok:
                    break

            frame = cv2.resize(frame, (1020, 600))

            # Draw line
            for p in self.region:
                cv2.circle(frame, p, 6, (0, 0, 255), -1)

            if len(self.region) == 2:
                cv2.line(frame, self.region[0], self.region[1], (0, 255, 0), 4)

            # Detection
            results = self.model.track(frame, persist=True, classes=self.classes, conf=0.8)

            if results and results[0].boxes.id is not None:
                ids = results[0].boxes.id.cpu().numpy().astype(int)
                boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)

                for tid, box in zip(ids, boxes):
                    x1,y1,x2,y2 = box
                    cx,cy = (x1+x2)//2, (y1+y2)//2

                    if tid not in self.tracks:
                        self.tracks[tid] = []
                    self.tracks[tid].append((cx,cy))

                    if len(self.tracks[tid]) > 1:
                        self.count_objects(self.tracks[tid][-2], self.tracks[tid][-1], tid)

                    color = (0,255,0) if tid not in self.counted_ids else (100,100,100)

                    cv2.rectangle(frame,(x1,y1),(x2,y2),color,2)
                    cv2.putText(frame,f"ID:{tid}",(x1,y1-5),
                                cv2.FONT_HERSHEY_SIMPLEX,0.6,color,2)
                    cv2.circle(frame,(cx,cy),4,color,-1)

            # HUD
            cv2.rectangle(frame,(0,0),(1020,50),(0,0,0),-1)
            cv2.putText(frame,f"IN: {self.in_count}",(30,35),
                        cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
            cv2.putText(frame,f"OUT: {self.out_count}",(200,35),
                        cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)

            cv2.imshow("ObjectCounter",frame)
            key=cv2.waitKey(1)

            if key==27:
                break
            elif key==ord('s'):
                self.delete_region()
            elif key==ord('o'):
                self.reset_all()

        self.end_session()
        self.print_summary()
        cv2.destroyAllWindows()
        self.cap.release()

    def reset_all(self):
        self.in_count=0
        self.out_count=0
        self.tracks.clear()
        self.counted_ids.clear()
        print("🔄 Reset done")


# ================== RUN ==================
if __name__ == "__main__":
    counter = ObjectCounter(
        source="your_video.mp4",   # OR rtsp://
        model="best_float32.tflite",
        classes_to_count=[0],
        show=True
    )
    counter.run()
