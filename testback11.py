import cv2
from ultralytics import YOLO
import json
import os
import time
from datetime import datetime
from imutils.video import VideoStream


class ObjectCounter:
    def __init__(self, source, model="best_float32.tflite",
                 classes_to_count=[0], show=True,
                 json_file="region_coords.json"):

        self.source = source
        self.model = YOLO(model)
        self.names = self.model.names
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

        # -------- Tracking --------
        self.hist = {}
        self.last_seen = {}
        self.counted_ids = set()

        # -------- Counters --------
        self.in_count = 0
        self.out_count = 0

        # -------- Miss tracking --------
        self.active_inside = set()
        self.active_outside = set()
        self.missed_inside = set()
        self.missed_outside = set()

        self.max_missing_frames = 40

        # -------- Region --------
        self.region = []
        self.region_initialized = False
        self.json_file = json_file
        self.load_region()

        self.frame_count = 0

        # -------- Window --------
        cv2.namedWindow("ObjectCounter")
        cv2.setMouseCallback("ObjectCounter", self.mouse_event)


    # ================= SESSION =================
    def start_new_session(self):
        self.session_start_time = datetime.now()
        self.current_session_data = {
            "day": self.session_start_time.strftime('%A'),
            "date": self.session_start_time.strftime('%Y-%m-%d'),
            "start_time": self.session_start_time.strftime('%H:%M:%S'),
            "end_time": None,
            "in": 0,
            "out": 0
        }


    def end_session(self):
        self.current_session_data["end_time"] = datetime.now().strftime('%H:%M:%S')
        self.current_session_data["in"] = self.in_count
        self.current_session_data["out"] = self.out_count


    def print_summary(self):
        print("\n" + "=" * 60)
        print("SESSION SUMMARY")
        print("=" * 60)
        for k, v in self.current_session_data.items():
            print(f"{k:12}: {v}")
        print("=" * 60)


    # ================= REGION =================
    def mouse_event(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(self.region) < 4:
                self.region.append((x, y))
                print("Point:", x, y)
                if len(self.region) == 4:
                    self.save_region()
                    print("Rectangle Saved!")


    def save_region(self):
        with open(self.json_file, "w") as f:
            json.dump({"region": self.region}, f)


    def load_region(self):
        if os.path.exists(self.json_file):
            with open(self.json_file) as f:
                self.region = json.load(f)["region"]


    def initialize_region(self):
        xs = [p[0] for p in self.region]
        ys = [p[1] for p in self.region]
        self.x1, self.x2 = min(xs), max(xs)
        self.y1, self.y2 = min(ys), max(ys)


    def is_inside(self, point):
        x, y = point
        return self.x1 <= x <= self.x2 and self.y1 <= y <= self.y2


    # ================= COUNT =================
    def count_objects(self, curr, prev, tid):
        if prev is None or tid in self.counted_ids:
            return

        was = self.is_inside(prev)
        now = self.is_inside(curr)

        if not was and now:
            self.in_count += 1
            self.counted_ids.add(tid)
            print(f"✅ ENTER ID {tid}")

        elif was and not now:
            self.out_count += 1
            self.counted_ids.add(tid)
            print(f"✅ EXIT ID {tid}")


    # ================= MISS =================
    def check_lost(self):
        lost = []
        for tid, last in self.last_seen.items():
            if self.frame_count - last > self.max_missing_frames:
                lost.append(tid)

        for tid in lost:
            if tid in self.active_inside:
                self.missed_inside.add(tid)
                print(f"❌ MISS INSIDE ID {tid}")

            if tid in self.active_outside:
                self.missed_outside.add(tid)
                print(f"❌ MISS OUTSIDE ID {tid}")

            self.hist.pop(tid, None)
            self.last_seen.pop(tid, None)
            self.active_inside.discard(tid)
            self.active_outside.discard(tid)


    # ================= RESET =================
    def reset_all(self):
        self.end_session()
        self.print_summary()

        self.in_count = 0
        self.out_count = 0
        self.hist.clear()
        self.last_seen.clear()
        self.active_inside.clear()
        self.active_outside.clear()
        self.missed_inside.clear()
        self.missed_outside.clear()
        self.counted_ids.clear()

        self.start_new_session()
        print("✅ RESET DONE")


    # ================= LOOP =================
    def run(self):
        print("Draw rectangle using 4 clicks")
        print("O = Reset | ESC = Exit")

        while True:
            if self.is_rtsp:
                frame = self.cap.read()
            else:
                ok, frame = self.cap.read()
                if not ok:
                    break

            self.frame_count += 1
            if self.frame_count % 3 != 0:
                continue

            frame = cv2.resize(frame, (1020, 600))

            for p in self.region:
                cv2.circle(frame, p, 5, (0, 0, 255), -1)

            if len(self.region) == 4 and not self.region_initialized:
                self.initialize_region()
                self.region_initialized = True

            if self.region_initialized:
                cv2.rectangle(frame, (self.x1, self.y1), (self.x2, self.y2), (0,255,0), 2)

            results = self.model.track(frame, persist=True,
                                       classes=self.classes,
                                       conf=0.80)

            if results[0].boxes.id is not None and self.region_initialized:
                ids = results[0].boxes.id.cpu().numpy().astype(int)
                boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)

                for tid, box in zip(ids, boxes):
                    x1, y1, x2, y2 = box
                    cx, cy = (x1+x2)//2, (y1+y2)//2

                    self.last_seen[tid] = self.frame_count

                    if tid in self.hist:
                        self.count_objects((cx,cy), self.hist[tid], tid)

                    self.hist[tid] = (cx,cy)

                    inside = self.is_inside((cx,cy))

                    if inside:
                        self.active_inside.add(tid)
                        self.active_outside.discard(tid)
                    else:
                        self.active_outside.add(tid)
                        self.active_inside.discard(tid)

                    color = (0,255,0) if inside else (0,0,255)
                    cv2.rectangle(frame, (x1,y1),(x2,y2), color, 2)
                    cv2.putText(frame, f"ID:{tid}", (x1,y1-5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            self.check_lost()

            # HUD
            cv2.rectangle(frame, (0, 0), (1020, 70), (0, 0, 0), -1)
            cv2.putText(frame, f"IN: {self.in_count}", (20,45), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
            cv2.putText(frame, f"OUT: {self.out_count}", (150,45), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,100,255), 2)
            cv2.putText(frame, f"MISS IN: {len(self.missed_inside)}", (300,45), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2)
            cv2.putText(frame, f"MISS OUT: {len(self.missed_outside)}", (520,45), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,100,255), 2)

            if self.show:
                cv2.imshow("ObjectCounter", frame)
                k = cv2.waitKey(1)
                if k == ord("o"):
                    self.reset_all()
                elif k == 27:
                    break

        self.end_session()
        self.print_summary()

        if self.is_rtsp:
            self.cap.stop()
        else:
            self.cap.release()

        cv2.destroyAllWindows()


# ================= RUN =================
if __name__ == "__main__":
    counter = ObjectCounter(
        source="your_video.mp4",
        model="best_float32.tflite",
        classes_to_count=[0],
        show=True
    )
    counter.run()
