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

        # -------- Region --------
        self.region = []
        self.region_initialized = False
        self.json_file = json_file
        self.load_region()

        self.frame_count = 0

        # -------- Keyboard Point Selection --------
        self.point_selection_mode = False
        self.current_mouse_pos = (0, 0)

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
        print("\n" + "=" * 50)
        print("SESSION SUMMARY")
        print("=" * 50)
        for k, v in self.current_session_data.items():
            print(f"{k:12}: {v}")
        print("=" * 50)


    # ================= REGION =================
    def mouse_event(self, event, x, y, flags, param):
        # Track mouse position for visual feedback
        if event == cv2.EVENT_MOUSEMOVE:
            self.current_mouse_pos = (x, y)
        
        # Only allow clicks when in point selection mode
        if event == cv2.EVENT_LBUTTONDOWN and self.point_selection_mode:
            if len(self.region) < 4:
                self.region.append((x, y))
                print(f"Point {len(self.region)}: ({x}, {y})")
                
                if len(self.region) == 4:
                    self.save_region()
                    self.point_selection_mode = False
                    self.region_initialized = False  # Force re-initialization
                    print("✅ Rectangle Saved! Press 'P' again to modify.")


    def save_region(self):
        with open(self.json_file, "w") as f:
            json.dump({"region": self.region}, f)


    def load_region(self):
        if os.path.exists(self.json_file):
            with open(self.json_file) as f:
                self.region = json.load(f)["region"]
                print(f"✅ Loaded region from {self.json_file}")


    def delete_region(self):
        """Delete current region coordinates"""
        self.region = []
        self.region_initialized = False
        if os.path.exists(self.json_file):
            os.remove(self.json_file)
            print("🗑️  Region coordinates deleted!")
        else:
            print("🗑️  No saved region to delete")


    def initialize_region(self):
        xs = [p[0] for p in self.region]
        ys = [p[1] for p in self.region]
        self.x1, self.x2 = min(xs), max(xs)
        self.y1, self.y2 = min(ys), max(ys)


    def is_inside(self, point):
        x, y = point
        return self.x1 <= x <= self.x2 and self.y1 <= y <= self.y2


    # ================= COUNT (FIXED LOGIC) =================
    def count_objects(self, curr, prev, tid):
        if prev is None or tid in self.counted_ids:
            return

        was = self.is_inside(prev)
        now = self.is_inside(curr)

        # CORRECT: Entering region = IN
        if not was and now:
            self.in_count += 1
            self.counted_ids.add(tid)
            print(f"✅ IN ID {tid} (entered region)")

        # CORRECT: Exiting region = OUT
        elif was and not now:
            self.out_count += 1
            self.counted_ids.add(tid)
            print(f"✅ OUT ID {tid} (exited region)")


    # ================= RESET =================
    def reset_all(self):
        self.end_session()
        self.print_summary()

        self.in_count = 0
        self.out_count = 0
        self.hist.clear()
        self.last_seen.clear()
        self.counted_ids.clear()

        self.start_new_session()
        print("✅ RESET DONE")


    # ================= LOOP =================
    def run(self):
        print("\n" + "=" * 50)
        print("CONTROLS:")
        print("=" * 50)
        print("P = Start drawing rectangle (4 clicks)")
        print("S = Delete saved coordinates")
        print("O = Reset counters")
        print("ESC = Exit")
        print("=" * 50)
        print("✅ LOGIC: Entering region = IN, Exiting region = OUT\n")

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

            # Draw existing points
            for i, p in enumerate(self.region):
                cv2.circle(frame, p, 5, (0, 0, 255), -1)
                cv2.putText(frame, str(i+1), (p[0]+10, p[1]), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

            # Draw lines between points
            if len(self.region) > 1:
                for i in range(len(self.region) - 1):
                    cv2.line(frame, self.region[i], self.region[i+1], (255, 0, 0), 2)
                if len(self.region) == 4:
                    cv2.line(frame, self.region[3], self.region[0], (255, 0, 0), 2)

            # Show cursor crosshair when in point selection mode
            if self.point_selection_mode:
                x, y = self.current_mouse_pos
                cv2.line(frame, (x-20, y), (x+20, y), (0, 255, 255), 1)
                cv2.line(frame, (x, y-20), (x, y+20), (0, 255, 255), 1)
                cv2.putText(frame, f"Point {len(self.region)+1}/4", (x+10, y-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

            # Initialize region if we have 4 points
            if len(self.region) == 4 and not self.region_initialized:
                self.initialize_region()
                self.region_initialized = True

            # Draw rectangle
            if self.region_initialized:
                cv2.rectangle(frame, (self.x1, self.y1), (self.x2, self.y2), (0,255,0), 2)

            # Detection and tracking
            results = self.model.track(frame, persist=True,
                                       classes=self.classes,
                                       conf=0.80)

            if results[0].boxes.id is not None and self.region_initialized:
                ids = results[0].boxes.id.cpu().numpy().astype(int)
                boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)

                for tid, box in zip(ids, boxes):
                    x1, y1, x2, y2 = box
                    cx, cy = (x1+x2)//2, (y1+y2)//2

                    if tid in self.hist:
                        self.count_objects((cx, cy), self.hist[tid], tid)

                    self.hist[tid] = (cx, cy)

                    inside = self.is_inside((cx, cy))
                    color = (0,255,0) if inside else (0,0,255)

                    cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)
                    cv2.putText(frame, f"ID:{tid}", (x1,y1-5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # HUD (IN / OUT)
            cv2.rectangle(frame, (0, 0), (1020, 50), (0, 0, 0), -1)
            cv2.putText(frame, f"IN: {self.in_count}", (30,35),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
            cv2.putText(frame, f"OUT: {self.out_count}", (220,35),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,120,255), 2)
            
            # Show mode indicator
            if self.point_selection_mode:
                cv2.putText(frame, f"DRAWING MODE [{len(self.region)}/4]", (450,35),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)

            if self.show:
                cv2.imshow("ObjectCounter", frame)
                key = cv2.waitKey(1)

                if key == ord('p') or key == ord('P'):
                    if not self.point_selection_mode:
                        self.point_selection_mode = True
                        self.region = []  # Clear existing points
                        self.region_initialized = False
                        print("📍 Point selection mode ON - Click 4 points")
                    
                elif key == ord('s') or key == ord('S'):
                    self.delete_region()
                    self.point_selection_mode = False
                    
                elif key == ord('o') or key == ord('O'):
                    self.reset_all()
                    
                elif key == 27:  # ESC
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
        source="your_video.mp4",   # or RTSP
        model="best_float32.tflite",
        classes_to_count=[0],
        show=True
    )
    counter.run()
