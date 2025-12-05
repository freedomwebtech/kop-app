import cv2
from ultralytics import YOLO
import json
import os
from imutils.video import VideoStream
import time
from datetime import datetime
from shapely.geometry import Point, Polygon
import numpy as np

# ==========================================================
#                     OBJECT COUNTER CLASS
# ==========================================================

class ObjectCounter:
    def __init__(self, source, model="best_float32.tflite",
                 classes_to_count=[0], show=True,
                 json_file="polygon_coords.json"):

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
        
        # Track which side object first appeared on
        self.origin_side = {}

        # -------- Counters --------
        self.in_count = 0
        self.out_count = 0

        # ✅ ONLY IN/OUT MISSED
        self.missed_in = set()
        self.missed_out = set()
        self.max_missing_frames = 40

        # -------- Polygon --------
        self.region = []  # For polygon points
        self.r_s = None   # Shapely polygon object
        self.Point = Point
        self.temp_points = []
        self.json_file = json_file
        self.load_geometry()

        self.frame_count = 0

        cv2.namedWindow("ObjectCounter")
        cv2.setMouseCallback("ObjectCounter", self.mouse_event)

    # ---------------- Session Management ----------------
    def start_new_session(self):
        """Start a new tracking session"""
        self.current_session_data = {
            'day': datetime.now().strftime('%A'),
            'date': datetime.now().strftime('%Y-%m-%d'),
            'start_time': datetime.now().strftime('%H:%M:%S'),
            'end_time': None,
            'in_count': 0,
            'out_count': 0,
            'missed_in': 0,
            'missed_out': 0
        }

    def end_current_session(self):
        """End the current session and save data"""
        if self.current_session_data:
            self.current_session_data['end_time'] = datetime.now().strftime('%H:%M:%S')
            self.current_session_data['in_count'] = self.in_count
            self.current_session_data['out_count'] = self.out_count
            self.current_session_data['missed_in'] = len(self.missed_in)
            self.current_session_data['missed_out'] = len(self.missed_out)

    def print_session_summary(self):
        """Print session summary to console"""
        print("\n" + "=" * 80)
        print("                    SESSION SUMMARY")
        print("=" * 80)
        print(f"Day:           {self.current_session_data['day']}")
        print(f"Date:          {self.current_session_data['date']}")
        print(f"Start Time:    {self.current_session_data['start_time']}")
        print(f"End Time:      {self.current_session_data['end_time']}")
        print(f"IN Count:      {self.current_session_data['in_count']}")
        print(f"OUT Count:     {self.current_session_data['out_count']}")
        print(f"Missed IN:     {self.current_session_data['missed_in']}")
        print(f"Missed OUT:    {self.current_session_data['missed_out']}")
        print("=" * 80 + "\n")

    # ---------------- Mouse ----------------
    def mouse_event(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.temp_points.append((x, y))
            print(f"Polygon point {len(self.temp_points)}: ({x}, {y})")

    # ---------------- Save / Load Geometry ----------------
    def save_geometry(self):
        if self.region:
            data = {"region": self.region, "type": "polygon"}
            with open(self.json_file, "w") as f:
                json.dump(data, f)
            print(f"✅ Geometry saved to {self.json_file}")

    def load_geometry(self):
        if os.path.exists(self.json_file):
            with open(self.json_file) as f:
                data = json.load(f)
                
                if data.get("type") == "polygon" and "region" in data:
                    self.region = [tuple(p) for p in data["region"]]
                    self.r_s = Polygon(self.region)
                    print(f"✅ Loaded polygon with {len(self.region)} points")

    # ================= MISSED TRACK HANDLER =================
    def check_lost_ids(self):
        """
        Check for objects that disappeared without being counted.
        Only tracks MISSED IN and MISSED OUT.
        """
        current = self.frame_count
        lost = []

        for tid, last in self.last_seen.items():
            if current - last > self.max_missing_frames:
                lost.append(tid)

        for tid in lost:
            # Only check crossed objects that weren't counted
            if tid in self.crossed_ids and tid not in self.counted:
                if tid in self.hist:
                    last_cx, last_cy = self.hist[tid]
                    
                    # Check last known position in polygon
                    if self.r_s and self.r_s.contains(Point(last_cx, last_cy)):
                        # Determine direction based on origin
                        if self.origin_side.get(tid) == "IN":
                            self.missed_out.add(tid)
                            print(f"⚠️ MISSED OUT (Polygon) - ID:{tid}")
                        else:
                            self.missed_in.add(tid)
                            print(f"⚠️ MISSED IN (Polygon) - ID:{tid}")

            # Cleanup
            self.hist.pop(tid, None)
            self.last_seen.pop(tid, None)
            self.origin_side.pop(tid, None)

    # ---------------- Reset Function ----------------
    def reset_all_data(self):
        """Reset all tracking data and start new session"""
        self.end_current_session()
        self.print_session_summary()
        
        self.hist.clear()
        self.last_seen.clear()
        self.crossed_ids.clear()
        self.counted.clear()
        self.origin_side.clear()
        self.missed_in.clear()
        self.missed_out.clear()
        self.in_count = 0
        self.out_count = 0
        
        self.start_new_session()
        print("✅ RESET DONE - New session started")

    # ================= POLYGON COUNTING LOGIC =================
    def count_with_polygon(self, tid, cx, cy, prev_cx, prev_cy):
        """Handle counting logic for polygon regions"""
        current_centroid = (cx, cy)
        prev_position = (prev_cx, prev_cy)
        
        # Check if current position is inside the polygon
        if self.r_s.contains(Point(current_centroid)):
            if tid not in self.counted:
                self.crossed_ids.add(tid)
                
                # Determine motion direction
                region_width = max(p[0] for p in self.region) - min(p[0] for p in self.region)
                region_height = max(p[1] for p in self.region) - min(p[1] for p in self.region)
                
                # Determine IN/OUT based on movement direction
                if region_width < region_height:
                    # Vertical region - check horizontal movement
                    if current_centroid[0] > prev_position[0]:
                        # Moving right = IN
                        self.in_count += 1
                        print(f"✅ IN (Polygon-Right) - ID:{tid}")
                        self.origin_side[tid] = "IN"
                    else:
                        # Moving left = OUT
                        self.out_count += 1
                        print(f"✅ OUT (Polygon-Left) - ID:{tid}")
                        self.origin_side[tid] = "OUT"
                else:
                    # Horizontal region - check vertical movement
                    if current_centroid[1] > prev_position[1]:
                        # Moving down = IN
                        self.in_count += 1
                        print(f"✅ IN (Polygon-Down) - ID:{tid}")
                        self.origin_side[tid] = "IN"
                    else:
                        # Moving up = OUT
                        self.out_count += 1
                        print(f"✅ OUT (Polygon-Up) - ID:{tid}")
                        self.origin_side[tid] = "OUT"
                
                self.counted.add(tid)

    # ---------------- Main Loop ----------------
    def run(self):
        print("RUNNING... [POLYGON MODE]")
        print("Press O to Reset & Show Summary | P to finish polygon | ESC to Exit")

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

            frame = cv2.resize(frame, (640, 360))

            # Draw temporary points
            for pt in self.temp_points:
                cv2.circle(frame, pt, 5, (0, 0, 255), -1)

            # Draw polygon
            if self.region:
                pts = np.array(self.region, np.int32).reshape((-1, 1, 2))
                cv2.polylines(frame, [pts], True, (255, 255, 0), 2)

            results = self.model.track(
                frame, persist=True, classes=self.classes, conf=0.80)

            if results[0].boxes.id is not None and self.region:
                ids = results[0].boxes.id.cpu().numpy().astype(int)
                boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)

                for tid, box in zip(ids, boxes):
                    x1, y1, x2, y2 = box
                    cx = int((x1 + x2) / 2)
                    cy = int((y1 + y2) / 2)

                    self.last_seen[tid] = self.frame_count

                    if tid not in self.hist:
                        self.origin_side[tid] = "UNKNOWN"

                    if tid in self.hist:
                        px, py = self.hist[tid]
                        
                        # Use polygon counting logic
                        self.count_with_polygon(tid, cx, cy, px, py)

                    self.hist[tid] = (cx, cy)

                    origin_label = self.origin_side.get(tid, "?")
                    
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f"ID:{tid} [{origin_label}]", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 0), 2)

            if self.region:
                self.check_lost_ids()

            # Display counts
            cv2.putText(frame, f"IN: {self.in_count}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"OUT: {self.out_count}", (10, 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            if self.show:
                cv2.imshow("ObjectCounter", frame)
                key = cv2.waitKey(1) & 0xFF

                if key == ord('o') or key == ord('O'):
                    self.reset_all_data()
                
                elif key == ord('p') or key == ord('P'):
                    if len(self.temp_points) >= 3:
                        self.region = self.temp_points.copy()
                        self.r_s = Polygon(self.region)
                        self.temp_points = []
                        self.save_geometry()
                        print(f"✅ Polygon completed with {len(self.region)} points")

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
#                        MAIN EXECUTION
# ==========================================================

if __name__ == "__main__":
    counter = ObjectCounter(
        source="your_video.mp4",
        model="best_float32.tflite",
        classes_to_count=[0],
        show=True
    )
    counter.run()
