import cv2
from ultralytics import YOLO
import json
import os
import time
from datetime import datetime
from imutils.video import VideoStream
import numpy as np
from shapely.geometry import Point, LineString


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
        self.counted_ids = []

        # -------- Counters --------
        self.in_count = 0
        self.out_count = 0

        # -------- Line Region --------
        self.region = []
        self.region_initialized = False
        self.json_file = json_file
        self.load_region()
        
        # Shapely line for intersection check
        self.r_s = None
        self.LineString = LineString

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


    # ================= LINE REGION =================
    def mouse_event(self, event, x, y, flags, param):
        # Track mouse position for visual feedback
        if event == cv2.EVENT_MOUSEMOVE:
            self.current_mouse_pos = (x, y)
        
        # Only allow clicks when in point selection mode
        if event == cv2.EVENT_LBUTTONDOWN and self.point_selection_mode:
            if len(self.region) < 2:
                self.region.append((x, y))
                print(f"Point {len(self.region)}: ({x}, {y})")
                
                if len(self.region) == 2:
                    self.save_region()
                    self.initialize_region()
                    self.point_selection_mode = False
                    print("✅ Line saved! Press 'P' to redraw.")


    def save_region(self):
        """Save line points to JSON file"""
        with open(self.json_file, "w") as f:
            json.dump({"region": self.region}, f)
        print(f"✅ Line with 2 points saved!")


    def load_region(self):
        """Load line points from JSON file"""
        if os.path.exists(self.json_file):
            with open(self.json_file) as f:
                self.region = json.load(f)["region"]
                print(f"✅ Loaded line from {self.json_file}")


    def delete_region(self):
        """Delete current region coordinates"""
        self.region = []
        self.region_initialized = False
        self.r_s = None
        if os.path.exists(self.json_file):
            os.remove(self.json_file)
            print("🗑️  Line coordinates deleted!")
        else:
            print("🗑️  No saved line to delete")


    def initialize_region(self):
        """Convert region points to Shapely LineString"""
        if len(self.region) == 2:
            # Create line from two points
            self.r_s = self.LineString(self.region)
            self.region_initialized = True
            
            # Determine if line is vertical or horizontal
            dx = abs(self.region[0][0] - self.region[1][0])
            dy = abs(self.region[0][1] - self.region[1][1])
            
            if dx < dy:
                self.line_orientation = "vertical"
                print(f"✅ Vertical line initialized: Right=IN, Left=OUT")
            else:
                self.line_orientation = "horizontal"
                print(f"✅ Horizontal line initialized: Down=IN, Up=OUT")
        else:
            print("⚠️  Need exactly 2 points to create a line")
            self.region_initialized = False


    # ================= COUNT LOGIC (LINE INTERSECTION) =================
    def count_objects(self, current_centroid, prev_position, track_id):
        """Count objects when they cross the line"""
        if prev_position is None or track_id in self.counted_ids:
            return

        # Check if movement path intersects with the counting line
        if len(self.region) == 2:
            movement_line = self.LineString([prev_position, current_centroid])
            
            if self.r_s.intersects(movement_line):
                # Determine orientation of the region (vertical or horizontal)
                if abs(self.region[0][0] - self.region[1][0]) < abs(self.region[0][1] - self.region[1][1]):
                    # Vertical region: Compare x-coordinates to determine direction
                    if current_centroid[0] > prev_position[0]:  # Moving right
                        self.in_count += 1
                        self.counted_ids.append(track_id)
                        print(f"✅ IN ID {track_id} (crossed line moving right)")
                    else:  # Moving left
                        self.out_count += 1
                        self.counted_ids.append(track_id)
                        print(f"✅ OUT ID {track_id} (crossed line moving left)")
                else:
                    # Horizontal region: Compare y-coordinates to determine direction
                    if current_centroid[1] > prev_position[1]:  # Moving downward
                        self.in_count += 1
                        self.counted_ids.append(track_id)
                        print(f"✅ IN ID {track_id} (crossed line moving down)")
                    else:  # Moving upward
                        self.out_count += 1
                        self.counted_ids.append(track_id)
                        print(f"✅ OUT ID {track_id} (crossed line moving up)")


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
        print("P = Start drawing line (2 clicks)")
        print("S = Delete saved coordinates")
        print("O = Reset counters")
        print("ESC = Exit")
        print("=" * 50)
        print("✅ LINE CROSSING LOGIC:")
        print("   - Vertical line: Right=IN, Left=OUT")
        print("   - Horizontal line: Down=IN, Up=OUT\n")

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

            # Draw existing line points
            for i, p in enumerate(self.region):
                cv2.circle(frame, p, 8, (0, 0, 255), -1)
                cv2.putText(frame, str(i+1), (p[0]+15, p[1]), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            # Draw the counting line
            if len(self.region) == 2:
                # Draw thick line with glow effect
                cv2.line(frame, self.region[0], self.region[1], (255, 255, 255), 8)
                cv2.line(frame, self.region[0], self.region[1], (0, 255, 0), 4)
                
                # Draw direction arrows
                mid_x = (self.region[0][0] + self.region[1][0]) // 2
                mid_y = (self.region[0][1] + self.region[1][1]) // 2
                
                if self.region_initialized:
                    if self.line_orientation == "vertical":
                        # Show horizontal arrows for vertical line
                        cv2.arrowedLine(frame, (mid_x - 30, mid_y), (mid_x - 50, mid_y), 
                                      (0, 120, 255), 3, tipLength=0.4)  # OUT (left)
                        cv2.arrowedLine(frame, (mid_x + 30, mid_y), (mid_x + 50, mid_y), 
                                      (0, 255, 0), 3, tipLength=0.4)  # IN (right)
                    else:
                        # Show vertical arrows for horizontal line
                        cv2.arrowedLine(frame, (mid_x, mid_y - 30), (mid_x, mid_y - 50), 
                                      (0, 120, 255), 3, tipLength=0.4)  # OUT (up)
                        cv2.arrowedLine(frame, (mid_x, mid_y + 30), (mid_x, mid_y + 50), 
                                      (0, 255, 0), 3, tipLength=0.4)  # IN (down)

            # Show cursor crosshair when in point selection mode
            if self.point_selection_mode:
                x, y = self.current_mouse_pos
                cv2.line(frame, (x-20, y), (x+20, y), (0, 255, 255), 2)
                cv2.line(frame, (x, y-20), (x, y+20), (0, 255, 255), 2)
                
                # Preview line from first point to cursor
                if len(self.region) == 1:
                    cv2.line(frame, self.region[0], (x, y), (255, 255, 0), 2)
                
                cv2.putText(frame, f"Point {len(self.region)+1}/2", (x+10, y-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

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

                    # Color based on whether already counted
                    color = (128, 128, 128) if tid in self.counted_ids else (255, 255, 0)

                    cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)
                    cv2.putText(frame, f"ID:{tid}", (x1,y1-5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                    
                    # Draw center point
                    cv2.circle(frame, (cx, cy), 4, color, -1)
                    
                    # Draw trajectory line if object has history
                    if tid in self.hist and len(self.hist) > 1:
                        cv2.line(frame, self.hist[tid], (cx, cy), color, 2)

            # HUD (IN / OUT)
            cv2.rectangle(frame, (0, 0), (1020, 50), (0, 0, 0), -1)
            cv2.putText(frame, f"IN: {self.in_count}", (30,35),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
            cv2.putText(frame, f"OUT: {self.out_count}", (220,35),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,120,255), 2)
            
            # Show mode indicator
            if self.point_selection_mode:
                cv2.putText(frame, f"DRAWING LINE [{len(self.region)}/2]", (450,35),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)
            elif len(self.region) == 2:
                orientation = self.line_orientation.upper()[0] if self.region_initialized else "?"
                cv2.putText(frame, f"LINE ({orientation})", (450,35),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

            if self.show:
                cv2.imshow("ObjectCounter", frame)
                key = cv2.waitKey(1)

                if key == ord('p') or key == ord('P'):
                    if not self.point_selection_mode:
                        self.point_selection_mode = True
                        self.region = []  # Clear existing points
                        self.region_initialized = False
                        self.r_s = None
                        print("📍 Line drawing mode ON - Click 2 points")
                
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
