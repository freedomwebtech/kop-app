import cv2
from ultralytics import YOLO
import json
import os
import time
from datetime import datetime
from imutils.video import VideoStream
import numpy as np
from shapely.geometry import Point, Polygon


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

        # -------- Polygon Region --------
        self.region = []
        self.region_initialized = False
        self.json_file = json_file
        self.load_region()
        
        # Shapely polygon for contains check
        self.r_s = None
        self.Point = Point

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


    # ================= POLYGON REGION =================
    def mouse_event(self, event, x, y, flags, param):
        # Track mouse position for visual feedback
        if event == cv2.EVENT_MOUSEMOVE:
            self.current_mouse_pos = (x, y)
        
        # Only allow clicks when in point selection mode
        if event == cv2.EVENT_LBUTTONDOWN and self.point_selection_mode:
            self.region.append((x, y))
            print(f"Point {len(self.region)}: ({x}, {y})")


    def save_region(self):
        """Save polygon points to JSON file"""
        with open(self.json_file, "w") as f:
            json.dump({"region": self.region}, f)
        print(f"✅ Polygon with {len(self.region)} points saved!")


    def load_region(self):
        """Load polygon points from JSON file"""
        if os.path.exists(self.json_file):
            with open(self.json_file) as f:
                self.region = json.load(f)["region"]
                print(f"✅ Loaded polygon with {len(self.region)} points from {self.json_file}")


    def delete_region(self):
        """Delete current region coordinates"""
        self.region = []
        self.region_initialized = False
        self.r_s = None
        if os.path.exists(self.json_file):
            os.remove(self.json_file)
            print("🗑️  Polygon coordinates deleted!")
        else:
            print("🗑️  No saved polygon to delete")


    def initialize_region(self):
        """Convert region points to numpy array and Shapely polygon"""
        if len(self.region) >= 3:
            # Create polygon from points (not rectangle!)
            self.polygon_points = np.array(self.region, dtype=np.int32)
            self.r_s = Polygon(self.region)
            self.region_initialized = True
            
            # Calculate bounding box dimensions for direction detection
            x_coords = [p[0] for p in self.region]
            y_coords = [p[1] for p in self.region]
            self.region_width = max(x_coords) - min(x_coords)
            self.region_height = max(y_coords) - min(y_coords)
            
            print(f"✅ Polygon initialized: {len(self.region)} points")
            print(f"   Bounding box: W={self.region_width}, H={self.region_height}")
            if self.region_width < self.region_height:
                print("   Vertical orientation: Right=IN, Left=OUT")
            else:
                print("   Horizontal orientation: Down=IN, Up=OUT")
        else:
            print("⚠️  Need at least 3 points to create a polygon")
            self.region_initialized = False


    # ================= COUNT LOGIC (POLYGON DIRECTION-BASED) =================
    def count_objects(self, current_centroid, prev_position, track_id):
        """Count objects based on movement direction within polygon"""
        if prev_position is None or track_id in self.counted_ids:
            return

        # Check if current position is inside the polygon
        if len(self.region) > 2 and self.r_s.contains(self.Point(current_centroid)):
            # Determine motion direction for vertical or horizontal polygons
            if self.region_width < self.region_height:
                # Vertical polygon: check horizontal movement
                if current_centroid[0] > prev_position[0]:  # Moving right
                    self.in_count += 1
                    self.counted_ids.append(track_id)
                    print(f"✅ IN ID {track_id} (moved right in vertical polygon)")
                else:  # Moving left
                    self.out_count += 1
                    self.counted_ids.append(track_id)
                    print(f"✅ OUT ID {track_id} (moved left in vertical polygon)")
            else:
                # Horizontal polygon: check vertical movement
                if current_centroid[1] > prev_position[1]:  # Moving downward
                    self.in_count += 1
                    self.counted_ids.append(track_id)
                    print(f"✅ IN ID {track_id} (moved down in horizontal polygon)")
                else:  # Moving upward
                    self.out_count += 1
                    self.counted_ids.append(track_id)
                    print(f"✅ OUT ID {track_id} (moved up in horizontal polygon)")


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
        print("P = Start drawing polygon (click multiple points)")
        print("ENTER = Finish polygon and save")
        print("S = Delete saved coordinates")
        print("O = Reset counters")
        print("ESC = Exit")
        print("=" * 50)
        print("✅ DIRECTION LOGIC:")
        print("   - Vertical polygon: Right=IN, Left=OUT")
        print("   - Horizontal polygon: Down=IN, Up=OUT\n")

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

            # Draw existing polygon points
            for i, p in enumerate(self.region):
                cv2.circle(frame, p, 5, (0, 0, 255), -1)
                cv2.putText(frame, str(i+1), (p[0]+10, p[1]), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

            # Draw polygon lines
            if len(self.region) > 1:
                for i in range(len(self.region) - 1):
                    cv2.line(frame, self.region[i], self.region[i+1], (255, 0, 0), 2)
                
                # Close the polygon if finished
                if self.region_initialized and len(self.region) > 2:
                    cv2.line(frame, self.region[-1], self.region[0], (255, 0, 0), 2)

            # Draw filled polygon with transparency
            if self.region_initialized and len(self.region) >= 3:
                overlay = frame.copy()
                cv2.fillPoly(overlay, [self.polygon_points], (0, 255, 0))
                cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
                cv2.polylines(frame, [self.polygon_points], True, (0, 255, 0), 2)

            # Show cursor crosshair when in point selection mode
            if self.point_selection_mode:
                x, y = self.current_mouse_pos
                cv2.line(frame, (x-20, y), (x+20, y), (0, 255, 255), 1)
                cv2.line(frame, (x, y-20), (x, y+20), (0, 255, 255), 1)
                
                # Preview line from last point to cursor
                if len(self.region) > 0:
                    cv2.line(frame, self.region[-1], (x, y), (255, 255, 0), 1)
                
                cv2.putText(frame, f"Point {len(self.region)+1}", (x+10, y-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

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

                    # Check if inside polygon using Shapely
                    inside = self.r_s.contains(self.Point((cx, cy)))
                    color = (0,255,0) if inside else (0,0,255)

                    cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)
                    cv2.putText(frame, f"ID:{tid}", (x1,y1-5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                    
                    # Draw trajectory line
                    if tid in self.hist:
                        cv2.line(frame, self.hist[tid], (cx, cy), color, 2)

            # HUD (IN / OUT)
            cv2.rectangle(frame, (0, 0), (1020, 50), (0, 0, 0), -1)
            cv2.putText(frame, f"IN: {self.in_count}", (30,35),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
            cv2.putText(frame, f"OUT: {self.out_count}", (220,35),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,120,255), 2)
            
            # Show mode indicator
            if self.point_selection_mode:
                cv2.putText(frame, f"DRAWING MODE [{len(self.region)} points]", (450,35),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)
            elif len(self.region) >= 3:
                direction = "V" if self.region_width < self.region_height else "H"
                cv2.putText(frame, f"POLYGON: {len(self.region)}pts ({direction})", (450,35),
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
                        print("📍 Polygon drawing mode ON - Click points, press ENTER to finish")
                    
                elif key == 13:  # ENTER key
                    if self.point_selection_mode and len(self.region) >= 3:
                        self.save_region()
                        self.initialize_region()
                        self.point_selection_mode = False
                        print(f"✅ Polygon finished with {len(self.region)} points")
                    elif self.point_selection_mode:
                        print("⚠️  Need at least 3 points to create a polygon")
                
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
