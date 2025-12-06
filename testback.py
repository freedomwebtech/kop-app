import cv2
from ultralytics import YOLO
import json
import os
from imutils.video import VideoStream
import time
from datetime import datetime
import numpy as np
from shapely.geometry import Point, Polygon

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

        # -------- Frame Counter --------
        self.frame_count = 0

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
        self.crossed_ids = set()
        self.counted = set()
        
        # Track previous position for direction
        self.prev_position = {}

        # -------- Counters --------
        self.in_count = 0
        self.out_count = 0

        # -------- Polygon Region (4 points) --------
        self.polygon_points = []
        self.temp_points = []
        self.polygon = None
        self.json_file = json_file
        self.load_polygon()

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
            'total_frames': 0
        }

    def end_current_session(self):
        """End the current session and save data"""
        if self.current_session_data:
            self.current_session_data['end_time'] = datetime.now().strftime('%H:%M:%S')
            self.current_session_data['in_count'] = self.in_count
            self.current_session_data['out_count'] = self.out_count
            self.current_session_data['total_frames'] = self.frame_count

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
        print(f"Total Frames:  {self.current_session_data['total_frames']}")
        print("=" * 80 + "\n")

    # ---------------- Mouse ----------------
    def mouse_event(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(self.temp_points) < 4:
                self.temp_points.append((x, y))
                print(f"Point {len(self.temp_points)}/4 added: ({x}, {y})")
                
                if len(self.temp_points) == 4:
                    self.polygon_points = self.temp_points.copy()
                    self.polygon = Polygon(self.polygon_points)
                    self.temp_points = []
                    self.save_polygon()
                    print("✅ Polygon complete and saved!")

    # ---------------- Save / Load Polygon ----------------
    def save_polygon(self):
        """Save polygon coordinates to JSON file"""
        try:
            with open(self.json_file, "w") as f:
                json.dump({"polygon_points": self.polygon_points}, f, indent=2)
            print(f"💾 Polygon saved to {self.json_file}")
        except Exception as e:
            print(f"❌ Error saving polygon: {e}")

    def load_polygon(self):
        """Load polygon coordinates from JSON file"""
        if os.path.exists(self.json_file):
            try:
                with open(self.json_file) as f:
                    data = json.load(f)
                    
                    if "polygon_points" not in data:
                        print(f"⚠️ No 'polygon_points' key found in {self.json_file}")
                        return
                    
                    points = data["polygon_points"]
                    
                    if len(points) != 4:
                        print(f"⚠️ Expected 4 points, found {len(points)}. Please redraw polygon.")
                        return
                    
                    self.polygon_points = [tuple(p) for p in points]
                    self.polygon = Polygon(self.polygon_points)
                    print(f"✅ Loaded polygon from {self.json_file}")
                    print(f"   Points: {self.polygon_points}")
                    
            except json.JSONDecodeError:
                print(f"❌ Invalid JSON in {self.json_file}. Please redraw polygon.")
            except Exception as e:
                print(f"❌ Error loading polygon: {e}")
        else:
            print(f"ℹ️ No saved polygon found. Click 4 points to create one.")

    # ---------------- Reset Function ----------------
    def reset_all_data(self):
        """Reset all tracking data and start new session"""
        self.end_current_session()
        self.print_session_summary()
        
        self.hist.clear()
        self.crossed_ids.clear()
        self.counted.clear()
        self.prev_position.clear()
        self.in_count = 0
        self.out_count = 0
        self.frame_count = 0
        
        self.start_new_session()
        print("✅ RESET DONE - New session started")

    # ---------------- Main Loop ----------------
    def run(self):
        print("RUNNING... Click 4 points to draw polygon | Press O to Reset | ESC to Exit")

        while True:
            if self.is_rtsp:
                frame = self.cap.read()
            else:
                ret, frame = self.cap.read()
                if not ret:
                    break

            # Increment frame counter
            self.frame_count += 1

            frame = cv2.resize(frame, (640, 360))

            # Draw temporary points
            for i, pt in enumerate(self.temp_points):
                cv2.circle(frame, pt, 5, (0, 0, 255), -1)
                cv2.putText(frame, str(i+1), (pt[0]+10, pt[1]-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                if i > 0:
                    cv2.line(frame, self.temp_points[i-1], pt, (0, 0, 255), 2)

            # Draw complete polygon
            if self.polygon_points:
                pts = np.array(self.polygon_points, np.int32)
                pts = pts.reshape((-1, 1, 2))
                cv2.polylines(frame, [pts], True, (255, 255, 0), 2)
                
                # Fill with semi-transparent overlay
                overlay = frame.copy()
                cv2.fillPoly(overlay, [pts], (0, 255, 255))
                cv2.addWeighted(overlay, 0.2, frame, 0.8, 0, frame)

            # Run detection and tracking
            results = self.model.track(
                frame, persist=True, classes=self.classes, conf=0.80)

            if results[0].boxes.id is not None and self.polygon:
                ids = results[0].boxes.id.cpu().numpy().astype(int)
                boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
                classes = results[0].boxes.cls.cpu().numpy().astype(int)

                for tid, box, cls in zip(ids, boxes, classes):
                    x1, y1, x2, y2 = box
                    cx = int((x1 + x2) / 2)
                    cy = int((y1 + y2) / 2)

                    current_point = Point(cx, cy)

                    # Check if object is inside polygon
                    is_inside = self.polygon.contains(current_point)

                    # Track crossing
                    if tid in self.prev_position:
                        prev_cx, prev_cy = self.prev_position[tid]
                        prev_point = Point(prev_cx, prev_cy)
                        was_inside = self.polygon.contains(prev_point)

                        # Detect crossing
                        if was_inside != is_inside and tid not in self.counted:
                            self.crossed_ids.add(tid)
                            
                            # Calculate region dimensions for direction
                            region_width = max(p[0] for p in self.polygon_points) - min(p[0] for p in self.polygon_points)
                            region_height = max(p[1] for p in self.polygon_points) - min(p[1] for p in self.polygon_points)
                            
                            # Determine IN/OUT based on movement direction
                            if is_inside:
                                # Object entered polygon
                                if (region_width < region_height and cx > prev_cx) or \
                                   (region_width >= region_height and cy > prev_cy):
                                    # Moving right or downward
                                    self.in_count += 1
                                    print(f"✅ IN - ID:{tid} (Frame: {self.frame_count})")
                                else:
                                    # Moving left or upward
                                    self.out_count += 1
                                    print(f"✅ OUT - ID:{tid} (Frame: {self.frame_count})")
                            else:
                                # Object exited polygon
                                if (region_width < region_height and cx > prev_cx) or \
                                   (region_width >= region_height and cy > prev_cy):
                                    # Moving right or downward
                                    self.out_count += 1
                                    print(f"✅ OUT - ID:{tid} (Frame: {self.frame_count})")
                                else:
                                    # Moving left or upward
                                    self.in_count += 1
                                    print(f"✅ IN - ID:{tid} (Frame: {self.frame_count})")
                            
                            self.counted.add(tid)

                    # Update position
                    self.prev_position[tid] = (cx, cy)
                    self.hist[tid] = (cx, cy)

                    # Draw bounding box
                    color = (0, 255, 0) if is_inside else (0, 165, 255)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    
                    status = "IN" if is_inside else "OUT"
                    cv2.putText(frame, f"ID:{tid} [{status}]", (x1, y1 - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 0), 2)

            # Display counts and frame info
            cv2.putText(frame, f"IN: {self.in_count} | OUT: {self.out_count}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(frame, f"Frame: {self.frame_count}", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            if self.show:
                cv2.imshow("ObjectCounter", frame)
                key = cv2.waitKey(1) & 0xFF

                if key == ord('o') or key == ord('O'):
                    self.reset_all_data()
                elif key == ord('c') or key == ord('C'):
                    # Clear polygon and start new
                    self.polygon_points = []
                    self.temp_points = []
                    self.polygon = None
                    print("🔄 Polygon cleared. Click 4 new points.")
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
