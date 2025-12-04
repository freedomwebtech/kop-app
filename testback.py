import cv2
from ultralytics import YOLO
import json
import os
import numpy as np
from imutils.video import VideoStream
import time
from datetime import datetime

# ==========================================================
#                HSV COLOR DETECTION (Brown + White)
# ==========================================================

# Adjusted HSV ranges for better detection
BROWN_LOWER = np.array([0, 40, 40])
BROWN_UPPER = np.array([30, 255, 200])

WHITE_LOWER = np.array([0, 0, 180])
WHITE_UPPER = np.array([180, 30, 255])

def detect_box_color_in_polygon(frame, box, polygon_points):
    """Detect color only within the polygon area"""
    if polygon_points is None or len(polygon_points) < 3:
        # If no polygon, detect in entire box
        return detect_box_color_full(frame, box)
    
    x1, y1, x2, y2 = box
    roi = frame[y1:y2, x1:x2]

    if roi.size == 0:
        return "Unknown"

    # Create mask for the bounding box
    box_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    cv2.rectangle(box_mask, (x1, y1), (x2, y2), 255, -1)
    
    # Create mask for polygon
    poly_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    pts = np.array(polygon_points, dtype=np.int32)
    cv2.fillPoly(poly_mask, [pts], 255)
    
    # Combine masks - only detect in overlapping area
    combined_mask = cv2.bitwise_and(box_mask, poly_mask)
    
    # Check if there's any overlap
    if cv2.countNonZero(combined_mask) == 0:
        return "Unknown"
    
    # Convert to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Detect brown
    brown_mask = cv2.inRange(hsv, BROWN_LOWER, BROWN_UPPER)
    brown_mask = cv2.bitwise_and(brown_mask, combined_mask)
    brown_pixels = cv2.countNonZero(brown_mask)
    
    # Detect white
    white_mask = cv2.inRange(hsv, WHITE_LOWER, WHITE_UPPER)
    white_mask = cv2.bitwise_and(white_mask, combined_mask)
    white_pixels = cv2.countNonZero(white_mask)
    
    total_pixels = cv2.countNonZero(combined_mask)
    
    if total_pixels == 0:
        return "Unknown"
    
    brown_percentage = (brown_pixels / total_pixels) * 100
    white_percentage = (white_pixels / total_pixels) * 100
    
    print(f"🎨 Detection - Brown: {brown_percentage:.1f}%, White: {white_percentage:.1f}%")
    
    # Lower threshold for better detection
    if brown_percentage > 15:
        return "Brown"
    elif white_percentage > 15:
        return "White"
    
    return "Unknown"

def detect_box_color_full(frame, box):
    """Detect color in full bounding box (fallback)"""
    x1, y1, x2, y2 = box
    roi = frame[y1:y2, x1:x2]

    if roi.size == 0:
        return "Unknown"

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    brown_mask = cv2.inRange(hsv, BROWN_LOWER, BROWN_UPPER)
    brown_pixels = cv2.countNonZero(brown_mask)

    white_mask = cv2.inRange(hsv, WHITE_LOWER, WHITE_UPPER)
    white_pixels = cv2.countNonZero(white_mask)
    
    total_pixels = roi.shape[0] * roi.shape[1]
    
    brown_percentage = (brown_pixels / total_pixels) * 100
    white_percentage = (white_pixels / total_pixels) * 100
    
    if brown_percentage > 15:
        return "Brown"
    elif white_percentage > 15:
        return "White"

    return "Unknown"


# ==========================================================
#                     OBJECT COUNTER CLASS
# ==========================================================

class ObjectCounter:
    def __init__(self, source, model="best_float32.tflite",
                 classes_to_count=[0], show=True,
                 json_file="line_coords.json",
                 polygon_file="polygon_coords.json"):

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

        # Get FPS for time calculation
        if not self.is_rtsp:
            self.fps = self.cap.get(cv2.CAP_PROP_FPS)
            if self.fps == 0:
                self.fps = 30
        else:
            self.fps = 30

        # -------- Session Data --------
        self.session_start_time = datetime.now()
        self.current_session_data = None
        self.start_new_session()

        # -------- Tracking Data --------
        self.hist = {}
        self.last_seen = {}
        self.crossed_ids = set()
        self.counted = set()
        
        self.color_at_crossing = {}

        # Track IN crossings with timestamp for delayed color detection
        self.pending_in_detections = {}
        self.delay_frames = int(0.4 * self.fps)

        # -------- Counters --------
        self.in_count = 0
        self.out_count = 0

        self.color_in_count = {}
        self.color_out_count = {}

        # MISSED LOGIC
        self.missed_in = set()
        self.missed_out = set()
        self.missed_cross = set()
        self.max_missing_frames = 40

        # -------- Line --------
        self.line_p1 = None
        self.line_p2 = None
        self.temp_points = []
        self.json_file = json_file
        self.load_line()

        # -------- Polygon for Color Detection --------
        self.polygon_points = []
        self.polygon_complete = False
        self.polygon_file = polygon_file
        self.drawing_mode = "line"  # "line" or "polygon"
        self.load_polygon()

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
            'missed_out': 0,
            'missed_cross': 0,
            'color_in': {},
            'color_out': {}
        }

    def end_current_session(self):
        """End the current session and save data"""
        if self.current_session_data:
            self.current_session_data['end_time'] = datetime.now().strftime('%H:%M:%S')
            self.current_session_data['in_count'] = self.in_count
            self.current_session_data['out_count'] = self.out_count
            self.current_session_data['missed_in'] = len(self.missed_in)
            self.current_session_data['missed_out'] = len(self.missed_out)
            self.current_session_data['missed_cross'] = len(self.missed_cross)
            self.current_session_data['color_in'] = dict(self.color_in_count)
            self.current_session_data['color_out'] = dict(self.color_out_count)

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
        print(f"Missed Cross:  {self.current_session_data['missed_cross']}")
        print("\nColor-wise Breakdown:")
        
        all_colors = set(
            list(self.current_session_data['color_in'].keys()) + 
            list(self.current_session_data['color_out'].keys())
        )
        
        if all_colors:
            for color in sorted(all_colors):
                in_c = self.current_session_data['color_in'].get(color, 0)
                out_c = self.current_session_data['color_out'].get(color, 0)
                print(f"  {color:10s} - IN: {in_c:3d}, OUT: {out_c:3d}")
        else:
            print("  No color data available")
        
        print("=" * 80 + "\n")

    # ---------------- Mouse Event Handler ----------------
    def mouse_event(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if self.drawing_mode == "line":
                # Line drawing mode (original functionality)
                self.temp_points.append((x, y))
                if len(self.temp_points) == 2:
                    self.line_p1, self.line_p2 = self.temp_points
                    self.temp_points = []
                    self.save_line()
                    print("✅ Counting line saved!")
                    
            elif self.drawing_mode == "polygon":
                # Polygon drawing mode
                self.polygon_points.append((x, y))
                print(f"📍 Polygon point {len(self.polygon_points)} added: ({x}, {y})")
        
        elif event == cv2.EVENT_RBUTTONDOWN:
            # Right click to complete polygon
            if self.drawing_mode == "polygon" and len(self.polygon_points) >= 3:
                self.polygon_complete = True
                self.save_polygon()
                print(f"✅ Polygon completed with {len(self.polygon_points)} points!")

    # ---------------- Save / Load Line ----------------
    def save_line(self):
        with open(self.json_file, "w") as f:
            json.dump({"line_p1": self.line_p1, "line_p2": self.line_p2}, f)

    def load_line(self):
        if os.path.exists(self.json_file):
            with open(self.json_file) as f:
                data = json.load(f)
                self.line_p1 = tuple(data["line_p1"])
                self.line_p2 = tuple(data["line_p2"])

    # ---------------- Save / Load Polygon ----------------
    def save_polygon(self):
        with open(self.polygon_file, "w") as f:
            json.dump({"polygon_points": self.polygon_points}, f)

    def load_polygon(self):
        if os.path.exists(self.polygon_file):
            with open(self.polygon_file) as f:
                data = json.load(f)
                self.polygon_points = [tuple(pt) for pt in data["polygon_points"]]
                if len(self.polygon_points) >= 3:
                    self.polygon_complete = True
                    print(f"✅ Loaded polygon with {len(self.polygon_points)} points")

    # ---------------- Utility ----------------
    def side(self, px, py, x1, y1, x2, y2):
        return (x2 - x1) * (py - y1) - (y2 - y1) * (px - x1)

    # ================= MISSED TRACK HANDLER =================
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
                s = self.side(cx, cy, *self.line_p2, *self.line_p1)

                if s < 0:
                    self.missed_in.add(tid)
                else:
                    self.missed_out.add(tid)

            self.hist.pop(tid, None)
            self.last_seen.pop(tid, None)
            self.color_at_crossing.pop(tid, None)
            self.pending_in_detections.pop(tid, None)

    # ---------------- Reset Function ----------------
    def reset_all_data(self):
        """Reset all tracking data and start new session"""
        self.end_current_session()
        self.print_session_summary()
        
        self.hist.clear()
        self.last_seen.clear()
        self.crossed_ids.clear()
        self.counted.clear()
        self.color_at_crossing.clear()
        self.color_in_count.clear()
        self.color_out_count.clear()
        self.missed_in.clear()
        self.missed_out.clear()
        self.missed_cross.clear()
        self.pending_in_detections.clear()
        self.in_count = 0
        self.out_count = 0
        
        self.start_new_session()
        print("✅ RESET DONE - New session started")

    # ---------------- Main Loop ----------------
    def run(self):
        print("=" * 80)
        print("CONTROLS:")
        print("  Press 'L' - Switch to LINE drawing mode (for counting line)")
        print("  Press 'P' - Switch to POLYGON drawing mode (for color detection area)")
        print("  Left Click - Add point (line or polygon)")
        print("  Right Click - Complete polygon (when in polygon mode)")
        print("  Press 'O' - Reset & Show Summary")
        print("  Press 'C' - Clear polygon")
        print("  Press 'D' - Show debug HSV values")
        print("  ESC - Exit")
        print("=" * 80)
        print(f"FPS: {self.fps}, Delay frames for 0.4 seconds: {self.delay_frames}")
        print(f"Current mode: {self.drawing_mode.upper()}")
        
        if self.polygon_complete:
            print(f"✅ Polygon loaded with {len(self.polygon_points)} points")

        debug_mode = False

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
            
            # Create debug window for HSV visualization
            debug_frame = frame.copy()

            # Draw temporary points
            for pt in self.temp_points:
                cv2.circle(frame, pt, 5, (0, 0, 255), -1)

            # Draw counting line
            if self.line_p1:
                cv2.line(frame, self.line_p1, self.line_p2, (255, 255, 255), 2)

            # Draw polygon with enhanced visibility
            if len(self.polygon_points) > 0:
                for i, pt in enumerate(self.polygon_points):
                    cv2.circle(frame, pt, 7, (0, 255, 255), -1)
                    cv2.circle(frame, pt, 9, (255, 255, 255), 2)
                    cv2.putText(frame, str(i+1), (pt[0]+12, pt[1]), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                
                if len(self.polygon_points) > 1:
                    pts = np.array(self.polygon_points, np.int32)
                    cv2.polylines(frame, [pts], self.polygon_complete, (0, 255, 255), 3)
                
                if self.polygon_complete:
                    overlay = frame.copy()
                    cv2.fillPoly(overlay, [pts], (0, 255, 255))
                    frame = cv2.addWeighted(overlay, 0.25, frame, 0.75, 0)
                    
                    # Show debug HSV if enabled
                    if debug_mode:
                        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                        poly_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
                        cv2.fillPoly(poly_mask, [pts], 255)
                        
                        brown_mask = cv2.inRange(hsv, BROWN_LOWER, BROWN_UPPER)
                        brown_mask = cv2.bitwise_and(brown_mask, poly_mask)
                        
                        white_mask = cv2.inRange(hsv, WHITE_LOWER, WHITE_UPPER)
                        white_mask = cv2.bitwise_and(white_mask, poly_mask)
                        
                        cv2.imshow("Brown Detection", brown_mask)
                        cv2.imshow("White Detection", white_mask)

            results = self.model.track(
                frame, persist=True, classes=self.classes, conf=0.80)

            if results[0].boxes.id is not None and self.line_p1:
                ids = results[0].boxes.id.cpu().numpy().astype(int)
                boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)

                for tid, box in zip(ids, boxes):
                    x1, y1, x2, y2 = box
                    cx = int((x1 + x2) / 2)
                    cy = int((y1 + y2) / 2)

                    self.last_seen[tid] = self.frame_count

                    if tid in self.hist:
                        px, py = self.hist[tid]
                        s1 = self.side(px, py, *self.line_p1, *self.line_p2)
                        s2 = self.side(cx, cy, *self.line_p1, *self.line_p2)

                        if s1 * s2 < 0:
                            self.crossed_ids.add(tid)

                            if tid not in self.counted:
                                if s2 > 0:
                                    self.pending_in_detections[tid] = self.frame_count
                                    self.in_count += 1
                                    print(f"📦 IN Detected - ID:{tid} (Color will be detected after 0.4 seconds)")
                                    
                                else:
                                    # Detect color immediately for OUT
                                    color_name = detect_box_color_in_polygon(frame, box, self.polygon_points if self.polygon_complete else None)
                                    
                                    self.out_count += 1
                                    self.color_out_count[color_name] = self.color_out_count.get(color_name, 0) + 1
                                    self.color_at_crossing[tid] = color_name
                                    print(f"✅ OUT - ID:{tid} Color:{color_name}")

                                self.counted.add(tid)
                    
                    # Check if delayed detection is ready
                    if tid in self.pending_in_detections:
                        frames_since_crossing = self.frame_count - self.pending_in_detections[tid]
                        
                        if frames_since_crossing >= self.delay_frames:
                            # Detect color now
                            color_name = detect_box_color_in_polygon(frame, box, self.polygon_points if self.polygon_complete else None)
                                
                            self.color_in_count[color_name] = self.color_in_count.get(color_name, 0) + 1
                            self.color_at_crossing[tid] = color_name
                            print(f"✅ IN Color Detected - ID:{tid} Color:{color_name} (after 0.4 seconds)")
                            
                            del self.pending_in_detections[tid]

                    self.hist[tid] = (cx, cy)

                    # Display color
                    if tid in self.color_at_crossing:
                        display_color = self.color_at_crossing[tid]
                    elif tid in self.pending_in_detections:
                        display_color = "Pending..."
                    else:
                        display_color = detect_box_color_in_polygon(frame, box, self.polygon_points if self.polygon_complete else None)
                    
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f"ID:{tid} {display_color}", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 0), 2)

            if self.line_p1:
                self.check_lost_ids()

            # Enhanced Display
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (1020, 180), (0, 0, 0), -1)
            frame = cv2.addWeighted(overlay, 0.35, frame, 0.65, 0)

            # Mode and polygon status
            mode_color = (100, 200, 255) if self.drawing_mode == "line" else (255, 100, 255)
            cv2.putText(frame, f"MODE: {self.drawing_mode.upper()}", (680, 32),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, mode_color, 2)
            
            if self.polygon_complete:
                cv2.putText(frame, "POLYGON: ON", (850, 32),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # Title
            cv2.putText(frame, "TRACKING SYSTEM", (15, 32),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 200, 255), 3)
            cv2.circle(frame, (250, 24), 7, (0, 255, 0), -1)

            # Total counts
            y_row1 = 65
            cv2.putText(frame, "IN:", (15, y_row1),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 150), 3)
            cv2.putText(frame, str(self.in_count), (75, y_row1),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 3)

            cv2.putText(frame, "OUT:", (150, y_row1),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 180, 255), 3)
            cv2.putText(frame, str(self.out_count), (230, y_row1),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 3)

            # Missed counts
            cv2.putText(frame, "MISS IN:", (320, y_row1),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (100, 255, 255), 2)
            cv2.putText(frame, str(len(self.missed_in)), (430, y_row1),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

            cv2.putText(frame, "MISS OUT:", (485, y_row1),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 100, 255), 2)
            cv2.putText(frame, str(len(self.missed_out)), (615, y_row1),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

            cv2.putText(frame, "CROSS:", (680, y_row1),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (100, 100, 255), 2)
            cv2.putText(frame, str(len(self.missed_cross)), (770, y_row1),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

            cv2.line(frame, (10, 85), (1010, 85), (100, 100, 100), 2)

            # Brown counts
            y_row2 = 118
            brown_in = self.color_in_count.get("Brown", 0)
            brown_out = self.color_out_count.get("Brown", 0)

            cv2.rectangle(frame, (15, y_row2 - 25), (50, y_row2 - 5), (19, 69, 139), -1)
            cv2.rectangle(frame, (15, y_row2 - 25), (50, y_row2 - 5), (255, 255, 255), 2)
            
            cv2.putText(frame, "BROWN IN:", (60, y_row2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 150, 255), 3)
            cv2.putText(frame, str(brown_in), (240, y_row2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 3)

            cv2.putText(frame, "BROWN OUT:", (350, y_row2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 150, 255), 3)
            cv2.putText(frame, str(brown_out), (570, y_row2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 3)

            # White counts
            y_row3 = 150
            white_in = self.color_in_count.get("White", 0)
            white_out = self.color_out_count.get("White", 0)

            cv2.rectangle(frame, (15, y_row3 - 25), (50, y_row3 - 5), (245, 245, 245), -1)
            cv2.rectangle(frame, (15, y_row3 - 25), (50, y_row3 - 5), (100, 100, 100), 2)
            
            cv2.putText(frame, "WHITE IN:", (60, y_row3),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 255, 200), 3)
            cv2.putText(frame, str(white_in), (240, y_row3),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 3)

            cv2.putText(frame, "WHITE OUT:", (350, y_row3),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 255, 200), 3)
            cv2.putText(frame, str(white_out), (570, y_row3),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 3)

            if self.show:
                cv2.imshow("ObjectCounter", frame)
                key = cv2.waitKey(1) & 0xFF

                if key == ord('l') or key == ord('L'):
                    self.drawing_mode = "line"
                    print("🔄 Switched to LINE mode")
                
                elif key == ord('p') or key == ord('P'):
                    self.drawing_mode = "polygon"
                    print("🔄 Switched to POLYGON mode")
                
                elif key == ord('c') or key == ord('C'):
                    self.polygon_points.clear()
                    self.polygon_complete = False
                    if os.path.exists(self.polygon_file):
                        os.remove(self.polygon_file)
                    print("🗑️ Polygon cleared")
                
                elif key == ord('d') or key == ord('D'):
                    debug_mode = not debug_mode
                    print(f"🔍 Debug mode: {'ON' if debug_mode else 'OFF'}")

                elif key == ord('o') or key == ord('O'):
                    self.reset_all_data()

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
