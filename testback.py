import cv2
from ultralytics import YOLO
import json
import os
import numpy as np
from imutils.video import VideoStream
import time
from datetime import datetime

# ==========================================================
#                     OBJECT COUNTER CLASS
# ==========================================================

class ObjectCounter:
    def __init__(self, source, model="best_float32.tflite",
                 classes_to_count=[0], show=True,
                 json_file="line_coords_back.json"):

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

        # -------- Tracking for Line Crossing (Previous Frame Data) --------
        self.prev_positions = []  # List of (cx, cy) from previous frame

        # -------- Counters --------
        self.in_count = 0
        self.out_count = 0

        # -------- Line --------
        self.line_p1 = None
        self.line_p2 = None
        self.temp_points = []
        self.json_file = json_file
        self.load_line()

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
            'out_count': 0
        }

    def end_current_session(self):
        """End the current session and save data"""
        if self.current_session_data:
            self.current_session_data['end_time'] = datetime.now().strftime('%H:%M:%S')
            self.current_session_data['in_count'] = self.in_count
            self.current_session_data['out_count'] = self.out_count

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
        print("=" * 80 + "\n")

    # ---------------- Mouse ----------------
    def mouse_event(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.temp_points.append((x, y))
            if len(self.temp_points) == 2:
                self.line_p1, self.line_p2 = self.temp_points
                self.temp_points = []
                self.save_line()

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

    # ---------------- Utility ----------------
    def side(self, px, py, x1, y1, x2, y2):
        """Calculate which side of line the point is on"""
        return (x2 - x1) * (py - y1) - (y2 - y1) * (px - x1)

    def find_closest_prev_point(self, current_pos, max_distance=100):
        """Find the closest previous position to match with current detection"""
        if not self.prev_positions:
            return None
        
        cx, cy = current_pos
        min_dist = float('inf')
        closest_prev = None
        closest_idx = -1
        
        for idx, (px, py) in enumerate(self.prev_positions):
            dist = np.sqrt((cx - px)**2 + (cy - py)**2)
            if dist < min_dist and dist < max_distance:
                min_dist = dist
                closest_prev = (px, py)
                closest_idx = idx
        
        # Remove the matched previous position
        if closest_idx >= 0:
            self.prev_positions.pop(closest_idx)
        
        return closest_prev

    # ---------------- Reset Function ----------------
    def reset_all_data(self):
        """Reset all tracking data and start new session"""
        self.end_current_session()
        self.print_session_summary()
        
        # Reset counters
        self.prev_positions = []
        self.in_count = 0
        self.out_count = 0
        
        self.start_new_session()
        print("✅ RESET DONE - New session started")

    # ---------------- Main Loop ----------------
    def run(self):
        print("RUNNING... Press O to Reset & Show Summary | ESC to Exit")

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

            # Draw temporary points for line setup
            for pt in self.temp_points:
                cv2.circle(frame, pt, 5, (0, 0, 255), -1)

            # Draw counting line
            if self.line_p1:
                cv2.line(frame, self.line_p1, self.line_p2, (255, 255, 255), 3)

            # Run YOLO detection (no tracking)
            results = self.model(frame, classes=self.classes, conf=0.80, verbose=False)

            current_positions = []
            
            if results[0].boxes is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
                
                for box in boxes:
                    x1, y1, x2, y2 = box
                    cx = int((x1 + x2) / 2)
                    cy = int((y1 + y2) / 2)
                    
                    current_positions.append((cx, cy))
                    
                    # NO DRAWING - No bounding box, No ID, No circles

            # Check for line crossings
            if self.line_p1 and len(current_positions) > 0:
                for current_pos in current_positions:
                    cx, cy = current_pos
                    
                    # Find matching previous position
                    prev_pos = self.find_closest_prev_point(current_pos)
                    
                    if prev_pos is not None:
                        px, py = prev_pos
                        
                        # Calculate which side of line
                        s1 = self.side(px, py, *self.line_p1, *self.line_p2)
                        s2 = self.side(cx, cy, *self.line_p1, *self.line_p2)
                        
                        # Check if crossed the line (sign change)
                        if s1 * s2 < 0:
                            if s2 > 0:  # Going IN
                                self.in_count += 1
                                print(f"✅ IN - Position: ({cx},{cy})")
                            else:  # Going OUT
                                self.out_count += 1
                                print(f"✅ OUT - Position: ({cx},{cy})")

            # Store current positions for next frame
            self.prev_positions = current_positions.copy()

            # ================= DISPLAY PANEL =================
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (1020, 100), (0, 0, 0), -1)
            frame = cv2.addWeighted(overlay, 0.4, frame, 0.6, 0)

            # Title
            cv2.putText(frame, "COUNTING SYSTEM", (15, 32),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 200, 255), 3)
            cv2.circle(frame, (250, 24), 7, (0, 255, 0), -1)

            # Counters
            y_row = 70
            font_size = 1.2
            thickness = 3
            
            # IN Counter
            cv2.putText(frame, "IN:", (15, y_row),
                        cv2.FONT_HERSHEY_SIMPLEX, font_size, (0, 255, 150), thickness)
            cv2.putText(frame, str(self.in_count), (120, y_row),
                        cv2.FONT_HERSHEY_SIMPLEX, font_size, (255, 255, 255), thickness)

            # OUT Counter
            cv2.putText(frame, "OUT:", (300, y_row),
                        cv2.FONT_HERSHEY_SIMPLEX, font_size, (100, 180, 255), thickness)
            cv2.putText(frame, str(self.out_count), (420, y_row),
                        cv2.FONT_HERSHEY_SIMPLEX, font_size, (255, 255, 255), thickness)

            if self.show:
                cv2.imshow("ObjectCounter", frame)
                key = cv2.waitKey(1) & 0xFF

                if key == ord('o') or key == ord('O'):
                    self.reset_all_data()
                elif key == 27:
                    break

        # Print final session summary before exit
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
        source="your_video.mp4",  # or 0 for webcam, or "rtsp://..."
        model="best_float32.tflite",
        classes_to_count=[0],
        show=True
    )
    counter.run()
