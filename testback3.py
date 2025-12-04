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

        # -------- Tracking Data --------
        self.hist = {}
        self.last_seen = {}
        self.crossed_ids = set()
        self.counted = set()
        self.false_detections = set()
        
        # -------- NEW: Enhanced Miss Tracking --------
        self.crossed_direction = {}  # Which direction object crossed: "IN" or "OUT"
        self.completion_zone = {}    # Track if object reached completion zone
        self.miss_tracked = set()    # IDs already counted as miss

        # -------- Counters --------
        self.in_count = 0
        self.out_count = 0
        self.false_count = 0
        self.miss_in_count = 0
        self.miss_out_count = 0

        # -------- MISSED LOGIC --------
        self.max_missing_frames = 40
        self.completion_distance = 80  # Distance from line to consider "completed"

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
            'out_count': 0,
            'false_count': 0,
            'miss_in_count': 0,
            'miss_out_count': 0
        }

    def end_current_session(self):
        """End the current session and save data"""
        if self.current_session_data:
            self.current_session_data['end_time'] = datetime.now().strftime('%H:%M:%S')
            self.current_session_data['in_count'] = self.in_count
            self.current_session_data['out_count'] = self.out_count
            self.current_session_data['false_count'] = self.false_count
            self.current_session_data['miss_in_count'] = self.miss_in_count
            self.current_session_data['miss_out_count'] = self.miss_out_count

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
        print(f"False Count:   {self.current_session_data['false_count']}")
        print(f"Miss IN:       {self.current_session_data['miss_in_count']}")
        print(f"Miss OUT:      {self.current_session_data['miss_out_count']}")
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
        return (x2 - x1) * (py - y1) - (y2 - y1) * (px - x1)

    def distance_from_line(self, px, py):
        """Calculate perpendicular distance from point to line"""
        if not self.line_p1 or not self.line_p2:
            return 0
        
        x1, y1 = self.line_p1
        x2, y2 = self.line_p2
        
        # Line equation: Ax + By + C = 0
        A = y2 - y1
        B = x1 - x2
        C = x2 * y1 - x1 * y2
        
        # Distance formula
        distance = abs(A * px + B * py + C) / np.sqrt(A**2 + B**2)
        return distance

    # ---------------- ENHANCED MISS TRACKING ----------------
    def check_completion_status(self, tid, cx, cy):
        """Check if object has moved far enough from line to be considered complete"""
        distance = self.distance_from_line(cx, cy)
        
        if tid in self.crossed_direction and tid not in self.completion_zone:
            if distance > self.completion_distance:
                self.completion_zone[tid] = True
                return True
        return False

    def check_lost_ids(self):
        """Enhanced lost ID checking with miss tracking"""
        current = self.frame_count
        lost = []

        for tid, last in self.last_seen.items():
            if current - last > self.max_missing_frames:
                lost.append(tid)

        for tid in lost:
            # Check if object crossed line but disappeared before counting
            if tid in self.crossed_direction and tid not in self.counted and tid not in self.miss_tracked:
                direction = self.crossed_direction[tid]
                
                # Check if it reached completion zone
                completed = tid in self.completion_zone
                
                if not completed:
                    # Object crossed but didn't complete the movement
                    if direction == "IN":
                        self.miss_in_count += 1
                        print(f"⚠️ MISS IN - ID:{tid} (crossed line going IN but didn't complete)")
                    else:  # OUT
                        self.miss_out_count += 1
                        print(f"⚠️ MISS OUT - ID:{tid} (crossed line going OUT but didn't complete)")
                    
                    self.miss_tracked.add(tid)

            # If object was detected but never crossed the line - mark as FALSE
            elif tid not in self.crossed_direction and tid not in self.false_detections:
                self.false_detections.add(tid)
                self.false_count += 1
                print(f"❌ FALSE - ID:{tid} (detected but didn't cross line)")

            # Clean up tracking data
            self.hist.pop(tid, None)
            self.last_seen.pop(tid, None)
            self.crossed_direction.pop(tid, None)
            self.completion_zone.pop(tid, None)

    # ---------------- Reset Function ----------------
    def reset_all_data(self):
        """Reset all tracking data and start new session"""
        # End current session before resetting
        self.end_current_session()
        
        # Print session summary to console
        self.print_session_summary()
        
        # Reset counters
        self.hist.clear()
        self.last_seen.clear()
        self.crossed_ids.clear()
        self.counted.clear()
        self.false_detections.clear()
        self.crossed_direction.clear()
        self.completion_zone.clear()
        self.miss_tracked.clear()
        self.in_count = 0
        self.out_count = 0
        self.false_count = 0
        self.miss_in_count = 0
        self.miss_out_count = 0
        
        # Start new session
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

            for pt in self.temp_points:
                cv2.circle(frame, pt, 5, (0, 0, 255), -1)

            # Draw main counting line (white)
            if self.line_p1:
                cv2.line(frame, self.line_p1, self.line_p2, (255, 255, 255), 3)

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

                    # Check completion status
                    self.check_completion_status(tid, cx, cy)

                    if tid in self.hist:
                        px, py = self.hist[tid]
                        s1 = self.side(px, py, *self.line_p1, *self.line_p2)
                        s2 = self.side(cx, cy, *self.line_p1, *self.line_p2)

                        # Detect line crossing
                        if s1 * s2 < 0:
                            self.crossed_ids.add(tid)
                            
                            # Record which direction object is crossing
                            if s2 > 0:
                                self.crossed_direction[tid] = "IN"
                            else:
                                self.crossed_direction[tid] = "OUT"

                        # Count only when object has crossed AND reached completion zone
                        if tid in self.crossed_direction and tid not in self.counted:
                            if tid in self.completion_zone:
                                direction = self.crossed_direction[tid]
                                
                                if direction == "IN":
                                    self.in_count += 1
                                    print(f"✅ IN - ID:{tid}")
                                else:  # OUT
                                    self.out_count += 1
                                    print(f"✅ OUT - ID:{tid}")
                                
                                self.counted.add(tid)

                    self.hist[tid] = (cx, cy)

                    # Draw bounding box with color based on status
                    color = (0, 255, 0)  # Default green
                    if tid in self.crossed_direction and tid not in self.counted:
                        color = (0, 255, 255)  # Yellow - crossed but not completed
                    elif tid in self.counted:
                        color = (0, 200, 0)  # Dark green - completed
                    
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    
                    # Show ID and status
                    status = ""
                    if tid in self.crossed_direction:
                        status = f" ({self.crossed_direction[tid]})"
                    cv2.putText(frame, f"ID:{tid}{status}", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 0), 2)

            # Check for missed objects and false detections
            if self.line_p1:
                self.check_lost_ids()

            # ================= DISPLAY PANEL =================

            # Main overlay panel - increased height for two rows
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (1020, 130), (0, 0, 0), -1)
            frame = cv2.addWeighted(overlay, 0.4, frame, 0.6, 0)

            # --------- TITLE BAR ---------
            cv2.putText(frame, "TRACKING SYSTEM", (15, 32),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 200, 255), 3)
            cv2.circle(frame, (250, 24), 7, (0, 255, 0), -1)

            # --------- ROW 1: MAIN COUNTS ---------
            y_row1 = 70
            font_size = 0.9
            thickness = 3
            
            # Total IN
            cv2.putText(frame, "IN:", (15, y_row1),
                        cv2.FONT_HERSHEY_SIMPLEX, font_size, (0, 255, 150), thickness)
            cv2.putText(frame, str(self.in_count), (90, y_row1),
                        cv2.FONT_HERSHEY_SIMPLEX, font_size, (255, 255, 255), thickness)

            # Total OUT
            cv2.putText(frame, "OUT:", (200, y_row1),
                        cv2.FONT_HERSHEY_SIMPLEX, font_size, (100, 180, 255), thickness)
            cv2.putText(frame, str(self.out_count), (300, y_row1),
                        cv2.FONT_HERSHEY_SIMPLEX, font_size, (255, 255, 255), thickness)

            # False Detections
            cv2.putText(frame, "FALSE:", (420, y_row1),
                        cv2.FONT_HERSHEY_SIMPLEX, font_size, (0, 100, 255), thickness)
            cv2.putText(frame, str(self.false_count), (560, y_row1),
                        cv2.FONT_HERSHEY_SIMPLEX, font_size, (255, 255, 255), thickness)

            # --------- ROW 2: MISS COUNTS ---------
            y_row2 = 110
            font_size_small = 0.75
            thickness_small = 2
            
            # Miss IN
            cv2.putText(frame, "MISS IN:", (15, y_row2),
                        cv2.FONT_HERSHEY_SIMPLEX, font_size_small, (100, 255, 255), thickness_small)
            cv2.putText(frame, str(self.miss_in_count), (160, y_row2),
                        cv2.FONT_HERSHEY_SIMPLEX, font_size_small, (255, 255, 255), thickness_small)

            # Miss OUT
            cv2.putText(frame, "MISS OUT:", (280, y_row2),
                        cv2.FONT_HERSHEY_SIMPLEX, font_size_small, (255, 100, 255), thickness_small)
            cv2.putText(frame, str(self.miss_out_count), (445, y_row2),
                        cv2.FONT_HERSHEY_SIMPLEX, font_size_small, (255, 255, 255), thickness_small)

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
    # Example usage:
    counter = ObjectCounter(
        source="your_video.mp4",  # or 0 for webcam, or "rtsp://..."
        model="best_float32.tflite",
        classes_to_count=[0],
        show=True
    )
    counter.run()
