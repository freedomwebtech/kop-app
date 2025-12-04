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

        # -------- Movement Detection --------
        self.movement_hist = {}  # Store position history for movement detection
        self.stationary_frames = {}  # Count frames object hasn't moved
        self.movement_threshold = 10  # Minimum pixels to consider as movement
        self.stationary_limit = 30  # Frames before considering object stationary
        self.stationary_ids = set()  # IDs that are stationary

        # -------- Counters --------
        self.in_count = 0
        self.out_count = 0

        # -------- MISSED LOGIC --------
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
            'missed_cross': 0
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

    # ---------------- Movement Detection ----------------
    def check_movement(self, tid, cx, cy):
        """Check if object is moving or stationary"""
        if tid not in self.movement_hist:
            self.movement_hist[tid] = [(cx, cy)]
            self.stationary_frames[tid] = 0
            return True
        
        # Add current position to history
        self.movement_hist[tid].append((cx, cy))
        
        # Keep only last 10 positions for efficiency
        if len(self.movement_hist[tid]) > 10:
            self.movement_hist[tid].pop(0)
        
        # Check movement from last position
        if len(self.movement_hist[tid]) >= 2:
            prev_x, prev_y = self.movement_hist[tid][-2]
            distance = np.sqrt((cx - prev_x)**2 + (cy - prev_y)**2)
            
            if distance < self.movement_threshold:
                self.stationary_frames[tid] += 1
            else:
                self.stationary_frames[tid] = 0
                
            # If stationary for too long, mark as stationary
            if self.stationary_frames[tid] >= self.stationary_limit:
                if tid not in self.stationary_ids:
                    self.stationary_ids.add(tid)
                    print(f"⚠️ STATIONARY DETECTED - ID:{tid} - Removing from tracking")
                return False
        
        return True

    # ---------------- Remove Stationary Object ----------------
    def remove_stationary_object(self, tid):
        """Remove stationary object from all tracking data"""
        # Remove from missed counts if it was there
        self.missed_in.discard(tid)
        self.missed_out.discard(tid)
        self.missed_cross.discard(tid)
        
        # Remove from tracking data
        self.hist.pop(tid, None)
        self.last_seen.pop(tid, None)
        self.crossed_ids.discard(tid)
        self.counted.discard(tid)
        self.movement_hist.pop(tid, None)
        self.stationary_frames.pop(tid, None)

    # ---------------- MISSED TRACK HANDLER ----------------
    def check_lost_ids(self):
        current = self.frame_count
        lost = []

        for tid, last in self.last_seen.items():
            if current - last > self.max_missing_frames:
                lost.append(tid)

        for tid in lost:
            # Skip if it's a stationary object
            if tid in self.stationary_ids:
                self.remove_stationary_object(tid)
                self.stationary_ids.discard(tid)
                continue
                
            if tid in self.crossed_ids and tid not in self.counted:
                self.missed_cross.add(tid)

            elif tid not in self.counted and tid in self.hist:
                cx, cy = self.hist[tid]
                s = self.side(cx, cy, *self.line_p1, *self.line_p2)

                if s < 0:
                    self.missed_in.add(tid)
                else:
                    self.missed_out.add(tid)

            self.hist.pop(tid, None)
            self.last_seen.pop(tid, None)
            self.movement_hist.pop(tid, None)
            self.stationary_frames.pop(tid, None)

    # ---------------- Reset Function ----------------
    def reset_all_data(self):
        """Reset all tracking data and start new session"""
        self.end_current_session()
        self.print_session_summary()
        
        # Reset all data
        self.hist.clear()
        self.last_seen.clear()
        self.crossed_ids.clear()
        self.counted.clear()
        self.missed_in.clear()
        self.missed_out.clear()
        self.missed_cross.clear()
        self.movement_hist.clear()
        self.stationary_frames.clear()
        self.stationary_ids.clear()
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

            for pt in self.temp_points:
                cv2.circle(frame, pt, 5, (0, 0, 255), -1)

            if self.line_p1:
                cv2.line(frame, self.line_p1, self.line_p2,
                         (255, 255, 255), 2)

            results = self.model.track(
                frame, persist=True, classes=self.classes, conf=0.80)

            if results[0].boxes.id is not None and self.line_p1:
                ids = results[0].boxes.id.cpu().numpy().astype(int)
                boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)

                for tid, box in zip(ids, boxes):
                    x1, y1, x2, y2 = box
                    cx = int((x1 + x2) / 2)
                    cy = int((y1 + y2) / 2)

                    # Check if object is moving
                    is_moving = self.check_movement(tid, cx, cy)
                    
                    # Skip stationary objects
                    if not is_moving:
                        self.remove_stationary_object(tid)
                        continue

                    self.last_seen[tid] = self.frame_count

                    if tid in self.hist:
                        px, py = self.hist[tid]
                        s1 = self.side(px, py, *self.line_p1, *self.line_p2)
                        s2 = self.side(cx, cy, *self.line_p1, *self.line_p2)

                        if s1 * s2 < 0:  # Crossed the line
                            self.crossed_ids.add(tid)

                            if tid not in self.counted:
                                if s2 > 0:  # Going IN
                                    self.in_count += 1
                                    print(f"✅ IN - ID:{tid}")
                                else:  # Going OUT
                                    self.out_count += 1
                                    print(f"✅ OUT - ID:{tid}")

                                self.counted.add(tid)

                    self.hist[tid] = (cx, cy)

                    # Draw bounding box (green for moving objects)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f"ID:{tid}", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 0), 2)

            # Check for missed objects
            if self.line_p1:
                self.check_lost_ids()

            # ================= DISPLAY PANEL =================

            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (1020, 100), (0, 0, 0), -1)
            frame = cv2.addWeighted(overlay, 0.4, frame, 0.6, 0)

            # --------- TITLE BAR ---------
            cv2.putText(frame, "TRACKING SYSTEM", (15, 32),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 200, 255), 3)
            cv2.circle(frame, (250, 24), 7, (0, 255, 0), -1)

            # --------- COUNTS ROW ---------
            y_row = 70
            font_size = 0.9
            thickness = 3
            
            # Total IN
            cv2.putText(frame, "IN:", (15, y_row),
                        cv2.FONT_HERSHEY_SIMPLEX, font_size, (0, 255, 150), thickness)
            cv2.putText(frame, str(self.in_count), (90, y_row),
                        cv2.FONT_HERSHEY_SIMPLEX, font_size, (255, 255, 255), thickness)

            # Total OUT
            cv2.putText(frame, "OUT:", (180, y_row),
                        cv2.FONT_HERSHEY_SIMPLEX, font_size, (100, 180, 255), thickness)
            cv2.putText(frame, str(self.out_count), (270, y_row),
                        cv2.FONT_HERSHEY_SIMPLEX, font_size, (255, 255, 255), thickness)

            # Missed IN
            cv2.putText(frame, "MISS IN:", (380, y_row),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (100, 255, 255), 2)
            cv2.putText(frame, str(len(self.missed_in)), (520, y_row),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

            # Missed OUT
            cv2.putText(frame, "MISS OUT:", (600, y_row),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 100, 255), 2)
            cv2.putText(frame, str(len(self.missed_out)), (750, y_row),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

            # Missed CROSS
            cv2.putText(frame, "CROSS:", (830, y_row),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (100, 100, 255), 2)
            cv2.putText(frame, str(len(self.missed_cross)), (940, y_row),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

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
