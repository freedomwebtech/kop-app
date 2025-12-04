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
                 json_file="line_coords_back.json",
                 movement_threshold=5):  # Minimum pixel movement to consider object as moving

        self.source = source
        self.model = YOLO(model)
        self.names = self.model.names
        self.classes = classes_to_count
        self.show = show
        
        # Movement detection threshold
        self.movement_threshold = movement_threshold

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
        self.hist = {}  # Position history
        self.last_seen = {}  # Last frame seen
        self.crossed_ids = set()
        self.counted = set()
        
        # -------- Movement Detection --------
        self.position_history = {}  # Store last N positions for each ID
        self.history_length = 5  # Number of frames to track
        self.stationary_ids = set()  # IDs that are currently stationary
        self.moving_ids = set()  # IDs that have shown movement
        
        # -------- Valid Track IDs --------
        self.valid_track_ids = set()  # Only track IDs that have moved

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
        self.line_mode = False
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
            'missed_cross': 0,
            'stationary_filtered': 0  # Count of filtered stationary objects
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
        print(f"Stationary Filtered: {self.current_session_data['stationary_filtered']}")
        print("=" * 80 + "\n")

    # ---------------- Mouse ----------------
    def mouse_event(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and self.line_mode:
            self.temp_points.append((x, y))
            print(f"📍 Point {len(self.temp_points)} set at: ({x}, {y})")
            
            if len(self.temp_points) == 2:
                self.line_p1, self.line_p2 = self.temp_points
                self.temp_points = []
                self.line_mode = False
                self.save_line()
                print(f"✅ Line saved: {self.line_p1} -> {self.line_p2}")

    # ---------------- Save / Load Line ----------------
    def save_line(self):
        """Save line coordinates to JSON file"""
        try:
            data = {
                "line_p1": self.line_p1,
                "line_p2": self.line_p2,
                "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                "source": str(self.source)
            }
            
            with open(self.json_file, "w") as f:
                json.dump(data, f, indent=4)
            
            print(f"💾 Line coordinates saved to '{self.json_file}'")
                
        except Exception as e:
            print(f"❌ Error saving line: {e}")

    def load_line(self):
        """Load line coordinates from JSON file"""
        if os.path.exists(self.json_file):
            try:
                with open(self.json_file) as f:
                    data = json.load(f)
                    self.line_p1 = tuple(data["line_p1"])
                    self.line_p2 = tuple(data["line_p2"])
                    
                print(f"✅ Line loaded from '{self.json_file}'")
                print(f"   Point 1: {self.line_p1}")
                print(f"   Point 2: {self.line_p2}")
                    
            except Exception as e:
                print(f"❌ Error loading line: {e}")
                print("   Please draw a new line by pressing 'L'")
        else:
            print(f"ℹ️  No saved line found. Press 'L' to draw a new line.")

    # ---------------- Utility ----------------
    def side(self, px, py, x1, y1, x2, y2):
        return (x2 - x1) * (py - y1) - (y2 - y1) * (px - x1)

    # ---------------- Movement Detection ----------------
    def is_object_moving(self, tid, current_pos):
        """
        Check if an object is moving by analyzing position history
        Returns: (is_moving, average_movement)
        """
        if tid not in self.position_history:
            self.position_history[tid] = []
        
        # Add current position to history
        self.position_history[tid].append(current_pos)
        
        # Keep only last N positions
        if len(self.position_history[tid]) > self.history_length:
            self.position_history[tid].pop(0)
        
        # Need at least 3 positions to determine movement
        if len(self.position_history[tid]) < 3:
            return False, 0
        
        # Calculate movement between consecutive positions
        movements = []
        for i in range(1, len(self.position_history[tid])):
            prev_x, prev_y = self.position_history[tid][i-1]
            curr_x, curr_y = self.position_history[tid][i]
            
            distance = np.sqrt((curr_x - prev_x)**2 + (curr_y - prev_y)**2)
            movements.append(distance)
        
        # Average movement
        avg_movement = np.mean(movements)
        
        # Object is moving if average movement exceeds threshold
        is_moving = avg_movement >= self.movement_threshold
        
        return is_moving, avg_movement

    def update_movement_status(self, tid, current_pos):
        """Update the movement status of a tracked object"""
        is_moving, avg_movement = self.is_object_moving(tid, current_pos)
        
        if is_moving:
            # Mark as moving and valid for tracking
            if tid not in self.moving_ids:
                self.moving_ids.add(tid)
                self.valid_track_ids.add(tid)
                print(f"🚀 ID:{tid} started moving (avg: {avg_movement:.1f}px)")
            
            # Remove from stationary if it was there
            self.stationary_ids.discard(tid)
            
        else:
            # Mark as stationary (but don't remove from valid if it was moving before)
            if tid not in self.stationary_ids and tid in self.moving_ids:
                self.stationary_ids.add(tid)
                print(f"⏸️  ID:{tid} became stationary")

    # ---------------- MISSED TRACK HANDLER ----------------
    def check_lost_ids(self):
        current = self.frame_count
        lost = []

        for tid, last in self.last_seen.items():
            if current - last > self.max_missing_frames:
                lost.append(tid)

        for tid in lost:
            # Only process if it was a valid moving object
            if tid not in self.valid_track_ids:
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

            # Cleanup
            self.hist.pop(tid, None)
            self.last_seen.pop(tid, None)
            self.position_history.pop(tid, None)
            self.stationary_ids.discard(tid)
            self.moving_ids.discard(tid)

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
        self.missed_in.clear()
        self.missed_out.clear()
        self.missed_cross.clear()
        self.position_history.clear()
        self.stationary_ids.clear()
        self.moving_ids.clear()
        self.valid_track_ids.clear()
        self.in_count = 0
        self.out_count = 0
        
        # Start new session
        self.start_new_session()
        print("✅ RESET DONE - New session started")

    # ---------------- Main Loop ----------------
    def run(self):
        print("\n" + "=" * 60)
        print("OBJECT COUNTER - CONTROLS")
        print("=" * 60)
        print("L      - Enter line drawing mode")
        print("O      - Reset & show session summary")
        print("+/-    - Adjust movement threshold")
        print("ESC    - Exit")
        print("=" * 60)
        print(f"Movement Threshold: {self.movement_threshold} pixels\n")

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

            # Draw temporary points during line creation
            for pt in self.temp_points:
                cv2.circle(frame, pt, 8, (0, 0, 255), -1)
                cv2.circle(frame, pt, 12, (255, 255, 255), 2)

            # Draw the counting line
            if self.line_p1:
                cv2.line(frame, self.line_p1, self.line_p2, (255, 255, 255), 3)
                cv2.circle(frame, self.line_p1, 6, (0, 255, 0), -1)
                cv2.circle(frame, self.line_p2, 6, (0, 255, 0), -1)

            # Show line drawing mode indicator
            if self.line_mode:
                cv2.putText(frame, "LINE DRAWING MODE - Click 2 points", 
                           (250, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.7, (0, 255, 255), 2)

            results = self.model.track(
                frame, persist=True, classes=self.classes, conf=0.80)

            if results[0].boxes.id is not None and self.line_p1:
                ids = results[0].boxes.id.cpu().numpy().astype(int)
                boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)

                for tid, box in zip(ids, boxes):
                    x1, y1, x2, y2 = box
                    cx = int((x1 + x2) / 2)
                    cy = int((y1 + y2) / 2)
                    current_pos = (cx, cy)

                    # Update movement status
                    self.update_movement_status(tid, current_pos)

                    # ✅ ONLY PROCESS IF OBJECT HAS SHOWN MOVEMENT
                    if tid not in self.valid_track_ids:
                        # Draw as ignored stationary object
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (128, 128, 128), 2)
                        cv2.putText(frame, f"STATIONARY", (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128, 128, 128), 2)
                        continue

                    # Update last seen
                    self.last_seen[tid] = self.frame_count

                    # Process tracking for moving objects
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

                    # Draw bounding box with status
                    if tid in self.stationary_ids:
                        # Currently stationary but was moving before
                        color = (0, 165, 255)  # Orange
                        status = "PAUSED"
                    else:
                        # Currently moving
                        color = (0, 255, 0)  # Green
                        status = "MOVING"
                    
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, f"ID:{tid} {status}", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # Check for missed objects
            if self.line_p1:
                self.check_lost_ids()

            # ================= DISPLAY PANEL =================

            # Main overlay panel
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (1020, 130), (0, 0, 0), -1)
            frame = cv2.addWeighted(overlay, 0.4, frame, 0.6, 0)

            # --------- TITLE BAR ---------
            cv2.putText(frame, "TRACKING SYSTEM", (15, 32),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 200, 255), 3)
            cv2.circle(frame, (250, 24), 7, (0, 255, 0), -1)

            # Movement threshold display
            cv2.putText(frame, f"Move Threshold: {self.movement_threshold}px", (800, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)

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

            # --------- TRACKING STATUS ROW ---------
            y_row2 = 105
            cv2.putText(frame, f"Active: {len(self.moving_ids)}", (15, y_row2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(frame, f"Paused: {len(self.stationary_ids)}", (150, y_row2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
            cv2.putText(frame, f"Filtered: {len(self.position_history) - len(self.valid_track_ids)}", 
                        (300, y_row2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (128, 128, 128), 2)

            if self.show:
                cv2.imshow("ObjectCounter", frame)
                key = cv2.waitKey(1) & 0xFF

                if key == ord('l') or key == ord('L'):
                    self.line_mode = True
                    self.temp_points = []
                    print("\n🎯 LINE DRAWING MODE ACTIVATED")
                    print("   Click 2 points on the video to draw the counting line")

                elif key == ord('o') or key == ord('O'):
                    self.reset_all_data()

                elif key == ord('+') or key == ord('='):
                    self.movement_threshold += 1
                    print(f"Movement threshold increased to: {self.movement_threshold}px")

                elif key == ord('-') or key == ord('_'):
                    self.movement_threshold = max(1, self.movement_threshold - 1)
                    print(f"Movement threshold decreased to: {self.movement_threshold}px")

                elif key == 27:  # ESC
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
        show=True,
        movement_threshold=5  # Adjust based on your video resolution and object speed
    )
    counter.run()
