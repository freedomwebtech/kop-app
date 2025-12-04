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

BROWN_LOWER = np.array([5, 80, 60])
BROWN_UPPER = np.array([20, 255, 255])

WHITE_LOWER = np.array([0, 0, 200])
WHITE_UPPER = np.array([180, 40, 255])

def detect_box_color(frame, box):
    x1, y1, x2, y2 = box
    roi = frame[y1:y2, x1:x2]

    if roi.size == 0:
        return "Unknown"

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    brown_mask = cv2.inRange(hsv, BROWN_LOWER, BROWN_UPPER)
    brown_intensity = brown_mask.mean()

    white_mask = cv2.inRange(hsv, WHITE_LOWER, WHITE_UPPER)
    white_intensity = white_mask.mean()

    if brown_intensity > 20:
        return "Brown"
    elif white_intensity > 20:
        return "White"

    return "Unknown"


# ==========================================================
#                     OBJECT COUNTER CLASS
# ==========================================================

class ObjectCounter:
    def __init__(self, source, model="best_float32.tflite",
                 classes_to_count=[0], show=True,
                 json_file="line_coords.json"):

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
                self.fps = 30  # Default fallback
        else:
            self.fps = 30  # Default for RTSP

        # -------- Session Data --------
        self.session_start_time = datetime.now()
        self.current_session_data = None
        self.start_new_session()

        # -------- Tracking Data --------
        self.hist = {}
        self.last_seen = {}
        self.crossed_ids = set()
        self.counted = set()
        
        # ✅ Store color detected at crossing point
        self.color_at_crossing = {}

        # ✅ NEW: Track IN crossings with timestamp for delayed color detection
        self.pending_in_detections = {}  # {track_id: frame_number_when_crossed}
        self.delay_frames = int(0.5 * self.fps)  # 0.5 seconds worth of frames

        # -------- Counters --------
        self.in_count = 0
        self.out_count = 0

        self.color_in_count = {}
        self.color_out_count = {}

        # ✅ MISSED LOGIC
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
                s = self.side(cx, cy, *self.line_p1, *self.line_p2)

                if s > 0:
                    self.missed_in.add(tid)
                else:
                    self.missed_out.add(tid)

            self.hist.pop(tid, None)
            self.last_seen.pop(tid, None)
            self.color_at_crossing.pop(tid, None)
            self.pending_in_detections.pop(tid, None)  # Clean up pending detections

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
        self.color_at_crossing.clear()
        self.color_in_count.clear()
        self.color_out_count.clear()
        self.missed_in.clear()
        self.missed_out.clear()
        self.missed_cross.clear()
        self.pending_in_detections.clear()
        self.in_count = 0
        self.out_count = 0
        
        # Start new session
        self.start_new_session()
        print("✅ RESET DONE - New session started")

    # ---------------- Main Loop ----------------
    def run(self):
        print("RUNNING... Press O to Reset & Show Summary | ESC to Exit")
        print(f"FPS: {self.fps}, Delay frames for 0.5 seconds: {self.delay_frames}")

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

                    self.last_seen[tid] = self.frame_count

                    if tid in self.hist:
                        px, py = self.hist[tid]
                        s1 = self.side(px, py, *self.line_p1, *self.line_p2)
                        s2 = self.side(cx, cy, *self.line_p1, *self.line_p2)

                        # Check if object crossed the line
                        if s1 * s2 < 0:  # Crossed the line
                            self.crossed_ids.add(tid)

                            if tid not in self.counted:
                                # Determine direction
                                if s2 > 0:  # Going IN (positive side)
                                    # ✅ Schedule color detection for 0.5 seconds later
                                    self.pending_in_detections[tid] = self.frame_count
                                    self.in_count += 1
                                    print(f"📦 IN Detected - ID:{tid} (Color will be detected after 0.5 seconds)")
                                    
                                else:  # Going OUT (negative side)
                                    # For OUT: Detect color immediately
                                    color_name = detect_box_color(frame, box)
                                    
                                    self.out_count += 1
                                    self.color_out_count[color_name] = self.color_out_count.get(color_name, 0) + 1
                                    print(f"✅ OUT - ID:{tid} Color:{color_name}")

                                self.counted.add(tid)
                    
                    # ✅ Check if it's time to detect color for pending IN detections
                    if tid in self.pending_in_detections:
                        frames_since_crossing = self.frame_count - self.pending_in_detections[tid]
                        
                        if frames_since_crossing >= self.delay_frames:
                            # Detect color now (0.5 seconds have passed)
                            color_name = detect_box_color(frame, box)
                            self.color_in_count[color_name] = self.color_in_count.get(color_name, 0) + 1
                            self.color_at_crossing[tid] = color_name
                            print(f"✅ IN Color Detected - ID:{tid} Color:{color_name} (after 0.5 seconds)")
                            
                            # Remove from pending
                            del self.pending_in_detections[tid]

                    self.hist[tid] = (cx, cy)

                    # Display current detected color (if available)
                    if tid in self.color_at_crossing:
                        display_color = self.color_at_crossing[tid]
                    elif tid in self.pending_in_detections:
                        display_color = "Pending..."
                    else:
                        display_color = detect_box_color(frame, box)
                    
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f"{display_color}", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 0), 2)

            # ✅ MISSED CHECK ACTIVE
            if self.line_p1:
                self.check_lost_ids()

            # ================= ENHANCED DISPLAY WITH LARGER FONTS =================

            # Main overlay panel
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (1020, 160), (0, 0, 0), -1)
            frame = cv2.addWeighted(overlay, 0.35, frame, 0.65, 0)

            # --------- TITLE BAR ---------
            cv2.putText(frame, "TRACKING SYSTEM", (15, 32),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 200, 255), 3)
            cv2.circle(frame, (250, 24), 7, (0, 255, 0), -1)

            # --------- TOTAL COUNTS ROW ---------
            y_row1 = 65
            font_large = 0.8
            thickness_bold = 3
            
            # Total IN
            cv2.putText(frame, "IN:", (15, y_row1),
                        cv2.FONT_HERSHEY_SIMPLEX, font_large, (0, 255, 150), thickness_bold)
            cv2.putText(frame, str(self.in_count), (75, y_row1),
                        cv2.FONT_HERSHEY_SIMPLEX, font_large, (255, 255, 255), thickness_bold)

            # Total OUT
            cv2.putText(frame, "OUT:", (150, y_row1),
                        cv2.FONT_HERSHEY_SIMPLEX, font_large, (100, 180, 255), thickness_bold)
            cv2.putText(frame, str(self.out_count), (230, y_row1),
                        cv2.FONT_HERSHEY_SIMPLEX, font_large, (255, 255, 255), thickness_bold)

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

            # --------- SEPARATOR LINE ---------
            cv2.line(frame, (10, 85), (1010, 85), (100, 100, 100), 2)

            # --------- BROWN BOX COUNTS ROW ---------
            y_row2 = 118
            brown_in = self.color_in_count.get("Brown", 0)
            brown_out = self.color_out_count.get("Brown", 0)

            cv2.rectangle(frame, (15, y_row2 - 25), (50, y_row2 - 5), (19, 69, 139), -1)
            cv2.rectangle(frame, (15, y_row2 - 25), (50, y_row2 - 5), (255, 255, 255), 2)
            
            cv2.putText(frame, "BROWN IN:", (60, y_row2),
                        cv2.FONT_HERSHEY_SIMPLEX, font_large, (100, 150, 255), thickness_bold)
            cv2.putText(frame, str(brown_in), (240, y_row2),
                        cv2.FONT_HERSHEY_SIMPLEX, font_large, (255, 255, 255), thickness_bold)

            cv2.putText(frame, "BROWN OUT:", (350, y_row2),
                        cv2.FONT_HERSHEY_SIMPLEX, font_large, (100, 150, 255), thickness_bold)
            cv2.putText(frame, str(brown_out), (570, y_row2),
                        cv2.FONT_HERSHEY_SIMPLEX, font_large, (255, 255, 255), thickness_bold)

            # --------- WHITE BOX COUNTS ROW ---------
            y_row3 = 150
            white_in = self.color_in_count.get("White", 0)
            white_out = self.color_out_count.get("White", 0)

            cv2.rectangle(frame, (15, y_row3 - 25), (50, y_row3 - 5), (245, 245, 245), -1)
            cv2.rectangle(frame, (15, y_row3 - 25), (50, y_row3 - 5), (100, 100, 100), 2)
            
            cv2.putText(frame, "WHITE IN:", (60, y_row3),
                        cv2.FONT_HERSHEY_SIMPLEX, font_large, (200, 255, 200), thickness_bold)
            cv2.putText(frame, str(white_in), (240, y_row3),
                        cv2.FONT_HERSHEY_SIMPLEX, font_large, (255, 255, 255), thickness_bold)

            cv2.putText(frame, "WHITE OUT:", (350, y_row3),
                        cv2.FONT_HERSHEY_SIMPLEX, font_large, (200, 255, 200), thickness_bold)
            cv2.putText(frame, str(white_out), (570, y_row3),
                        cv2.FONT_HERSHEY_SIMPLEX, font_large, (255, 255, 255), thickness_bold)

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
