import cv2
from ultralytics import YOLO
import json
import os
import numpy as np
from imutils.video import VideoStream
import time
from datetime import datetime

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

        # -------- Counters --------
        self.in_count = 0
        self.out_count = 0
        self.miss_in_count = 0
        self.miss_out_count = 0

        # ✅ NEW: Active tracking status for moving objects
        self.active_in_direction = set()   # IDs currently moving toward IN
        self.active_out_direction = set()  # IDs currently moving toward OUT
        
        # ✅ Permanent missed IDs
        self.permanent_missed_in = set()
        self.permanent_missed_out = set()
        
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
            'miss_in_count': 0,
            'miss_out_count': 0
        }

    def end_current_session(self):
        """End the current session and save data"""
        if self.current_session_data:
            self.current_session_data['end_time'] = datetime.now().strftime('%H:%M:%S')
            self.current_session_data['in_count'] = self.in_count
            self.current_session_data['out_count'] = self.out_count
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

    # ✅ NEW: Check lost IDs and convert active misses to permanent
    def check_lost_ids(self):
        """
        Check for disappeared objects:
        - If ID was in active_in_direction but disappeared → permanent MISS IN
        - If ID was in active_out_direction but disappeared → permanent MISS OUT
        """
        current = self.frame_count
        lost = []

        for tid, last in self.last_seen.items():
            if current - last > self.max_missing_frames:
                lost.append(tid)

        for tid in lost:
            # ✅ ID was moving IN but disappeared without counting
            if tid in self.active_in_direction:
                if tid not in self.permanent_missed_in:
                    self.permanent_missed_in.add(tid)
                    print(f"❌ PERMANENT MISS IN - ID:{tid} (disappeared)")
                self.active_in_direction.discard(tid)
            
            # ✅ ID was moving OUT but disappeared without counting
            if tid in self.active_out_direction:
                if tid not in self.permanent_missed_out:
                    self.permanent_missed_out.add(tid)
                    print(f"❌ PERMANENT MISS OUT - ID:{tid} (disappeared)")
                self.active_out_direction.discard(tid)

            # Cleanup
            self.hist.pop(tid, None)
            self.last_seen.pop(tid, None)

        # ✅ Update miss counts (active + permanent)
        self.miss_in_count = len(self.active_in_direction) + len(self.permanent_missed_in)
        self.miss_out_count = len(self.active_out_direction) + len(self.permanent_missed_out)

    # ---------------- Reset Function ----------------
    def reset_all_data(self):
        """Reset all tracking data and start new session"""
        self.end_current_session()
        self.print_session_summary()
        
        self.hist.clear()
        self.last_seen.clear()
        self.crossed_ids.clear()
        self.counted.clear()
        self.active_in_direction.clear()
        self.active_out_direction.clear()
        self.permanent_missed_in.clear()
        self.permanent_missed_out.clear()
        self.in_count = 0
        self.out_count = 0
        self.miss_in_count = 0
        self.miss_out_count = 0
        
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

                    if tid in self.hist:
                        px, py = self.hist[tid]
                        s1 = self.side(px, py, *self.line_p1, *self.line_p2)
                        s2 = self.side(cx, cy, *self.line_p1, *self.line_p2)

                        # ✅ Detect line crossing
                        if s1 * s2 < 0:
                            self.crossed_ids.add(tid)

                            # ✅ Moving toward IN side (s2 > 0)
                            if s2 > 0:
                                if tid not in self.counted:
                                    # Add to active IN tracking
                                    self.active_in_direction.add(tid)
                                    print(f"🔄 ACTIVE IN - ID:{tid} (crossing toward IN)")
                                else:
                                    # Already counted, just update count
                                    self.in_count += 1
                                    # Remove from active tracking
                                    self.active_in_direction.discard(tid)
                                    print(f"✅ IN COUNTED - ID:{tid}")
                            
                            # ✅ Moving toward OUT side (s2 < 0)
                            else:
                                if tid not in self.counted:
                                    # Add to active OUT tracking
                                    self.active_out_direction.add(tid)
                                    print(f"🔄 ACTIVE OUT - ID:{tid} (crossing toward OUT)")
                                else:
                                    # Already counted, just update count
                                    self.out_count += 1
                                    # Remove from active tracking
                                    self.active_out_direction.discard(tid)
                                    print(f"✅ OUT COUNTED - ID:{tid}")
                        
                        # ✅ Check if object completed the crossing
                        else:
                            # Object is on IN side and was being tracked
                            if s2 > 0 and tid in self.active_in_direction:
                                if tid not in self.counted:
                                    # Successfully completed IN crossing
                                    self.in_count += 1
                                    self.counted.add(tid)
                                    self.active_in_direction.discard(tid)
                                    print(f"✅ IN COMPLETED - ID:{tid}")
                            
                            # Object is on OUT side and was being tracked
                            elif s2 < 0 and tid in self.active_out_direction:
                                if tid not in self.counted:
                                    # Successfully completed OUT crossing
                                    self.out_count += 1
                                    self.counted.add(tid)
                                    self.active_out_direction.discard(tid)
                                    print(f"✅ OUT COMPLETED - ID:{tid}")

                    self.hist[tid] = (cx, cy)

                    # ✅ Draw bounding box with status
                    color = (0, 255, 0)  # Default green
                    status = ""
                    
                    if tid in self.active_in_direction:
                        color = (0, 255, 255)  # Yellow for active IN
                        status = " [→IN]"
                    elif tid in self.active_out_direction:
                        color = (255, 100, 255)  # Pink for active OUT
                        status = " [→OUT]"
                    elif tid in self.counted:
                        color = (0, 255, 0)  # Green for counted
                        status = " [✓]"
                    
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, f"ID:{tid}{status}", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            if self.line_p1:
                self.check_lost_ids()

            # ================= DISPLAY =================
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (1020, 130), (0, 0, 0), -1)
            frame = cv2.addWeighted(overlay, 0.4, frame, 0.6, 0)

            # Title
            cv2.putText(frame, "ACTIVE TRACKING SYSTEM", (15, 32),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 200, 255), 3)
            cv2.circle(frame, (290, 24), 7, (0, 255, 0), -1)

            # Main counts row
            y_row1 = 70
            font_size = 0.9
            thickness = 3
            
            # IN Count
            cv2.putText(frame, "IN:", (15, y_row1),
                        cv2.FONT_HERSHEY_SIMPLEX, font_size, (0, 255, 150), thickness)
            cv2.putText(frame, str(self.in_count), (90, y_row1),
                        cv2.FONT_HERSHEY_SIMPLEX, font_size, (255, 255, 255), thickness)

            # OUT Count
            cv2.putText(frame, "OUT:", (200, y_row1),
                        cv2.FONT_HERSHEY_SIMPLEX, font_size, (100, 180, 255), thickness)
            cv2.putText(frame, str(self.out_count), (300, y_row1),
                        cv2.FONT_HERSHEY_SIMPLEX, font_size, (255, 255, 255), thickness)

            # Second row - Active and Missed counts
            y_row2 = 110
            font_size_small = 0.65
            thickness_small = 2
            
            # Active IN
            cv2.putText(frame, f"Active→IN: {len(self.active_in_direction)}", (15, y_row2),
                        cv2.FONT_HERSHEY_SIMPLEX, font_size_small, (0, 255, 255), thickness_small)

            # Active OUT
            cv2.putText(frame, f"Active→OUT: {len(self.active_out_direction)}", (200, y_row2),
                        cv2.FONT_HERSHEY_SIMPLEX, font_size_small, (255, 100, 255), thickness_small)

            # Total MISS IN (active + permanent)
            cv2.putText(frame, f"MISS IN: {self.miss_in_count}", (420, y_row2),
                        cv2.FONT_HERSHEY_SIMPLEX, font_size_small, (100, 255, 255), thickness_small)

            # Total MISS OUT (active + permanent)
            cv2.putText(frame, f"MISS OUT: {self.miss_out_count}", (640, y_row2),
                        cv2.FONT_HERSHEY_SIMPLEX, font_size_small, (255, 100, 255), thickness_small)

            if self.show:
                cv2.imshow("ObjectCounter", frame)
                key = cv2.waitKey(1) & 0xFF

                if key == ord('o') or key == ord('O'):
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
