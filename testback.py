import cv2
from ultralytics import YOLO
import json
import os
from imutils.video import VideoStream
import time
from datetime import datetime

# ==========================================================
#                     OBJECT COUNTER CLASS (RTSP ONLY)
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

        # -------- RTSP Stream Only --------
        print(f"🔄 Connecting to RTSP stream: {source}")
        self.cap = VideoStream(source).start()
        time.sleep(3.0)  # Give camera time to initialize
        print("✅ RTSP stream started")

        # -------- Session Data --------
        self.session_start_time = datetime.now()
        self.current_session_data = None
        self.start_new_session()

        # -------- Tracking Data --------
        self.hist = {}
        self.last_seen = {}
        self.crossed_ids = set()
        self.counted = set()
        
        # Track which side object first appeared on
        self.origin_side = {}

        # -------- Counters --------
        self.in_count = 0
        self.out_count = 0

        # ✅ ONLY IN/OUT MISSED
        self.missed_in = set()
        self.missed_out = set()
        self.max_missing_frames = 40

        # -------- Line --------
        self.line_p1 = None
        self.line_p2 = None
        self.temp_points = []
        self.json_file = json_file
        self.load_line()

        self.frame_count = 0

        cv2.namedWindow("ObjectCounter - RTSP Back")
        cv2.setMouseCallback("ObjectCounter - RTSP Back", self.mouse_event)

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
            'missed_out': 0
        }

    def end_current_session(self):
        """End the current session and save data"""
        if self.current_session_data:
            self.current_session_data['end_time'] = datetime.now().strftime('%H:%M:%S')
            self.current_session_data['in_count'] = self.in_count
            self.current_session_data['out_count'] = self.out_count
            self.current_session_data['missed_in'] = len(self.missed_in)
            self.current_session_data['missed_out'] = len(self.missed_out)

    def print_session_summary(self):
        """Print session summary to console"""
        print("\n" + "=" * 80)
        print("                    SESSION SUMMARY (BACK CAMERA)")
        print("=" * 80)
        print(f"Day:           {self.current_session_data['day']}")
        print(f"Date:          {self.current_session_data['date']}")
        print(f"Start Time:    {self.current_session_data['start_time']}")
        print(f"End Time:      {self.current_session_data['end_time']}")
        print(f"IN Count:      {self.current_session_data['in_count']}")
        print(f"OUT Count:     {self.current_session_data['out_count']}")
        print(f"Missed IN:     {self.current_session_data['missed_in']}")
        print(f"Missed OUT:    {self.current_session_data['missed_out']}")
        print("=" * 80 + "\n")

    # ---------------- Mouse ----------------
    def mouse_event(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.temp_points.append((x, y))
            print(f"Line point {len(self.temp_points)}: ({x}, {y})")
            
            if len(self.temp_points) == 2:
                self.line_p1, self.line_p2 = self.temp_points
                self.temp_points = []
                self.save_line()
                print("✅ Line completed and saved")

    # ---------------- Save / Load Line ----------------
    def save_line(self):
        with open(self.json_file, "w") as f:
            json.dump({"line_p1": self.line_p1, "line_p2": self.line_p2}, f)
        print(f"✅ Line saved to {self.json_file}")

    def load_line(self):
        if os.path.exists(self.json_file):
            with open(self.json_file) as f:
                data = json.load(f)
                self.line_p1 = tuple(data["line_p1"])
                self.line_p2 = tuple(data["line_p2"])
                print(f"✅ Line loaded from {self.json_file}")

    # ---------------- Utility ----------------
    def side(self, px, py, x1, y1, x2, y2):
        return (x2 - x1) * (py - y1) - (y2 - y1) * (px - x1)

    # ================= MISSED TRACK HANDLER (ONLY IN/OUT) =================
    def check_lost_ids(self):
        """
        Check for objects that disappeared without being counted.
        Only tracks MISSED IN and MISSED OUT.
        """
        current = self.frame_count
        lost = []

        # ✅ FIX: Convert to list to avoid modification during iteration
        for tid, last in list(self.last_seen.items()):
            if current - last > self.max_missing_frames:
                lost.append(tid)

        for tid in lost:
            # Only check crossed objects that weren't counted
            if tid in self.crossed_ids and tid not in self.counted:
                if tid in self.hist:
                    last_cx, last_cy = self.hist[tid]
                    last_side = self.side(last_cx, last_cy, *self.line_p1, *self.line_p2)
                    
                    # IN logic: ended on positive side
                    if last_side > 0:
                        self.missed_in.add(tid)
                        print(f"⚠️ MISSED IN (Back) - ID:{tid}")
                    
                    # OUT logic: ended on negative side
                    elif last_side < 0:
                        self.missed_out.add(tid)
                        print(f"⚠️ MISSED OUT (Back) - ID:{tid}")

            # Cleanup
            self.hist.pop(tid, None)
            self.last_seen.pop(tid, None)
            self.origin_side.pop(tid, None)

    # ---------------- Reset Function ----------------
    def reset_all_data(self):
        """Reset all tracking data and start new session"""
        self.end_current_session()
        self.print_session_summary()
        
        self.hist.clear()
        self.last_seen.clear()
        self.crossed_ids.clear()
        self.counted.clear()
        self.origin_side.clear()
        self.missed_in.clear()
        self.missed_out.clear()
        self.in_count = 0
        self.out_count = 0
        
        self.start_new_session()
        print("✅ RESET DONE - New session started (Back Camera)")

    # ---------------- Main Loop (RTSP Only) ----------------
    def run(self):
        print("RUNNING... [RTSP BACK CAMERA]")
        print("Click 2 points to draw counting line | Press O to Reset | ESC to Exit")

        while True:
            # ✅ RTSP Stream frame reading
            frame = self.cap.read()
            
            # ✅ Validate frame
            if frame is None:
                print("⚠️ No frame from RTSP stream, retrying...")
                time.sleep(0.1)
                continue

            self.frame_count += 1
            
            # Optional: Process every frame or skip frames
            # if self.frame_count % 3 != 0:
            #     continue
            
            frame = cv2.resize(frame, (640, 360))

            # Draw temporary line points
            for pt in self.temp_points:
                cv2.circle(frame, pt, 5, (0, 0, 255), -1)

            # Draw counting line
            if self.line_p1:
                cv2.line(frame, self.line_p1, self.line_p2,
                         (255, 255, 255), 2)

            # YOLO tracking
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

                    # Set origin side on first detection
                    if tid not in self.hist:
                        s_init = self.side(cx, cy, *self.line_p1, *self.line_p2)
                        if s_init < 0:
                            self.origin_side[tid] = "IN"
                        else:
                            self.origin_side[tid] = "OUT"

                    # Check for line crossing
                    if tid in self.hist:
                        px, py = self.hist[tid]
                        s1 = self.side(px, py, *self.line_p1, *self.line_p2)
                        s2 = self.side(cx, cy, *self.line_p1, *self.line_p2)

                        # Line crossed (different signs)
                        if s1 * s2 < 0:
                            self.crossed_ids.add(tid)

                            if tid not in self.counted:
                                if s2 > 0:
                                    self.in_count += 1
                                    print(f"✅ IN (Back) - ID:{tid}")
                                else:
                                    self.out_count += 1
                                    print(f"✅ OUT (Back) - ID:{tid}")

                                self.counted.add(tid)

                    self.hist[tid] = (cx, cy)

                    origin_label = self.origin_side.get(tid, "?")
                    
                    # Draw bounding box and ID
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f"ID:{tid} [{origin_label}]", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 0), 2)

            # Check for lost objects
            if self.line_p1:
                self.check_lost_ids()

            # Display counts on frame
            cv2.putText(frame, f"IN: {self.in_count}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"OUT: {self.out_count}", (10, 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(frame, "BACK CAMERA", (10, 110),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

            if self.show:
                cv2.imshow("ObjectCounter - RTSP Back", frame)
                key = cv2.waitKey(1) & 0xFF

                if key == ord('o') or key == ord('O'):
                    self.reset_all_data()

                elif key == 27:  # ESC
                    print("👋 Exiting...")
                    break

        # Cleanup
        self.end_current_session()
        self.print_session_summary()
        self.cap.stop()
        cv2.destroyAllWindows()


# ==========================================================
#                        MAIN EXECUTION
# ==========================================================

if __name__ == "__main__":
    # RTSP stream configuration
    RTSP_URL = "rtsp://username:password@192.168.1.100:554/stream1"
    
    counter = ObjectCounter(
        source=RTSP_URL,
        model="best_float32.tflite",
        classes_to_count=[0],
        show=True,
        json_file="line_coords_back.json"
    )
    counter.run()
