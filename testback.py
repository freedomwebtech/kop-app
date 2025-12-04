import cv2
from ultralytics import YOLO
import json
import os
import numpy as np
from imutils.video import VideoStream
import time
from datetime import datetime
from scipy.spatial import distance as dist

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

        # -------- Custom Tracking Data --------
        self.next_object_id = 0
        self.objects = {}  # custom_id -> {'centroid': (x,y), 'bbox': (x1,y1,x2,y2)}
        self.disappeared = {}  # custom_id -> frame_count
        
        # -------- Tracking Data --------
        self.hist = {}  # custom_id -> (cx, cy) - PREVIOUS position
        self.last_seen = {}
        self.crossed_ids = set()
        self.counted = set()

        # -------- Counters --------
        self.in_count = 0
        self.out_count = 0

        # -------- MISSED LOGIC --------
        self.missed_in = set()
        self.missed_out = set()
        self.missed_cross = set()
        self.max_missing_frames = 40
        self.max_distance = 80  # Maximum pixel distance for ID matching

        # -------- Line --------
        self.line_p1 = None
        self.line_p2 = None
        self.temp_points = []
        self.json_file = json_file
        self.load_line()

        self.frame_count = 0

        cv2.namedWindow("ObjectCounter")
        cv2.setMouseCallback("ObjectCounter", self.mouse_event)

    # ---------------- Custom Tracking Methods ----------------
    def register(self, centroid, bbox):
        """Register a new object with a unique ID"""
        self.objects[self.next_object_id] = {
            'centroid': centroid,
            'bbox': bbox
        }
        self.disappeared[self.next_object_id] = 0
        self.next_object_id += 1

    def deregister(self, object_id):
        """Remove an object from tracking"""
        del self.objects[object_id]
        del self.disappeared[object_id]

    def update_tracking(self, detections):
        """
        Update object tracking with new detections
        detections: list of (centroid, bbox) tuples
        """
        # If no detections, mark all as disappeared
        if len(detections) == 0:
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_missing_frames:
                    self.deregister(object_id)
            return self.objects

        # If no existing objects, register all detections
        if len(self.objects) == 0:
            for i in range(len(detections)):
                self.register(detections[i][0], detections[i][1])
        else:
            # Get current object IDs and centroids
            object_ids = list(self.objects.keys())
            object_centroids = [self.objects[oid]['centroid'] for oid in object_ids]

            # Get detection centroids
            detection_centroids = [d[0] for d in detections]

            # Compute distance matrix
            D = dist.cdist(np.array(object_centroids), np.array(detection_centroids))

            # Find minimum distance matches
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]

            used_rows = set()
            used_cols = set()

            # Match existing objects to detections
            for (row, col) in zip(rows, cols):
                if row in used_rows or col in used_cols:
                    continue

                # Only match if distance is reasonable
                if D[row, col] > self.max_distance:
                    continue

                object_id = object_ids[row]
                self.objects[object_id] = {
                    'centroid': detections[col][0],
                    'bbox': detections[col][1]
                }
                self.disappeared[object_id] = 0

                used_rows.add(row)
                used_cols.add(col)

            # Handle unused rows (disappeared objects)
            unused_rows = set(range(D.shape[0])) - used_rows
            for row in unused_rows:
                object_id = object_ids[row]
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_missing_frames:
                    self.deregister(object_id)

            # Handle unused cols (new objects)
            unused_cols = set(range(D.shape[1])) - used_cols
            for col in unused_cols:
                self.register(detections[col][0], detections[col][1])

        return self.objects

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

    # ---------------- MISSED TRACK HANDLER ----------------
    def check_lost_ids(self):
        """Check for objects that have been lost during tracking"""
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

                if s < 0:
                    self.missed_in.add(tid)
                else:
                    self.missed_out.add(tid)

            self.hist.pop(tid, None)
            self.last_seen.pop(tid, None)

    # ---------------- Reset Function ----------------
    def reset_all_data(self):
        """Reset all tracking data and start new session"""
        self.end_current_session()
        self.print_session_summary()
        
        # Reset custom tracking
        self.objects.clear()
        self.disappeared.clear()
        self.next_object_id = 0
        
        # Reset counters
        self.hist.clear()
        self.last_seen.clear()
        self.crossed_ids.clear()
        self.counted.clear()
        self.missed_in.clear()
        self.missed_out.clear()
        self.missed_cross.clear()
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
                cv2.line(frame, self.line_p1, self.line_p2, (255, 255, 255), 2)

            # Run YOLO detection (without tracking)
            results = self.model(frame, classes=self.classes, conf=0.80, verbose=False)

            detections = []
            if results[0].boxes is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
                
                for box in boxes:
                    x1, y1, x2, y2 = box
                    cx = int((x1 + x2) / 2)
                    cy = int((y1 + y2) / 2)
                    detections.append(((cx, cy), (x1, y1, x2, y2)))

            # Update custom tracking
            tracked_objects = self.update_tracking(detections)

            # Process tracked objects for line crossing
            if self.line_p1:
                for tid, obj_data in tracked_objects.items():
                    cx, cy = obj_data['centroid']
                    x1, y1, x2, y2 = obj_data['bbox']

                    self.last_seen[tid] = self.frame_count

                    # Check for line crossing
                    if tid in self.hist:
                        px, py = self.hist[tid]
                        s1 = self.side(px, py, *self.line_p1, *self.line_p2)
                        s2 = self.side(cx, cy, *self.line_p1, *self.line_p2)

                        if s1 * s2 < 0:  # Crossed the line
                            self.crossed_ids.add(tid)

                            if tid not in self.counted:
                                if s2 > 0:  # Going IN
                                    self.in_count += 1
                                    print(f"✅ IN - ID:{tid} | Position: ({cx},{cy})")
                                else:  # Going OUT
                                    self.out_count += 1
                                    print(f"✅ OUT - ID:{tid} | Position: ({cx},{cy})")

                                self.counted.add(tid)

                    # Update history with current position
                    self.hist[tid] = (cx, cy)

                    # Draw bounding box and trajectory
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.circle(frame, (cx, cy), 4, (0, 255, 255), -1)
                    cv2.putText(frame, f"ID:{tid}", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 0), 2)

                # Check for lost objects
                self.check_lost_ids()

            # ================= DISPLAY PANEL =================
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (1020, 100), (0, 0, 0), -1)
            frame = cv2.addWeighted(overlay, 0.4, frame, 0.6, 0)

            cv2.putText(frame, "TRACKING SYSTEM", (15, 32),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 200, 255), 3)
            cv2.circle(frame, (250, 24), 7, (0, 255, 0), -1)

            y_row = 70
            font_size = 0.9
            thickness = 3
            
            cv2.putText(frame, "IN:", (15, y_row),
                        cv2.FONT_HERSHEY_SIMPLEX, font_size, (0, 255, 150), thickness)
            cv2.putText(frame, str(self.in_count), (90, y_row),
                        cv2.FONT_HERSHEY_SIMPLEX, font_size, (255, 255, 255), thickness)

            cv2.putText(frame, "OUT:", (180, y_row),
                        cv2.FONT_HERSHEY_SIMPLEX, font_size, (100, 180, 255), thickness)
            cv2.putText(frame, str(self.out_count), (270, y_row),
                        cv2.FONT_HERSHEY_SIMPLEX, font_size, (255, 255, 255), thickness)

            cv2.putText(frame, "MISS IN:", (380, y_row),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (100, 255, 255), 2)
            cv2.putText(frame, str(len(self.missed_in)), (520, y_row),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

            cv2.putText(frame, "MISS OUT:", (600, y_row),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 100, 255), 2)
            cv2.putText(frame, str(len(self.missed_out)), (750, y_row),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

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
