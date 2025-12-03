import cv2
from ultralytics import YOLO
import cvzone
import json
import os
from imutils.video import VideoStream
import time

class ObjectCounter:
    def __init__(self, source, model="yolo12n.pt", classes_to_count=[0], show=True, json_file="line_coords.json"):
        self.source = source
        self.model = YOLO(model)
        self.names = self.model.names

        self.classes = classes_to_count
        self.show = show

        # --- RTSP support ---
        if isinstance(source, str) and source.startswith("rtsp://"):
            self.cap = VideoStream(source).start()
            time.sleep(2.0)  # warm-up
            self.is_rtsp = True
        else:
            self.cap = cv2.VideoCapture(source)
            self.is_rtsp = False

        self.hist = {}
        self.counted = set()
        self.in_count = 0
        self.out_count = 0

        self.line_p1 = None
        self.line_p2 = None
        self.temp_points = []
        self.json_file = json_file
        self.load_line()

        self.frame_count = 0

        cv2.namedWindow("ObjectCounter")
        cv2.setMouseCallback("ObjectCounter", self.mouse_event)

    # --------------- Mouse and line functions ----------------
    def mouse_event(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.temp_points.append((x, y))
            print(f"Point selected: {x}, {y}")
            if len(self.temp_points) == 2:
                self.line_p1, self.line_p2 = self.temp_points
                self.temp_points = []
                self.save_line()
                print(f"Line set: {self.line_p1}, {self.line_p2}")

    def save_line(self):
        data = {"line_p1": self.line_p1, "line_p2": self.line_p2}
        with open(self.json_file, "w") as f:
            json.dump(data, f)
        print(f"Line coordinates saved to {self.json_file}")

    def load_line(self):
        if os.path.exists(self.json_file):
            with open(self.json_file, "r") as f:
                data = json.load(f)
                self.line_p1 = tuple(data["line_p1"])
                self.line_p2 = tuple(data["line_p2"])
            print(f"Loaded line from {self.json_file}: {self.line_p1}, {self.line_p2}")

    # --------------- Utility ----------------
    def side(self, px, py, x1, y1, x2, y2):
        return (x2 - x1)*(py - y1) - (y2 - y1)*(px - x1)

    # --------------- Main loop ----------------
    def run(self):
        print("Starting detection...")
        while True:
            # Read frame
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

            # Draw temp points
            for pt in self.temp_points:
                cv2.circle(frame, pt, 5, (0, 0, 255), -1)

            # Draw line
            if self.line_p1 and self.line_p2:
                cv2.line(frame, self.line_p1, self.line_p2, (255, 255, 255), 2)

            # YOLO tracking
            results = self.model.track(frame, persist=True, classes=self.classes, conf=0.8)

            if results[0].boxes.id is not None and self.line_p1 and self.line_p2:
                ids = results[0].boxes.id.cpu().numpy().astype(int)
                boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
                class_ids = results[0].boxes.cls.int().cpu().tolist()

                for track_id, box, cls_id in zip(ids, boxes, class_ids):
                    x1, y1, x2, y2 = box
                    cx = int((x1 + x2)/2)
                    cy = int((y1 + y2)/2)

                    if track_id in self.hist:
                        prev_cx, prev_cy = self.hist[track_id]
                        prev_side = self.side(prev_cx, prev_cy, *self.line_p1, *self.line_p2)
                        curr_side = self.side(cx, cy, *self.line_p1, *self.line_p2)

                        if prev_side * curr_side < 0 and track_id not in self.counted:
                            if curr_side > 0:
                                self.in_count += 1
                            else:
                                self.out_count += 1
                            self.counted.add(track_id)

                    self.hist[track_id] = (cx, cy)

                    # Draw boxes
                    cv2.circle(frame, (cx, cy), 4, (255, 0, 0), -1)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cvzone.putTextRect(frame, f'ID:{track_id}', (x1, y1), 1, 1)

            # Display counts
            cvzone.putTextRect(frame, f'IN: {self.in_count}', (50, 30), 2, 2, colorR=(0, 255, 0))
            cvzone.putTextRect(frame, f'OUT: {self.out_count}', (50, 80), 2, 2, colorR=(0, 0, 255))

            if self.show:
                cv2.imshow("ObjectCounter", frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('r'):
                    print("Resetting line coordinates. Select two new points.")
                    self.line_p1 = None
                    self.line_p2 = None
                    self.temp_points = []
                    self.counted = set()
                    self.in_count = 0
                    self.out_count = 0
                    if os.path.exists(self.json_file):
                        os.remove(self.json_file)
                elif key == 27:  # ESC to quit
                    break

        # Cleanup
        if self.is_rtsp:
            self.cap.stop()
        else:
            self.cap.release()
        cv2.destroyAllWindows()
        print("Detection stopped.")
