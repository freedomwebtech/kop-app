import cv2
import time
from ultralytics import YOLO
from collections import defaultdict

class ObjectCounter:
    def __init__(self, video_path, model_path="yolov8n.pt", classes=None, show=True):
        self.cap = cv2.VideoCapture(video_path)
        self.model = YOLO(model_path)

        self.classes = classes
        self.show = show

        self.line_p1 = None
        self.line_p2 = None
        self.temp_points = []

        self.hist = {}
        self.origin_side = {}
        self.counted = set()
        self.crossed_ids = set()
        self.last_seen = defaultdict(int)

        self.in_count = 0
        self.out_count = 0
        self.frame_count = 0

        self.is_rtsp = video_path.startswith("rtsp://")

    # ---------------------------------------
    # Select Line by Clicking 2 Points
    # ---------------------------------------
    def click_event(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.temp_points.append((x, y))

            if len(self.temp_points) == 2:
                self.line_p1 = self.temp_points[0]
                self.line_p2 = self.temp_points[1]
                print(f"Selected Line: {self.line_p1} -> {self.line_p2}")

    # ---------------------------------------
    # Side Calculation
    # ---------------------------------------
    def side(self, px, py, x1, y1, x2, y2):
        return (px - x1) * (y2 - y1) - (py - y1) * (x2 - x1)

    # ---------------------------------------
    # Remove Lost IDs Automatically
    # ---------------------------------------
    def check_lost_ids(self, max_lost=20):
        lost = []
        for tid in list(self.last_seen.keys()):
            if self.frame_count - self.last_seen[tid] > max_lost:
                lost.append(tid)

        for tid in lost:
            if tid in self.hist:
                del self.hist[tid]

            if tid in self.origin_side:
                del self.origin_side[tid]

            if tid in self.counted:
                self.counted.remove(tid)

            if tid in self.crossed_ids:
                self.crossed_ids.remove(tid)

            if tid in self.last_seen:
                del self.last_seen[tid]

    # ---------------------------------------
    # Reset All Data
    # ---------------------------------------
    def reset_all_data(self):
        self.hist.clear()
        self.origin_side.clear()
        self.counted.clear()
        self.crossed_ids.clear()
        self.last_seen.clear()

        self.in_count = 0
        self.out_count = 0

        print("\n--- RESET DONE ---\n")

    # ---------------------------------------
    # Main RUN Function
    # ---------------------------------------
    def run(self):
        print("\nRUNNING... Press O to Reset | ESC to Exit\n")

        cv2.namedWindow("ObjectCounter")
        cv2.setMouseCallback("ObjectCounter", self.click_event)

        while True:

            ret, frame = self.cap.read()
            if not ret:
                break

            self.frame_count += 1

            # FIXED — Resize to 640×360
            frame = cv2.resize(frame, (640, 360))

            # DISPLAY IN & OUT
            cv2.putText(frame, f"IN  : {self.in_count}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            cv2.putText(frame, f"OUT : {self.out_count}", (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

            # Draw temp click points
            for pt in self.temp_points:
                cv2.circle(frame, pt, 5, (0, 0, 255), -1)

            # Draw selected line
            if self.line_p1:
                cv2.line(frame, self.line_p1, self.line_p2, (255, 255, 255), 2)

            # YOLO Tracking
            results = self.model.track(frame, persist=True, classes=self.classes, conf=0.70)

            if results and results[0].boxes.id is not None and self.line_p1:

                ids = results[0].boxes.id.cpu().numpy().astype(int)
                boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)

                for tid, box in zip(ids, boxes):
                    x1, y1, x2, y2 = box

                    # Center point
                    cx = int((x1 + x2) / 2)
                    cy = int((y1 + y2) / 2)

                    self.last_seen[tid] = self.frame_count

                    # First time detected
                    if tid not in self.hist:
                        s_init = self.side(cx, cy, *self.line_p1, *self.line_p2)
                        self.origin_side[tid] = "IN" if s_init < 0 else "OUT"

                    # If previously seen
                    if tid in self.hist:
                        px, py = self.hist[tid]

                        s1 = self.side(px, py, *self.line_p1, *self.line_p2)
                        s2 = self.side(cx, cy, *self.line_p1, *self.line_p2)

                        if s1 * s2 < 0:  # SIDE CHANGE = crossing
                            if tid not in self.counted:

                                if s2 > 0:
                                    self.in_count += 1
                                    print(f"IN +1 (ID:{tid})")
                                else:
                                    self.out_count += 1
                                    print(f"OUT +1 (ID:{tid})")

                                self.counted.add(tid)

                    # Save center
                    self.hist[tid] = (cx, cy)

                    # Draw Box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f"ID:{tid}", (x1, y1 - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 0), 2)

            # Clean lost IDs
            self.check_lost_ids()

            # SHOW WINDOW
            cv2.imshow("ObjectCounter", frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('o'):
                self.reset_all_data()

            elif key == 27:
                break

        self.cap.release()
        cv2.destroyAllWindows()
