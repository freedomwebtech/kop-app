import cv2
import numpy as np
from ultralytics import YOLO
from collections import defaultdict
from bytetrack import ByteTrack

class ObjectCounter:
    def __init__(self, source, model, classes_to_count=[0], show=True):
        self.source = source
        self.model = YOLO(model)
        self.classes_to_count = classes_to_count
        self.show = show

        # Line for in/out count
        self.line_y = 300  # change if needed
        self.in_count = 0
        self.out_count = 0

        # Tracker
        self.tracker = ByteTrack()

        # Store previous positions
        self.track_history = defaultdict(lambda: [])

    def detect_and_track(self, frame):
        results = self.model(frame, verbose=False)[0]

        detections = []
        for box in results.boxes:
            cls = int(box.cls)
            if cls in self.classes_to_count:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf)
                detections.append([x1, y1, x2, y2, conf, cls])

        # tracking
        tracks = self.tracker.update_tracks(detections, frame=frame)

        for track in tracks:
            if not track.is_confirmed():
                continue

            track_id = track.track_id
            l, t, r, b = track.tlbr
            cx = int((l + r) / 2)
            cy = int((t + b) / 2)

            # Save center history
            self.track_history[track_id].append((cx, cy))

            # Check crossing
            if len(self.track_history[track_id]) > 2:
                y_prev = self.track_history[track_id][-2][1]
                y_now = cy

                if y_prev < self.line_y and y_now >= self.line_y:
                    self.in_count += 1

                if y_prev > self.line_y and y_now <= self.line_y:
                    self.out_count += 1

            # Draw tracking box
            cv2.rectangle(frame, (int(l), int(t)), (int(r), int(b)), (0,255,0), 2)
            cv2.putText(frame, f"ID {track_id}", (int(l), int(t)-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

            # Draw center dot
            cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)

        return frame

    def run(self):
        cap = cv2.VideoCapture(self.source)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = self.detect_and_track(frame)

            # Draw IN/OUT line
            cv2.line(frame, (0, self.line_y), (frame.shape[1], self.line_y), (0, 0, 255), 2)

            # Display counts
            cv2.putText(frame, f"IN: {self.in_count}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 3)
            cv2.putText(frame, f"OUT: {self.out_count}", (20, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 3)

            if self.show:
                cv2.imshow("Counter", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
