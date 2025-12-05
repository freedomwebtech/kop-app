import cv2
import numpy as np
from ultralytics import YOLO
from collections import defaultdict

# ----------------------------
# SIMPLE BYTE-TRACK IMPLEMENTATION
# ----------------------------
class Track:
    def __init__(self, track_id, tlbr):
        self.track_id = track_id
        self.tlbr = tlbr
        self.time_since_update = 0
        self.hits = 1
    
    def update(self, tlbr):
        self.tlbr = tlbr
        self.time_since_update = 0
        self.hits += 1
    
    def is_confirmed(self):
        return self.hits >= 2

class ByteTrack:
    def __init__(self, iou_threshold=0.3):
        self.tracks = []
        self.next_id = 1
        self.iou_threshold = iou_threshold

    def iou(self, boxA, boxB):
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        interArea = max(0, xB - xA) * max(0, yB - yA)
        boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
        return interArea / float(boxAArea + boxBArea - interArea + 1e-6)

    def update_tracks(self, detections, frame):
        new_tracks = []

        for det in detections:
            best_iou = 0
            best_track = None
            det_box = det[:4]

            for track in self.tracks:
                iou_score = self.iou(det_box, track.tlbr)
                if iou_score > best_iou:
                    best_iou = iou_score
                    best_track = track

            if best_iou > self.iou_threshold:
                best_track.update(det_box)
                new_tracks.append(best_track)
            else:
                new_track = Track(self.next_id, det_box)
                self.next_id += 1
                new_tracks.append(new_track)

        self.tracks = new_tracks
        return self.tracks


# ----------------------------
# MAIN OBJECT COUNTER
# ----------------------------
class ObjectCounter:
    def __init__(self, source, model, classes_to_count=[0], show=True):
        self.source = source
        self.model = YOLO(model)
        self.classes_to_count = classes_to_count
        self.show = show

        self.line_y = 300         # Crossing line position
        self.in_count = 0
        self.out_count = 0

        self.tracker = ByteTrack()
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

        tracks = self.tracker.update_tracks(detections, frame)

        for track in tracks:
            if not track.is_confirmed():
                continue

            track_id = track.track_id
            l, t, r, b = map(int, track.tlbr)
            cx, cy = (l + r) // 2, (t + b) // 2

            self.track_history[track_id].append((cx, cy))

            # IN / OUT counting
            if len(self.track_history[track_id]) > 2:
                y_prev = self.track_history[track_id][-2][1]
                y_now  = cy

                if y_prev < self.line_y <= y_now:
                    self.in_count += 1

                if y_prev > self.line_y >= y_now:
                    self.out_count += 1

            # Draw box + ID
            cv2.rectangle(frame, (l, t), (r, b), (0,255,0), 2)
            cv2.putText(frame, f"ID {track_id}", (l, t-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

            cv2.circle(frame, (cx, cy), 4, (0,0,255), -1)

        return frame

    def run(self):
        cap = cv2.VideoCapture(self.source)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = self.detect_and_track(frame)

            # Draw line + counts
            cv2.line(frame, (0, self.line_y), (frame.shape[1], self.line_y), (0,0,255), 2)
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
