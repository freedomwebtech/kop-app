import cv2
from ultralytics import YOLO
import cvzone

class ObjectCounter:
    def __init__(self, source=0, model="yolo12n.pt", classes_to_count=[0], show=True):
        self.source = source
        self.model = YOLO(model)
        self.names = self.model.names

        self.classes = classes_to_count
        self.cap = cv2.VideoCapture(self.source)

        self.hist = {}
        self.counted = set()
        self.in_count = 0
        self.out_count = 0

        self.line_p1 = (706, 216)
        self.line_p2 = (983, 311)

        self.frame_count = 0
        self.show = show

    # ----------------- SIDE CHECK FUNCTION -----------------
    def side(self, px, py, x1, y1, x2, y2):
        return (x2 - x1)*(py - y1) - (y2 - y1)*(px - x1)

    # ----------------- PROCESS ONE FRAME ONLY -----------------
    def __call__(self):
        """
        Process only 1 frame.
        Caller controls the loop.
        """
        ret, frame = self.cap.read()
        if not ret:
            print("Stream ended or file finished.")
            return None

        self.frame_count += 1
        if self.frame_count % 3 != 0:
            return frame  # skip frame but still return valid frame

        frame = cv2.resize(frame, (1020, 600))

        results = self.model.track(frame, persist=True, classes=self.classes)

        if results[0].boxes.id is not None:
            ids = results[0].boxes.id.cpu().numpy().astype(int)
            boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
            class_ids = results[0].boxes.cls.int().cpu().tolist()

            for track_id, box, cls_id in zip(ids, boxes, class_ids):
                x1, y1, x2, y2 = box
                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)

                if track_id in self.hist:
                    prev_cx, prev_cy = self.hist[track_id]
                    prev_side = self.side(prev_cx, prev_cy, *self.line_p1, *self.line_p2)
                    curr_side = self.side(cx, cy, *self.line_p1, *self.line_p2)

                    if prev_side * curr_side < 0 and track_id not in self.counted:
                        if curr_side < 0:
                            self.in_count += 1
                        else:
                            self.out_count += 1
                        self.counted.add(track_id)

                self.hist[track_id] = (cx, cy)

                # Draw
                cv2.circle(frame, (cx, cy), 4, (255, 0, 0), -1)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cvzone.putTextRect(frame, f'ID:{track_id}', (x1, y1), 1, 1)

        # Display counts
        cvzone.putTextRect(frame, f'IN: {self.in_count}', (50, 30), 2, 2, colorR=(0, 255, 0))
        cvzone.putTextRect(frame, f'OUT: {self.out_count}', (50, 80), 2, 2, colorR=(0, 0, 255))
        cv2.line(frame, self.line_p1, self.line_p2, (255, 255, 255), 2)

        if self.show:
            cv2.imshow("ObjectCounter", frame)
            cv2.waitKey(1)

        return frame
