import cv2
from ultralytics import YOLO
import json
import os
import numpy as np
from imutils.video import VideoStream
import time
from datetime import datetime
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch

# ==========================================================
#                HSV COLOR DETECTION (Brown + White)
# ==========================================================

BROWN_LOWER = np.array([5, 80, 60])
BROWN_UPPER = np.array([20, 255, 255])

WHITE_LOWER = np.array([0, 0, 200])
WHITE_UPPER = np.array([180, 40, 255])

# ==========================================================
#                  UTILITY FUNCTIONS
# ==========================================================

def clamp_box(box, frame_shape):
    x1, y1, x2, y2 = box
    h, w = frame_shape[:2]
    x1 = max(0, int(x1)); y1 = max(0, int(y1))
    x2 = min(w - 1, int(x2)); y2 = min(h - 1, int(y2))
    return x1, y1, x2, y2

def detect_box_color(frame, box):
    x1, y1, x2, y2 = clamp_box(box, frame.shape)
    roi = frame[y1:y2, x1:x2]
    if roi.size == 0:
        return "Unknown"

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    brown = cv2.inRange(hsv, BROWN_LOWER, BROWN_UPPER).mean()
    white = cv2.inRange(hsv, WHITE_LOWER, WHITE_UPPER).mean()

    if brown > 25:
        return "Brown"
    if white > 25:
        return "White"
    return "Unknown"

# ==========================================================
#                     OBJECT COUNTER
# ==========================================================

class ObjectCounter:
    def __init__(self, source, model="best_float32.tflite",
                 classes_to_count=[0], show=False,
                 json_file="line_coords.json",
                 pdf_folder="pdf_report",
                 display_size=(1020, 600),
                 translucent_overlay=False):

        self.show = show
        self.source = source
        self.model = YOLO(model)
        self.names = self.model.names
        self.classes = classes_to_count
        self.display_size = display_size
        self.translucent_overlay = translucent_overlay
        self.json_file = json_file
        self.pdf_folder = pdf_folder

        os.makedirs(self.pdf_folder, exist_ok=True)

        # Video Source
        if isinstance(source, str) and source.startswith("rtsp://"):
            self.cap = VideoStream(source).start()
            time.sleep(2)
            self.is_rtsp = True
        else:
            self.cap = cv2.VideoCapture(source)
            self.is_rtsp = False

        # Tracking & Counters
        self.hist = {}
        self.last_seen = {}
        self.crossed_ids = set()
        self.counted = set()
        self.missed_in = set()
        self.missed_out = set()
        self.missed_cross = set()
        self.color_in_count = {}
        self.color_out_count = {}
        self.in_count = 0
        self.out_count = 0
        self.max_missing_frames = 40

        # Line
        self.line_p1 = None
        self.line_p2 = None
        self.temp_points = []
        self.load_line()

        # Sessions
        self.sessions_data = []
        self.start_new_session()

        self.frame_count = 0

        cv2.namedWindow("ObjectCounter", cv2.WINDOW_NORMAL)
        cv2.setMouseCallback("ObjectCounter", self.mouse_event)

    # ---------------- SESSION ----------------

    def start_new_session(self):
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
        self.current_session_data['end_time'] = datetime.now().strftime('%H:%M:%S')
        self.current_session_data['in_count'] = self.in_count
        self.current_session_data['out_count'] = self.out_count
        self.current_session_data['missed_in'] = len(self.missed_in)
        self.current_session_data['missed_out'] = len(self.missed_out)
        self.current_session_data['missed_cross'] = len(self.missed_cross)
        self.current_session_data['color_in'] = dict(self.color_in_count)
        self.current_session_data['color_out'] = dict(self.color_out_count)
        self.sessions_data.append(self.current_session_data.copy())

    # ---------------- PDF ----------------

    def generate_pdf_report(self):
        pdf_path = os.path.join(self.pdf_folder, "tracking_report.pdf")
        doc = SimpleDocTemplate(pdf_path, pagesize=A4)
        styles = getSampleStyleSheet()
        elements = []

        title = Paragraph("OBJECT TRACKING REPORT", styles['Title'])
        elements.append(title)
        elements.append(Spacer(1, 20))

        data = [['Date', 'Start', 'End', 'IN', 'OUT', 'Miss IN', 'Miss OUT', 'Cross']]
        for s in self.sessions_data:
            data.append([s['date'], s['start_time'], s['end_time'], s['in_count'], s['out_count'],
                         s['missed_in'], s['missed_out'], s['missed_cross']])

        table = Table(data)
        table.setStyle(TableStyle([
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey)
        ]))
        elements.append(table)

        doc.build(elements)
        print("✅ PDF SAVED:", pdf_path)

    # ---------------- MOUSE ----------------

    def mouse_event(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.temp_points.append((x, y))
            if len(self.temp_points) == 2:
                self.line_p1, self.line_p2 = self.temp_points
                self.temp_points = []
                self.save_line()

    def save_line(self):
        with open(self.json_file, "w") as f:
            json.dump({"line_p1": self.line_p1, "line_p2": self.line_p2}, f)

    def load_line(self):
        if os.path.exists(self.json_file):
            with open(self.json_file) as f:
                data = json.load(f)
                self.line_p1 = tuple(data["line_p1"])
                self.line_p2 = tuple(data["line_p2"])

    # ---------------- UTILS ----------------

    def side(self, px, py, x1, y1, x2, y2):
        return (x2 - x1) * (py - y1) - (y2 - y1) * (px - x1)

    def check_lost_ids(self):
        current = self.frame_count
        for tid in list(self.last_seen):
            if current - self.last_seen[tid] > self.max_missing_frames:
                self.missed_cross.add(tid)
                self.hist.pop(tid, None)
                self.last_seen.pop(tid, None)

    # ---------------- RESET ----------------

    def reset_all_data(self):
        self.end_current_session()
        self.generate_pdf_report()
        self.hist.clear(); self.last_seen.clear(); self.crossed_ids.clear()
        self.counted.clear(); self.missed_in.clear(); self.missed_out.clear()
        self.missed_cross.clear(); self.color_in_count.clear(); self.color_out_count.clear()
        self.in_count = 0; self.out_count = 0
        self.start_new_session()
        print("🔄 RESET DONE")

    # ---------------- MAIN ----------------

    def run(self):
        print("D = Toggle Display | O = Reset | ESC = Exit")

        while True:
            frame = self.cap.read()
            if isinstance(frame, tuple):
                ret, frame = frame
            else:
                ret = frame is not None

            if not ret:
                break

            self.frame_count += 1
            frame = cv2.resize(frame, self.display_size)

            if self.line_p1:
                cv2.line(frame, self.line_p1, self.line_p2, (255, 255, 255), 2)

            results = self.model.track(frame, persist=True, classes=self.classes)

            if results and results[0].boxes.id is not None:
                ids = results[0].boxes.id.int().cpu().tolist()
                boxes = results[0].boxes.xyxy.cpu().numpy()

                for tid, box in zip(ids, boxes):
                    x1, y1, x2, y2 = map(int, box)
                    cx = (x1 + x2) // 2
                    cy = (y1 + y2) // 2

                    if tid in self.hist and self.line_p1:
                        if self.side(*self.hist[tid], *self.line_p1, *self.line_p2) * self.side(cx, cy, *self.line_p1, *self.line_p2) < 0:
                            if tid not in self.counted:
                                color = detect_box_color(frame, (x1, y1, x2, y2))
                                if self.side(cx, cy, *self.line_p1, *self.line_p2) > 0:
                                    self.in_count += 1
                                    self.color_in_count[color] = self.color_in_count.get(color, 0) + 1
                                else:
                                    self.out_count += 1
                                    self.color_out_count[color] = self.color_out_count.get(color, 0) + 1
                                self.counted.add(tid)

                    self.hist[tid] = (cx, cy)
                    self.last_seen[tid] = self.frame_count

                    if self.show:
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(frame, detect_box_color(frame, (x1, y1, x2, y2)), (x1, y1 - 8),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

            self.check_lost_ids()

            key = cv2.waitKey(1) & 0xFF

            # TOGGLE DISPLAY
            if key == ord('d'):
                self.show = not self.show
                print("DISPLAY:", "ON" if self.show else "OFF")
                if not self.show:
                    cv2.destroyWindow("ObjectCounter")

            # RESET
            if key == ord('o'):
                self.reset_all_data()

            # EXIT
            if key == 27:
                break

            # SHOW IF ENABLED
            if self.show:
                cv2.imshow("ObjectCounter", frame)

        self.end_current_session()
        self.generate_pdf_report()
        self.cap.release()
        cv2.destroyAllWindows()

# ==========================================================
#                       RUN APP
# ==========================================================

if __name__ == "__main__":
    counter = ObjectCounter(
        source=0,
        model="best_float32.tflite",
        classes_to_count=[0],
        show=False
    )
    counter.run()
