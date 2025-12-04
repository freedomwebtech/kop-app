import cv2
from ultralytics import YOLO
import json
import os
import numpy as np
from imutils.video import VideoStream
import time
from datetime import datetime

# ✅ FIXED REPORTLAB IMPORTS
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, A4   # ✅ Correct path
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


def detect_box_color(frame, box):
    x1, y1, x2, y2 = box
    roi = frame[y1:y2, x1:x2]
    if roi.size == 0:
        return "Unknown"

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    brown_mask = cv2.inRange(hsv, BROWN_LOWER, BROWN_UPPER)
    white_mask = cv2.inRange(hsv, WHITE_LOWER, WHITE_UPPER)

    if brown_mask.mean() > 20:
        return "Brown"
    elif white_mask.mean() > 20:
        return "White"

    return "Unknown"


# ==========================================================
#                     OBJECT COUNTER CLASS
# ==========================================================

class ObjectCounter:
    def __init__(self, source, model="best_float32.tflite",
                 classes_to_count=[0], show=True,
                 json_file="line_coords.json",
                 pdf_folder="pdf_report"):

        self.source = source
        self.model = YOLO(model)
        self.classes = classes_to_count
        self.show = show
        self.pdf_folder = pdf_folder

        os.makedirs(self.pdf_folder, exist_ok=True)

        if isinstance(source, str) and source.startswith("rtsp"):
            self.cap = VideoStream(source).start()
            time.sleep(2)
            self.is_rtsp = True
        else:
            self.cap = cv2.VideoCapture(source)
            self.is_rtsp = False

        self.sessions_data = []
        self.start_new_session()

        self.hist = {}
        self.last_seen = {}
        self.crossed_ids = set()
        self.counted = set()
        self.color_at_crossing = {}

        self.in_count = 0
        self.out_count = 0
        self.color_in_count = {}
        self.color_out_count = {}

        self.missed_in = set()
        self.missed_out = set()
        self.missed_cross = set()
        self.max_missing_frames = 40

        self.json_file = json_file
        self.line_p1 = None
        self.line_p2 = None
        self.temp_points = []
        self.load_line()

        self.frame_count = 0

        cv2.namedWindow("ObjectCounter")
        cv2.setMouseCallback("ObjectCounter", self.mouse_event)


    # ---------------- Session Control ----------------
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


    # ---------------- PDF REPORT ----------------
    def generate_pdf_report(self):
        pdf_file = os.path.join(self.pdf_folder, "tracking_report.pdf")
        doc = SimpleDocTemplate(pdf_file, pagesize=A4)
        styles = getSampleStyleSheet()
        elements = []

        title = Paragraph("OBJECT COUNT REPORT", styles['Title'])
        elements.append(title)
        elements.append(Spacer(1, 0.2 * inch))

        table_data = [['Day', 'Date', 'Start', 'End', 'IN', 'OUT', 'Miss IN', 'Miss OUT', 'Cross']]

        for s in self.sessions_data:
            table_data.append([
                s['day'], s['date'], s['start_time'], s['end_time'],
                s['in_count'], s['out_count'],
                s['missed_in'], s['missed_out'], s['missed_cross']
            ])

        table = Table(table_data, colWidths=1.2*inch)
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONT', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ]))

        elements.append(table)
        doc.build(elements)
        print(f"✅ PDF SAVED: {pdf_file}")


    # ---------------- Mouse ----------------
    def mouse_event(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.temp_points.append((x, y))
            if len(self.temp_points) == 2:
                self.line_p1, self.line_p2 = self.temp_points
                self.temp_points = []
                self.save_line()


    # ---------------- Line Save/Load ----------------
    def save_line(self):
        with open(self.json_file, "w") as f:
            json.dump({"line_p1": self.line_p1, "line_p2": self.line_p2}, f)


    def load_line(self):
        if os.path.exists(self.json_file):
            with open(self.json_file) as f:
                data = json.load(f)
                self.line_p1 = tuple(data["line_p1"])
                self.line_p2 = tuple(data["line_p2"])


    # ---------------- Side ----------------
    def side(self, px, py, x1, y1, x2, y2):
        return (x2-x1)*(py-y1) - (y2-y1)*(px-x1)


    # ---------------- Reset ----------------
    def reset_all_data(self):
        self.end_current_session()
        self.generate_pdf_report()

        self.hist.clear()
        self.last_seen.clear()
        self.crossed_ids.clear()
        self.counted.clear()
        self.color_at_crossing.clear()

        self.in_count = self.out_count = 0
        self.color_in_count.clear()
        self.color_out_count.clear()
        self.missed_in.clear()
        self.missed_out.clear()
        self.missed_cross.clear()

        self.start_new_session()
        print("🔄 RESET COMPLETE")


    # ---------------- Main Loop ----------------
    def run(self):
        print("▶ RUNNING | Press O = Reset & Save | ESC = Exit")

        while True:
            frame = self.cap.read() if self.is_rtsp else self.cap.read()[1]
            if frame is None:
                break

            frame = cv2.resize(frame, (1020, 600))
            self.frame_count += 1

            if self.line_p1:
                cv2.line(frame, self.line_p1, self.line_p2, (255,255,255), 2)

            results = self.model.track(frame, persist=True, classes=self.classes, conf=0.80)

            if results[0].boxes.id is not None and self.line_p1:
                ids = results[0].boxes.id.cpu().numpy().astype(int)
                boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)

                for tid, box in zip(ids, boxes):
                    x1,y1,x2,y2 = box
                    cx, cy = (x1+x2)//2, (y1+y2)//2

                    if tid in self.hist:
                        px,py = self.hist[tid]
                        s1 = self.side(px,py,*self.line_p1,*self.line_p2)
                        s2 = self.side(cx,cy,*self.line_p1,*self.line_p2)

                        if s1 * s2 < 0 and tid not in self.counted:
                            color = detect_box_color(frame, box)
                            if s2 > 0:
                                self.in_count += 1
                                self.color_in_count[color] = self.color_in_count.get(color,0)+1
                            else:
                                self.out_count += 1
                                self.color_out_count[color] = self.color_out_count.get(color,0)+1
                            self.counted.add(tid)

                    self.hist[tid] = (cx,cy)

                    cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)

            cv2.rectangle(frame,(0,0),(300,70),(0,0,0),-1)
            cv2.putText(frame,f"IN: {self.in_count}",(10,25),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,255,0),2)
            cv2.putText(frame,f"OUT: {self.out_count}",(10,55),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,150,255),2)

            cv2.imshow("ObjectCounter", frame)
            key = cv2.waitKey(1) & 0xFF

            if key in [ord('o'), ord('O')]:
                self.reset_all_data()
            elif key == 27:
                break

        self.end_current_session()
        self.generate_pdf_report()

        self.cap.stop() if self.is_rtsp else self.cap.release()
        cv2.destroyAllWindows()


# ==========================================================
#                        MAIN
# ==========================================================
if __name__ == "__main__":
    counter = ObjectCounter(
        source=0,    # Webcam | Or "video.mp4" | Or "rtsp://...."
        model="best_float32.tflite",
        classes_to_count=[0],
        show=True
    )
    counter.run()
