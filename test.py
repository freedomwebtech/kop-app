import cv2
from ultralytics import YOLO
import json
import os
import numpy as np
from imutils.video import VideoStream
import time
from datetime import datetime
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, A4
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
    brown_intensity = brown_mask.mean()

    white_mask = cv2.inRange(hsv, WHITE_LOWER, WHITE_UPPER)
    white_intensity = white_mask.mean()

    if brown_intensity > 20:
        return "Brown"
    elif white_intensity > 20:
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
        self.names = self.model.names
        self.classes = classes_to_count
        self.show = show
        self.pdf_folder = pdf_folder

        # Create PDF folder if not exists
        if not os.path.exists(self.pdf_folder):
            os.makedirs(self.pdf_folder)

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
        self.sessions_data = []
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

        self.color_in_count = {}
        self.color_out_count = {}

        # ✅ MISSED LOGIC ADDED
        self.missed_in = set()
        self.missed_out = set()
        self.missed_cross = set()
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
            'missed_in': 0,
            'missed_out': 0,
            'missed_cross': 0,
            'color_in': {},
            'color_out': {}
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
            self.current_session_data['color_in'] = dict(self.color_in_count)
            self.current_session_data['color_out'] = dict(self.color_out_count)
            
            self.sessions_data.append(self.current_session_data.copy())

    def generate_pdf_report(self):
        """Generate PDF report with all session data - Single PDF file"""
        pdf_filename = os.path.join(self.pdf_folder, "tracking_report.pdf")
        
        doc = SimpleDocTemplate(pdf_filename, pagesize=A4)
        elements = []
        
        # Styles
        styles = getSampleStyleSheet()
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=24,
            textColor=colors.HexColor('#1a5490'),
            spaceAfter=30,
            alignment=1  # Center
        )
        
        # Title
        title = Paragraph("OBJECT TRACKING REPORT", title_style)
        elements.append(title)
        elements.append(Spacer(1, 0.3*inch))
        
        # Report Generation Info
        info_style = styles['Normal']
        report_info = Paragraph(f"<b>Report Generated:</b> {datetime.now().strftime('%A, %B %d, %Y at %H:%M:%S')}", info_style)
        elements.append(report_info)
        elements.append(Spacer(1, 0.2*inch))
        
        # Session Summary Table
        if self.sessions_data:
            # Main data table
            data = [['Day', 'Date', 'Start Time', 'End Time', 'IN', 'OUT', 'Miss IN', 'Miss OUT', 'Cross']]
            
            for session in self.sessions_data:
                data.append([
                    session['day'],
                    session['date'],
                    session['start_time'],
                    session['end_time'] if session['end_time'] else 'N/A',
                    str(session['in_count']),
                    str(session['out_count']),
                    str(session['missed_in']),
                    str(session['missed_out']),
                    str(session['missed_cross'])
                ])
            
            table = Table(data, colWidths=[0.8*inch, 1*inch, 0.9*inch, 0.9*inch, 0.6*inch, 0.6*inch, 0.7*inch, 0.7*inch, 0.6*inch])
            table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1a5490')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 10),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
                ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
                ('FONTSIZE', (0, 1), (-1, -1), 9),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.lightgrey])
            ]))
            
            elements.append(table)
            elements.append(Spacer(1, 0.4*inch))
            
            # Color-wise breakdown for each session
            subtitle = Paragraph("<b>Color-wise Breakdown</b>", styles['Heading2'])
            elements.append(subtitle)
            elements.append(Spacer(1, 0.2*inch))
            
            for idx, session in enumerate(self.sessions_data, 1):
                session_title = Paragraph(
                    f"<b>Session {idx}:</b> {session['date']} ({session['start_time']} - {session['end_time'] if session['end_time'] else 'N/A'})",
                    styles['Heading3']
                )
                elements.append(session_title)
                elements.append(Spacer(1, 0.1*inch))
                
                # Color data table
                color_data = [['Color', 'IN Count', 'OUT Count']]
                
                all_colors = set(list(session['color_in'].keys()) + list(session['color_out'].keys()))
                
                if all_colors:
                    for color in sorted(all_colors):
                        in_count = session['color_in'].get(color, 0)
                        out_count = session['color_out'].get(color, 0)
                        color_data.append([color, str(in_count), str(out_count)])
                else:
                    color_data.append(['No Data', '0', '0'])
                
                color_table = Table(color_data, colWidths=[2*inch, 1.5*inch, 1.5*inch])
                color_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#4a90e2')),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, 0), 10),
                    ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                    ('BACKGROUND', (0, 1), (-1, -1), colors.lightblue),
                    ('GRID', (0, 0), (-1, -1), 1, colors.black),
                    ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
                    ('FONTSIZE', (0, 1), (-1, -1), 9)
                ]))
                
                elements.append(color_table)
                elements.append(Spacer(1, 0.3*inch))
        
        # Build PDF
        doc.build(elements)
        print(f"✅ PDF Report Generated: {pdf_filename}")
        return pdf_filename

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

    # ================= MISSED TRACK HANDLER =================
    def check_lost_ids(self):
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

                if s > 0:
                    self.missed_in.add(tid)
                else:
                    self.missed_out.add(tid)

            self.hist.pop(tid, None)
            self.last_seen.pop(tid, None)

    # ---------------- Reset Function ----------------
    def reset_all_data(self):
        """Reset all tracking data, generate PDF, and start new session"""
        # End current session before resetting
        self.end_current_session()
        
        # Generate PDF with all sessions
        if self.sessions_data:
            pdf_file = self.generate_pdf_report()
            print(f"📄 PDF Updated: {pdf_file}")
        
        # Reset counters
        self.hist.clear()
        self.last_seen.clear()
        self.crossed_ids.clear()
        self.counted.clear()
        self.color_in_count.clear()
        self.color_out_count.clear()
        self.missed_in.clear()
        self.missed_out.clear()
        self.missed_cross.clear()
        self.in_count = 0
        self.out_count = 0
        
        # Start new session
        self.start_new_session()
        print("✅ RESET DONE - New session started | PDF saved")

    # ---------------- Main Loop ----------------
    def run(self):
        print("RUNNING... Press O to Reset & Save PDF | ESC to Exit")

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
                cv2.line(frame, self.line_p1, self.line_p2,
                         (255, 255, 255), 2)

            results = self.model.track(
                frame, persist=True, classes=self.classes, conf=0.7)

            if results[0].boxes.id is not None and self.line_p1:
                ids = results[0].boxes.id.cpu().numpy().astype(int)
                boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)

                for tid, box in zip(ids, boxes):
                    x1, y1, x2, y2 = box
                    cx = int((x1 + x2) / 2)
                    cy = int((y1 + y2) / 2)

                    self.last_seen[tid] = self.frame_count
                    color_name = detect_box_color(frame, box)

                    if tid in self.hist:
                        px, py = self.hist[tid]
                        s1 = self.side(px, py, *self.line_p1, *self.line_p2)
                        s2 = self.side(cx, cy, *self.line_p1, *self.line_p2)

                        if s1 * s2 < 0:
                            self.crossed_ids.add(tid)

                            if tid not in self.counted:
                                if s2 > 0:
                                    self.in_count += 1
                                    self.color_in_count[color_name] = self.color_in_count.get(color_name, 0) + 1
                                else:
                                    self.out_count += 1
                                    self.color_out_count[color_name] = self.color_out_count.get(color_name, 0) + 1

                                self.counted.add(tid)

                    self.hist[tid] = (cx, cy)

                    cv2.rectangle(frame, (x1, y1),
                                  (x2, y2), (0, 255, 0), 2)

                    cv2.putText(frame, color_name, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.6, (255, 200, 0), 2)

            # ✅ MISSED CHECK ACTIVE
            if self.line_p1:
                self.check_lost_ids()

            # ================= ENHANCED DISPLAY WITH COLOR SEPARATION =================

            # Main overlay panel (increased height for better spacing)
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (1020, 140), (0, 0, 0), -1)
            frame = cv2.addWeighted(overlay, 0.3, frame, 0.7, 0)

            # --------- TITLE BAR ---------
            cv2.putText(frame, "TRACKING SYSTEM", (15, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 200, 255), 2)
            cv2.circle(frame, (230, 21), 6, (0, 255, 0), -1)

            # --------- TOTAL COUNTS (First Row) ---------
            y_row1 = 60
            font_scale = 0.65
            thickness = 2
            
            # Total IN
            cv2.putText(frame, "TOTAL IN:", (15, y_row1),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 150), thickness)
            cv2.putText(frame, str(self.in_count), (150, y_row1),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)

            # Total OUT
            cv2.putText(frame, "TOTAL OUT:", (240, y_row1),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, (100, 180, 255), thickness)
            cv2.putText(frame, str(self.out_count), (390, y_row1),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)

            # Missed counts
            cv2.putText(frame, "MISS IN:", (480, y_row1),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 255, 255), 1)
            cv2.putText(frame, str(len(self.missed_in)), (575, y_row1),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            cv2.putText(frame, "MISS OUT:", (620, y_row1),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 100, 255), 1)
            cv2.putText(frame, str(len(self.missed_out)), (730, y_row1),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            cv2.putText(frame, "CROSS:", (780, y_row1),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 255), 1)
            cv2.putText(frame, str(len(self.missed_cross)), (860, y_row1),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # --------- SEPARATOR LINE ---------
            cv2.line(frame, (10, 75), (1010, 75), (80, 80, 80), 1)

            # --------- BROWN BOX COUNTS (Second Row) ---------
            y_row2 = 105
            brown_in = self.color_in_count.get("Brown", 0)
            brown_out = self.color_out_count.get("Brown", 0)

            # Brown box icon/indicator
            cv2.rectangle(frame, (15, y_row2 - 20), (40, y_row2 - 5), (19, 69, 139), -1)
            
            cv2.putText(frame, "BROWN IN:", (50, y_row2),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, (100, 150, 255), thickness)
            cv2.putText(frame, str(brown_in), (200, y_row2),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)

            cv2.putText(frame, "BROWN OUT:", (290, y_row2),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, (100, 150, 255), thickness)
            cv2.putText(frame, str(brown_out), (460, y_row2),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)

            # --------- WHITE BOX COUNTS (Third Row) ---------
            y_row3 = 130
            white_in = self.color_in_count.get("White", 0)
            white_out = self.color_out_count.get("White", 0)

            # White box icon/indicator
            cv2.rectangle(frame, (15, y_row3 - 20), (40, y_row3 - 5), (245, 245, 245), -1)
            cv2.rectangle(frame, (15, y_row3 - 20), (40, y_row3 - 5), (100, 100, 100), 2)
            
            cv2.putText(frame, "WHITE IN:", (50, y_row3),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, (200, 255, 200), thickness)
            cv2.putText(frame, str(white_in), (200, y_row3),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)

            cv2.putText(frame, "WHITE OUT:", (290, y_row3),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, (200, 255, 200), thickness)
            cv2.putText(frame, str(white_out), (460, y_row3),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)

            if self.show:
                cv2.imshow("ObjectCounter", frame)
                key = cv2.waitKey(1) & 0xFF

                if key == ord('o') or key == ord('O'):
                    self.reset_all_data()

                elif key == 27:
                    break

        # Save final session and generate PDF before exit
        self.end_current_session()
        if self.sessions_data:
            self.generate_pdf_report()
            print("📄 Final PDF generated on exit")

        if self.is_rtsp:
            self.cap.stop()
        else:
            self.cap.release()

        cv2.destroyAllWindows()


# ==========================================================
#                        MAIN EXECUTION
# ==========================================================

if __name__ == "__main__":
    # Example usage:
    counter = ObjectCounter(
        source="your_video.mp4",  # or 0 for webcam, or "rtsp://..."
        model="best_float32.tflite",
        classes_to_count=[0],
        show=True,
        pdf_folder="pdf_report"
    )
    counter.run()
