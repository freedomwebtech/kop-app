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

def clamp_box(box, frame_shape):
    """Clamp box coords to frame and return ints (x1,y1,x2,y2)."""
    x1, y1, x2, y2 = box
    h, w = frame_shape[:2]
    x1 = max(0, int(x1))
    y1 = max(0, int(y1))
    x2 = min(w - 1, int(x2))
    y2 = min(h - 1, int(y2))
    if x2 <= x1 or y2 <= y1:
        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)
        x1 = max(0, cx - 5)
        y1 = max(0, cy - 5)
        x2 = min(w - 1, cx + 5)
        y2 = min(h - 1, cy + 5)
    return x1, y1, x2, y2

def detect_box_color(frame, box):
    """
    Detect whether ROI contains Brown or White using HSV masks.
    Returns "Brown", "White", or "Unknown".
    """
    x1, y1, x2, y2 = clamp_box(box, frame.shape)
    roi = frame[y1:y2, x1:x2]

    if roi.size == 0:
        return "Unknown"

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    brown_mask = cv2.inRange(hsv, BROWN_LOWER, BROWN_UPPER)
    brown_intensity = brown_mask.mean()

    white_mask = cv2.inRange(hsv, WHITE_LOWER, WHITE_UPPER)
    white_intensity = white_mask.mean()

    # thresholds tuned for typical ROI sizes; adjust if necessary
    if brown_intensity > 25:
        return "Brown"
    elif white_intensity > 25:
        return "White"

    return "Unknown"

# ==========================================================
#                     OBJECT COUNTER CLASS
# ==========================================================

class ObjectCounter:
    def __init__(self, source, model="best_float32.tflite",
                 classes_to_count=[0], show=True,
                 json_file="line_coords.json",
                 pdf_folder="pdf_report",
                 display_size=(1020, 600),
                 translucent_overlay=False):
        self.source = source
        # initialize YOLO model (can be .pt or .tflite path)
        self.model = YOLO(model)
        self.names = self.model.names
        self.classes = classes_to_count
        self.show = show
        self.pdf_folder = pdf_folder
        self.display_size = display_size  # (width, height)
        self.translucent_overlay = translucent_overlay

        # Create PDF folder if not exists
        if not os.path.exists(self.pdf_folder):
            os.makedirs(self.pdf_folder)

        # -------- RTSP or File / Webcam --------
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
        self.hist = {}            # last centroid per track id
        self.last_seen = {}       # last seen frame index per track id
        self.crossed_ids = set()
        self.counted = set()

        # -------- Counters --------
        self.in_count = 0
        self.out_count = 0

        self.color_in_count = {}
        self.color_out_count = {}

        # MISSED TRACK LOGIC
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

        cv2.namedWindow("ObjectCounter", cv2.WINDOW_NORMAL)
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
                try:
                    data = json.load(f)
                    self.line_p1 = tuple(data["line_p1"])
                    self.line_p2 = tuple(data["line_p2"])
                except Exception:
                    # corrupted file: ignore
                    self.line_p1 = None
                    self.line_p2 = None

    # ---------------- Utility ----------------
    def side(self, px, py, x1, y1, x2, y2):
        # Determinant-based side test
        return (x2 - x1) * (py - y1) - (y2 - y1) * (px - x1)

    # ================= MISSED TRACK HANDLER =================
    def check_lost_ids(self):
        current = self.frame_count
        lost = []

        for tid, last in list(self.last_seen.items()):
            if current - last > self.max_missing_frames:
                lost.append(tid)

        for tid in lost:
            if tid in self.crossed_ids and tid not in self.counted:
                self.missed_cross.add(tid)

            elif tid not in self.counted and tid in self.hist:
                cx, cy = self.hist[tid]
                if self.line_p1 and self.line_p2:
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

        display_w, display_h = self.display_size

        while True:
            if self.is_rtsp:
                frame = self.cap.read()
                if isinstance(frame, tuple):
                    ret, frame = frame
                else:
                    ret = frame is not None
            else:
                ret, frame = self.cap.read()

            if not ret or frame is None:
                break

            self.frame_count += 1
            # Throttle processing frequency if needed
            if self.frame_count % 3 != 0:
                # show the raw frame (resized) even when skipping logic
                if self.show:
                    display_frame = cv2.resize(frame, (display_w, display_h), interpolation=cv2.INTER_LINEAR)
                    for pt in self.temp_points:
                        cv2.circle(display_frame, pt, 5, (0, 0, 255), -1)
                    if self.line_p1:
                        cv2.line(display_frame, self.line_p1, self.line_p2, (255, 255, 255), 2)
                    cv2.imshow("ObjectCounter", display_frame)
                    if cv2.waitKey(1) & 0xFF == 27:
                        break
                continue

            # Resize frame to display size first (draw UI on this size to keep consistent)
            frame = cv2.resize(frame, (display_w, display_h), interpolation=cv2.INTER_LINEAR)

            # Draw temp points while user selects line
            for pt in self.temp_points:
                cv2.circle(frame, pt, 5, (0, 0, 255), -1)

            # Draw line if set
            if self.line_p1:
                cv2.line(frame, self.line_p1, self.line_p2, (255, 255, 255), 2)

            # Run YOLO tracking
            results = self.model.track(frame, persist=True, classes=self.classes, conf=0.7)

            # results[0] contains boxes; ensure structure exists
            if results and len(results) > 0 and self.line_p1 and hasattr(results[0].boxes, "xyxy"):
                boxes_xy = results[0].boxes.xyxy
                ids_attr = getattr(results[0].boxes, "id", None)

                if ids_attr is not None:
                    try:
                        ids = ids_attr.cpu().numpy().astype(int)
                    except Exception:
                        ids = np.array(ids_attr).astype(int)
                else:
                    ids = []

                try:
                    boxes = boxes_xy.cpu().numpy().astype(int)
                except Exception:
                    boxes = np.array(boxes_xy).astype(int)

                if len(ids) == len(boxes):
                    for tid, box in zip(ids, boxes):
                        x1, y1, x2, y2 = box
                        cx = int((x1 + x2) / 2)
                        cy = int((y1 + y2) / 2)

                        # update last seen frame index
                        self.last_seen[tid] = self.frame_count

                        # compute side positions
                        prev_side = None
                        if tid in self.hist:
                            px, py = self.hist[tid]
                            prev_side = self.side(px, py, *self.line_p1, *self.line_p2)

                        curr_side = self.side(cx, cy, *self.line_p1, *self.line_p2)

                        # CROSSING DETECTION
                        if prev_side is not None and prev_side * curr_side < 0:
                            # object crossed the line this frame
                            self.crossed_ids.add(tid)

                            if tid not in self.counted:
                                if curr_side < 0:
                                    # OUT direction -> detect color BEFORE crossing using previous centroid area
                                    px, py = self.hist.get(tid, (cx, cy))
                                    prev_box = (px - 20, py - 20, px + 20, py + 20)
                                    color_name = detect_box_color(frame, prev_box)
                                    self.out_count += 1
                                    self.color_out_count[color_name] = self.color_out_count.get(color_name, 0) + 1
                                else:
                                    # IN direction -> detect color AFTER crossing using current bounding box
                                    color_name = detect_box_color(frame, (x1, y1, x2, y2))
                                    self.in_count += 1
                                    self.color_in_count[color_name] = self.color_in_count.get(color_name, 0) + 1

                                self.counted.add(tid)

                        # update history (centroid)
                        self.hist[tid] = (cx, cy)

                        # Draw bounding box and label
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        display_color = detect_box_color(frame, (x1, y1, x2, y2))
                        cv2.putText(frame, display_color, (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 0), 2, lineType=cv2.LINE_AA)

            # MISSED CHECK
            if self.line_p1:
                self.check_lost_ids()

            # ================= ENHANCED DISPLAY (CRISP PANEL) =================

            # Panel area (top)
            panel_h = 160
            w, h = display_w, display_h

            if self.translucent_overlay:
                # Blend only the top ROI to keep rest crisp
                roi = frame[0:panel_h, 0:w].copy()
                overlay = roi.copy()
                cv2.rectangle(overlay, (0, 0), (w, panel_h), (0, 0, 0), -1)
                alpha = 0.35
                blended = cv2.addWeighted(overlay, alpha, roi, 1.0 - alpha, 0)
                frame[0:panel_h, 0:w] = blended
            else:
                # Draw solid rectangle (crisp text) — recommended for sharpness
                cv2.rectangle(frame, (0, 0), (w, panel_h), (0, 0, 0), -1)

            # --------- TITLE BAR ---------
            cv2.putText(frame, "TRACKING SYSTEM", (15, 32),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 200, 255), 3, lineType=cv2.LINE_AA)
            cv2.circle(frame, (250, 24), 7, (0, 255, 0), -1)

            # --------- TOTAL COUNTS ROW ---------
            y_row1 = 65
            font_large = 0.8
            thickness_bold = 3

            # Total IN
            cv2.putText(frame, "IN:", (15, y_row1),
                        cv2.FONT_HERSHEY_SIMPLEX, font_large, (0, 255, 150), thickness_bold, lineType=cv2.LINE_AA)
            cv2.putText(frame, str(self.in_count), (75, y_row1),
                        cv2.FONT_HERSHEY_SIMPLEX, font_large, (255, 255, 255), thickness_bold, lineType=cv2.LINE_AA)

            # Total OUT
            cv2.putText(frame, "OUT:", (150, y_row1),
                        cv2.FONT_HERSHEY_SIMPLEX, font_large, (100, 180, 255), thickness_bold, lineType=cv2.LINE_AA)
            cv2.putText(frame, str(self.out_count), (230, y_row1),
                        cv2.FONT_HERSHEY_SIMPLEX, font_large, (255, 255, 255), thickness_bold, lineType=cv2.LINE_AA)

            # Missed counts (smaller but visible)
            cv2.putText(frame, "MISS IN:", (320, y_row1),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (100, 255, 255), 2, lineType=cv2.LINE_AA)
            cv2.putText(frame, str(len(self.missed_in)), (430, y_row1),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, lineType=cv2.LINE_AA)

            cv2.putText(frame, "MISS OUT:", (485, y_row1),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 100, 255), 2, lineType=cv2.LINE_AA)
            cv2.putText(frame, str(len(self.missed_out)), (615, y_row1),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, lineType=cv2.LINE_AA)

            cv2.putText(frame, "CROSS:", (680, y_row1),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (100, 100, 255), 2, lineType=cv2.LINE_AA)
            cv2.putText(frame, str(len(self.missed_cross)), (770, y_row1),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, lineType=cv2.LINE_AA)

            # --------- SEPARATOR LINE ---------
            cv2.line(frame, (10, 85), (w-10, 85), (100, 100, 100), 2)

            # --------- BROWN BOX COUNTS ROW ---------
            y_row2 = 118
            brown_in = self.color_in_count.get("Brown", 0)
            brown_out = self.color_out_count.get("Brown", 0)

            # Brown box indicator (larger)
            cv2.rectangle(frame, (15, y_row2 - 25), (50, y_row2 - 5), (19, 69, 139), -1)
            cv2.rectangle(frame, (15, y_row2 - 25), (50, y_row2 - 5), (255, 255, 255), 2)

            cv2.putText(frame, "BROWN IN:", (60, y_row2),
                        cv2.FONT_HERSHEY_SIMPLEX, font_large, (100, 150, 255), thickness_bold, lineType=cv2.LINE_AA)
            cv2.putText(frame, str(brown_in), (240, y_row2),
                        cv2.FONT_HERSHEY_SIMPLEX, font_large, (255, 255, 255), thickness_bold, lineType=cv2.LINE_AA)

            cv2.putText(frame, "BROWN OUT:", (350, y_row2),
                        cv2.FONT_HERSHEY_SIMPLEX, font_large, (100, 150, 255), thickness_bold, lineType=cv2.LINE_AA)
            cv2.putText(frame, str(brown_out), (570, y_row2),
                        cv2.FONT_HERSHEY_SIMPLEX, font_large, (255, 255, 255), thickness_bold, lineType=cv2.LINE_AA)

            # --------- WHITE BOX COUNTS ROW ---------
            y_row3 = 150
            white_in = self.color_in_count.get("White", 0)
            white_out = self.color_out_count.get("White", 0)

            # White box indicator (larger)
            cv2.rectangle(frame, (15, y_row3 - 25), (50, y_row3 - 5), (245, 245, 245), -1)
            cv2.rectangle(frame, (15, y_row3 - 25), (50, y_row3 - 5), (100, 100, 100), 2)

            cv2.putText(frame, "WHITE IN:", (60, y_row3),
                        cv2.FONT_HERSHEY_SIMPLEX, font_large, (200, 255, 200), thickness_bold, lineType=cv2.LINE_AA)
            cv2.putText(frame, str(white_in), (240, y_row3),
                        cv2.FONT_HERSHEY_SIMPLEX, font_large, (255, 255, 255), thickness_bold, lineType=cv2.LINE_AA)

            cv2.putText(frame, "WHITE OUT:", (350, y_row3),
                        cv2.FONT_HERSHEY_SIMPLEX, font_large, (200, 255, 200), thickness_bold, lineType=cv2.LINE_AA)
            cv2.putText(frame, str(white_out), (570, y_row3),
                        cv2.FONT_HERSHEY_SIMPLEX, font_large, (255, 255, 255), thickness_bold, lineType=cv2.LINE_AA)

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
            try:
                self.cap.stop()
            except Exception:
                pass
        else:
            self.cap.release()

        cv2.destroyAllWindows()

# ==========================================================
#                        MAIN EXECUTION
# ==========================================================

if __name__ == "__main__":
    # Example usage:
    counter = ObjectCounter(
        source=0,  # "your_video.mp4" or 0 for webcam, or "rtsp://..."
        model="best_float32.tflite",  # or "yolov8n.pt"
        classes_to_count=[0],  # class ids as per your model
        show=True,
        pdf_folder="pdf_report",
        display_size=(1020, 600),
        translucent_overlay=False  # set True if you want the translucent top bar (may be slightly faded)
    )
    counter.run()
