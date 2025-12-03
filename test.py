import cv2
from ultralytics import YOLO
import cvzone
import json
import os
from imutils.video import VideoStream
import time
from datetime import datetime
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.platypus import Table, TableStyle

class ObjectCounter:
    def __init__(self, source, model="yolo12n.pt", classes_to_count=[0], show=True, json_file="line_coords.json"):
        self.source = source
        self.model = YOLO(model)
        self.names = self.model.names
        self.classes = classes_to_count
        self.show = show

        # ---- RTSP ----
        if isinstance(source, str) and source.startswith("rtsp://"):
            self.cap = VideoStream(source).start()
            time.sleep(2.0)
            self.is_rtsp = True
        else:
            self.cap = cv2.VideoCapture(source)
            self.is_rtsp = False

        # ---- counters ----
        self.hist = {}
        self.counted = set()
        self.track_info = {}

        self.in_count = 0
        self.out_count = 0
        self.missed_in = 0
        self.missed_out = 0

        self.max_missing_frames = 30
        self.frame_count = 0
        
        # ---- Session tracking ----
        self.session_start_time = datetime.now()

        # ---- line storage ----
        self.line_p1 = None
        self.line_p2 = None
        self.temp_points = []
        self.json_file = json_file
        self.load_line()

        cv2.namedWindow("ObjectCounter")
        cv2.setMouseCallback("ObjectCounter", self.mouse_event)

    # ---------------- Save to PDF ----------------
    def save_to_pdf(self):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"count_report_{timestamp}.pdf"
        
        c = canvas.Canvas(filename, pagesize=letter)
        width, height = letter
        
        # Header Background
        c.setFillColorRGB(0.2, 0.4, 0.7)  # Blue header
        c.rect(0, height - 2.2*inch, width, 2.2*inch, fill=True, stroke=False)
        
        # Title
        c.setFillColorRGB(1, 1, 1)  # White text
        c.setFont("Helvetica-Bold", 28)
        c.drawCentredString(width/2, height - 1*inch, "OBJECT COUNTER REPORT")
        
        # Subtitle line
        c.setFont("Helvetica", 12)
        c.drawCentredString(width/2, height - 1.4*inch, "Automated Traffic Monitoring System")
        
        # Session Times
        session_start = self.session_start_time.strftime("%B %d, %Y - %I:%M:%S %p")
        session_end = datetime.now().strftime("%B %d, %Y - %I:%M:%S %p")
        
        c.setFont("Helvetica-Bold", 10)
        c.drawString(1*inch, height - 1.9*inch, f"Session Start: {session_start}")
        c.drawString(1*inch, height - 2.1*inch, f"Session End:   {session_end}")
        
        # Calculate duration
        duration = datetime.now() - self.session_start_time
        hours = duration.seconds // 3600
        minutes = (duration.seconds % 3600) // 60
        seconds = duration.seconds % 60
        duration_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
        
        c.setFillColorRGB(1, 1, 1)
        c.drawString(width - 3*inch, height - 1.9*inch, f"Duration: {duration_str}")
        
        # Main Table Section
        y_position = height - 3*inch
        
        c.setFont("Helvetica-Bold", 16)
        c.setFillColorRGB(0.2, 0.2, 0.2)
        c.drawString(1*inch, y_position, "Traffic Count Summary")
        
        # Table Data
        table_y = y_position - 0.5*inch
        
        # Table settings
        col_width = (width - 2*inch) / 2
        row_height = 0.5*inch
        
        data = [
            ["Category", "Count"],
            ["IN Count", str(self.in_count)],
            ["OUT Count", str(self.out_count)],
            ["Missed IN", str(self.missed_in)],
            ["Missed OUT", str(self.missed_out)],
            ["Total Traffic", str(self.in_count + self.out_count)],
            ["Net Flow", str(self.in_count - self.out_count)]
        ]
        
        # Draw table
        start_x = 1*inch
        current_y = table_y
        
        for i, row in enumerate(data):
            if i == 0:  # Header row
                c.setFillColorRGB(0.2, 0.4, 0.7)
                c.rect(start_x, current_y - row_height, col_width * 2, row_height, fill=True, stroke=False)
                c.setFillColorRGB(1, 1, 1)
                c.setFont("Helvetica-Bold", 14)
            elif i == len(data) - 2 or i == len(data) - 1:  # Total and Net Flow rows
                c.setFillColorRGB(0.9, 0.9, 0.9)
                c.rect(start_x, current_y - row_height, col_width * 2, row_height, fill=True, stroke=True)
                c.setFillColorRGB(0.2, 0.2, 0.2)
                c.setFont("Helvetica-Bold", 13)
            else:  # Data rows
                # Alternating row colors
                if i % 2 == 1:
                    c.setFillColorRGB(0.95, 0.95, 0.95)
                else:
                    c.setFillColorRGB(1, 1, 1)
                c.rect(start_x, current_y - row_height, col_width * 2, row_height, fill=True, stroke=True)
                c.setFillColorRGB(0.2, 0.2, 0.2)
                c.setFont("Helvetica", 12)
            
            # Draw cell borders
            c.setStrokeColorRGB(0.7, 0.7, 0.7)
            c.setLineWidth(1)
            c.rect(start_x, current_y - row_height, col_width, row_height, fill=False, stroke=True)
            c.rect(start_x + col_width, current_y - row_height, col_width, row_height, fill=False, stroke=True)
            
            # Draw text
            c.drawString(start_x + 0.2*inch, current_y - row_height + 0.15*inch, row[0])
            
            # Color code the count values
            if i > 0:
                if "IN" in row[0] and "Missed" not in row[0]:
                    c.setFillColorRGB(0.2, 0.7, 0.3)  # Green for IN
                elif "OUT" in row[0] and "Missed" not in row[0]:
                    c.setFillColorRGB(0.9, 0.3, 0.3)  # Red for OUT
                elif "Missed IN" in row[0]:
                    c.setFillColorRGB(0.95, 0.7, 0.1)  # Yellow for Missed IN
                elif "Missed OUT" in row[0]:
                    c.setFillColorRGB(1, 0.5, 0.1)  # Orange for Missed OUT
                    
                if i < len(data) - 2:  # Not total rows
                    c.setFont("Helvetica-Bold", 14)
                
            c.drawString(start_x + col_width + 0.2*inch, current_y - row_height + 0.15*inch, row[1])
            
            current_y -= row_height
        
        # Statistics Summary Box
        summary_y = current_y - 0.8*inch
        c.setFillColorRGB(0.95, 0.95, 1)
        c.rect(1*inch, summary_y, width - 2*inch, 0.6*inch, fill=True, stroke=True)
        
        c.setFillColorRGB(0.2, 0.2, 0.2)
        c.setFont("Helvetica-Bold", 11)
        
        total_detected = self.in_count + self.out_count + self.missed_in + self.missed_out
        accuracy = ((self.in_count + self.out_count) / total_detected * 100) if total_detected > 0 else 100
        
        c.drawString(1.3*inch, summary_y + 0.35*inch, f"Total Objects Detected: {total_detected}")
        c.drawString(1.3*inch, summary_y + 0.1*inch, f"Counting Accuracy: {accuracy:.1f}%")
        
        # Footer
        c.setFont("Helvetica-Oblique", 9)
        c.setFillColorRGB(0.5, 0.5, 0.5)
        c.drawCentredString(width/2, 0.5*inch, "Generated by Object Counter System | Powered by YOLO")
        
        c.save()
        print(f"✓ Professional report saved: {filename}")
        return filename

    # ---------------- Mouse ----------------
    def mouse_event(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.temp_points.append((x, y))
            print(f"Point selected: {x},{y}")
            if len(self.temp_points) == 2:
                self.line_p1, self.line_p2 = self.temp_points
                self.temp_points = []
                self.save_line()
                print(f"Line set: {self.line_p1},{self.line_p2}")

    # ---------------- Save/Load ----------------
    def save_line(self):
        with open(self.json_file, "w") as f:
            json.dump({"line_p1": self.line_p1, "line_p2": self.line_p2}, f)

    def load_line(self):
        if os.path.exists(self.json_file):
            with open(self.json_file, "r") as f:
                data = json.load(f)
                self.line_p1 = tuple(data["line_p1"])
                self.line_p2 = tuple(data["line_p2"])
                print("Loaded saved line:", self.line_p1, self.line_p2)

    # ---------------- Side Check ----------------
    def side(self, px, py, x1, y1, x2, y2):
        return (x2 - x1)*(py - y1) - (y2 - y1)*(px - x1)

    # ---------------- Main Loop ----------------
    def run(self):
        print("Starting Object Counter...")
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

            # temp points
            for pt in self.temp_points:
                cv2.circle(frame, pt, 5, (0,0,255), -1)

            if self.line_p1 and self.line_p2:
                cv2.line(frame, self.line_p1, self.line_p2, (255,255,255), 2)

            results = self.model.track(frame, persist=True, classes=self.classes, conf=0.8)

            visible_ids = set()

            if results[0].boxes.id is not None and self.line_p1 and self.line_p2:

                ids = results[0].boxes.id.cpu().numpy().astype(int)
                boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)

                for track_id, box in zip(ids, boxes):
                    visible_ids.add(track_id)

                    x1, y1, x2, y2 = box
                    cx = int((x1 + x2) / 2)
                    cy = int((y1 + y2) / 2)

                    curr_side = self.side(cx, cy, *self.line_p1, *self.line_p2)

                    # create track record
                    if track_id not in self.track_info:
                        self.track_info[track_id] = {
                            "first_side": curr_side,
                            "last_side": curr_side,
                            "last_seen": self.frame_count,
                            "counted": False
                        }

                    # crossing logic
                    if track_id in self.hist:
                        prev_cx, prev_cy = self.hist[track_id]
                        prev_side = self.side(prev_cx, prev_cy, *self.line_p1, *self.line_p2)

                        if prev_side * curr_side < 0 and not self.track_info[track_id]["counted"]:
                            if curr_side > 0:
                                self.in_count += 1
                                direction = "IN"
                            else:
                                self.out_count += 1
                                direction = "OUT"

                            self.track_info[track_id]["counted"] = True
                            self.counted.add(track_id)
                            print(f"[COUNT] {track_id} -> {direction}")

                    self.track_info[track_id]["last_side"] = curr_side
                    self.track_info[track_id]["last_seen"] = self.frame_count
                    self.hist[track_id] = (cx, cy)

                    # draw
                    cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
                    cv2.circle(frame,(cx,cy),4,(255,0,0),-1)

                    status = "COUNTED" if self.track_info[track_id]["counted"] else "ACTIVE"
                    cvzone.putTextRect(frame, f"ID:{track_id} {status}", (x1,y1), 1, 1)

            # -------- MISSED CHECK --------
            for tid in list(self.track_info.keys()):
                info = self.track_info[tid]

                if tid not in visible_ids and (self.frame_count - info["last_seen"] > self.max_missing_frames):

                    if not info["counted"]:
                        # Check if object crossed the line but wasn't counted
                        if tid in self.hist:
                            first_side = info["first_side"]
                            last_side = info["last_side"]
                            
                            # Check if sides changed (crossed the line)
                            if first_side * last_side < 0:
                                if last_side > 0:
                                    self.missed_in += 1
                                    missed = "MISSED IN"
                                else:
                                    self.missed_out += 1
                                    missed = "MISSED OUT"
                                
                                print(f"[MISS] {tid} -> {missed}")

                    del self.track_info[tid]

            # -------- DISPLAY --------
            cvzone.putTextRect(frame,f"IN: {self.in_count}",(50,30),2,2,colorR=(0,255,0))
            cvzone.putTextRect(frame,f"OUT: {self.out_count}",(50,80),2,2,colorR=(0,0,255))

            cvzone.putTextRect(frame,f"MISSED IN: {self.missed_in}",(50,130),2,2,colorR=(255,255,0))
            cvzone.putTextRect(frame,f"MISSED OUT: {self.missed_out}",(50,180),2,2,colorR=(255,50,50))

            if self.show:
                cv2.imshow("ObjectCounter", frame)

            key = cv2.waitKey(1) & 0xFF

            if key == ord('r'):
                print("RESET LINE ONLY")
                self.line_p1 = None
                self.line_p2 = None
                self.temp_points = []
                if os.path.exists(self.json_file):
                    os.remove(self.json_file)

            elif key == ord('o'):
                print("SAVING DATA TO PDF & RESETTING COUNTERS...")
                self.save_to_pdf()
                self.in_count = 0
                self.out_count = 0
                self.session_start_time = datetime.now()  # Reset session time
                print("IN & OUT COUNTERS RESET - NEW SESSION STARTED")

            elif key == 27:
                break

        # -------- CLEANUP --------
        if self.is_rtsp:
            self.cap.stop()
        else:
            self.cap.release()
        cv2.destroyAllWindows()
        print("Stopped.")

# ---------------- RUN ----------------
if __name__ == "__main__":
    src = 0   # 0 for webcam or "rtsp://...." or video file path
    counter = ObjectCounter(source=src, model="yolo12n.pt", classes_to_count=[0], show=True)
    counter.run()
