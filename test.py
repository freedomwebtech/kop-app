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
    def __init__(self, source, model="yolo12n.pt", classes_to_count=[0], show=True, json_file="line_coords.json", pdf_file="count_report.pdf"):
        self.source = source
        self.model = YOLO(model)
        self.names = self.model.names
        self.classes = classes_to_count
        self.show = show
        self.pdf_file = pdf_file

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
        
        # ---- NEW: Track missed IDs for display ----
        self.missed_track_ids = []  # Store recently missed track IDs
        self.max_missed_display = 10  # Maximum number of missed IDs to show

        self.max_missing_frames = 30
        self.frame_count = 0
        
        # ---- Session tracking ----
        self.session_start_time = datetime.now()
        self.session_history = []
        self.current_session_saved = False
        self.load_session_history()

        # ---- line storage ----
        self.line_p1 = None
        self.line_p2 = None
        self.temp_points = []
        self.json_file = json_file
        self.load_line()

        cv2.namedWindow("ObjectCounter")
        cv2.setMouseCallback("ObjectCounter", self.mouse_event)

    # ---------------- Load Session History ----------------
    def load_session_history(self):
        """Load previous session data from JSON"""
        history_file = "session_history.json"
        if os.path.exists(history_file):
            try:
                with open(history_file, "r") as f:
                    self.session_history = json.load(f)
                print(f"Loaded {len(self.session_history)} previous sessions")
            except:
                self.session_history = []

    # ---------------- Save Session History ----------------
    def save_session_history(self):
        """Save session data to JSON"""
        history_file = "session_history.json"
        with open(history_file, "w") as f:
            json.dump(self.session_history, f, indent=2)

    # ---------------- Save to PDF ----------------
    def save_to_pdf(self):
        """Save all sessions to a single PDF file (overwrites)"""
        
        if not self.current_session_saved:
            current_session = {
                "start": self.session_start_time.strftime("%Y-%m-%d %H:%M:%S"),
                "end": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "in_count": self.in_count,
                "out_count": self.out_count,
                "missed_in": self.missed_in,
                "missed_out": self.missed_out
            }
            self.session_history.append(current_session)
            self.save_session_history()
            self.current_session_saved = True
        
        c = canvas.Canvas(self.pdf_file, pagesize=letter)
        width, height = letter
        
        # Header Background
        c.setFillColorRGB(0.2, 0.4, 0.7)
        c.rect(0, height - 2.2*inch, width, 2.2*inch, fill=True, stroke=False)
        
        # Title
        c.setFillColorRGB(1, 1, 1)
        c.setFont("Helvetica-Bold", 28)
        c.drawCentredString(width/2, height - 1*inch, "OBJECT COUNTER REPORT")
        
        c.setFont("Helvetica", 12)
        c.drawCentredString(width/2, height - 1.4*inch, "Automated Traffic Monitoring System")
        
        c.setFont("Helvetica-Bold", 10)
        c.drawCentredString(width/2, height - 1.8*inch, f"Report Generated: {datetime.now().strftime('%B %d, %Y - %I:%M:%S %p')}")
        
        y_position = height - 2.8*inch
        
        # ============ ALL SESSIONS TABLE ============
        c.setFont("Helvetica-Bold", 16)
        c.setFillColorRGB(0.2, 0.2, 0.2)
        c.drawString(1*inch, y_position, f"All Sessions Summary ({len(self.session_history)} Total)")
        
        y_position -= 0.4*inch
        
        total_in = sum(s["in_count"] for s in self.session_history)
        total_out = sum(s["out_count"] for s in self.session_history)
        total_missed_in = sum(s["missed_in"] for s in self.session_history)
        total_missed_out = sum(s["missed_out"] for s in self.session_history)
        
        col_width = (width - 2*inch) / 2
        row_height = 0.4*inch
        
        grand_data = [
            ["Total Metric", "Count"],
            ["Total IN", str(total_in)],
            ["Total OUT", str(total_out)],
            ["Total Missed IN", str(total_missed_in)],
            ["Total Missed OUT", str(total_missed_out)],
            ["Grand Total Traffic", str(total_in + total_out)],
            ["Net Flow", str(total_in - total_out)]
        ]
        
        start_x = 1*inch
        current_y = y_position
        
        for i, row in enumerate(grand_data):
            if i == 0:
                c.setFillColorRGB(0.2, 0.4, 0.7)
                c.rect(start_x, current_y - row_height, col_width * 2, row_height, fill=True, stroke=False)
                c.setFillColorRGB(1, 1, 1)
                c.setFont("Helvetica-Bold", 12)
            else:
                c.setFillColorRGB(0.95, 0.95, 0.95) if i % 2 == 0 else c.setFillColorRGB(1, 1, 1)
                c.rect(start_x, current_y - row_height, col_width * 2, row_height, fill=True, stroke=True)
                c.setFillColorRGB(0.2, 0.2, 0.2)
                c.setFont("Helvetica", 11)
            
            c.setStrokeColorRGB(0.7, 0.7, 0.7)
            c.setLineWidth(1)
            c.rect(start_x, current_y - row_height, col_width, row_height, fill=False, stroke=True)
            c.rect(start_x + col_width, current_y - row_height, col_width, row_height, fill=False, stroke=True)
            
            c.drawString(start_x + 0.2*inch, current_y - row_height + 0.12*inch, row[0])
            
            if i > 0:
                if "IN" in row[0] and "Missed" not in row[0]:
                    c.setFillColorRGB(0.2, 0.7, 0.3)
                elif "OUT" in row[0] and "Missed" not in row[0]:
                    c.setFillColorRGB(0.9, 0.3, 0.3)
                c.setFont("Helvetica-Bold", 12)
                
            c.drawString(start_x + col_width + 0.2*inch, current_y - row_height + 0.12*inch, row[1])
            current_y -= row_height
        
        # ============ INDIVIDUAL SESSIONS ============
        current_y -= 0.6*inch
        
        if current_y < 3*inch:
            c.showPage()
            current_y = height - 1*inch
        
        c.setFont("Helvetica-Bold", 14)
        c.setFillColorRGB(0.2, 0.2, 0.2)
        c.drawString(1*inch, current_y, "Individual Session Details")
        
        current_y -= 0.3*inch
        
        for idx, session in enumerate(self.session_history[-10:], 1):
            if current_y < 2*inch:
                c.showPage()
                current_y = height - 1*inch
            
            box_height = 1.2*inch
            c.setFillColorRGB(0.98, 0.98, 1)
            c.rect(1*inch, current_y - box_height, width - 2*inch, box_height, fill=True, stroke=True)
            
            c.setFillColorRGB(0.2, 0.2, 0.2)
            c.setFont("Helvetica-Bold", 11)
            c.drawString(1.2*inch, current_y - 0.25*inch, f"Session {len(self.session_history) - 10 + idx}")
            
            c.setFont("Helvetica", 9)
            c.drawString(1.2*inch, current_y - 0.45*inch, f"Start: {session['start']}")
            c.drawString(1.2*inch, current_y - 0.65*inch, f"End:   {session['end']}")
            
            c.setFont("Helvetica-Bold", 10)
            c.setFillColorRGB(0.2, 0.7, 0.3)
            c.drawString(1.2*inch, current_y - 0.9*inch, f"IN: {session['in_count']}")
            
            c.setFillColorRGB(0.9, 0.3, 0.3)
            c.drawString(2.2*inch, current_y - 0.9*inch, f"OUT: {session['out_count']}")
            
            c.setFillColorRGB(0.6, 0.6, 0.6)
            c.drawString(3.3*inch, current_y - 0.9*inch, f"Missed IN: {session['missed_in']}")
            c.drawString(4.8*inch, current_y - 0.9*inch, f"Missed OUT: {session['missed_out']}")
            
            current_y -= (box_height + 0.2*inch)
        
        # Footer
        c.setFont("Helvetica-Oblique", 9)
        c.setFillColorRGB(0.5, 0.5, 0.5)
        c.drawCentredString(width/2, 0.5*inch, "Generated by Object Counter System | Powered by YOLO")
        
        c.save()
        print(f"✓ Report saved to: {self.pdf_file} (Contains {len(self.session_history)} sessions)")
        return self.pdf_file

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
                        if tid in self.hist:
                            first_side = info["first_side"]
                            last_side = info["last_side"]
                            
                            if first_side * last_side < 0:
                                if last_side > 0:
                                    self.missed_in += 1
                                    missed = "MISSED IN"
                                else:
                                    self.missed_out += 1
                                    missed = "MISSED OUT"
                                
                                # ---- NEW: Add to missed display list ----
                                self.missed_track_ids.append({
                                    "id": tid,
                                    "type": missed,
                                    "frame": self.frame_count
                                })
                                
                                # Keep only recent missed IDs
                                if len(self.missed_track_ids) > self.max_missed_display:
                                    self.missed_track_ids.pop(0)
                                
                                print(f"[MISS] {tid} -> {missed}")

                    del self.track_info[tid]

            # -------- DISPLAY --------
            cvzone.putTextRect(frame,f"IN: {self.in_count}",(50,30),2,2,colorR=(0,255,0))
            cvzone.putTextRect(frame,f"OUT: {self.out_count}",(50,80),2,2,colorR=(0,0,255))

            cvzone.putTextRect(frame,f"MISSED IN: {self.missed_in}",(50,130),2,2,colorR=(255,255,0))
            cvzone.putTextRect(frame,f"MISSED OUT: {self.missed_out}",(50,180),2,2,colorR=(255,50,50))
            
            cvzone.putTextRect(frame,f"Sessions: {len(self.session_history)}",(50,230),1,1,colorR=(100,100,255))
            
            # -------- NEW: DISPLAY MISSED TRACK IDs PANEL --------
            if self.missed_track_ids:
                panel_y = 280
                cvzone.putTextRect(frame, "MISSED TRACKS:", (50, panel_y), 1, 2, colorR=(255,165,0))
                
                panel_y += 40
                for missed_info in self.missed_track_ids[-5:]:  # Show last 5
                    tid = missed_info["id"]
                    miss_type = missed_info["type"]
                    
                    # Color based on type
                    if "IN" in miss_type:
                        color = (0, 255, 255)  # Yellow
                    else:
                        color = (50, 50, 255)  # Red
                    
                    cvzone.putTextRect(frame, f"ID:{tid} - {miss_type}", (50, panel_y), 1, 1, colorR=color)
                    panel_y += 25

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
                print("SAVING SESSION TO PDF & RESETTING COUNTERS...")
                self.save_to_pdf()
                self.in_count = 0
                self.out_count = 0
                self.missed_in = 0
                self.missed_out = 0
                self.missed_track_ids = []  # Clear missed track display
                self.session_start_time = datetime.now()
                self.current_session_saved = False
                print("COUNTERS RESET - NEW SESSION STARTED")

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
