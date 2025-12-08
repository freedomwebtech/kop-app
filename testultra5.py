# Ultralytics 🚀 AGPL-3.0 License

from __future__ import annotations
from collections import defaultdict
from typing import Any
from datetime import datetime
import math
import cv2
from ultralytics.solutions.solutions import BaseSolution, SolutionAnnotator, SolutionResults
from ultralytics.utils.plotting import colors


class ObjectCounter(BaseSolution):
    """Object Counter with strict confidence filtering + IN / OUT / MISS + STATIONARY + Dimension Logger + Width Filter"""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

        # ========= COUNTS =========
        self.in_count = 0
        self.out_count = 0
        self.miss_in = 0
        self.miss_out = 0
        self.stationary_count = 0

        # ========= STATE =========
        self.counted_ids = set()
        self.last_motion = {}
        self.stationary_ids = set()
        self.stationary_frames = defaultdict(int)

        # ========= CONFIG =========
        self.STATIONARY_THRESH = 4
        self.STATIONARY_FRAME_LIMIT = 20
        self.conf = float(self.CFG.get("conf", 0.80))   # 🔥 STRICT CONF THRESH
        self.min_area = self.CFG.get("min_area", 28999)  # 🔥 MINIMUM AREA FILTER

        # ========= CLASSWISE =========
        self.classwise_count = defaultdict(lambda: {"IN": 0, "OUT": 0})

        # ========= DISPLAY =========
        self.region_initialized = False
        self.show_in = self.CFG.get("show_in", True)
        self.show_out = self.CFG.get("show_out", True)
        self.margin = self.line_width * 2

        # ========= DIMENSION LOGGING =========
        self.log_file = self.CFG.get("log_file", "object_dimensions.txt")
        self.logged_ids = set()  # Track which IDs we've already logged
        self.rejected_ids = set()  # Track IDs rejected due to area


    # --------------------------------------------------
    # CALCULATE AREA
    # --------------------------------------------------
    def calculate_area(self, width, height):
        """Calculate area of rectangle"""
        return width * height


    # --------------------------------------------------
    # CHECK AREA VALIDITY
    # --------------------------------------------------
    def is_valid_area(self, box):
        """Check if object area meets minimum requirement"""
        x1, y1, x2, y2 = box
        width = int(x2 - x1)
        height = int(y2 - y1)
        area = self.calculate_area(width, height)
        return area >= self.min_area


    # --------------------------------------------------
    # CONVERT PIXELS TO CENTIMETERS
    # --------------------------------------------------
    def pixels_to_cm(self, pixels):
        """Convert pixels to centimeters"""
        return round(pixels / self.pixels_per_cm, 2)
    # --------------------------------------------------
    # LOG DIMENSIONS TO FILE
    # --------------------------------------------------
    def log_dimensions(self, track_id, box, cls):
        """Log width, height, area, track ID, class, date and time to file"""
        
        # Only log each ID once to avoid duplicate entries
        if track_id in self.logged_ids:
            return
        
        x1, y1, x2, y2 = box
        width = int(x2 - x1)
        height = int(y2 - y1)
        area = self.calculate_area(width, height)
        
        # Get current date and time
        now = datetime.now()
        date_str = now.strftime("%Y-%m-%d")
        time_str = now.strftime("%H:%M:%S")
        
        # Class name
        class_name = self.names[cls]
        
        # Create log entry with area
        log_entry = f"ID: {track_id} | Class: {class_name} | Width: {width}px | Height: {height}px | Area: {area}px² | Date: {date_str} | Time: {time_str}\n"
        
        # Append to file
        try:
            with open(self.log_file, 'a') as f:
                f.write(log_entry)
            self.logged_ids.add(track_id)
            print(f"✅ Logged: {log_entry.strip()}")
        except Exception as e:
            print(f"❌ Error logging dimensions: {e}")


    # --------------------------------------------------
    # CHECK WIDTH AND HEIGHT VALIDITY
    # --------------------------------------------------
    def is_valid_size(self, box):
        """Check if object width AND height meet minimum requirements"""
        x1, y1, x2, y2 = box
        width = int(x2 - x1)
        height = int(y2 - y1)
        return width >= self.min_width and height >= self.min_height


    # --------------------------------------------------
    # REMOVE LOW CONF DETECTIONS
    # --------------------------------------------------
    def apply_conf_filter(self):
        if len(self.confs) == 0:
            return

        boxes2, clss2, confs2, ids2 = [], [], [], []

        for i, c in enumerate(self.confs):
            if float(c) >= self.conf:
                boxes2.append(self.boxes[i])
                clss2.append(self.clss[i])
                confs2.append(self.confs[i])
                ids2.append(self.track_ids[i])

        self.boxes = boxes2
        self.clss = clss2
        self.confs = confs2
        self.track_ids = ids2


    # --------------------------------------------------
    # DIRECTION + STATIONARY
    # --------------------------------------------------
    def get_natural_direction(self, prev, curr):
        dx = curr[0] - prev[0]
        dy = curr[1] - prev[1]
        dist = math.sqrt(dx * dx + dy * dy)

        if dist < self.STATIONARY_THRESH:
            return "STATIONARY"

        if abs(dx) > abs(dy):
            return "RIGHT" if dx > 0 else "LEFT"
        else:
            return "DOWN" if dy > 0 else "UP"


    # --------------------------------------------------
    # SIDE OF LINE
    # --------------------------------------------------
    def get_side_of_line(self, point):
        x1, y1 = self.region[0]
        x2, y2 = self.region[1]
        px, py = point
        return (x2 - x1) * (py - y1) - (y2 - y1) * (px - x1)


    # --------------------------------------------------
    # COUNT OBJECTS
    # --------------------------------------------------
    def count_objects(self, curr, track_id, prev, cls, box):

        if prev is None or track_id in self.counted_ids:
            return

        # 🔥 CHECK AREA - DON'T COUNT IF TOO SMALL
        if not self.is_valid_area(box):
            if track_id not in self.rejected_ids:
                self.rejected_ids.add(track_id)
                x1, y1, x2, y2 = box
                width = int(x2 - x1)
                height = int(y2 - y1)
                area = self.calculate_area(width, height)
                print(f"❌ REJECTED ID {track_id} - Area {area}px² < {self.min_area}px² minimum")
            return

        # -------- LINE MODE --------
        if len(self.region) == 2:
            line = self.LineString(self.region)
            motion = self.LineString([prev, curr])

            curr_side = self.get_side_of_line(curr)
            prev_side = self.get_side_of_line(prev)

            if line.intersects(motion):

                if prev_side < 0 and curr_side > 0:
                    direction = "IN"
                elif prev_side > 0 and curr_side < 0:
                    direction = "OUT"
                else:
                    direction = "IN" if curr_side > prev_side else "OUT"

                if direction == "IN":
                    self.in_count += 1
                    self.classwise_count[self.names[cls]]["IN"] += 1
                else:
                    self.out_count += 1
                    self.classwise_count[self.names[cls]]["OUT"] += 1

                self.counted_ids.add(track_id)

        # -------- POLYGON MODE --------
        elif len(self.region) > 2 and self.r_s.contains(self.Point(curr)):

            w = max(p[0] for p in self.region) - min(p[0] for p in self.region)
            h = max(p[1] for p in self.region) - min(p[1] for p in self.region)

            direction = "IN" if (w < h and curr[0] > prev[0]) or (w >= h and curr[1] > prev[1]) else "OUT"

            if direction == "IN":
                self.in_count += 1
                self.classwise_count[self.names[cls]]["IN"] += 1
            else:
                self.out_count += 1
                self.classwise_count[self.names[cls]]["OUT"] += 1

            self.counted_ids.add(track_id)


    # --------------------------------------------------
    # MISSED OBJECT LOGIC
    # --------------------------------------------------
    def check_missed_objects(self):

        active = set(self.track_ids) if hasattr(self, "track_ids") else set()

        for track_id in list(self.last_motion.keys()):
            if track_id not in active and track_id not in self.counted_ids and track_id not in self.rejected_ids:

                direction = self.last_motion.get(track_id)

                if direction == "IN":
                    self.miss_in += 1
                elif direction == "OUT":
                    self.miss_out += 1

                self.counted_ids.add(track_id)
                self.last_motion.pop(track_id, None)


    # --------------------------------------------------
    # DISPLAY COUNTS
    # --------------------------------------------------
    def display_counts(self, frame):

        labels = {}

        for cls, count in self.classwise_count.items():
            parts = []
            if self.show_in:
                parts.append(f"IN {count['IN']}")
            if self.show_out:
                parts.append(f"OUT {count['OUT']}")
            labels[cls] = " | ".join(parts)

        labels["MISS"] = f"IN {self.miss_in} | OUT {self.miss_out}"
        labels["STATIONARY"] = str(self.stationary_count)

        self.annotator.display_analytics(frame, labels, (0, 0, 200), (255, 255, 255), self.margin)


    # --------------------------------------------------
    # MAIN PROCESS
    # --------------------------------------------------
    def process(self, im0):

        if not self.region_initialized:
            self.initialize_region()
            self.region_initialized = True

        # -- run detection and tracking
        self.extract_tracks(im0)

        # ✅ KILL LOW CONF BEFORE DOING ANYTHING
        self.apply_conf_filter()

        self.annotator = SolutionAnnotator(im0, line_width=self.line_width)
        self.annotator.draw_region(self.region, color=(160, 0, 255), thickness=self.line_width * 2)

        for box, track_id, cls, conf in zip(self.boxes, self.track_ids, self.clss, self.confs):

            # ========= CHECK AREA VALIDITY =========
            is_valid = self.is_valid_area(box)
            
            # ========= LOG DIMENSIONS (ONCE PER ID) - ONLY IF VALID =========
            if is_valid:
                self.log_dimensions(track_id, box, cls)

            self.store_tracking_history(track_id, box)

            prev = self.track_history[track_id][-2] if len(self.track_history[track_id]) > 1 else None
            curr = self.track_history[track_id][-1]

            arrow = "STILL"
            if prev is not None:
                arrow = self.get_natural_direction(prev, curr)

            # ========= STATIONARY (ONLY FOR VALID OBJECTS) =========
            if is_valid:
                if arrow == "STATIONARY":
                    self.stationary_frames[track_id] += 1
                else:
                    self.stationary_frames[track_id] = 0

                if self.stationary_frames[track_id] >= self.STATIONARY_FRAME_LIMIT:
                    if track_id not in self.stationary_ids:
                        self.stationary_ids.add(track_id)
                        self.stationary_count += 1

            # ========= DIRECTION MEMORY (ONLY FOR VALID OBJECTS) =========
            if is_valid:
                if arrow in ["UP", "LEFT"]:
                    self.last_motion[track_id] = "OUT"
                elif arrow in ["DOWN", "RIGHT"]:
                    self.last_motion[track_id] = "IN"

            # ========= LABEL WITH CONF + DIMENSIONS + AREA =========
            x1, y1, x2, y2 = box
            width = int(x2 - x1)
            height = int(y2 - y1)
            area = self.calculate_area(width, height)
            
            if is_valid:
                label = f"{self.names[cls]} {conf:.2f} | ID {track_id} | {arrow} | W:{width} H:{height} | Area:{area}px²"
                if track_id in self.stationary_ids:
                    label += " | STOPPED"
                box_color = colors(cls, True)
            else:
                label = f"REJECTED | ID {track_id} | W:{width} H:{height} | Area:{area}px² | MIN:{self.min_area}px²"
                box_color = (0, 0, 255)  # 🔴 RED for rejected objects

            self.annotator.box_label(box, label, color=box_color)

            # ========= COUNT (ONLY IF VALID AREA) =========
            if is_valid:
                self.count_objects(curr, track_id, prev, cls, box)

        self.check_missed_objects()

        plot_im = self.annotator.result()
        self.display_counts(plot_im)
        self.display_output(plot_im)

        return SolutionResults(
            plot_im=plot_im,
            in_count=self.in_count,
            out_count=self.out_count,
            miss_in=self.miss_in,
            miss_out=self.miss_out,
            stationary=self.stationary_count,
            classwise_count=dict(self.classwise_count),
            total_tracks=len(self.track_ids),
        )