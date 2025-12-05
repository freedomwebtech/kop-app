from __future__ import annotations
from collections import defaultdict
from typing import Any

from ultralytics.solutions.solutions import BaseSolution, SolutionAnnotator, SolutionResults
from ultralytics.utils.plotting import colors


class ObjectCounter(BaseSolution):
    """
    Rectangular-Region Object Counter
    Only works with rectangle region: [(x1,y1),(x2,y1),(x2,y2),(x1,y2)]
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

        self.in_count = 0
        self.out_count = 0
        self.counted_ids = set()
        self.classwise_count = defaultdict(lambda: {"IN": 0, "OUT": 0})
        self.region_initialized = False

        self.show_in = self.CFG.get("show_in", True)
        self.show_out = self.CFG.get("show_out", True)
        self.margin = self.line_width * 2

        self.rect = None


    def initialize_region(self):
        """Convert 4-Point Region into Rectangle Bounds"""
        xs = [p[0] for p in self.region]
        ys = [p[1] for p in self.region]

        self.x1, self.x2 = int(min(xs)), int(max(xs))
        self.y1, self.y2 = int(min(ys)), int(max(ys))

        self.rect = True


    def is_inside(self, point):
        """Check if centroid is inside rectangle"""
        x, y = point
        return self.x1 <= x <= self.x2 and self.y1 <= y <= self.y2


    def count_objects(self, curr, prev, track_id, cls):
        """Count entering and exiting rectangle"""

        if prev is None or track_id in self.counted_ids:
            return

        was_inside = self.is_inside(prev)
        is_inside = self.is_inside(curr)

        # ✅ Entering rectangle
        if not was_inside and is_inside:
            self.in_count += 1
            self.classwise_count[self.names[cls]]["IN"] += 1
            self.counted_ids.add(track_id)

        # ✅ Leaving rectangle
        elif was_inside and not is_inside:
            self.out_count += 1
            self.classwise_count[self.names[cls]]["OUT"] += 1
            self.counted_ids.add(track_id)


    def display_counts(self, plot_im):
        labels = {}
        for k, v in self.classwise_count.items():
            if v["IN"] or v["OUT"]:
                labels[k.upper()] = f"IN {v['IN']}  OUT {v['OUT']}"

        if labels:
            self.annotator.display_analytics(
                plot_im, labels, (0, 0, 255), (255, 255, 255), self.margin
            )


    def process(self, im0):
        """Main frame processing method"""

        if not self.region_initialized:
            self.initialize_region()
            self.region_initialized = True

        self.extract_tracks(im0)
        self.annotator = SolutionAnnotator(im0, line_width=self.line_width)

        # ✅ Draw rectangle region
        self.annotator.draw_region(
            reg_pts=self.region, color=(0, 255, 0), thickness=self.line_width * 2
        )

        for box, tid, cls, conf in zip(self.boxes, self.track_ids, self.clss, self.confs):
            self.annotator.box_label(
                box, label=self.adjust_box_label(cls, conf, tid), color=colors(cls, True)
            )

            self.store_tracking_history(tid, box)
            prev = self.track_history[tid][-2] if len(self.track_history[tid]) > 1 else None
            curr = self.track_history[tid][-1]

            self.count_objects(curr, prev, tid, cls)

        plot_im = self.annotator.result()
        self.display_counts(plot_im)
        self.display_output(plot_im)

        return SolutionResults(
            plot_im=plot_im,
            in_count=self.in_count,
            out_count=self.out_count,
            classwise_count=dict(self.classwise_count),
            total_tracks=len(self.track_ids),
        )
