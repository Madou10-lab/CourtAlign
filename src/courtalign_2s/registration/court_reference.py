from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from sympy import Line


class CourtReference:
    """Reference-court model used by the CourtAlign-2S tennis geometry stage."""

    def __init__(self, asset_dir: str | Path | None = None):
        self.court_conf2 = {1: [(426, 2389), (1239, 2389), (426, 2932), (1239, 2932)]}
        self.court_conf1 = {1: [(284, 559), (1381, 559), (284, 2937), (1379, 2935)]}

        self.baseline_top = ((286, 561), (1379, 561))
        self.baseline_bottom = ((286, 2935), (1379, 2935))
        self.net = ((286, 1748), (1379, 1748))
        self.left_court_line = ((286, 561), (286, 2935))
        self.right_court_line = ((1379, 561), (1379, 2935))
        self.left_inner_line = ((423, 561), (423, 2935))
        self.right_inner_line = ((1242, 561), (1242, 2935))
        self.middle_line = ((832, 1110), (832, 2386))
        self.top_inner_line = ((423, 1110), (1242, 1110))
        self.bottom_inner_line = ((423, 2386), (1242, 2386))
        self.court_width = 1117
        self.court_height = 2408
        self.top_bottom_border = 549
        self.right_left_border = 274

        if asset_dir is None:
            asset_dir = Path("templates")
        else:
            asset_dir = Path(asset_dir)
        court_path = asset_dir / "tennis_court_reference.png"
        court = cv2.imread(str(court_path))
        if court is None:
            raise FileNotFoundError(f"Missing CourtAlign-2S court reference image: {court_path}")
        self.court = cv2.cvtColor(court, cv2.COLOR_BGR2GRAY)

    def get_important_lines(self):
        lines = [
            *self.baseline_top,
            *self.baseline_bottom,
            *self.left_court_line,
            *self.right_court_line,
            *self.left_inner_line,
            *self.right_inner_line,
            *self.middle_line,
            *self.top_inner_line,
            *self.bottom_inner_line,
        ]
        return lines

    def draw_yellow_lines(self, img):
        line1 = (self.bottom_inner_line[0], self.left_inner_line[1])
        line2 = (self.left_inner_line[0], self.top_inner_line[0])

        y11, y12 = self.split_line_into_3_sections(line1)
        y21, y22 = self.split_line_into_3_sections(line2)

        start_point11 = (line1[0][0], y11)
        end_point11 = (self.right_inner_line[0][0], y11)
        start_point12 = (line1[0][0], y12)
        end_point12 = (self.right_inner_line[0][0], y12)
        start_point21 = (line2[0][0], y21)
        end_point21 = (self.right_inner_line[0][0], y21)
        start_point22 = (line2[0][0], y22)
        end_point22 = (self.right_inner_line[0][0], y22)

        self.draw_dotted_line((start_point11, end_point11), img)
        self.draw_dotted_line((start_point12, end_point12), img)
        self.draw_dotted_line((start_point21, end_point21), img)
        self.draw_dotted_line((start_point22, end_point22), img)
        return [
            (start_point11, end_point11),
            (start_point12, end_point12),
            (start_point21, end_point21),
            (start_point22, end_point22),
        ]

    def segment_court_zones(self):
        x1 = np.round((self.baseline_top[0][0] + self.baseline_top[1][0]) / 2)
        y1 = self.baseline_top[0][1]
        center1 = (x1, y1)
        x2 = np.round((self.baseline_bottom[0][0] + self.baseline_bottom[1][0]) / 2)
        y2 = self.baseline_bottom[0][1]
        center2 = (x2, y2)

        inter1 = line_intersection(self.left_inner_line, self.net)
        inter2 = line_intersection(self.middle_line, self.net)
        inter3 = line_intersection(self.right_inner_line, self.net)

        zone_top_left = [self.left_inner_line[0], center1, inter1, inter2]
        zone_top_right = [center1, self.right_inner_line[0], inter2, inter3]
        zone_bottom_left = [inter1, inter2, self.left_inner_line[1], center2]
        zone_bottom_right = [inter2, inter3, center2, self.right_inner_line[1]]
        zones = [zone_top_left, zone_bottom_left, zone_top_right, zone_bottom_right]

        court = np.zeros(
            (self.court_height + 2 * self.top_bottom_border, self.court_width + 2 * self.right_left_border),
            dtype=np.uint8,
        )
        court = cv2.cvtColor(court, cv2.COLOR_GRAY2BGR)
        cv2.line(court, *self.baseline_top, (255, 255, 255), 1)
        cv2.line(court, *self.baseline_bottom, (255, 255, 255), 1)
        cv2.line(court, *self.net, (255, 255, 255), 1)
        cv2.line(court, *self.top_inner_line, (255, 255, 255), 1)
        cv2.line(court, *self.bottom_inner_line, (255, 255, 255), 1)
        cv2.line(court, *self.left_court_line, (255, 255, 255), 1)
        cv2.line(court, *self.right_court_line, (255, 255, 255), 1)
        cv2.line(court, *self.left_inner_line, (255, 255, 255), 1)
        cv2.line(court, *self.right_inner_line, (255, 255, 255), 1)
        cv2.line(court, *self.middle_line, (255, 255, 255), 1)

        overlay = court.copy()
        for zone in zones[0:2]:
            p1, p2, p3, p4 = self.split_zone(zone)
            cv2.rectangle(overlay, (int(zone[0][0]), int(zone[0][1])), p3, color=(240, 255, 255), thickness=-1)
            cv2.rectangle(overlay, p1, p4, color=(0, 255, 0), thickness=-1)
            cv2.rectangle(overlay, p2, (int(zone[3][0]), int(zone[3][1])), color=(191, 0, 230), thickness=-1)

        for zone in zones[2:4]:
            p1, p2, p3, p4 = self.split_zone(zone)
            cv2.rectangle(overlay, (int(zone[0][0]), int(zone[0][1])), p3, color=(191, 0, 230), thickness=-1)
            cv2.rectangle(overlay, p1, p4, color=(0, 255, 0), thickness=-1)
            cv2.rectangle(overlay, p2, (int(zone[3][0]), int(zone[3][1])), color=(240, 255, 255), thickness=-1)

        alpha = 0.4
        image_new = cv2.addWeighted(overlay, alpha, court, 1 - alpha, 0)
        image_new = cv2.dilate(image_new, np.ones((5, 5), dtype=np.uint8))
        lines = self.draw_yellow_lines(image_new)
        return zones, lines

    def split_line_into_3_sections(self, line):
        p1 = line[0]
        p2 = line[1]
        part = (p2[1] - p1[1]) // 3
        return p1[1] + part, p2[1] - part

    def draw_dotted_line(self, line, img):
        start_point = line[0]
        end_point = line[1]
        dot_spacing = 10
        dx = end_point[0] - start_point[0]
        dy = end_point[1] - start_point[1]
        distance = int(np.sqrt(dx * dx + dy * dy))
        num_dots = int(distance / dot_spacing)
        x_increment = float(dx) / num_dots
        y_increment = float(dy) / num_dots
        dot_color = (0, 0, 0)
        for i in range(num_dots):
            x = int(start_point[0] + i * x_increment)
            y = int(start_point[1] + i * y_increment)
            cv2.circle(img, (x, y), 6, dot_color, -1)
        cv2.line(img, start_point, end_point, dot_color, thickness=1)

    def split_zone(self, zone):
        a = zone[0][0]
        b = zone[1][0]
        part = (b - a) // 3
        p1 = (int(a + part), int(zone[0][1]))
        p2 = (int(b - part), int(zone[1][1]))
        p3 = (int(a + part), int(zone[2][1]))
        p4 = (int(b - part), int(zone[3][1]))
        return p1, p2, p3, p4

def line_intersection(line1, line2):
    l1 = Line(line1[0], line1[1])
    l2 = Line(line2[0], line2[1])
    intersection = l1.intersection(l2)
    return intersection[0].coordinates
