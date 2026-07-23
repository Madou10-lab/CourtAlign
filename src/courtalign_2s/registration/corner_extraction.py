from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import cv2
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression


@dataclass(frozen=True)
class CornerExtractionResult:
    corners_tl_tr_bl_br: np.ndarray | None
    status: str
    details: dict


def contour_to_lines(contour):
    return [[contour[i][0][0], contour[i][0][1], contour[i + 1][0][0], contour[i + 1][0][1]] for i in range(len(contour) - 1)] + [
        [contour[-1][0][0], contour[-1][0][1], contour[0][0][0], contour[0][0][1]]
    ]


def classify_lines(lines):
    horizontal = []
    vertical = []
    highest_vertical_y = np.inf
    lowest_vertical_y = 0
    for line in lines:
        x1, y1, x2, y2 = line
        dx = abs(x1 - x2)
        dy = abs(y1 - y2)
        if dx > 2 * dy:
            horizontal.append(line)
        else:
            vertical.append(line)
            highest_vertical_y = min(highest_vertical_y, y1, y2)
            lowest_vertical_y = max(lowest_vertical_y, y1, y2)

    clean_horizontal = []
    h = lowest_vertical_y - highest_vertical_y
    lowest_vertical_y += h / 15
    highest_vertical_y -= h * 2 / 15
    for line in horizontal:
        x1, y1, _, _ = line
        if lowest_vertical_y > y1 > highest_vertical_y:
            clean_horizontal.append(line)
    return clean_horizontal, vertical


def cluster_horizontals(segments, y_threshold=5, x_threshold=10):
    if not segments:
        return [], []
    segments.sort(key=lambda x: (x[1] + x[3]) / 2)
    clusters = {0: [segments[0]]}
    current_cluster = 0

    def are_segments_in_same_cluster(seg1, seg2):
        avg_y1 = (seg1[1] + seg1[3]) / 2
        avg_y2 = (seg2[1] + seg2[3]) / 2
        if abs(avg_y1 - avg_y2) > y_threshold:
            return False
        max_left = max(min(seg1[0], seg1[2]), min(seg2[0], seg2[2]))
        min_right = min(max(seg1[0], seg1[2]), max(seg2[0], seg2[2]))
        if max_left - min_right > x_threshold and min_right - max_left > x_threshold:
            return False
        return True

    for i in range(1, len(segments)):
        if are_segments_in_same_cluster(segments[i], segments[i - 1]):
            clusters[current_cluster].append(segments[i])
        else:
            current_cluster += 1
            clusters[current_cluster] = [segments[i]]

    sorted_clusters = sorted(clusters.values(), key=len, reverse=True)[:2]
    cluster1 = sorted_clusters[0] if sorted_clusters else []
    cluster2 = sorted_clusters[1] if len(sorted_clusters) > 1 else []
    return cluster1, cluster2


def are_connected(seg1, seg2):
    return seg1[2:] == seg2[:2] or seg1[2:] == seg2[2:]


def cluster_verticals(segments):
    clusters = []
    used_segments = []
    for segment in segments:
        if segment not in used_segments:
            current_cluster = [segment]
            used_segments.append(segment)
            added = True
            while added:
                added = False
                for seg in segments:
                    if seg not in used_segments and are_connected(current_cluster[-1], seg):
                        current_cluster.append(seg)
                        used_segments.append(seg)
                        added = True
                        break
            clusters.append(current_cluster)
    return clusters


def fit_cluster(cluster):
    if not cluster:
        return []
    X = []
    Y = []
    for segment in cluster:
        midpoint_x = (segment[0] + segment[2]) / 2
        midpoint_y = (segment[1] + segment[3]) / 2
        X.append([midpoint_x])
        Y.append(midpoint_y)
    reg = LinearRegression().fit(X, Y)
    slope = reg.coef_[0]
    intercept = reg.intercept_
    min_x = min(seg[i] for seg in cluster for i in [0, 2])
    max_x = max(seg[i] for seg in cluster for i in [0, 2])
    y1 = slope * min_x + intercept
    y2 = slope * max_x + intercept
    return [min_x, y1, max_x, y2]


def fit_vertical_line(cluster):
    if not cluster:
        return []
    if len(cluster) == 1:
        return cluster[0]
    midpoints = np.array([((seg[0] + seg[2]) / 2, (seg[1] + seg[3]) / 2) for seg in cluster])
    pca = PCA(n_components=2)
    pca.fit(midpoints)
    direction = pca.components_[0]
    mean = pca.mean_
    if abs(direction[0]) > abs(direction[1]):
        min_x = min(midpoints[:, 0])
        max_x = max(midpoints[:, 0])
        y1 = mean[1] - (mean[0] - min_x) * (direction[1] / direction[0])
        y2 = mean[1] + (max_x - mean[0]) * (direction[1] / direction[0])
        return [min_x, y1, max_x, y2]
    min_y = min(midpoints[:, 1])
    max_y = max(midpoints[:, 1])
    x1 = mean[0] - (mean[1] - min_y) * (direction[0] / direction[1])
    x2 = mean[0] + (max_y - mean[1]) * (direction[0] / direction[1])
    return [x1, min_y, x2, max_y]


def line_intersection(line1, line2):
    x1, y1, x2, y2 = line1
    x3, y3, x4, y4 = line2
    det_line1 = x1 * y2 - y1 * x2
    det_line2 = x3 * y4 - y3 * x4
    x1_x2_diff = x1 - x2
    y1_y2_diff = y1 - y2
    x3_x4_diff = x3 - x4
    y3_y4_diff = y3 - y4
    det_divisor = x1_x2_diff * y3_y4_diff - y1_y2_diff * x3_x4_diff
    if det_divisor == 0:
        return None
    x = (det_line1 * x3_x4_diff - x1_x2_diff * det_line2) / det_divisor
    y = (det_line1 * y3_y4_diff - y1_y2_diff * det_line2) / det_divisor
    return (x, y)


def sort_intersection_points(intersections: Iterable[tuple[float, float]]):
    y_sorted = sorted(intersections, key=lambda x: x[1])
    p12 = y_sorted[:2]
    p34 = y_sorted[2:]
    p12 = sorted(p12, key=lambda x: x[0])
    p34 = sorted(p34, key=lambda x: x[0])
    return p12 + p34


def extract_corners_from_binary_mask(
    binary_mask: np.ndarray,
    *,
    frame_shape_hw: tuple[int, int],
    x_scale: float = 1.0,
    y_scale: float = 1.0,
    closing_iterations: int = 2,
    erosion_iterations: int = 1,
    min_area_fraction: float | None = None,
    min_area_px: float | None = None,
) -> CornerExtractionResult:
    pred_zone_mask = binary_mask.astype(np.uint8)
    kernel = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]], np.uint8)
    dilated = cv2.dilate(pred_zone_mask, kernel, iterations=closing_iterations)
    pred_zone_mask = cv2.erode(dilated, kernel, iterations=closing_iterations + erosion_iterations)
    smoothed = cv2.GaussianBlur(pred_zone_mask, (5, 5), 0)
    contours, _ = cv2.findContours(smoothed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return CornerExtractionResult(None, "missing_contour", {"n_contours": 0})

    largest_contour = max(contours, key=cv2.contourArea)
    largest_contour = cv2.convexHull(largest_contour)
    area_px = float(cv2.contourArea(largest_contour))
    frame_area = float(frame_shape_hw[0] * frame_shape_hw[1])
    area_fraction = area_px / frame_area if frame_area > 0 else 0.0
    if min_area_fraction is not None and area_fraction < min_area_fraction:
        return CornerExtractionResult(
            None,
            "area_fraction_too_small",
            {"area_px": area_px, "area_fraction": area_fraction, "min_area_fraction": min_area_fraction},
        )
    if min_area_px is not None and area_px < min_area_px:
        return CornerExtractionResult(
            None,
            "area_px_too_small",
            {"area_px": area_px, "area_fraction": area_fraction, "min_area_px": min_area_px},
        )

    lines = contour_to_lines(largest_contour)
    clean_horizontal, vertical = classify_lines(lines)
    cluster1, cluster2 = cluster_horizontals(clean_horizontal)
    vertical_clusters = sorted(cluster_verticals(vertical), key=len, reverse=True)
    vcluster1 = vertical_clusters[0] if len(vertical_clusters) > 0 else []
    vcluster2 = vertical_clusters[1] if len(vertical_clusters) > 1 else []
    line1 = fit_cluster(cluster1)
    line2 = fit_cluster(cluster2)
    line3 = fit_vertical_line(vcluster1)
    line4 = fit_vertical_line(vcluster2)

    details = {
        "area_px": area_px,
        "area_fraction": area_fraction,
        "n_contours": len(contours),
        "n_lines": len(lines),
        "n_clean_horizontal": len(clean_horizontal),
        "n_vertical": len(vertical),
        "horizontal_cluster_sizes": [len(cluster1), len(cluster2)],
        "vertical_cluster_sizes": [len(vcluster1), len(vcluster2)],
    }
    if any(x == [] for x in (line1, line2, line3, line4)):
        return CornerExtractionResult(None, "insufficient_fitted_lines", details)

    intersection_points = [
        line_intersection(line1, line3),
        line_intersection(line1, line4),
        line_intersection(line2, line3),
        line_intersection(line2, line4),
    ]
    if any(p is None for p in intersection_points):
        return CornerExtractionResult(None, "parallel_or_invalid_lines", details)

    pts = sort_intersection_points(
        [
            (int(x * x_scale), int(y * y_scale))
            for x, y in np.squeeze(intersection_points)
            if x is not None and y is not None
        ]
    )
    if len(pts) != 4:
        return CornerExtractionResult(None, "invalid_corner_count", details | {"n_points": len(pts)})
    return CornerExtractionResult(np.asarray(pts, dtype=np.float64), "ok", details)
