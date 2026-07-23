from __future__ import annotations

import logging
import os
from time import perf_counter
from pathlib import Path

import albumentations as album
import cv2
import numpy as np
import segmentation_models_pytorch as smp
import torch

from courtalign_2s.registration.court_reference import CourtReference
from courtalign_2s.registration.corner_extraction import (
    extract_corners_from_binary_mask,
    sort_intersection_points,
)
from courtalign_2s.segmentation import dataset_utils as du
from courtalign_2s.segmentation import model_utils as mu
from courtalign_2s.segmentation import utils
from courtalign_common.utils.paths import resolve_repo_path

logger = logging.getLogger(__name__)


class CourtAlign2STennisFullCourtRegistration:
    """CourtAlign-2S tennis mask-to-homography registration."""

    def __init__(
        self,
        weight_path: str | Path | None = None,
        asset_dir: str | Path | None = None,
        width: int = 1280,
        height: int = 720,
        collapse_foreground_to_fullcourt: bool = False,
    ) -> None:
        self.checkpoint_path = resolve_repo_path(weight_path or "weights/courtalign_2s/tennis/best_model.pth")
        self.asset_dir = resolve_repo_path(asset_dir or "assets/courtalign_2s/tennis/templates")
        self.court_reference = CourtReference(self.asset_dir)
        self.width = width
        self.height = height
        self.collapse_foreground_to_fullcourt = bool(collapse_foreground_to_fullcourt)
        self.ref_to_frame_matrix = None
        self.frame_to_ref_matrix = None
        self.yellow_lines = []
        self.x_scale = 1920 / 1280
        self.y_scale = 1080 / 720
        self.closing_iterations = 2
        self.erosion_iterations = 1
        self.harris_block_size = 9
        self.harris_ksize = 3
        self.harris_k = 0.05
        self.harris_thresh = 0.15
        self.subpix_winsize = 3
        self.distance_thresh = 7
        self.area_thresh = 1
        self.polygon_thresh = 15
        self.angle_thresh = 35
        self.angle_thresh_i = 15
        self.angle_thresh_7 = 15
        self.dilation_iterations = 2
        self.n_classes = 2
        self.opacity = 0.7
        self.points = {}
        self.colour_palette = [[0, 0, 0], [255, 255, 255]]
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.encoder = "resnet34"
        self.encoder_weights = "imagenet"
        self.preprocessing_fn = mu.get_preprocessing(
            smp.encoders.get_preprocessing_fn(self.encoder, self.encoder_weights)
        )
        self.is_visible = False
        self.is_loaded = False
        self.last_pred_mask = None
        self.last_semantic_mask = None
        test_transform = [album.Resize(height=self.height, width=self.width, always_apply=True)]
        test_transform.append(du.ToHSV(always_apply=True))
        self.test_augmentation = album.Compose(test_transform)
        self.court_conf = {}

    def reset_run_state(self) -> None:
        self.is_visible = False
        self.ref_to_frame_matrix = None
        self.frame_to_ref_matrix = None
        self.points = {}
        self.court_conf = {}
        self.yellow_lines = []
        self.last_pred_mask = None
        self.last_semantic_mask = None

    def load_best_model(self) -> None:
        if self.is_loaded:
            return
        if not self.checkpoint_path.is_file():
            raise FileNotFoundError(f"Missing CourtAlign-2S checkpoint: {self.checkpoint_path}")
        self.inference_model = torch.load(self.checkpoint_path, map_location=self.device)
        self.inference_model.eval()
        self.is_loaded = True

    def post_processing(self, tensor):
        return utils.transpose_reverse_one_hot(tensor.detach().squeeze().cpu().numpy())

    def inference(self, image_path, preprocessing=True, checkpoint_filename=None):
        total_start = perf_counter()
        image = cv2.imread(str(image_path))
        self.frame = image
        base_name = os.path.basename(str(image_path))
        file_name, _ = os.path.splitext(base_name)
        read_done = perf_counter()
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (self.width, self.height))

        load_start = perf_counter()
        self.load_best_model()
        load_done = perf_counter()
        preprocess_start = perf_counter()
        if preprocessing:
            image = self.preprocessing_fn(image=self.test_augmentation(image=image)["image"])["image"]
        preprocess_done = perf_counter()
        x_tensor = torch.from_numpy(image).to(self.device).unsqueeze(0)
        forward_start = perf_counter()
        predicted_mask = self.inference_model(x_tensor)
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        forward_done = perf_counter()
        predicted_mask = self.post_processing(predicted_mask)
        self.last_semantic_mask = predicted_mask.astype(np.uint8, copy=True)
        if self.collapse_foreground_to_fullcourt:
            predicted_mask = (predicted_mask > 0).astype(np.uint8)
        self.last_pred_mask = predicted_mask.astype(np.uint8, copy=True)
        postprocess_done = perf_counter()
        preds = self.homography(predicted_mask, base_name)
        homography_done = perf_counter()
        self.transform_points_using_homography()
        transform_done = perf_counter()
        if self.is_visible:
            court = self.add_court_overlay(self.frame)
        else:
            court = self.frame
        total_done = perf_counter()
        self.last_timings = {
            "read_sec": read_done - total_start,
            "load_model_sec": load_done - load_start,
            "preprocess_sec": preprocess_done - preprocess_start,
            "forward_sec": forward_done - forward_start,
            "postprocess_sec": postprocess_done - forward_done,
            "homography_sec": homography_done - postprocess_done,
            "transform_sec": transform_done - homography_done,
            "overlay_sec": total_done - transform_done,
            "total_sec": total_done - total_start,
        }
        return court, file_name, preds

    def detect_court(self, image):
        try:
            self.frame = image.copy()
            img_for_model = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            img_for_model = cv2.resize(img_for_model, (self.width, self.height))
            self.load_best_model()
            processed_img = self.preprocessing_fn(image=self.test_augmentation(image=img_for_model)["image"])["image"]
            x_tensor = torch.from_numpy(processed_img).to(self.device).unsqueeze(0)
            pred_mask = self.inference_model(x_tensor)
            pred_mask = self.post_processing(pred_mask)
            self.last_semantic_mask = pred_mask.astype(np.uint8, copy=True)
            if self.collapse_foreground_to_fullcourt:
                pred_mask = (pred_mask > 0).astype(np.uint8)
            self.last_pred_mask = pred_mask.astype(np.uint8, copy=True)
            self.homography(pred_mask, "test.png")
            self.transform_points_using_homography()
            if self.is_visible:
                return self.get_overlay_full_size_img(image)
            return self.frame
        except Exception as exc:
            logger.exception("Court detection failed: %s", exc)
            self.is_visible = False
            return image

    def add_court_overlay(self, frame):
        return cv2.warpPerspective(self.court_reference.court, self.ref_to_frame_matrix, frame.shape[1::-1])

    def homography(self, predicted_mask, base_name):
        prediction_points = []
        reference_points = []

        for i in range(1, self.n_classes):
            pred_zone_mask = (predicted_mask == i).astype(np.uint8)
            extraction = extract_corners_from_binary_mask(
                pred_zone_mask,
                frame_shape_hw=self.frame.shape[:2],
                x_scale=self.x_scale,
                y_scale=self.y_scale,
                closing_iterations=self.closing_iterations,
                erosion_iterations=self.erosion_iterations,
                min_area_fraction=0.05,
            )
            if extraction.corners_tl_tr_bl_br is not None:
                prediction_points.extend(extraction.corners_tl_tr_bl_br.tolist())
                reference_points.extend(sort_intersection_points(self.court_reference.court_conf1[i]))

        try:
            if len(prediction_points) < 4:
                raise ValueError(f"Not enough valid points found ({len(prediction_points)})")
            best_homography, _ = cv2.findHomography(
                np.array(reference_points, dtype=float),
                np.array(prediction_points, dtype=float),
                cv2.RANSAC,
                ransacReprojThreshold=35,
                maxIters=2000,
            )
            if best_homography is None:
                raise ValueError("findHomography returned None")
            retval, inverted_matrix = cv2.invert(best_homography)
            if not retval or inverted_matrix is None:
                raise ValueError("Failed to invert the homography matrix")
            self.is_visible = True
            self.ref_to_frame_matrix = best_homography
            self.frame_to_ref_matrix = inverted_matrix
        except Exception as exc:
            logger.debug("Homography calculation failed: %s", exc)
            self.is_visible = False
            self.ref_to_frame_matrix = None
            self.frame_to_ref_matrix = None

    def transform_points_using_homography(self):
        if not self.is_visible:
            text = "Court not detected"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_size = 2
            font_thickness = 3
            font_color = (0, 0, 255)
            (text_width, text_height), _ = cv2.getTextSize(text, font, font_size, font_thickness)
            start_x = (self.frame.shape[1] - text_width) // 2
            start_y = (self.frame.shape[0] + text_height) // 2
            cv2.putText(self.frame, text, (start_x, start_y), font, font_size, font_color, font_thickness)
            return 1

        for conf_id, ref_points in self.court_reference.court_conf2.items():
            src_points = np.float32(ref_points).reshape(-1, 1, 2)
            dst_points = cv2.perspectiveTransform(src_points, self.ref_to_frame_matrix)
            reshaped_dst_points = dst_points.reshape(-1, 2).tolist()
            self.court_conf[conf_id] = sort_intersection_points([tuple(point) for point in reshaped_dst_points])

        if not self.is_visible:
            self.court_conf = {}

    def draw_yellow_lines(self, lines, img):
        for line in lines:
            points = np.array([line[0], line[1]], dtype=np.float32).reshape((-1, 1, 2))
            start_point, end_point = cv2.perspectiveTransform(points, self.ref_to_frame_matrix).squeeze().round().tolist()
            start_point = tuple(map(int, start_point))
            end_point = tuple(map(int, end_point))
            draw_dotted_line((start_point, end_point), img)
            self.yellow_lines.append((start_point, end_point))

    def segment_zones(self, frame):
        overlay = frame.copy()
        zones, lines = self.court_reference.segment_court_zones()
        self.zones = zones

        for zone in zones[0:2]:
            p1, p2, p3, p4 = split_zone(zone)
            points = np.array(
                [(int(zone[0][0]), int(zone[0][1])), p1, (int(zone[2][0]), int(zone[2][1])), p3],
                dtype=np.float32,
            ).reshape((-1, 1, 2))
            first, second, third, fourth = cv2.perspectiveTransform(points, self.ref_to_frame_matrix).squeeze().round().tolist()
            l = np.intp([third, fourth, second, first])
            cv2.drawContours(overlay, [l], 0, (248, 248, 121), thickness=cv2.FILLED)

            points = np.array([p1, p2, p3, p4], dtype=np.float32).reshape((-1, 1, 2))
            first, second, third, fourth = cv2.perspectiveTransform(points, self.ref_to_frame_matrix).squeeze().round().tolist()
            l = np.intp([third, fourth, second, first])
            cv2.drawContours(overlay, [l], 0, (0, 255, 0), thickness=cv2.FILLED)

            points = np.array(
                [p2, (int(zone[1][0]), int(zone[1][1])), p4, (int(zone[3][0]), int(zone[3][1]))],
                dtype=np.float32,
            ).reshape((-1, 1, 2))
            first, second, third, fourth = cv2.perspectiveTransform(points, self.ref_to_frame_matrix).squeeze().round().tolist()
            l = np.intp([third, fourth, second, first])
            cv2.drawContours(overlay, [l], 0, (191, 0, 230), thickness=cv2.FILLED)

        for zone in zones[2:4]:
            p1, p2, p3, p4 = split_zone(zone)
            points = np.array(
                [(int(zone[0][0]), int(zone[0][1])), p1, (int(zone[2][0]), int(zone[2][1])), p3],
                dtype=np.float32,
            ).reshape((-1, 1, 2))
            first, second, third, fourth = cv2.perspectiveTransform(points, self.ref_to_frame_matrix).squeeze().round().tolist()
            l = np.intp([third, fourth, second, first])
            cv2.drawContours(overlay, [l], -1, (191, 0, 230), thickness=cv2.FILLED)

            points = np.array([p1, p2, p3, p4], dtype=np.float32).reshape((-1, 1, 2))
            first, second, third, fourth = cv2.perspectiveTransform(points, self.ref_to_frame_matrix).squeeze().round().tolist()
            l = np.intp([third, fourth, second, first])
            cv2.drawContours(overlay, [l], -1, (0, 255, 0), thickness=cv2.FILLED)

            points = np.array(
                [p2, (int(zone[1][0]), int(zone[1][1])), p4, (int(zone[3][0]), int(zone[3][1]))],
                dtype=np.float32,
            ).reshape((-1, 1, 2))
            first, second, third, fourth = cv2.perspectiveTransform(points, self.ref_to_frame_matrix).squeeze().round().tolist()
            l = np.intp([third, fourth, second, first])
            cv2.drawContours(overlay, [l], -1, (248, 248, 121), thickness=cv2.FILLED)

        alpha = 0.2
        frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
        self.draw_yellow_lines(lines, frame)
        return frame

    def get_overlay_full_size_img(self, img):
        if not self.is_visible:
            img = self.frame.copy()
            return img
        self.frame = img.copy()
        img = self.segment_zones(img)
        lines = cv2.perspectiveTransform(
            np.array(self.court_reference.get_important_lines(), dtype=np.float32).reshape((-1, 1, 2)),
            self.ref_to_frame_matrix,
        ).reshape(-1)

        for i in range(0, len(lines), 4):
            x1, y1, x2, y2 = lines[i], lines[i + 1], lines[i + 2], lines[i + 3]
            cv2.line(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 0), 2)

        for i in range(0, len(lines), 4):
            x1, y1, x2, y2 = lines[i], lines[i + 1], lines[i + 2], lines[i + 3]
            cv2.circle(img, (int(x1), int(y1)), 1, (0, 255, 0), 6)
            cv2.circle(img, (int(x2), int(y2)), 1, (0, 255, 0), 6)
        return img


def split_zone(zone):
    a = zone[0][0]
    b = zone[1][0]
    part = (b - a) // 3
    p1 = (int(a + part), int(zone[0][1]))
    p2 = (int(b - part), int(zone[1][1]))
    p3 = (int(a + part), int(zone[2][1]))
    p4 = (int(b - part), int(zone[3][1]))
    return p1, p2, p3, p4


def draw_dotted_line(line, img):
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
