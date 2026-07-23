from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[3]


@dataclass
class RegistrationPrediction:
    status: str
    metric_to_image: np.ndarray | None
    reason: str | None = None
    diagnostics: dict = field(default_factory=dict)

    @property
    def valid(self) -> bool:
        return self.status == "valid" and self.metric_to_image is not None


class RegistrationPredictor(Protocol):
    method: str
    sport: str

    def predict(self, frame_bgr: np.ndarray) -> RegistrationPrediction: ...


class CourtAlign2SPredictor:
    method = "CourtAlign-2S"

    def __init__(self, sport: str, checkpoint: Path):
        import albumentations as album
        import segmentation_models_pytorch as smp
        import torch

        from courtalign_common.evaluation.official_protocol import load_official_spec
        from courtalign_2s.segmentation import dataset_utils as du
        from courtalign_2s.segmentation import model_utils as mu

        self.sport = sport
        self.checkpoint = checkpoint
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dataset = "tennis_fullcourt" if sport == "tennis" else "badminton_zones"
        self.spec = load_official_spec(ROOT / "data/benchmark_gt/official/court_spec_tables.json", dataset)
        if sport == "tennis":
            from courtalign_2s.registration import CourtAlign2STennisFullCourtRegistration

            self.tennis = CourtAlign2STennisFullCourtRegistration(
                weight_path=checkpoint,
                asset_dir=ROOT / "assets/courtalign_2s/tennis/templates",
            )
            self.tennis.load_best_model()
            self.model = None
            return

        self.tennis = None
        self.model = torch.load(checkpoint, map_location=self.device)
        self.model.to(self.device).eval()
        self.preprocessing = mu.get_preprocessing(
            smp.encoders.get_preprocessing_fn("resnet34", "imagenet")
        )
        self.transform = album.Compose([
            album.Resize(height=720, width=1280, always_apply=True),
            du.ToHSV(always_apply=True),
        ])

    def _predict_tennis(self, frame_bgr: np.ndarray) -> RegistrationPrediction:
        from courtalign_common.evaluation.metric_homography import homography_from_corners

        frame_height, frame_width = frame_bgr.shape[:2]
        native_width, native_height = 1920, 1080
        native_frame = cv2.resize(
            frame_bgr,
            (native_width, native_height),
            interpolation=cv2.INTER_LINEAR,
        )
        self.tennis.reset_run_state()
        self.tennis.detect_court(native_frame)
        if not self.tennis.is_visible or self.tennis.ref_to_frame_matrix is None:
            return RegistrationPrediction("skipped", None, "court_geometry_not_found")
        reference = np.asarray(self.tennis.court_reference.court_conf1[1], dtype=np.float32)
        native_corners = cv2.perspectiveTransform(
            reference.reshape(-1, 1, 2), self.tennis.ref_to_frame_matrix
        ).reshape(-1, 2)
        native_to_frame = np.asarray(
            [[frame_width / native_width, 0.0, 0.0],
             [0.0, frame_height / native_height, 0.0],
             [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        image_corners = cv2.perspectiveTransform(
            native_corners.astype(np.float32).reshape(-1, 1, 2),
            native_to_frame,
        ).reshape(-1, 2)
        metric_to_image = homography_from_corners(self.spec.outer_corners, image_corners)
        return RegistrationPrediction(
            "valid",
            metric_to_image,
            diagnostics={"semantic_mask_shape": list(self.tennis.last_pred_mask.shape)},
        )

    def _predict_badminton(self, frame_bgr: np.ndarray) -> RegistrationPrediction:
        import torch

        from courtalign_common.evaluation.official_protocol import estimate_badminton_homography
        from courtalign_2s.segmentation import utils as courtalign_2s_utils

        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        transformed = self.transform(image=rgb)["image"]
        transformed = self.preprocessing(image=transformed)["image"]
        tensor = torch.from_numpy(transformed).to(self.device).unsqueeze(0)
        with torch.inference_mode():
            logits = self.model(tensor)
        mask = courtalign_2s_utils.transpose_reverse_one_hot(
            logits.detach().squeeze().cpu().numpy()
        ).astype(np.uint8)
        height, width = frame_bgr.shape[:2]
        result = estimate_badminton_homography(
            mask,
            self.spec,
            image_width=width,
            image_height=height,
            ransac_reprojection_threshold_px=8.0,
            min_classes=8,
            min_inlier_ratio=0.75,
            min_zone_miou=0.88,
            min_fullcourt_iou=0.94,
        )
        matrix = np.asarray(result["metric_to_image"], dtype=np.float64) if result["status"] == "valid" else None
        status = "failed" if result["status"] == "rejected" else result["status"]
        return RegistrationPrediction(status, matrix, result["skipped_reason"], result.get("validation", {}))

    def predict(self, frame_bgr: np.ndarray) -> RegistrationPrediction:
        return self._predict_tennis(frame_bgr) if self.sport == "tennis" else self._predict_badminton(frame_bgr)


class CourtAlignE2EPredictor:
    method = "CourtAlign-E2E"

    def __init__(self, sport: str, checkpoint: Path):
        import torch

        from courtalign_e2e.geometry import templates
        from courtalign_e2e.models.checkpoint import load_checkpoint_state
        from courtalign_e2e.models.factory import build_model, resolve_lattice

        self.sport = sport
        self.checkpoint = checkpoint
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        payload = torch.load(checkpoint, map_location="cpu")
        self.config = payload["config"]
        if self.config["sport"] != sport:
            raise ValueError(f"Checkpoint sport is {self.config['sport']!r}, not {sport!r}")
        self.pack = templates.sport_pack(sport)
        self.model = build_model(self.config, self.pack, resolve_lattice(self.config, self.pack)).to(self.device)
        load_checkpoint_state(self.model, payload["model"])
        self.model.eval()
        self.ih, self.iw = self.config["input_hw"]

    def predict(self, frame_bgr: np.ndarray) -> RegistrationPrediction:
        import torch

        from courtalign_e2e.data.dataset import IMAGENET_MEAN, IMAGENET_STD

        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (self.iw, self.ih), interpolation=cv2.INTER_LINEAR)
        array = (resized.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
        tensor = torch.from_numpy(array.transpose(2, 0, 1)).unsqueeze(0).to(self.device)
        with torch.inference_mode():
            output = self.model(tensor)

        segmentation = output["seg_logits"].argmax(1)[0].cpu().numpy()
        court_fraction = float((segmentation > 0).mean())
        coordinates = output["coords"][0].cpu().numpy()
        confidence = output["conf"][0].cpu().numpy()
        spread = float(coordinates.std(axis=0).mean()) / float((self.ih**2 + self.iw**2) ** 0.5)
        diagnostics = {
            "court_area_fraction": court_fraction,
            "mean_correspondence_confidence": float(confidence.mean()),
            "mean_landmark_spread": spread,
        }
        if court_fraction < float(self.config["no_court_area_frac"]):
            return RegistrationPrediction("skipped", None, "no_court_detected", diagnostics)
        if spread < float(self.config["min_spread_frac"]) or float(confidence.mean()) < float(self.config["min_conf_mean"]):
            return RegistrationPrediction("failed", None, "degenerate_geometry", diagnostics)

        height, width = frame_bgr.shape[:2]
        input_to_frame = np.diag([width / self.iw, height / self.ih, 1.0])
        metric_to_image = input_to_frame @ output["H_pred"][0].cpu().numpy().astype(np.float64)
        return RegistrationPrediction("valid", metric_to_image, diagnostics=diagnostics)


def build_predictor(method: str, sport: str, checkpoint: str | Path) -> RegistrationPredictor:
    checkpoint = Path(checkpoint).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Missing checkpoint: {checkpoint}")
    if method == "courtalign-2s":
        return CourtAlign2SPredictor(sport, checkpoint)
    if method == "courtalign-e2e":
        return CourtAlignE2EPredictor(sport, checkpoint)
    raise ValueError(f"Unknown method: {method}")
