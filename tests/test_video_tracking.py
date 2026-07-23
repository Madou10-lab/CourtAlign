from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

from courtalign.video.predictors import CourtAlign2SPredictor, RegistrationPrediction
from courtalign.video.tracking import process_video


class _IdentityPredictor:
    method = "CourtAlign-Test"
    sport = "tennis"

    def predict(self, frame_bgr: np.ndarray) -> RegistrationPrediction:
        metric_to_image = np.array(
            [[10.0, 0.0, 20.0], [0.0, 4.0, 10.0], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        return RegistrationPrediction("valid", metric_to_image)


class _TennisGeometryStub:
    def __init__(self) -> None:
        reference = np.asarray(
            [[0.0, 0.0], [100.0, 0.0], [0.0, 100.0], [100.0, 100.0]],
            dtype=np.float32,
        )
        self.court_reference = SimpleNamespace(court_conf1={1: reference})
        self.ref_to_frame_matrix = np.asarray(
            [[10.0, 0.0, 200.0], [0.0, 5.0, 100.0], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        self.is_visible = True
        self.last_pred_mask = np.zeros((720, 1280), dtype=np.uint8)
        self.input_shape = None

    def reset_run_state(self) -> None:
        return None

    def detect_court(self, frame_bgr: np.ndarray) -> None:
        self.input_shape = frame_bgr.shape[:2]


def _write_video(path: Path, n_frames: int = 3) -> None:
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10.0, (160, 96)
    )
    assert writer.isOpened()
    for index in range(n_frames):
        frame = np.full((96, 160, 3), 30 + index * 20, dtype=np.uint8)
        writer.write(frame)
    writer.release()


def test_video_application_writes_registered_video_and_sidecar(tmp_path: Path):
    source = tmp_path / "input.avi"
    output = tmp_path / "registered.mp4"
    _write_video(source)

    summary = process_video(_IdentityPredictor(), source, output)

    assert output.is_file() and output.stat().st_size > 0
    assert summary["n_frames"] == 3
    assert summary["n_inference_calls"] == 3
    assert summary["n_frames_with_registration"] == 3
    records = [json.loads(line) for line in output.with_suffix(".homographies.jsonl").read_text().splitlines()]
    assert len(records) == 3
    assert all(record["status"] == "valid" for record in records)
    assert all(np.asarray(record["metric_to_image"]).shape == (3, 3) for record in records)


def test_tennis_video_predictor_maps_native_geometry_to_input_resolution():
    predictor = CourtAlign2SPredictor.__new__(CourtAlign2SPredictor)
    predictor.tennis = _TennisGeometryStub()
    predictor.spec = SimpleNamespace(
        outer_corners=np.asarray(
            [[0.0, 0.0], [10.0, 0.0], [0.0, 20.0], [10.0, 20.0]],
            dtype=np.float64,
        )
    )
    frame = np.zeros((360, 640, 3), dtype=np.uint8)

    prediction = predictor._predict_tennis(frame)

    assert prediction.valid
    assert predictor.tennis.input_shape == (1080, 1920)
    projected = cv2.perspectiveTransform(
        predictor.spec.outer_corners.astype(np.float32).reshape(-1, 1, 2),
        prediction.metric_to_image,
    ).reshape(-1, 2)
    expected_native = cv2.perspectiveTransform(
        predictor.tennis.court_reference.court_conf1[1].reshape(-1, 1, 2),
        predictor.tennis.ref_to_frame_matrix,
    ).reshape(-1, 2)
    expected = expected_native * np.asarray([640 / 1920, 360 / 1080])
    assert np.allclose(projected, expected, atol=1e-4)
