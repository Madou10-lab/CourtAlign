from __future__ import annotations

import json
import time
from pathlib import Path

import cv2
import numpy as np

from .geometry import official_line_segments, project_points
from .predictors import RegistrationPrediction, RegistrationPredictor


def draw_registration(frame: np.ndarray, prediction: RegistrationPrediction, sport: str) -> np.ndarray:
    output = frame.copy()
    if not prediction.valid:
        cv2.putText(output, "No registration", (24, 46), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2, cv2.LINE_AA)
        return output
    thickness = max(2, int(round(frame.shape[1] / 640)))
    for start, end in official_line_segments(sport):
        projected = project_points(np.asarray([start, end], dtype=np.float64), prediction.metric_to_image)
        if np.isfinite(projected).all():
            a, b = np.round(projected).astype(int)
            cv2.line(output, tuple(a), tuple(b), (0, 255, 255), thickness, cv2.LINE_AA)
    return output


def foreground_fraction(subtractor, kernel: np.ndarray, frame: np.ndarray) -> float:
    mask = subtractor.apply(frame)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return float((mask > 0).mean())


def process_video(
    predictor: RegistrationPredictor,
    video_path: Path,
    output_path: Path,
    *,
    tracking_mode: str = "every-frame",
    refresh_interval: int = 30,
    motion_threshold: float = 0.08,
) -> dict:
    related_outputs = [
        output_path,
        output_path.with_suffix(".homographies.jsonl"),
        output_path.with_suffix(".summary.json"),
    ]
    existing = [path for path in related_outputs if path.exists()]
    if existing:
        raise FileExistsError(
            "Refusing to overwrite existing video-tracking output: "
            + ", ".join(str(path) for path in existing)
        )
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise OSError(f"Could not open video: {video_path}")
    fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        capture.release()
        raise OSError(f"Could not create output video: {output_path}")

    subtractor = cv2.createBackgroundSubtractorMOG2()
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    sidecar = output_path.with_suffix(".homographies.jsonl")
    frame_index = 0
    inference_count = 0
    accepted_count = 0
    last_prediction: RegistrationPrediction | None = None
    started = time.perf_counter()

    with sidecar.open("w") as records:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            motion = foreground_fraction(subtractor, kernel, frame)
            refresh = last_prediction is None or frame_index % max(1, refresh_interval) == 0
            run_inference = tracking_mode == "every-frame" or refresh or motion >= motion_threshold
            if run_inference:
                last_prediction = predictor.predict(frame)
                inference_count += 1
            prediction = last_prediction or RegistrationPrediction("skipped", None, "not_initialized")
            accepted_count += int(prediction.valid)
            writer.write(draw_registration(frame, prediction, predictor.sport))
            records.write(json.dumps({
                "frame_index": frame_index,
                "time_sec": frame_index / fps,
                "method": predictor.method,
                "sport": predictor.sport,
                "status": prediction.status,
                "reason": prediction.reason,
                "inference_executed": run_inference,
                "foreground_fraction": motion,
                "metric_to_image": prediction.metric_to_image.tolist() if prediction.valid else None,
                "diagnostics": prediction.diagnostics,
            }) + "\n")
            frame_index += 1

    capture.release()
    writer.release()
    summary = {
        "input": str(video_path),
        "output": str(output_path),
        "homographies": str(sidecar),
        "method": predictor.method,
        "sport": predictor.sport,
        "tracking_mode": tracking_mode,
        "n_frames": frame_index,
        "n_inference_calls": inference_count,
        "n_frames_with_registration": accepted_count,
        "elapsed_sec": time.perf_counter() - started,
    }
    output_path.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2))
    return summary
