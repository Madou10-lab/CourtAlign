#!/usr/bin/env python3
"""Track and project a tennis or badminton court in one video or a directory."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _common import default_checkpoint, method_name, relative_or_absolute
from courtalign.video import build_predictor, process_video


VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}


def collect_videos(path: Path) -> list[Path]:
    if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS:
        return [path]
    if path.is_dir():
        return sorted(item for item in path.rglob("*") if item.suffix.lower() in VIDEO_EXTENSIONS)
    return []


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", required=True, choices=["courtalign-2s", "courtalign-e2e", "2s", "e2e"])
    parser.add_argument("--sport", required=True, choices=["tennis", "badminton"])
    parser.add_argument("--input", required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--output-dir", default="outputs/video_tracking")
    parser.add_argument("--tracking-mode", choices=["every-frame", "motion"], default="every-frame")
    parser.add_argument("--refresh-interval", type=int, default=30)
    parser.add_argument("--motion-threshold", type=float, default=0.08)
    args = parser.parse_args()

    method = method_name(args.method)
    checkpoint = relative_or_absolute(args.checkpoint) if args.checkpoint else default_checkpoint(method, args.sport)
    videos = collect_videos(relative_or_absolute(args.input))
    if not videos:
        raise SystemExit(f"No supported videos found under {args.input}")
    output_dir = relative_or_absolute(args.output_dir)
    predictor = build_predictor(method, args.sport, checkpoint)
    for video in videos:
        output = output_dir / f"{video.stem}_{method}_{args.sport}.mp4"
        summary = process_video(
            predictor,
            video,
            output,
            tracking_mode=args.tracking_mode,
            refresh_interval=args.refresh_interval,
            motion_threshold=args.motion_threshold,
        )
        print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
