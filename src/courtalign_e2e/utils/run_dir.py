"""Run-directory discipline: every run is self-describing and self-contained."""

from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path


def set_seed(seed: int):
    import random
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def environment_snapshot() -> dict:
    pkgs = {}
    for mod in ("torch", "torchvision", "numpy", "cv2", "scipy", "pandas"):
        try:
            m = __import__(mod)
            pkgs[mod] = getattr(m, "__version__", "?")
        except ImportError:
            pkgs[mod] = "missing"
    snap = {"python": sys.version.split()[0], "platform": platform.platform(), "packages": pkgs}
    try:
        import torch
        snap["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            snap["gpu"] = torch.cuda.get_device_name(0)
    except ImportError:
        pass
    return snap


def init_run_dir(run_dir: Path, config: dict, config_path: str | None, argv: list[str],
                 seed: int) -> Path:
    run_dir = Path(run_dir)
    if (run_dir / "metrics.csv").exists():
        raise FileExistsError(
            f"{run_dir} already contains a run (metrics.csv). Refusing to overwrite — "
            "choose a new experiment_id or delete the folder yourself.")
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (run_dir / "metadata").mkdir(exist_ok=True)
    (run_dir / "metadata" / "config_used.json").write_text(json.dumps(config, indent=2))
    if config_path and Path(config_path).exists():
        shutil.copy2(config_path, run_dir / "metadata" / Path(config_path).name)
    (run_dir / "metadata" / "command.txt").write_text(
        f"{time.strftime('%Y-%m-%d %H:%M:%S')}\ncwd: {os.getcwd()}\n"
        f"cmd: {' '.join(argv)}\nseed: {seed}\n")
    (run_dir / "metadata" / "environment.json").write_text(
        json.dumps(environment_snapshot(), indent=2))
    return run_dir
