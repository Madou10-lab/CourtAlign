from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

METHOD_ALIASES = {
    "2s": "courtalign-2s",
    "courtalign-2s": "courtalign-2s",
    "e2e": "courtalign-e2e",
    "courtalign-e2e": "courtalign-e2e",
}


def method_name(value: str) -> str:
    try:
        return METHOD_ALIASES[value.lower()]
    except KeyError as exc:
        raise ValueError(f"Unknown method {value!r}. Use courtalign-2s or courtalign-e2e.") from exc


def dataset_id(sport: str) -> str:
    return "tennis_fullcourt" if sport == "tennis" else "badminton_zones"


def default_config(method: str, sport: str) -> Path:
    folder = "courtalign_2s" if method == "courtalign-2s" else "courtalign_e2e"
    return ROOT / "configs" / folder / f"{sport}.json"


def default_checkpoint(method: str, sport: str) -> Path:
    suffix = ".pth" if method == "courtalign-2s" else ".pt"
    folder = "courtalign_2s" if method == "courtalign-2s" else "courtalign_e2e"
    return ROOT / "weights" / folder / sport / f"best_model{suffix}"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def run(command: list[str]) -> None:
    printable = " ".join(str(item) for item in command)
    print(f"$ {printable}", flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def require_new_output(path: Path) -> None:
    if path.exists() and any(path.iterdir()):
        raise SystemExit(f"Refusing to overwrite non-empty output directory: {path}")
    path.mkdir(parents=True, exist_ok=True)


def relative_or_absolute(path: str | Path) -> Path:
    path = Path(path).expanduser()
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


PYTHON = sys.executable
