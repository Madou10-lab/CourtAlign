#!/usr/bin/env python3
"""Sanity checks before CourtAlign-E2E training.

Checks: environment, data paths, supervision artifacts, splits, dataset item
geometry, synthetic DLT recovery, final model construction,
one forward+backward step, output-dir writability.

Usage:
  python scripts/courtalign_e2e/sanity_check.py \
    --config configs/courtalign_e2e/tennis.json [--smoke-train]
  --smoke-train additionally checks that the loss decreases over 20 updates.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

OK, FAIL = "  [ok]", "  [FAIL]"
failures = []


def check(name, fn):
    try:
        msg = fn() or ""
        print(f"{OK} {name} {msg}")
    except Exception as e:  # noqa: BLE001
        failures.append((name, str(e)))
        print(f"{FAIL} {name}: {e}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--smoke-train", action="store_true")
    args = ap.parse_args()
    cfg = json.loads(Path(args.config).read_text())
    sport = cfg["sport"]

    def env():
        import torch
        import cv2
        return f"(torch {torch.__version__}, cuda={torch.cuda.is_available()}, cv2 {cv2.__version__})"
    check("environment", env)

    def paths_ok():
        from courtalign_e2e import paths
        dataset = paths.SOURCES[sport]["dataset"]
        manifest = paths.split_manifest(dataset)
        assert manifest.exists(), f"missing {manifest}"
        rows = paths.manifest_rows(dataset)
        assert rows, f"empty manifest {manifest}"
        for row in rows:
            for key in ("image_path", "mask_path"):
                p = paths.data_path(row[key])
                assert p.exists(), f"missing {p}"
        assert paths.official_gt_dir().exists(), "official ground truth missing"
        return f"({len(rows)} manifest rows)"
    check("data paths", paths_ok)

    def supervision():
        from courtalign_e2e import paths
        f = paths.supervision_path(sport)
        assert f.exists(), f"run scripts/courtalign_e2e/build_supervision.py first ({f} missing)"
        d = json.load(open(f))
        from courtalign_e2e.geometry import templates as T
        K = len(T.sport_pack(sport)["lattice"])
        assert d["lattice_K"] == K, f"lattice K mismatch {d['lattice_K']} vs {K}"
        return f"(n={d['n']}, K={K})"
    check("supervision artifact", supervision)

    if sport == "tennis":
        def groups():
            from courtalign_e2e import paths
            f = paths.tennis_group_split()
            assert f.exists(), "run scripts/courtalign_e2e/build_tennis_groups.py first"
            import csv as _csv
            rows = list(_csv.DictReader(open(f)))
            tr = {r["group"] for r in rows if r["split"] == "train"}
            va = {r["group"] for r in rows if r["split"] == "val"}
            assert not (tr & va), "group leakage between train and val!"
            return f"(train_groups={len(tr)}, val_groups={len(va)}, disjoint)"
        check("rally-group split", groups)

    def dlt():
        import torch
        from courtalign_e2e.geometry.homography_torch import weighted_dlt, project_h
        rng = np.random.RandomState(0)
        Hgt = np.array([[1.2, 0.1, 30], [-0.05, 1.4, 60], [1e-4, 2e-4, 1.0]])
        src = rng.uniform(0, 10, size=(1, 24, 2))
        ones = np.ones((1, 24, 1))
        dsth = np.concatenate([src, ones], -1) @ Hgt.T
        dst = dsth[..., :2] / dsth[..., 2:]
        H = weighted_dlt(torch.tensor(src, dtype=torch.float64),
                         torch.tensor(dst, dtype=torch.float64),
                         torch.ones(1, 24, dtype=torch.float64))
        err = float((project_h(H, torch.tensor(src, dtype=torch.float64))
                     - torch.tensor(dst)).norm(dim=-1).max())
        assert err < 1e-6, f"DLT recovery err {err}"
        # gradient flows
        d2 = torch.tensor(dst, dtype=torch.float64, requires_grad=True)
        H2 = weighted_dlt(torch.tensor(src, dtype=torch.float64), d2,
                          torch.ones(1, 24, dtype=torch.float64))
        H2.sum().backward()
        assert d2.grad is not None and float(d2.grad.abs().sum()) > 0
        return f"(recovery err {err:.1e}, grads flow)"
    check("weighted DLT", dlt)

    def dataset():
        from courtalign_e2e.data.dataset import CourtAlignE2EDataset
        from courtalign_e2e.models.factory import resolve_lattice
        from courtalign_e2e.geometry import templates as T
        ih, iw = cfg.get("input_hw", [448, 784])
        pack = T.sport_pack(sport)
        lat = resolve_lattice(cfg, pack)
        ov = lat if cfg.get("lattice_mode") == "zone_centroids" else None
        ds = CourtAlignE2EDataset(sport, "train", (ih, iw), augment=True, seed=0, lattice_metric=ov)
        it = ds[0]
        assert it["image"].shape == (3, ih, iw)
        assert it["seg"].shape == (ih, iw)
        K = it["lattice_uv"].shape[0]
        assert K == len(lat) and it["vis"].shape == (K,)
        # supervision consistency: project lattice via H_sup ~= lattice_uv
        from courtalign_e2e.geometry.homography_np import project
        uv = project(it["H_sup"].numpy().astype(np.float64), lat)
        d = np.linalg.norm(uv - it["lattice_uv"].numpy(), axis=1)
        assert d.max() < 0.5, f"H_sup vs lattice_uv inconsistent ({d.max():.2f}px)"
        return f"(train n={len(ds)}, item ok, K={K}, H~lattice max {d.max():.3f}px)"
    check("dataset geometry", dataset)

    def model():
        import torch
        from courtalign_e2e.geometry import templates as T
        from courtalign_e2e.models.factory import build_model, resolve_lattice
        pack = T.sport_pack(sport)
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        net = build_model(cfg, pack, resolve_lattice(cfg, pack)).to(dev)
        ih, iw = cfg.get("input_hw", [448, 784])           # real input (sam3 needs 1008x1008)
        x = torch.randn(1, 3, ih, iw, device=dev)
        out = net(x)
        assert out["H_pred"].shape == (1, 3, 3)
        n = sum(p.numel() for p in net.trainable_parameters())
        mode = "trainable" if cfg.get("backbone_trainable", False) else "frozen"
        return f"(H_pred ok, trainable {n/1e6:.2f}M, backbone {mode}, model {cfg.get('model', 'courtalign_e2e')})"
    check(f"model build [{cfg['backbone']}]", model)

    def step():
        import torch
        from torch.utils.data import DataLoader
        from courtalign_e2e.data.dataset import CourtAlignE2EDataset
        from courtalign_e2e.geometry import templates as T
        from courtalign_e2e.models.factory import build_model, resolve_lattice
        from courtalign_e2e.models.losses import CourtAlignE2ELoss
        pack = T.sport_pack(sport)
        ih, iw = cfg.get("input_hw", [448, 784])
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        lat = resolve_lattice(cfg, pack)
        ov = lat if cfg.get("lattice_mode") == "zone_centroids" else None
        ds = CourtAlignE2EDataset(sport, "train", (ih, iw), augment=False, lattice_metric=ov)
        dl = DataLoader(ds, batch_size=2, shuffle=False)
        batch = next(iter(dl))
        batch = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in batch.items()}
        net = build_model(cfg, pack, lat).to(dev)
        lf = CourtAlignE2ELoss(cfg, lat, pack["zone_polys"], pack["zone_class_of_poly"],
                    (ih, iw), dev)
        out = net(batch["image"])
        loss, logs = lf(out, batch, pack["n_classes"])
        loss.backward()
        if args.smoke_train:
            opt = torch.optim.AdamW(net.trainable_parameters(), lr=1e-3)
            l0 = float(loss)
            for _ in range(20):
                out = net(batch["image"])
                l, _ = lf(out, batch, pack["n_classes"])
                opt.zero_grad()
                l.backward()
                opt.step()
            l1 = float(l)
            assert l1 < l0, f"smoke-training loss did not decrease ({l0:.3f}->{l1:.3f})"
            return f"(loss {l0:.3f}->{l1:.3f} over 20 steps)"
        return f"(loss {float(loss):.3f}, components {{{', '.join(f'{k}:{v:.3f}' for k, v in logs.items())}}})"
    check("forward+backward", step)

    def outdirs():
        (ROOT / "runs").mkdir(exist_ok=True)
        t = ROOT / "runs" / ".write_test"
        t.write_text("ok")
        t.unlink()
    check("output dirs writable", outdirs)

    print()
    if failures:
        print(f"{len(failures)} CHECK(S) FAILED:")
        for n, e in failures:
            print(f"  - {n}: {e}")
        return 1
    print("ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
