# Reproducibility record

This document identifies the exact method configurations represented by the
public repository. Model behavior is defined by the versioned source, configs,
frozen protocol assets, and checkpoint hashes below.

The reported training and evaluation jobs used one NVIDIA RTX A6000 with
48 GiB of GPU memory, an Intel Core i9-13900KF processor, and 62 GiB of system
memory. No multi-GPU training was used.

## Final method configurations

| Method | Sport | Encoder or trunk | Epochs | Batch | Seed | Validation selection |
|---|---|---|---:|---:|---:|---|
| CourtAlign-2S | Tennis | ImageNet ResNet-34 | 50 | 4 | 20260602 | Minimum validation Dice loss |
| CourtAlign-2S | Badminton | ImageNet ResNet-34 | 80 | 2 | 20260602 | Minimum comparable all-class validation Dice loss |
| CourtAlign-E2E | Tennis | Frozen SAM 3 ViT and trainable SAM 3 FPN neck | 60 | 2 | 1337 | Minimum validation lattice reprojection error |
| CourtAlign-E2E | Badminton | Frozen SAM 3 ViT and trainable SAM 3 FPN neck | 80 | 2 | 1337 | Minimum validation lattice reprojection error |

The CourtAlign-E2E reprojection objective applies a Huber penalty directly to
input-pixel errors. Its weight is 0.15, its transition is 1 pixel, and errors
are clamped at 50 pixels for stability. The selected checkpoints are epoch 43
for tennis and epoch 60 for badminton.

The badminton CourtAlign-2S configuration applies the three-phase class-focused
Dice schedule at epochs 1--14, 15, and 16--80. Epoch 15 activates only
`bl_singles` and `br_singles`. The tennis CourtAlign-2S configuration uses
binary full-court supervision.

## Downloaded checkpoint hashes

After placing the checkpoints under `weights/`, their SHA-256 hashes should be:

| Checkpoint | Bytes | SHA-256 |
|---|---:|---|
| `weights/courtalign_2s/tennis/best_model.pth` | 89,963,091 | `616fca950f2fbf3fc7e6e9818148511e2479dbc5fbc7ecedc83888e279afd35b` |
| `weights/courtalign_2s/badminton/best_model.pth` | 89,975,379 | `5bdae8e774cf6d9276791b5a152f2aa9958a76bc905510ba6a9274a174541633` |
| `weights/courtalign_e2e/tennis/best_model.pt` | 1,819,646,593 | `77607584497e3bce018b08cf92dfd2f8a9a8bd76c5c5b580a8e626217fcfc706` |
| `weights/courtalign_e2e/badminton/best_model.pt` | 1,819,674,625 | `c0689a660201d06d0edb8d9a7f365382c9c8491f92a8b4e96dc1bf09ce3615f2` |

The released checkpoints are the complete selected models used for the
reported evaluations. New CourtAlign-E2E training jobs omit the unchanged SAM 3
vision trunk from their saved checkpoints to avoid duplicating frozen
parameters. The checkpoint loader accepts both formats.

## Data and split integrity

The frozen manifest hashes are:

| Artifact | SHA-256 |
|---|---|
| `data/benchmark_gt/official/FREEZE_MANIFEST.json` | `5eff9eaa12dfc67963e871d7a73a49770d8542abe474062e5b149f53ead1c868` |
| `data/benchmark_gt/official/HASHES.json` | `ce2a4af1f97954a619ec94ddf832674d4e700eb686c0e68521093af99e62818b` |
| `data/splits/tennis_fullcourt.csv` | `fe40f8bc69d5e902a98fa33146b8b6fbd3e44bf6ef107c0635e85d45a7b40609` |
| `data/splits/badminton_zones.csv` | `d67bb370418c80fdd5f90b7aa5198c6f511ca8cea8d7cf1f7a4c199e21cd93f3` |
| `data/courtalign_e2e/splits/tennis_groups.csv` | `aac5ac14df6396a6efc4d93ccc973c104440a120196349fcc7c89989297de4d9` |
| `data/courtalign_e2e/supervision/tennis.json` | `e506fefbea31c9ac4295daf8c4e15cf042f1982aa6db6ff693f46355698e560a` |
| `data/courtalign_e2e/supervision/badminton.json` | `76579b82c2bfea345eb137de8d9554d6333d050c910634964fc07e12c0edc303` |

CourtAlign-2S sees 904 tennis training frames, 160 validation frames, and 119
test frames. CourtAlign-E2E uses a rally-group-disjoint tennis training and
validation assignment. Including non-registrable frames, its loaders see 918
training frames, 146 validation frames, and the same 119 test frames. The
corresponding visible and non-registrable counts are 909 and 9 for training,
143 and 3 for validation, and 99 and 20 for testing.

Both methods use the same badminton split. The loaders see 436 training frames,
95 validation frames, and 44 test frames. CourtAlign-E2E includes 125 and 27
non-registrable frames in the training and validation subsets for rejection
supervision. The test set contains 20 non-registrable frames used only for final
evaluation.

No test frame is used for checkpoint or threshold selection.

## Frozen geometric protocol

The geometric ground truth under `data/benchmark_gt/official/` is frozen by a
hash manifest. `scripts/verify_setup.py` verifies those files, the split
manifests, and the expected split sizes. The held-out sets contain:

- tennis: 99 registrable frames and 20 non-registrable frames
- badminton: 24 registrable frames and 20 non-registrable frames

Both methods export the same schema. A valid record contains a 3-by-3
metric-court-to-image homography. A failed or skipped record contains an
explicit status and no homography. The canonical evaluator then computes IoU,
projection error, reprojection error, PCK-H, Line-IoU, and non-court false
registrations from the complete test manifest.

The published comparison table reports PCK-H@5.

## Official commands

The public entry points reproduce the final jobs:

```bash
python scripts/train.py --method courtalign-2s --sport tennis
python scripts/train.py --method courtalign-2s --sport badminton
python scripts/train.py --method courtalign-e2e --sport tennis
python scripts/train.py --method courtalign-e2e --sport badminton
```

```bash
python scripts/evaluate.py --method courtalign-2s --sport tennis
python scripts/evaluate.py --method courtalign-2s --sport badminton
python scripts/evaluate.py --method courtalign-e2e --sport tennis
python scripts/evaluate.py --method courtalign-e2e --sport badminton
```

Use the method-specific environment documented in the main README for each
command. The evaluation command refuses to overwrite a non-empty result folder.

## Portable data interfaces

Dataset images and masks are resolved from the repository-relative paths in
the frozen CSV manifests. CourtAlign-E2E reads its court specifications and
supervision from the versioned files under `data/courtalign_e2e/`. The command-line entry
points map each method and sport to its released configuration, checkpoint, and
the common evaluator.

CourtAlign-2S checkpoints serialize the segmentation model. CourtAlign-E2E
checkpoints store a model state dictionary, the selected epoch, and the
scientific settings represented by `configs/courtalign_e2e/`.

## End-to-end inference record

The final CourtAlign-E2E checkpoints produced complete prediction exports for
both held-out test sets:

| Sport | Frames | Prediction SHA-256 |
|---|---:|---|
| Tennis | 119 | `8b9adf7ac420bf383f7c96305252867fcf98db9520da9e1a55f8462801266ef0` |
| Badminton | 44 | `872de90f88e00e5e30a08f1bb6911659273f529f282abfc198767cdc3558f2d1` |

The canonical evaluator reproduced every CourtAlign-E2E value reported in the
comparison table, including the non-registrable-frame decisions.

## Video application verification

The public video entry point was tested with the four combinations of method
and sport. Two-frame H.264 MP4 inputs at 1280 by 720 pixels were created from
visible held-out frames 286 for tennis and 190 for badminton. Every combination
produced a readable registered MP4, two valid homography records, and a JSON
summary. The rendered output was inspected to confirm that the projected court
lines were correctly scaled and aligned. The updated CourtAlign-E2E release
retains the model architecture, inference path, and checkpoint schema exercised
by these tests.

The optional motion-triggered mode was also tested with CourtAlign-2S on the
tennis clip. It performed one inference call and reused the accepted homography
for the second frame, while retaining valid registration on both frames. These
checks cover checkpoint loading, video decoding, method-specific inference,
homography conversion, output-resolution mapping, line rendering, video
encoding, and sidecar serialization.
