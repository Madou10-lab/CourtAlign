# CourtAlign: A Deep Learning-Based Framework for Racket Sports Court Registration

CourtAlign provides two methods for monocular tennis and badminton court
registration. Both methods estimate a homography from the metric court model to
the input image and are evaluated with the same frozen test protocol.

- **CourtAlign-2S** combines semantic court segmentation with classical
  correspondence extraction, RANSAC, and DLT. The final model uses a
  DeepLabV3+ decoder and an ImageNet-pretrained ResNet-34 encoder. Tennis uses a
  binary full-court mask, while badminton uses 13 semantic court zones.

  [![Paper](https://img.shields.io/badge/Springer-Paper-538135.svg?style=for-the-badge)](https://doi.org/10.1007/978-3-031-63219-8_2)

- **CourtAlign-E2E** predicts semantic masks and geometric correspondences from
  a frozen SAM 3 vision trunk, a fine-tuned feature-pyramid neck, and learned
  task heads. A confidence-weighted differentiable DLT layer estimates the
  homography during training and inference.

The repository contains the final training and evaluation code, frozen splits,
official geometric ground truth, pretrained-weight instructions, and a video
court-tracking application for both methods.

## Method overview

### CourtAlign-2S

[![CourtAlign-2S pipeline](docs/figures/courtalign_2s_pipeline.png)](docs/figures/courtalign_2s_pipeline.pdf)

CourtAlign-2S builds on the method introduced in our published
[AIAI 2024 paper](https://doi.org/10.1007/978-3-031-63219-8_2). The implementation
provided here uses ResNet-34 for both sports. The learned segmentation stage is
followed by the method-specific classical registration stage.

### CourtAlign-E2E

[![CourtAlign-E2E pipeline](docs/figures/courtalign_e2e_pipeline.png)](docs/figures/courtalign_e2e_pipeline.pdf)

CourtAlign-E2E jointly supervises semantic prediction, image correspondences,
and homography accuracy. The SAM 3 vision transformer remains frozen, while
the feature-pyramid neck and prediction heads are optimized on the target
sport. Both final configurations predict the official line-axis landmarks.
The registration objective applies a Huber penalty to reprojection distances
in input pixels, using a 1-pixel transition and a 50-pixel stability clamp.

### Registration validity

Both variants return an explicit decision for every frame rather than a
homography unconditionally, so replays, close-ups and crowd views are rejected
instead of receiving a projected court. CourtAlign-2S applies a deterministic
check to the recovered geometry. CourtAlign-E2E gates the candidate homography
on three signals read from the network: the fraction of the frame the auxiliary
segmentation assigns to a court class (`no_court_area_frac`), the spatial spread
of the predicted keypoints (`min_spread_frac`), and their mean visibility
confidence (`min_conf_mean`). The thresholds are declared in
[`configs/courtalign_e2e/`](configs/courtalign_e2e) and are identical for both
sports. A frame that fails any of them is written with an explicit failure
status, which is what the NC-FP column below counts.

## Quantitative comparison

IoU, PCK-H and Line-IoU are reported as percentages. Projection error is
measured in metres on the reference court. Reprojection error is measured in
native image pixels. Each method is trained with several seeds; a run is first
averaged over the court-visible test frames, and the table reports the mean and
standard deviation across seeds. NC-FP is the mean seed-level number of false
registrations among the 20 non-registrable test frames of each sport, so a
method that never asserts a court on a frame without one scores `0/20`. The
LaTeX source of this table is
[`docs/benchmark/comparison_table.tex`](docs/benchmark/comparison_table.tex).

### Tennis

| Method | IoU (%) ↑ | Proj. (m) ↓ | Reproj. (px) ↓ | PCK-H@5px (%) ↑ | Line-IoU@0 (%) ↑ | @3 ↑ | @5 ↑ | NC-FP ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| KpSFR | 96.43 ± 0.31 | 0.11 ± 0.01 | 3.71 ± 0.33 | 48.65 ± 5.32 | 7.34 ± 0.67 | 42.85 ± 2.82 | 57.18 ± 2.58 | 0/20 |
| No-Bells-Just-Whistles | 98.99 ± 0.11 | 0.06 ± 0.00 | 2.51 ± 0.15 | 84.70 ± 0.96 | 11.56 ± 0.52 | 57.71 ± 0.63 | 69.45 ± 0.55 | 18.5/20 |
| KaliCalib | 95.53 ± 0.22 | 0.08 ± 0.01 | 3.60 ± 1.33 | 56.20 ± 7.20 | 9.94 ± 0.72 | 48.08 ± 2.60 | 61.95 ± 2.78 | 20/20 |
| TVCalib | 98.11 ± 0.31 | 0.11 ± 0.00 | 4.61 ± 0.36 | 52.09 ± 2.66 | 7.36 ± 0.22 | 47.06 ± 0.70 | 60.90 ± 1.05 | 0/20 |
| PnLCalib | 98.93 ± 0.15 | 0.05 ± 0.00 | 2.20 ± 0.32 | 89.39 ± 8.22 | 13.78 ± 2.03 | 61.52 ± 1.68 | 72.40 ± 1.35 | 2/20 |
| **CourtAlign-2S** | **99.68 ± 0.05** | 0.05 ± 0.00 | 2.20 ± 0.29 | 83.41 ± 1.12 | 13.72 ± 0.54 | 60.44 ± 0.78 | 71.41 ± 0.58 | **0/20** |
| **CourtAlign-E2E** | 99.60 ± 0.08 | **0.02 ± 0.01** | **0.87 ± 0.12** | **97.33 ± 0.73** | **18.54 ± 0.17** | **65.07 ± 0.15** | **75.11 ± 0.11** | **0/20** |

### Badminton

| Method | IoU (%) ↑ | Proj. (m) ↓ | Reproj. (px) ↓ | PCK-H@5px (%) ↑ | Line-IoU@0 (%) ↑ | @3 ↑ | @5 ↑ | NC-FP ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| KpSFR | 99.27 ± 0.20 | 0.03 ± 0.00 | **1.67 ± 0.08** | **97.95 ± 0.66** | 12.51 ± 0.53 | 55.93 ± 0.65 | 67.25 ± 0.51 | 0.75/20 |
| No-Bells-Just-Whistles | 99.33 ± 0.12 | 0.05 ± 0.00 | 2.90 ± 0.17 | 97.19 ± 1.20 | 11.13 ± 0.63 | 53.26 ± 0.72 | 65.12 ± 0.57 | 0/20 |
| KaliCalib | 97.53 ± 0.25 | 0.10 ± 0.03 | 6.18 ± 2.19 | 65.00 ± 9.48 | 7.91 ± 0.89 | 45.43 ± 3.11 | 57.70 ± 3.00 | 20/20 |
| TVCalib | 98.70 ± 0.36 | 0.15 ± 0.01 | 9.06 ± 0.47 | 19.76 ± 3.88 | 5.64 ± 0.38 | 32.27 ± 1.52 | 45.40 ± 1.12 | 5.25/20 |
| PnLCalib | 99.06 ± 0.42 | 0.06 ± 0.02 | 3.22 ± 0.73 | 90.97 ± 10.43 | 11.37 ± 1.16 | 52.60 ± 3.31 | 64.42 ± 2.93 | 4.00/20 |
| **CourtAlign-2S** | 99.45 ± 0.19 | 0.05 ± 0.01 | 2.28 ± 0.36 | 97.27 ± 2.25 | 11.85 ± 0.85 | 54.79 ± 1.32 | 66.29 ± 1.05 | **0/20** |
| **CourtAlign-E2E** | **99.47 ± 0.01** | **0.03 ± 0.00** | 1.69 ± 0.05 | 94.86 ± 0.69 | **13.20 ± 0.16** | **56.84 ± 0.12** | **67.97 ± 0.09** | **0/20** |

## Auxiliary cross-sport transfer on WC14

This is an auxiliary study, not the primary CourtAlign benchmark. It asks
whether the geometry-aware end-to-end formulation still applies outside racket
sports, on a soccer field that is larger, more often only partly visible, and
more locally ambiguous. Only CourtAlign-E2E is evaluated here: the CourtAlign-2S
zone representation is tied to the racket-court geometry and does not transfer
to a soccer field without a different sport-specific representation and
correspondence extractor.

Note the units differ from the racket-sports table above. Projection error is in
**metres**, while reprojection error is in **normalized image coordinates**, not
pixels. `*` marks SoccerNet pre-training, `†` marks WC14 fine-tuning, and
Rob-Solv is an inference-only variant that replaces the confidence-weighted
differentiable solver with RANSAC–DLT, leaving the trained network and the
selected checkpoint unchanged. The LaTeX source is
[`docs/benchmark/wc14_table.tex`](docs/benchmark/wc14_table.tex).

| Method | Proj. mean (m) ↓ | Proj. median (m) ↓ | Reproj. mean ↓ | Reproj. median ↓ |
|---|---:|---:|---:|---:|
| Nie et al. | 0.84 | 0.65 | 0.019 | 0.014 |
| KpSFR | 0.81 | 0.63 | 0.019 | 0.014 |
| No-Bells-Just-Whistles* | 1.23 | 0.58 | 0.026 | 0.014 |
| PnLCalib*† | **0.60** | **0.42** | **0.014** | **0.010** |
| **CourtAlign-E2E*†** | 0.83 | 0.46 | 0.020 | 0.012 |
| **CourtAlign-E2E w/Rob-Solv*†** | 0.78 | 0.48 | 0.020 | 0.013 |

## Accuracy--complexity comparison

The figure relates the number of trainable parameters to mean reprojection
error on the tennis test split. Lower reprojection error indicates more
accurate geometric registration. Red stars identify the CourtAlign methods,
while colored circles identify the compared methods. The horizontal axis counts
trainable parameters only. It is not a measure of total model size, runtime, or
memory: CourtAlign-E2E keeps a large pretrained backbone frozen, so its
trainable count is far smaller than the parameters it evaluates at inference.

[![Trainable parameters versus mean reprojection error on tennis](docs/figures/params_vs_reprojection_tennis.png)](docs/figures/params_vs_reprojection_tennis.pdf)

## Qualitative comparison

The tennis example compares projected court lines on the same test frame. The
badminton example shows behavior on a non-registrable frame. CourtAlign
projections are yellow and baseline projections are blue.

[![Tennis qualitative comparison](docs/figures/qualitative_tennis.png)](docs/figures/qualitative_tennis.pdf)

[![Badminton qualitative comparison](docs/figures/qualitative_badminton.png)](docs/figures/qualitative_badminton.pdf)

These figures use the final ResNet-34 CourtAlign-2S homographies, the final
CourtAlign-E2E homographies, and the baseline outputs reported in the table.

## Repository structure

```text
CourtAlign/
├── assets/                         # CourtAlign-2S tennis template assets
├── configs/
│   ├── courtalign_2s/              # final ResNet-34 training configs
│   ├── courtalign_e2e/             # final SAM 3 training configs
│   ├── datasets/                   # label schemas and manifest references
│   ├── evaluation/                 # frozen metric protocol
│   └── registration/               # CourtAlign-2S geometric stages
├── data/
│   ├── splits/                     # frozen train, validation, and test manifests
│   ├── benchmark_gt/official/      # frozen geometric ground truth
│   └── courtalign_e2e/             # CourtAlign-E2E supervision and group split
├── docs/                           # figures, benchmark table, and reproducibility records
├── environments/                   # method-specific Conda environments
├── scripts/
│   ├── train.py                    # public training entry point
│   ├── evaluate.py                 # public frozen-protocol evaluator
│   ├── track_video.py              # video application
│   ├── verify_setup.py             # data, weight, split, and GT checks
│   ├── courtalign_2s/              # CourtAlign-2S training and evaluation jobs
│   └── courtalign_e2e/             # CourtAlign-E2E preparation, training, and evaluation jobs
├── src/
│   ├── courtalign_2s/              # segmentation and classical registration
│   ├── courtalign_e2e/             # end-to-end model, losses, and geometry
│   ├── courtalign_common/          # shared data and evaluation components
│   └── courtalign/video/           # shared video application
└── weights/                        # downloaded checkpoints, not tracked by Git
```

The two method implementations are separate public Python packages under
`src/`. The top-level scripts provide a common interface for training,
evaluation, and video inference.

## Data

Download the datasets from:

**[CourtAlign datasets on Google Drive](https://drive.google.com/drive/folders/1rLmik07lxm5ameDtHscQsUtbdjEjEuIn?usp=sharing)**

Extract them under `data/` according to [`data/README.md`](data/README.md). The
expected top-level directories are:

```text
data/tennis_fullcourt/
data/badminton_zones/
```

The repository already contains the frozen manifests and official geometric
ground truth. Dataset images and masks are intentionally excluded from Git.

The shared held-out test sets contain 100 tennis frames and 33 badminton
frames. CourtAlign-2S uses the frozen 904/160 tennis training and validation
partition. CourtAlign-E2E keeps the same held-out test set but uses the included
rally-group-disjoint tennis training and validation partition to prevent
near-duplicate rally frames from crossing those two subsets. Both methods use
the frozen 436/95/33 badminton partition.

## Pretrained weights

Download the four final checkpoints from:

**[CourtAlign checkpoints on Google Drive](https://drive.google.com/drive/folders/1zhD7T0JxcJGemRNj33cutR9GjJb17Fvh?usp=sharing)**

Place them according to [`weights/README.md`](weights/README.md):

```text
weights/courtalign_2s/tennis/best_model.pth
weights/courtalign_2s/badminton/best_model.pth
weights/courtalign_e2e/tennis/best_model.pt
weights/courtalign_e2e/badminton/best_model.pt
```

The downloaded files are the complete selected checkpoints used for the
reported evaluations. CourtAlign-E2E initializes the SAM 3 architecture from
`facebook/sam3` before loading its selected checkpoint. New CourtAlign-E2E
training jobs save a compact checkpoint without duplicating the unchanged
vision trunk, and the loader supports both formats.

Verify the complete setup before training or evaluation:

```bash
python scripts/verify_setup.py --require-data --require-weights
```

## Environment setup

The two methods use separate environments because their frozen PyTorch stacks
are not interchangeable.

### CourtAlign-2S

```bash
conda env create -f environments/courtalign-2s.yml
conda activate courtalign-2s
python -m pip install -e . --no-deps
```

### CourtAlign-E2E

```bash
conda env create -f environments/courtalign-e2e.yml
conda activate courtalign-e2e
python -m pip install -e . --no-deps
```

CourtAlign-E2E loads the gated `facebook/sam3` checkpoint through Hugging Face.
Accept its model terms and authenticate once before training or inference:

```bash
hf auth login
```

## Tests

Run the CourtAlign-2S and shared protocol tests in the CourtAlign-2S
environment:

```bash
conda activate courtalign-2s
python -m pytest -q \
  tests/test_courtalign_2s_augmentation.py \
  tests/test_courtalign_2s_three_phase_dice.py \
  tests/test_metric_homography.py \
  tests/test_metrics.py \
  tests/test_official_registration_protocol.py \
  tests/test_public_layout.py \
  tests/test_video_tracking.py
```

Run the CourtAlign-E2E unit and public-layout tests in the CourtAlign-E2E
environment:

```bash
conda activate courtalign-e2e
python -m pytest -q \
  tests/test_e2e_core.py \
  tests/test_metrics.py \
  tests/test_public_layout.py
```

## Training

Run all commands from the repository root.

### CourtAlign-2S

```bash
conda activate courtalign-2s

python scripts/train.py --method courtalign-2s --sport tennis
python scripts/train.py --method courtalign-2s --sport badminton
```

The tennis job trains for 50 epochs with batch size 4 and uses the all-class
Dice loss. The badminton job trains for 80 epochs with batch size 2 and applies
the final three-phase class-focused Dice schedule. Selected checkpoints are
written to:

```text
runs/courtalign_2s/tennis/segmentation/checkpoints/best_model.pth
runs/courtalign_2s/badminton/segmentation/checkpoints/best_model.pth
```

### CourtAlign-E2E

```bash
conda activate courtalign-e2e

python scripts/train.py --method courtalign-e2e --sport tennis
python scripts/train.py --method courtalign-e2e --sport badminton
```

The tennis and badminton jobs run for 60 and 80 epochs, respectively. Both use
seed 1337 and select the checkpoint with the lowest validation lattice
reprojection error. The reprojection loss is evaluated directly in input
pixels with a weight of `0.15`, a 1-pixel Huber transition, and a 50-pixel
clamp. Interrupted jobs can restore the model, optimizer, scheduler, and
best-checkpoint state:

```bash
python scripts/train.py --method courtalign-e2e --sport tennis --resume
```

## Official evaluation

The evaluator runs every held-out frame, exports either a metric-court-to-image
homography or an explicit failure status, and then applies the frozen metric
implementation to the complete test manifest.

### CourtAlign-2S

```bash
conda activate courtalign-2s

python scripts/evaluate.py --method courtalign-2s --sport tennis
python scripts/evaluate.py --method courtalign-2s --sport badminton
```

### CourtAlign-E2E

```bash
conda activate courtalign-e2e

python scripts/evaluate.py --method courtalign-e2e --sport tennis
python scripts/evaluate.py --method courtalign-e2e --sport badminton
```

Each command writes the canonical summary to:

```text
runs/evaluation/<method>/<sport>/official/official_metrics.json
```

To evaluate a newly trained checkpoint instead of the downloaded model, pass
its path explicitly:

```bash
python scripts/evaluate.py \
  --method courtalign-2s \
  --sport tennis \
  --checkpoint runs/courtalign_2s/tennis/segmentation/checkpoints/best_model.pth \
  --output-dir runs/evaluation/courtalign-2s/tennis_retrained
```

Evaluation directories are never overwritten. Choose a new `--output-dir` for
another run.

## Video court tracking

The application accepts one video or a directory of videos and supports either
method. The default mode estimates a registration independently on every
frame, preserving the selected method's final inference behavior. Homographies
are returned in the original video resolution, including when it differs from
the native evaluation resolution.

```bash
conda activate courtalign-2s
python scripts/track_video.py \
  --method courtalign-2s \
  --sport tennis \
  --input path/to/input.mp4 \
  --output-dir outputs/video_tracking
```

```bash
conda activate courtalign-e2e
python scripts/track_video.py \
  --method courtalign-e2e \
  --sport badminton \
  --input path/to/input.mp4 \
  --output-dir outputs/video_tracking
```

The optional motion mode re-estimates the court when camera motion is detected
or when the refresh interval is reached. Otherwise, it reuses the last accepted
homography:

```bash
python scripts/track_video.py \
  --method courtalign-2s \
  --sport tennis \
  --input path/to/input.mp4 \
  --tracking-mode motion \
  --refresh-interval 30 \
  --motion-threshold 0.08
```

For each output video, the application also writes a JSON Lines file containing
the per-frame status, homography, inference decision, and diagnostics. Existing
video outputs and sidecar files are never overwritten.

## Reproducibility

The exact configurations, checkpoint hashes, split hashes, selection rules,
and protocol notes are recorded in
[`docs/reproducibility.md`](docs/reproducibility.md). The official ground-truth
hashes can be verified without downloading either dataset:

```bash
python scripts/verify_setup.py
```

## Citation

If you use CourtAlign-2S, please cite:

```bibtex
@inproceedings{jouini2024deep,
  title     = {A deep learning-based framework for racket sports court registration},
  author    = {Jouini, Ahmed and Elloumi, Melek and Chaieb, Faten},
  booktitle = {IFIP International Conference on Artificial Intelligence Applications and Innovations},
  year      = {2024},
  pages     = {17--29}
}
```

The benchmark includes the following related methods:

```bibtex
@inproceedings{chu2022sports,
  title     = {Sports field registration via keypoints-aware label condition},
  author    = {Chu, Yen-Jui and Su, Jheng-Wei and Hsiao, Kai-Wen and Lien, Chi-Yu and Fan, Shu-Ho and Hu, Min-Chun and Lee, Ruen-Rone and Yao, Chih-Yuan and Chu, Hung-Kuo},
  booktitle = {CVPR},
  year      = {2022},
  pages     = {3523--3530}
}

@inproceedings{Gutierrez-Perez_2024_CVPR,
  title     = {No bells just whistles: Sports field registration by leveraging geometric properties},
  author    = {Guti{\'e}rrez-P{\'e}rez, Marc and Agudo, Antonio},
  booktitle = {CVPRW},
  year      = {2024},
  pages     = {3325--3334}
}

@inproceedings{maglo2022kalicalib,
  title     = {{KaliCalib}: A framework for basketball court registration},
  author    = {Maglo, Adrien and Orcesi, Astrid and Pham, Quoc-Cuong},
  booktitle = {International ACM Workshop on Multimedia Content Analysis in Sports},
  year      = {2022},
  pages     = {111--116}
}

@inproceedings{theiner2023tvcalib,
  title     = {{TVCalib}: Camera calibration for sports field registration in soccer},
  author    = {Theiner, Jonas and Ewerth, Ralph},
  booktitle = {WACV},
  year      = {2023},
  pages     = {1166--1175}
}
```
