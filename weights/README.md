# Model checkpoints

Download the four released CourtAlign checkpoints from:

https://drive.google.com/drive/folders/1zhD7T0JxcJGemRNj33cutR9GjJb17Fvh?usp=sharing

Place and rename them exactly as follows:

```text
weights/
├── courtalign_2s/
│   ├── tennis/best_model.pth
│   └── badminton/best_model.pth
└── courtalign_e2e/
    ├── tennis/best_model.pt
    └── badminton/best_model.pt
```

The four downloaded files are the complete selected checkpoints used for the
reported evaluations. CourtAlign-2S checkpoints must be loaded with the
CourtAlign-2S environment. CourtAlign-E2E initializes the SAM 3 architecture
from `facebook/sam3` before loading the selected checkpoint. New training jobs
save a compact CourtAlign-E2E format that omits the unchanged vision trunk. The
loader supports both the released full checkpoints and newly trained compact
checkpoints.

The CourtAlign-E2E checkpoints were updated on 30 July 2026. Replace earlier
downloads under the two CourtAlign-E2E paths before running the official
evaluation. Their expected sizes and SHA-256 hashes are recorded in
`docs/reproducibility.md` and checked by `scripts/verify_setup.py`.
