# Dataset placement

Download the tennis and badminton data from:

https://drive.google.com/drive/folders/1rLmik07lxm5ameDtHscQsUtbdjEjEuIn?usp=sharing

Extract the two dataset directories under `data/` so that the paths stored in
the frozen manifests resolve directly:

```text
data/
├── tennis_fullcourt/
│   ├── images/{train,val,test}/
│   ├── masks/{train,val,test}/
│   └── line_masks/test/
└── badminton_zones/
    ├── images/{train,val,test}/
    ├── masks/{train,val,test}/
    └── line_masks/all/
```

Do not edit `data/splits/*.csv` or `data/benchmark_gt/official/`. They are the
frozen split and geometric ground-truth resources used for the reported
experiments.

The split manifests contain repository-relative image and mask paths, file
hashes, dimensions, and label schemas. This makes the protocol portable while
preserving the exact sample membership and ordering used in the experiments.

After extraction, verify every image, mask, checkpoint, split, and ground-truth
hash from the repository root:

```bash
python scripts/verify_setup.py --require-data --require-weights
```
