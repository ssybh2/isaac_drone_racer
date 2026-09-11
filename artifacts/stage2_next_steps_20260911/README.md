# Stage 2 next-step validation and training artifacts

This directory contains the tracked outputs from executing
`docs/stage2_next_steps_review_2026-09-11.md` through Checkpoint D.
Checkpoint E was intentionally not started because the offline pose thresholds
were not reached.

Contents:

- `physical_validation/`: all 32 RGB oracle overlays, contact sheet, and the
  configured-versus-truth extrinsic report;
- `params/`: 20,480-frame training manifest, independent seed-2 validation and
  seed-3 test manifests, and both training summaries;
- `evaluations/`: full aggregate, distance-binned, projected-size-binned,
  latency, memory, and model-size reports;
- `checkpoints/gate_keypoint_net_best.pt`: retrained compact detector selected
  on the independent seed-2 validation set;
- `checkpoints/torchvision_keypointrcnn_best.pt`: mature torchvision comparison
  model selected on the same validation set;
- `SHA256SUMS`: integrity hashes for this artifact set.

The rendered datasets remain in the ignored local `logs/stage2/datasets/`
tree because the expanded training set is about 1.9 GiB. It is reproducible
from the tracked collection script and `params/train_manifest.json`.

Pull Git LFS objects and verify this directory with:

```bash
git lfs pull
cd artifacts/stage2_next_steps_20260911
sha256sum -c SHA256SUMS
```

See `docs/stage2_next_steps_execution_report_2026-09-11.md` for conclusions and
the exact commands used.
