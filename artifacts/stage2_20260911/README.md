# Stage 2 gate-perception baseline artifacts

These artifacts were generated on September 11, 2026 from the calibrated
Stage2A oracle pipeline and the first Stage2B spatial-heatmap detector baseline.

Contents:

- `checkpoints/best_detector.pt`: best detector selected on the fixed validation split;
- `evaluations/stage2a_equivalence.json`: oracle projection/PnP equivalence over 1,835 complete samples;
- `evaluations/stage2b_evaluation.json`: detector pixel and downstream PnP errors;
- `params/dataset_manifest.json`: generation parameters for the 2,048-frame dataset;
- `params/training_summary.json`: selected epoch and training/validation summary;
- `SHA256SUMS`: integrity hashes for all artifacts above.

The 118 MB rendered PNG/JSON dataset remains under the ignored local `logs/`
directory and is reproducible with:

```bash
OMNI_KIT_ACCEPT_EULA=YES PYTHONPATH="$PWD" python \
  scripts/perception/collect_stage2_dataset.py \
  --headless --enable_cameras \
  --output logs/stage2/datasets/2026-09-11_oracle_gate_256_seed1 \
  --num_envs 32 --batches 64 --width 256 --height 256 --seed 1
```

Download Git LFS objects and verify the tracked artifact copies with:

```bash
git lfs pull
cd artifacts/stage2_20260911
sha256sum -c SHA256SUMS
cd ../..
```

Load the trained detector through the repository adapter:

```python
from perception.keypoint_detector import TorchGateCornerDetector

detector = TorchGateCornerDetector(
    "artifacts/stage2_20260911/checkpoints/best_detector.pt",
    device="cuda",
)
observation = detector.detect(rgb_image)
```

The fixed validation split contains 410 frames. The detector reached 3.57 px
corner RMSE at the original 256 x 256 resolution and 99.2% complete-detection /
PnP success on complete labels. Its mean gate translation/rotation errors are
0.132 m / 9.14 degrees. This is an offline baseline, not a closed-loop racing
release: planar PnP amplifies the remaining corner errors, and the forward
pinhole camera cannot always observe the next gate on the current track.
