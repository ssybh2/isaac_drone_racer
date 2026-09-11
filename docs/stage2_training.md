# Stage 2 calibration, dataset, and detector training

This is the executable workflow for the architecture in
[`stage2_perception_architecture.md`](stage2_perception_architecture.md).
Stage2A contains no learned model: it calibrates and validates the oracle-corner
projection and shared PnP backend. Training begins at Stage2B with the RGB corner
detector.

## Calibrated repository assets

The checked-in [`gate_keypoints.json`](../assets/gate/gate_keypoints.json) was
measured from `/gate/geometry/mesh` in `gate.usd`:

- opening size: 1.524 m x 1.524 m;
- approach-side corner plane: approximately `x = -0.0661411 m`;
- opening center in actor coordinates: `(-0.0661411, 0, 1.0668) m`.

The actor origin is at floor level, not at the opening center. Pose recovery
therefore transforms the calibrated corner centroid to produce `target_pos_b`.
It must not use the raw actor translation.

The Stage2 data environment uses an explicit pinhole camera. At 1000 x 1000 its
current Isaac calibration is approximately `fx = fy = 1145.3 px`, `cx = cy =
500 px`. The mount maps the ROS/OpenCV optical frame into the drone body frame:

```text
R_bc = [[ 0,  0,  1],
        [-1,  0,  0],
        [ 0, -1,  0]]
t_bc = [0.14, 0, 0.05] m
```

Isaac Lab 2.1's tiled camera otherwise retains its initialization pose. The
Stage2 config requests the latest pose and the pinned-version adapter refreshes
that buffer before writing a label.

## Dataset collection

The collector randomizes distance, lateral/vertical position, roll, pitch, and
yaw around gate 0. It writes RGB plus exact ordered corners, visibility, K, gate
index, and `T_wg`/`T_wc`/`T_wb` truth.

```bash
OMNI_KIT_ACCEPT_EULA=YES PYTHONPATH="$PWD" python \
  scripts/perception/collect_stage2_dataset.py \
  --headless --enable_cameras \
  --output logs/stage2/datasets/oracle_gate_256 \
  --num_envs 32 --batches 64 --width 256 --height 256 --seed 1
```

Confirm that exact labels close through PnP before training:

```bash
PYTHONPATH="$PWD" python scripts/perception/evaluate_stage2a_dataset.py \
  --dataset logs/stage2/datasets/oracle_gate_256 \
  --output logs/stage2/datasets/oracle_gate_256/stage2a_equivalence.json
```

## Stage2B detector training and evaluation

The baseline detector uses a compact encoder/decoder, four ordered spatial
heatmaps with soft-argmax coordinates, and a four-corner visibility head.

```bash
PYTHONPATH="$PWD" python scripts/perception/train_gate_detector.py \
  --dataset logs/stage2/datasets/oracle_gate_256 \
  --output logs/stage2/models/gate_heatmap/best_detector.pt \
  --epochs 80 --batch_size 64 --input_size 128 --width 32 --seed 1

PYTHONPATH="$PWD" python scripts/perception/evaluate_gate_detector.py \
  --dataset logs/stage2/datasets/oracle_gate_256 \
  --checkpoint logs/stage2/models/gate_heatmap/best_detector.pt \
  --output logs/stage2/models/gate_heatmap/evaluation.json \
  --validation_fraction 0.2 --seed 1 --device cuda
```

The second command measures both detector pixel error and the downstream pose
error through the same OpenCV IPPE backend used by Stage2A.

## Verified baseline (September 11, 2026)

The local 2,048-frame, 256 x 256 dataset contains 1,835 complete and 213
partially visible samples. On the fixed 410-frame validation split:

| Metric | Result |
| --- | ---: |
| Complete detection rate on complete labels | 99.2% |
| Corner RMSE at original resolution | 3.57 px |
| Mean / p95 gate translation error | 0.132 / 0.375 m |
| Mean / p95 gate rotation error | 9.14 / 23.15 deg |
| Mean / p95 recovered body translation error | 0.996 / 2.534 m |

The oracle Stage2A path over all 1,835 complete samples has mean reprojection
error `1.36e-13 px`, mean gate translation error `1.25e-14 m`, and mean body
translation error `1.29e-7 m`. The learned detector is therefore the current
accuracy bottleneck.

This baseline is not ready for closed-loop racing. Planar PnP amplifies corner
and orientation error at distance, and the current forward pinhole camera does
not always see the next gate on the existing multi-turn track. Before Stage2B.3,
add broader domain randomization and temporal/map handling for missing gates,
improve pose-level accuracy, and define explicit closed-loop acceptance limits.
