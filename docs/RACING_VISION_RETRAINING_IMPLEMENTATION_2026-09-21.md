# Racing Vision Retraining Implementation

Date: 2026-09-21

Branch:

```text
feature/racing-vision-retraining
```

This branch starts from the documented `feature/swift-ctbr-rl` checkpoint and implements the first part of the next estimator/perception phase:

```text
frozen successful GT racing policy
        ->
high-speed racing image collection
        ->
exact simulator gate-corner labels
        ->
aggressive racing vision fine-tuning
        ->
held-out hybrid availability evaluation
```

The policy remains GT-controlled during data collection. The new visual model does not affect the trajectory used to generate its own training data.

## 1. New files

```text
scripts/perception/collect_racing_vision_dataset.py
scripts/perception/train_racing_vision_models.py
scripts/perception/evaluate_racing_vision_hybrid.py
tests/perception/test_racing_vision_static.py
```

The parent branch also contains:

```text
docs/RACING_VISION_DATASET_AND_ESTIMATOR_PLAN_2026-09-21.md
```

## 2. Data collection design

The collector launches:

```text
Isaac-Drone-Racer-Swift-CTBR-GT-Racing-v0
```

with:

- one environment;
- the exact successful GT actor observation;
- the existing CTBR control path;
- the production Stage2 pinhole camera re-enabled at 256x256;
- no estimator or detector feedback into control.

The default capture interval is every four 100-Hz policy steps:

```text
25 Hz image/label collection
```

Each frame stores the standard Stage2 label plus:

- episode/attempt index;
- policy step;
- gates passed so far;
- active gate index;
- GT world position;
- GT world velocity;
- GT quaternion;
- GT body rate;
- speed;
- body-rate norm;
- normalized policy action;
- physical CTBR command.

Only episodes passing `--min-gates` are retained.

Train/validation/test split is performed by complete accepted episode, not by frame.

## 3. Sync the new branch

```bash
cd ~/isaac_projects/isaac_drone_racer

git fetch ssybh2 --prune
git switch feature/racing-vision-retraining
git pull --ff-only ssybh2 feature/racing-vision-retraining

git log -6 --oneline
```

## 4. Static validation

```bash
./.conda-env/bin/python -m pytest -q   tests/perception/test_racing_vision_static.py   tests/rl/test_swift_ctbr_static.py   tests/rl/test_learned_inertial_rl_interface_static.py

./.conda-env/bin/python -m compileall -q   scripts/perception   perception   tasks/drone_racer
```

Do not start a large collection until these commands pass.

## 5. First smoke collection

Use the known GT racing checkpoint:

```bash
export BEST="logs/skrl/swift_ctbr_gt_racing/2026-09-21_20-47-02_ppo_torch_easy7_4096env_roll24_256x3_gt/checkpoints/best_agent.pt"
```

Collect one successful episode first:

```bash
rm -rf artifacts/racing_vision/smoke

./.conda-env/bin/python   scripts/perception/collect_racing_vision_dataset.py   --checkpoint "$BEST"   --output-dir artifacts/racing_vision/smoke   --target-successful-episodes 1   --val-episodes 0   --test-episodes 0   --min-gates 30   --max-attempts 10   --capture-every-steps 4   --width 256   --height 256   --seed 1   --device cuda:0   --headless
```

Expected structure:

```text
artifacts/racing_vision/smoke/
  manifest.json
  episodes.csv
  train/
    manifest.json
    images/
    labels/
  val/
  test/
```

Inspect:

```bash
cat artifacts/racing_vision/smoke/manifest.json
cat artifacts/racing_vision/smoke/train/manifest.json
column -s, -t < artifacts/racing_vision/smoke/episodes.csv | head -30
```

The smoke goal is not detector performance yet. It is to confirm that:

- the GT policy still races with the camera enabled;
- one >=30-gate episode is retained;
- images and labels are written;
- speed/body-rate metadata is non-zero and racing-like;
- the visible-corner histogram contains useful 2/3/4-corner frames.

## 6. Main racing dataset

After the smoke passes, collect several successful episodes.

Recommended first dataset:

```bash
rm -rf artifacts/racing_vision/easy7_gt_racing_v1

./.conda-env/bin/python   scripts/perception/collect_racing_vision_dataset.py   --checkpoint "$BEST"   --output-dir artifacts/racing_vision/easy7_gt_racing_v1   --target-successful-episodes 10   --val-episodes 2   --test-episodes 2   --min-gates 30   --max-attempts 40   --capture-every-steps 4   --width 256   --height 256   --seed 1   --device cuda:0   --headless
```

This produces:

```text
6 train episodes
2 validation episodes
2 test episodes
```

with no episode leakage between splits.

If 30 gates admits too many incomplete trajectories, raise:

```text
--min-gates 35
```

or eventually:

```text
--min-gates 37
```

once collection reliability is established.

## 7. Baseline the old visual stack on racing data

Before retraining, measure the old production model on the held-out racing test split:

```bash
./.conda-env/bin/python   scripts/perception/evaluate_racing_vision_hybrid.py   --dataset artifacts/racing_vision/easy7_gt_racing_v1/test   --coordinate-checkpoint     artifacts/stage2_next_steps_20260911/checkpoints/torchvision_keypointrcnn_best.pt   --visibility-checkpoint     artifacts/stage2_next_steps_20260911/checkpoints/gate_keypoint_net_best.pt   --output artifacts/racing_vision/easy7_gt_racing_v1/old_hybrid_test.json   --device cuda:0
```

Primary metric:

```text
overall.hybrid_availability_given_gt_ge2
```

This directly asks:

> when simulator truth says at least two semantic gate corners are actually in the image, how often does the production hybrid return at least two usable corners?

The evaluator also bins availability by:

- speed;
- body-rate magnitude.

This is the direct test for racing-distribution visual OOD.

## 8. Train the aggressive racing model

First fine-tune only on racing data plus the existing production checkpoints:

```bash
rm -rf artifacts/racing_vision/models_v1

./.conda-env/bin/python   scripts/perception/train_racing_vision_models.py   --racing-train artifacts/racing_vision/easy7_gt_racing_v1/train   --racing-val artifacts/racing_vision/easy7_gt_racing_v1/val   --output-dir artifacts/racing_vision/models_v1   --model both   --racing-repeat 3   --rcnn-init     artifacts/stage2_next_steps_20260911/checkpoints/torchvision_keypointrcnn_best.pt   --guard-init     artifacts/stage2_next_steps_20260911/checkpoints/gate_keypoint_net_best.pt   --rcnn-image-size 256   --guard-input-size 256   --device cuda:0
```

Outputs:

```text
artifacts/racing_vision/models_v1/
  torchvision_keypointrcnn_racing_best.pt
  torchvision_keypointrcnn_racing_best.json
  gate_keypoint_net_racing_best.pt
  gate_keypoint_net_racing_best.json
  training_summary.json
```

The training augmentation is deliberately more aggressive than the original static Stage2 profile:

- brightness gain 0.55-1.45;
- contrast 0.70-1.30;
- image noise up to 0.045 in normalized intensity;
- random directional motion blur;
- blur kernels up to 15 px;
- 256x256 model input.

Racing samples are repeated three times by default so they dominate fine-tuning.

## 9. Optional legacy-data retention

Once the racing-only fine-tune works, mix the original static Stage2 dataset to avoid catastrophic forgetting:

```bash
./.conda-env/bin/python   scripts/perception/train_racing_vision_models.py   --racing-train artifacts/racing_vision/easy7_gt_racing_v1/train   --racing-val artifacts/racing_vision/easy7_gt_racing_v1/val   --legacy-train <OLD_STAGE2_TRAIN_DATASET_ROOT>   --output-dir artifacts/racing_vision/models_v2_mixed   --model both   --racing-repeat 3   --device cuda:0
```

The racing validation split remains the model-selection distribution.

## 10. Evaluate the new hybrid

```bash
./.conda-env/bin/python   scripts/perception/evaluate_racing_vision_hybrid.py   --dataset artifacts/racing_vision/easy7_gt_racing_v1/test   --coordinate-checkpoint     artifacts/racing_vision/models_v1/torchvision_keypointrcnn_racing_best.pt   --visibility-checkpoint     artifacts/racing_vision/models_v1/gate_keypoint_net_racing_best.pt   --output artifacts/racing_vision/easy7_gt_racing_v1/new_hybrid_test.json   --device cuda:0
```

Compare:

```bash
./.conda-env/bin/python - <<'PY'
import json
from pathlib import Path

root = Path("artifacts/racing_vision/easy7_gt_racing_v1")
for name in ("old_hybrid_test.json", "new_hybrid_test.json"):
    x = json.loads((root / name).read_text())
    print("=" * 90)
    print(name)
    print("availability | GT>=2:",
          x["overall"]["hybrid_availability_given_gt_ge2"])
    print("coordinate >=2 rate:",
          x["overall"]["coordinate_ge2_rate"])
    print("guard >=2 rate:",
          x["overall"]["guard_ge2_rate"])
    print("hybrid >=2 rate:",
          x["overall"]["hybrid_ge2_rate"])
    print("corner RMSE:",
          x["corner_rmse_px_on_gt_visible"])
PY
```

## 11. Replacement gate

Do not replace the production visual checkpoints only because validation corner RMSE improves.

The new model should show a clear improvement in:

```text
hybrid_availability_given_gt_ge2
```

especially in the high-speed and high-body-rate bins.

The desired direction is:

```text
GT says >=2 corners visible
        ->
coordinate detector finds the gate/corners
        ->
visibility guard preserves >=2 useful corners
        ->
direct reprojection receives measurements continuously
```

After this held-out test passes, wire the new checkpoints into a new GTShadow configuration and re-run the 20 s racing shadow benchmark.

## 12. What is deliberately not changed yet

This branch does not yet:

- loosen the 80 px association threshold;
- loosen the normalized NIS threshold;
- change V6.6 stochastic-clone EKF mathematics;
- retrain V6.2 Body-Delta-v;
- feed estimated state back into the racing policy.

Those steps should wait until the visual front end is measured on the real racing distribution.

The next policy change after vision retraining is to fine-tune the current 37-gate checkpoint with a stronger image-space perception-aware objective.
