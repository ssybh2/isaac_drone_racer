# Learned Inertial V6.2 — Endpoint-Body Displacement Training

Date: 2026-09-20

Branch:

```text
feature/v6.2-uzh-stochastic-cloning
```

## Decision from the Oracle ablations

Three seed-0 estimator experiments isolated the measurement-model failure:

```text
IMU-only A
position RMSE ≈ 2.495 m

V6.2 two-clone kinematic-residual Oracle
position RMSE ≈ 2.544 m
K_position,max ≈ 1248
dx_position,max ≈ 2.716 m

V6.2 pure UZH world-displacement Oracle
position RMSE ≈ 0.183 m
K_position,max ≈ 3.75
dx_position,max ≈ 0.072 m

V6.2 endpoint-body full-displacement Oracle
position RMSE ≈ 0.348 m
K_position,max ≈ 9.33
dx_position,max ≈ 0.140 m
```

The endpoint-body displacement factor retains the deployment-friendly
body-frame representation while removing the V6.1 kinematic residualization
term involving the cloned start velocity.

The V6.2 deployment target is therefore:

```text
target_mode = displacement_body_end_gyro_aligned

target = R_end^T (p_end - p_start)
features = gyro/thrust aligned into endpoint body frame using gyro-only
relative rotation
```

The filter remains the V6.2 UZH-style stochastic-cloning EKF with 9D [R,v,p]
clones and a full Kalman gain.

## Reuse the V6.1 dataset

No recollection is required initially. The existing V6.1 traces contain
position, velocity, attitude, gyro and thrust, so the new target can be
recomputed offline from:

```text
config/learned_inertial/imo_v6_1_manifest.json
artifacts/imo_dataset_v6_1/
```

This keeps the trajectory distribution identical and isolates the target
representation.

## Validate

```bash
cd ~/isaac_projects/isaac_drone_racer

./.conda-env/bin/python -m pytest -q \
  tests/estimation/test_learned_motion.py \
  tests/estimation/test_learned_motion_dataset.py \
  tests/estimation/test_learned_inertial_odometry.py \
  tests/estimation/test_v6_1_manifest.py
```

## Audit the recomputed V6.2 target

```bash
mkdir -p artifacts/imo_tcn/v6_2

./.conda-env/bin/python \
  scripts/estimation/audit_imo_dataset.py \
  config/learned_inertial/imo_v6_1_manifest.json \
  --window_time_s 0.5 \
  --sample_rate_hz 100 \
  --stride_time_s 0.05 \
  --target_mode displacement_body_end_gyro_aligned \
  --output artifacts/imo_tcn/v6_2/dataset_audit.json
```

Inspect held-out support warnings before training.

## Train V6.2

```bash
./.conda-env/bin/python \
  scripts/estimation/train_learned_motion.py \
  --manifest config/learned_inertial/imo_v6_1_manifest.json \
  --output artifacts/imo_tcn/model_v6_2_body_displacement_balanced.pt \
  --window_time_s 0.5 \
  --sample_rate_hz 100 \
  --stride_time_s 0.01 \
  --target_mode displacement_body_end_gyro_aligned \
  --train_sampling trace_balanced \
  --selection_metric nll \
  --epochs 200 \
  --batch_size 128 \
  --lr 1e-4 \
  --device cuda
```

## Offline checkpoint evaluation

```bash
./.conda-env/bin/python \
  scripts/estimation/evaluate_learned_motion_checkpoint.py \
  --checkpoint artifacts/imo_tcn/model_v6_2_body_displacement_balanced.pt \
  --manifest config/learned_inertial/imo_v6_1_manifest.json \
  --split all \
  --device cuda \
  --output artifacts/imo_tcn/v6_2/checkpoint_eval.json
```

Accept the network only after checking aggregate and per-trace RMSE/bias,
especially the held-out Lissajous, racing-like, circle and translate+yaw
trajectories.

## First estimator acceptance run

Use the existing 30 s seed-0 Lissajous replay with the new checkpoint. Do not
use an Oracle flag and do not use a legacy freeze_* gain mode.

```bash
export V6_CKPT="artifacts/imo_tcn/model_v6_2_body_displacement_balanced.pt"
export SEED=0
export REPLAY="artifacts/learned_inertial_diagnostics/v6_30s_lissajous_multiseed/seed_${SEED}/replay.npz"
export OUT="artifacts/learned_inertial_diagnostics/v6_2_body_displacement/seed_${SEED}/network_cov30"

mkdir -p "$OUT"

./.conda-env/bin/python \
  scripts/estimation/evaluate_learned_inertial_mode.py \
  --mode B \
  --steps 3000 \
  --profile lissajous \
  --seed "$SEED" \
  --imu-noise-seed "$SEED" \
  --imu-accel-white-noise-sigma-mps2 0.01 \
  --imu-gyro-white-noise-sigma-radps 0.001 \
  --imu-accel-bias-rw-sigma-mps2-sqrt-s 0.001 \
  --imu-gyro-bias-rw-sigma-radps-sqrt-s 0.0001 \
  --learned-update-rate-hz 20 \
  --learned-fusion-rate-hz 2 \
  --learned-covariance-multiplier 30 \
  --learned-checkpoint "$V6_CKPT" \
  --replay-npz "$REPLAY" \
  --output-dir "$OUT" \
  --device cuda:0 \
  --headless
```

Primary acceptance diagnostics:

```text
position / velocity RMSE
network prediction error vs GT
K_position and K_clone_position
dx_position and dx_velocity
measurement translation nullspace
propagated vs instantaneous gauge projection
innovation covariance conditioning
```

Only after seed 0 is stable should the same checkpoint be evaluated on the
five-seed replay suite.
