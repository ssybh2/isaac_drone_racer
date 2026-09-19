# Learned Inertial V6.1: Coupled-Motion Training

## Goal

V6.1 keeps the V6 estimator representation unchanged and addresses the remaining
trajectory-dependent network error through a better training distribution.

Unchanged from V6:

- 0.5 s window at 100 Hz.
- Input: body gyro + mass-normalized collective thrust.
- Gyro-only relative-attitude alignment into the endpoint body frame.
- Target:
  `kinematic_residual_body_end_gravity_compensated`.
- Runtime estimator ablation candidate:
  `freeze_clones_attitude_bias` (current velocity/position update only).

V6.1 changes only:

1. Dataset coverage: substantially more long coupled motion.
2. Training sampling: optional trace-balanced sampling.
3. Diagnostics: train/val/test per-trace bias and RMSE are saved with the
   checkpoint report.

The purpose is to test whether the V6 Lissajous/racing-like bias comes from
training/deployment distribution mismatch before changing the TCN architecture
or the EKF again.

## Dataset

Manifest:

`config/learned_inertial/imo_v6_1_manifest.json`

The split is fixed at the whole-trajectory level. Training, validation and test
all contain circle, Lissajous, racing-like and translate+yaw trajectories with
different amplitudes, frequencies, phases and seeds.

Long coupled trajectories (30--45 s) are intentionally common. Trace-balanced
sampling prevents those longer traces from dominating merely because they
produce more overlapping 0.5 s windows.

## 1. Validate the code and manifest

```bash
cd ~/isaac_projects/isaac_drone_racer

./.conda-env/bin/python -m pytest -q \
  tests/estimation/test_learned_motion.py \
  tests/estimation/test_learned_motion_dataset.py \
  tests/estimation/test_v6_1_manifest.py

./.conda-env/bin/python \
  scripts/estimation/collect_imo_manifest.py \
  config/learned_inertial/imo_v6_1_manifest.json \
  --dry-run \
  --headless \
  --device cuda:0
```

## 2. Collect V6.1 traces

```bash
./.conda-env/bin/python \
  scripts/estimation/collect_imo_manifest.py \
  config/learned_inertial/imo_v6_1_manifest.json \
  --headless \
  --device cuda:0 \
  --resume
```

The collector stores traces under:

`artifacts/imo_dataset_v6_1/`

Ground truth is used only to create supervised labels and offline evaluation.

## 3. Audit the actual V6 target distribution

Do not audit ordinary displacement when training V6.1. Audit the exact
gravity-compensated endpoint-body residual target:

```bash
mkdir -p artifacts/imo_tcn/v6_1

./.conda-env/bin/python \
  scripts/estimation/audit_imo_dataset.py \
  config/learned_inertial/imo_v6_1_manifest.json \
  --window_time_s 0.5 \
  --sample_rate_hz 100 \
  --stride_time_s 0.05 \
  --target_mode kinematic_residual_body_end_gravity_compensated \
  --output artifacts/imo_tcn/v6_1/dataset_audit.json
```

Investigate any held-out support warning before training.

## 4. Train V6.1

First V6.1 experiment keeps the model and loss unchanged and selects the best
checkpoint using validation NLL. The only training-distribution change is
trace-balanced sampling.

```bash
./.conda-env/bin/python \
  scripts/estimation/train_learned_motion.py \
  --manifest config/learned_inertial/imo_v6_1_manifest.json \
  --output artifacts/imo_tcn/model_v6_1_coupled_balanced.pt \
  --window_time_s 0.5 \
  --sample_rate_hz 100 \
  --stride_time_s 0.01 \
  --target_mode kinematic_residual_body_end_gravity_compensated \
  --train_sampling trace_balanced \
  --selection_metric nll \
  --epochs 200 \
  --batch_size 128 \
  --lr 1e-4 \
  --device cuda
```

The generated `.pt.json` report contains `train_by_trace`,
`validation_by_trace` and `test_by_trace` metrics including:

- axis bias,
- axis RMSE,
- norm RMSE,
- predicted sigma mean.

## 5. Offline checkpoint evaluation

```bash
./.conda-env/bin/python \
  scripts/estimation/evaluate_learned_motion_checkpoint.py \
  --checkpoint artifacts/imo_tcn/model_v6_1_coupled_balanced.pt \
  --manifest config/learned_inertial/imo_v6_1_manifest.json \
  --split all \
  --device cuda \
  --output artifacts/imo_tcn/v6_1/checkpoint_eval.json
```

Do not accept V6.1 based only on aggregate RMSE. Inspect per-trajectory bias,
especially Lissajous and racing-like validation/test traces.

## 6. Estimator acceptance sequence

Only after offline evaluation passes:

1. 30 s, five-seed Lissajous, Network vs IMU-only A vs Oracle.
2. 30--60 s racing-like, multiple seeds.
3. Re-audit fusion-selected bias and covariance.
4. Revisit the V6 measurement sigma floor only after the mean error is stable.
5. Freeze the inertial core only when learned fusion is consistently beneficial.

Gate PnP and estimator-in-loop RL remain downstream of this acceptance.

## Blackbird

Blackbird is a valuable later external real-world benchmark because it contains
aggressive multirotor motion, 100 Hz IMU, motor-speed telemetry and accurate
motion-capture ground truth. It should not replace the V6.1 Isaac dataset:

- the airframe, mass, motors, propellers and IMU differ from our platform;
- motor RPM must be converted/calibrated into the thrust representation used by
  our network;
- mixing uncalibrated Blackbird thrust with Isaac mass-normalized thrust would
  create a domain inconsistency.

Recommended use after V6.1:

1. Build a Blackbird -> V6 feature/target converter.
2. Validate V6/V6.1 zero-shot on selected aggressive trajectories.
3. Calibrate RPM -> mass-normalized collective thrust.
4. Use Blackbird for real-data fine-tuning/domain adaptation while retaining
   held-out Blackbird trajectories for external evaluation.
