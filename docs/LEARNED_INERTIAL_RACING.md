# Learned-Inertial Drone Racing (OpenVINS-free)

Branch: `feature/learned-inertial-racing`

Base branch: `debug/openvins-imu-convention-isolation`, which already contains
Stage2 gate detection/PnP/known-track infrastructure and is descended from
`feature/stage2a-perfect-pnp-framework`.

## Goal

Replace Raw OpenVINS as the primary estimator with an IMO-style learned
inertial odometry stack:

```text
fixed known start pose
        |
        v
IMU gyro + accelerometer --------------------------.
        |                                          |
        v                                          |
error-state EKF propagation                        |
        |                                          |
        +<-- TCN relative displacement <--- gyro + mass-normalized thrust
        |                                          |
        +<-- mapped gate PnP absolute pose <--- camera + Stage2 detector
        |
        v
estimated 6-DoF state (p, q, v, ba, bg)
        |
        v
RL racing policy
```

Simulator ground truth is allowed in two places only:

1. supervised dataset labels/evaluation;
2. the task's known fixed initial pose contract.

It must not enter the deployed estimator or policy observations after reset.

## What is implemented

### 1. 20 Hz fixed-lag multi-clone EKF

`estimation/learned_inertial_odometry.py`

Nominal state:

```text
R_wb, v_wb, p_wb, b_a, b_g
```

The current error-state block is 15-D:

```text
[dtheta, dv, dp, dba, dbg]
```

Every 0.05 s the runtime clones the current 3-D position. Clone augmentation
keeps the full current/clone and clone/clone cross-covariance. With a 0.5 s
learned window, the default 11-clone bank covers both fixed-lag endpoints.

Each learned displacement is fused as:

```text
z = dp_NN(i,j)
h(x) = p_j - p_i
r = z - h(x)
H_current_p = +I
H_historical_clone_i = -I
```

The used historical clone is marginalized after its relative update. IMU
propagation evolves only the current 15-D state dynamics while the augmented
transition preserves all cross-covariances. Absolute gate position/orientation
anchors remain compatible with the augmented state.

### 2. Supervised learned-motion model reuse

Existing modules are reused:

- `estimation/learned_motion.py`
- `estimation/learned_motion_dataset.py`
- `scripts/estimation/train_learned_motion.py`

The TCN input contract is:

```text
[gyro_w_x, gyro_w_y, gyro_w_z,
 thrust_w_x, thrust_w_y, thrust_w_z]
```

over 0.5 s, 100 Hz. The target is GT world-frame relative displacement
`p_gt(t+0.5)-p_gt(t)`. The network also predicts log variance.

### 3. Clean Isaac Lab supervised data collection

`scripts/estimation/collect_imo_supervised_data.py`

The collector stores:

- GT p/v/q for labels/evaluation;
- IMU gyro/accel;
- applied collective thrust;
- mass-normalized body thrust;
- control action.

Unlike the older hybrid-VIO collector, the new trace metadata explicitly marks
`thrust_b` as mass-normalized acceleration units.

### 4. Stage2 gate detector + known map reuse

The new runtime reuses:

- `TorchvisionGateCornerDetector`;
- optional visibility guard;
- `GatePoseMeasurementBuilder`;
- mapped `TrackLayout`;
- calibrated body-camera extrinsics.

An unlabeled gate observation is associated against the known map using the
current learned-inertial position. The resulting mapped PnP body pose becomes
an absolute EKF correction. A 3-DoF Mahalanobis gate rejects globally
inconsistent associations before the update.

### 5. RL observation boundary

`tasks/drone_racer/mdp/learned_inertial_observations.py`

The actor gets:

```text
estimated p_w
estimated q_wb
estimated body-frame velocity
measured body angular rate
next mapped gate position expressed using estimated pose
last action
```

The actor does not read Isaac root pose.

Rewards/terminations may still use simulator truth during training; that is an
RL training signal, not a deployable observation.

## Current task

```text
Isaac-Drone-Racer-Learned-Inertial-v0
```

Runtime:

```text
tasks/drone_racer/learned_inertial_racing_env.py
```

Config:

```text
tasks/drone_racer/drone_racer_learned_inertial_env_cfg.py
```

## Current estimator limitations

The runtime now uses overlapping 0.5 s learned-displacement windows at 20 Hz
with timestamped position clones. It is still not a claim of line-for-line
paper equivalence. Current remaining differences include:

- clones store historical position rather than a full historical pose state;
- the repository TCN predicts both displacement and log variance rather than
  reproducing the paper training code exactly;
- held-out V1 traces showed rare horizontal uncertainty collapse, so runtime
  covariance is protected by an axis-wise sigma floor and scale;
- gate PnP orientation uncertainty still uses a configurable fixed sigma;
- more yaw/braking/vertical/racing-like supervised trajectories are needed.

The default covariance protection is:

```text
sigma_floor_xyz_m = (0.10, 0.10, 0.01)
covariance_scale = 1.25
sigma_used = max(sigma_NN, sigma_floor)
R_used = diag((covariance_scale * sigma_used)^2)
```

## Recommended experiment order

Do not train the final racing policy first.

1. Collect many independent trajectory files. Include translate-X, circle,
   Lissajous, braking, mixed yaw rates, different speeds and later racing-like
   trajectories.
2. Split by whole trajectory, never by adjacent windows from the same trace.
3. Train the TCN and validate held-out displacement RMSE/calibration.
4. Run estimator-only evaluation with no camera:
   IMU propagation vs IMU+20 Hz TCN.
5. Add Stage2 Gate-PnP anchors and compare drift between visible-gate intervals.
6. Verify learned-update timing, clone count and skipped-update diagnostics.
7. Only after the estimator is stable, train PPO/SKRL with estimator-backed
   actor observations.

## IMO V2 dataset: explicit speed coverage and fixed splits

The first V1 dataset exposed an important generalization failure. Its
`translate_x` traces were 6000 steps long, so the legacy target ramp occupied
about 42 s. The later 3 m smoke evaluation used a 7 s target ramp. The V1 TCN
therefore saw much slower forward motion during training and under-predicted
the faster held-out displacement (roughly 0.26 * GT plus an offset in the
diagnostic linear fit).

V2 fixes this at the data contract level:

- motion duration is explicit and independent of trace length;
- quintic S-curve translation covers acceleration and braking;
- vertical, yaw, translate+yaw, circle, Lissajous and mixed racing-like motion
  are included;
- train/validation/test membership is declared before collection;
- the camera is disabled during supervised IMO collection because the TCN only
  needs IMU/thrust plus GT labels;
- dataset displacement coverage can be audited before training.

The canonical manifest is:

```text
config/learned_inertial/imo_v2_manifest.json
```

Collect all V2 traces in fresh Isaac processes:

```bash
./.conda-env/bin/python scripts/estimation/collect_imo_manifest.py \
  config/learned_inertial/imo_v2_manifest.json \
  --headless \
  --resume
```

Audit 0.5 s displacement coverage before training:

```bash
./.conda-env/bin/python scripts/estimation/audit_imo_dataset.py \
  config/learned_inertial/imo_v2_manifest.json \
  --output artifacts/imo_dataset_v2/audit.json
```

Train from the explicit manifest. Selecting the checkpoint by held-out RMSE is
recommended for the current displacement-quality milestone even though the
network is still optimized with Gaussian NLL:

```bash
./.conda-env/bin/python scripts/estimation/train_learned_motion.py \
  --manifest config/learned_inertial/imo_v2_manifest.json \
  --output artifacts/imo_tcn/model_v2.pt \
  --window_time_s 0.5 \
  --sample_rate_hz 100 \
  --stride_time_s 0.01 \
  --epochs 200 \
  --batch_size 128 \
  --lr 1e-4 \
  --device cuda \
  --selection_metric rmse
```

The training report now contains aggregate validation/test metrics plus
`validation_by_trace` and `test_by_trace` so held-out translation, yaw,
vertical and coupled-motion failures are visible rather than hidden in one
aggregate score.

## Legacy manual data collection

Run several separate traces so train/validation/test can be split at the
trajectory-file level:

```bash
cd ~/isaac_projects/isaac_drone_racer
git switch feature/learned-inertial-racing

./.conda-env/bin/python scripts/estimation/collect_imo_supervised_data.py \
  --headless --steps 6000 --profile translate_x \
  --output artifacts/imo_dataset/translate_x_01.csv

./.conda-env/bin/python scripts/estimation/collect_imo_supervised_data.py \
  --headless --steps 6000 --profile circle \
  --amplitude_m 1.5 --frequency_hz 0.10 \
  --output artifacts/imo_dataset/circle_01.csv

./.conda-env/bin/python scripts/estimation/collect_imo_supervised_data.py \
  --headless --steps 8000 --profile lissajous \
  --amplitude_m 1.5 --frequency_hz 0.10 \
  --output artifacts/imo_dataset/lissajous_01.csv
```

Repeat each profile with several seeds/trajectory parameters.

## Example TCN training

```bash
./.conda-env/bin/python scripts/estimation/train_learned_motion.py \
  artifacts/imo_dataset/*.csv \
  --output artifacts/imo_tcn/model.pt \
  --window_time_s 0.5 \
  --sample_rate_hz 100 \
  --stride_time_s 0.01 \
  --epochs 200 \
  --device cuda
```

## Diagnostic interpretation from the first 20 Hz smoke test

The 10 s 3 m translation diagnostic produced a useful separation:

```text
A  IMU only                    position RMSE ~0.284 m
S  IMU + TCN shadow            same state as A; TCN dp RMSE ~0.155 m
B  IMU + fused 20 Hz TCN       position RMSE ~1.65 m
P  B + Gate position only      position RMSE ~0.40 m, attitude drifted
C  B + Gate position+attitude  position RMSE ~0.40 m, attitude improved
```

The 20 Hz scheduler itself was healthy (20 Hz, ten retained clones, zero
skips). The shadow mode showed that the dominant issue was the V1 TCN's
held-out forward-displacement bias, not the clone cadence. Position-only gate
updates also demonstrated strong pose cross-covariance effects; the gate
orientation anchor reduced the resulting attitude drift, so it should not be
removed based on that test.

Do not tune covariance floors to hide a biased displacement model. Fix held-out
motion coverage first, retrain V2, then rerun the same diagnostics.

## Unified estimator A/B/C evaluation

Use the unified evaluator before training the racing policy. One invocation runs
three estimator modes with the same action replay:

```text
A: IMU propagation only
B: IMU + 20 Hz overlapping TCN
C: IMU + 20 Hz TCN + mapped Gate-PnP
```

Mode A generates the action sequence once with a deterministic GT-feedback
trajectory controller. Modes B and C replay those exact actions. Ground truth
is used only for evaluation and for generating the replay trajectory; it is not
fed into the estimator after the fixed known start.

Quick smoke evaluation:

```bash
./.conda-env/bin/python scripts/estimation/evaluate_learned_inertial_ab.py \
  --headless \
  --steps 1000 \
  --profile translate_x \
  --translation_m 3.0 \
  --learned-checkpoint artifacts/imo_tcn/model_v1.pt \
  --output-dir artifacts/learned_inertial_ab/smoke
```

Longer dynamic evaluation:

```bash
./.conda-env/bin/python scripts/estimation/evaluate_learned_inertial_ab.py \
  --headless \
  --steps 6000 \
  --profile lissajous \
  --amplitude_m 1.5 \
  --frequency_hz 0.10 \
  --learned-checkpoint artifacts/imo_tcn/model_v1.pt \
  --gate-checkpoint artifacts/stage2_next_steps_20260911/checkpoints/torchvision_keypointrcnn_best.pt \
  --visibility-checkpoint artifacts/stage2_next_steps_20260911/checkpoints/gate_keypoint_net_best.pt \
  --output-dir artifacts/learned_inertial_ab/lissajous_v1
```

Outputs:

```text
estimator_ab_trace.csv
estimator_ab_summary.json
```

The summary reports position/velocity/orientation RMSE, maximum position error,
learned-update frequency, clone count, skipped learned updates, TCN innovation
statistics, Gate-PnP attempts/accepted/rejected counts, and the GT replay
difference of modes B/C versus A. The replay-difference fields verify that the
three estimators were compared on the same physical trajectory.

## Estimator task with a trained checkpoint

The config currently leaves both learned-motion and gate-detector checkpoints
unset by default. Set them in a run/evaluation script before `gym.make`:

```python
env_cfg.learned_motion_checkpoint = "artifacts/imo_tcn/model.pt"
env_cfg.swift_detector_checkpoint = "<stage2 detector checkpoint>"
env_cfg.swift_visibility_checkpoint = "<optional visibility checkpoint>"
```

Then create:

```text
Isaac-Drone-Racer-Learned-Inertial-v0
```

## RL phase

The final policy should learn control, not repair estimator bugs.

Recommended actor observation:

```text
[p_est, q_est, v_est_body, omega_meas_body,
 next_gate_position_est_body, previous_action]
```

Add uncertainty/health only if experiments show it helps:

```text
trace(P_pose), learned measurement variance,
gate measurement age, gate accepted flag
```

The reward can remain simulator-ground-truth based during training:

- track progress;
- gate pass;
- collision penalty;
- angular-rate/action regularization;
- optional lap-time pressure.

Before claiming sensor-faithful racing, verify the trained actor with all
policy pose/velocity inputs sourced from the estimator, not Isaac root state.
