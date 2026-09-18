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

## What is implemented in the first branch slice

### 1. Standalone EKF

`estimation/learned_inertial_odometry.py`

State:

```text
R_wb, v_wb, p_wb, b_a, b_g
```

Propagation uses gyroscope and accelerometer measurements. A cloned
window-start position gives the learned displacement update the relative
measurement:

```text
r = dp_NN - (p_j - p_i)
H = [ ... +I(current p) ... -I(cloned p) ]
```

The module also accepts absolute position/orientation anchors.

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

## Important MVP limitation

The paper performs overlapping learned-displacement updates at 20 Hz while
keeping multiple past filter states. The first implementation on this branch
uses one cloned position and therefore non-overlapping 0.5 s updates (2 Hz).

Do not call the estimator paper-equivalent yet. The next estimator milestone is
a fixed-lag multi-clone covariance state so a new clone can be created every
0.05 s and the 0.5 s learned constraint can be applied at 20 Hz.

## Recommended experiment order

Do not train the final racing policy first.

1. Collect many independent trajectory files. Include translate-X, circle,
   Lissajous, braking, mixed yaw rates, different speeds and later racing-like
   trajectories.
2. Split by whole trajectory, never by adjacent windows from the same trace.
3. Train the TCN and validate held-out displacement RMSE/calibration.
4. Run estimator-only evaluation with no camera:
   IMU propagation vs IMU+TCN.
5. Add Stage2 Gate-PnP anchors and compare drift between visible-gate intervals.
6. Upgrade the EKF to 20 Hz overlapping multi-clone updates.
7. Only after the estimator is stable, train PPO/SKRL with estimator-backed
   actor observations.

## Example data collection

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
