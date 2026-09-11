# SWIFT1 next-step handoff for local Codex

Branch: `swift1`  
Base commit: `2286419abc3dc7e9c818e2d7b1aafec059644ce9`  
Primary reference: Kaufmann et al., **Champion-level drone racing using deep reinforcement learning**, Nature 620, 982-987 (2023), DOI `10.1038/s41586-023-06419-4`.

## Goal

Build the Swift-2023-style perception/state-estimation layer on top of the
already validated Stage2A geometry, camera extrinsics, and IPPE backend.

Do **not** start RL training in this phase. The immediate goal is to show that:

1. the existing fake VIO provides a smooth high-rate state with drift;
2. learned gate corners produce a noisy mapped world-pose measurement;
3. 20 perturbed-corner IPPE solves produce a meaningful measurement covariance;
4. the Kalman filter trusts near/clear gate measurements more than far/uncertain measurements;
5. the fused state is smoother and more accurate than feeding raw PnP pose directly to the policy.

## Repository audit before SWIFT1

### Known track layout already exists, but was implicit

`tasks/drone_racer/drone_racer_env_cfg.py` creates all seven gates from a fixed
`track_config`. At runtime the `RigidObjectCollection` therefore already
contains the known world pose of every gate.

`tasks/drone_racer/mdp/commands.py` also tracks the current `next_gate_idx`.

`perception/isaac_adapter.py` already extracts the active gate actor/link pose
`T_wg` for Stage2A. It correctly avoids the COM pose because Stage2 gate
keypoints are calibrated in the gate actor frame.

Before this branch there was no explicit perception-side "track map" object.
SWIFT1 adds `perception.track_layout.TrackLayout` plus an Isaac adapter that
snapshots all mapped gate actor poses.

### VIO already exists in simulation

The repository already contains:

- `estimation/fake_vio.py`
- `estimation/fake_imu.py`
- `estimation/pipeline.py`
- `tasks/drone_racer/stage1_env.py`

`FakeVio` already models bias, random-walk drift, measurement noise, source
latency, update timing, independent dropouts, and burst dropouts. The
`Stage1StatePipeline` publishes a world-aligned `StateEstimate`.

This is a **simulation VIO surrogate**, not a real image+IMU VIO algorithm.
That is acceptable for the next simulator stage because Swift's Kalman layer
only requires a VIO state interface. Future hardware VIO should satisfy the
same `VioWorldEstimate` contract rather than changing the fusion math.

## New SWIFT1 reference modules

### `perception/track_layout.py`

Explicit known-map abstraction.

Key identities:

```text
T_wc = T_wg @ inverse(T_cg)
T_wb = T_wc @ inverse(T_bc)
```

`TrackLayout.associate_to_nearest_gate(...)` implements the paper's map
association idea for unlabeled detections: every candidate `T_wg` implies a
body world pose, and the candidate closest to the current VIO position wins.

When the task already knows `next_gate_idx`, pass that index directly instead
of unnecessarily performing association.

### `perception/swift_isaac_adapter.py`

Builds a `TrackLayout` from every gate actor/link pose in the Isaac
`RigidObjectCollection`. Do not replace actor/link pose with COM pose.

### `perception/swift_gate_measurement.py`

Builds a gate-derived world-pose measurement from learned corners.

For every accepted gate observation:

1. run nominal IPPE and obtain noisy `T_cg`;
2. combine `T_cg`, known `T_wg`, and calibrated `T_bc` to obtain `T_wb_gate`;
3. perturb all four image corners;
4. run IPPE again for every perturbed set;
5. transform every sample to `T_wb`;
6. compute the sample covariance of the body world positions.

The default perturbation count is exactly **20**, matching the Nature paper.

Important: `corner_sigma_px=2.0` is a temporary repository-side starting value,
not a number claimed by the Swift paper. Before formal experiments, estimate
the corner-error distribution from the existing Stage2B validation/test data
and configure this value from data.

### `estimation/swift_vio_drift.py`

Reference implementation of the translational drift filter described in
Swift 2023.

State:

```text
x = [p_d, v_d] in R^6
```

Prediction:

```text
F = [[I, dt I],
     [0,    I]]

Q = [[sigma_pos I, 0],
     [0, sigma_vel I]]
```

Defaults follow the paper:

```text
sigma_pos = 0.05
sigma_vel = 0.1
```

Gate-derived body position gives a drift measurement:

```text
z = p_vio - p_gate
```

because final corrected position is:

```text
p_corrected = p_vio - p_d
v_corrected = v_vio - v_d
```

Gate/IPPE orientation does **not** replace VIO orientation. This is deliberate:
the paper states that only translational VIO components are refined because
VIO orientation is high quality and gate detections are lower rate.

Multiple gate measurements can be stacked in one update.

### `estimation/swift_fusion.py`

Reference orchestration:

```text
VIO ------------------------------+
                                   |
RGB -> detector -> corners -> IPPE |
        -> mapped pose -> R -------+-> VIO drift Kalman -> fused state
```

A missing/partial/failed gate measurement falls back to VIO prediction only.
Raw `T_cg` is never directly substituted for the policy state.

## Why this addresses the current "2 px -> large 3D error" issue

Do not try to force every single `T_cg` to be accurate.

The Swift approach asks how uncertain that `T_cg`-derived position is.

For the same image-plane corner sigma:

- a far/small gate produces a wider distribution of 20 IPPE world positions;
- therefore `R` is larger;
- Kalman gain becomes smaller;
- the estimator mostly follows VIO.

For a near/large gate:

- the same pixel perturbation produces a tighter 3D distribution;
- `R` is smaller;
- Kalman gain increases;
- the mapped gate gives a stronger VIO-drift correction.

This is the paper-faithful replacement for a hard-coded FAR/MID/NEAR switch.

## Required local validation before Isaac integration

Run:

```bash
PYTHONPATH=. pytest -q \
  tests/perception/test_swift_gate_measurement.py \
  tests/estimation/test_swift_vio_drift.py
```

Then run the existing perception suite:

```bash
PYTHONPATH=. pytest -q tests/perception
```

Also run Python compile/static checks used by this repository.

## Next local-Codex implementation task: single-environment diagnostic

After unit tests pass, create a diagnostic script; do not wire PPO yet.

Suggested file:

```text
scripts/perception/evaluate_swift_fusion.py
```

Use one Isaac environment with camera enabled and log at least:

```text
timestamp
active gate index
GT p_wb
raw Fake-VIO p_wb
raw gate-derived p_wb
fused p_wb
VIO position error
gate position error
fused position error
corner confidence
nominal IPPE reprojection RMSE
trace(R)
eigenvalues(R)
estimated p_d
estimated v_d
gate measurement accepted/rejected
```

Target source rates should mimic the paper as closely as practical:

```text
IMU:           200 Hz
VIO:           100 Hz
gate detector:  30 Hz
fusion output: 100 Hz
```

The existing Stage1 fake-sensor timing framework should be reused rather than
creating a second unrelated VIO simulator.

## Detector/outlier work required before closed-loop control

The 20-sample covariance handles ordinary corner uncertainty. It does not make
a catastrophic hallucinated corner safe.

Before online learned-detector fusion:

1. verify the semantics/calibration of torchvision `keypoints_scores`;
2. stop forcing every detected keypoint to `visible=True`;
3. reject partial/incomplete four-corner observations for IPPE;
4. add a configurable nominal reprojection-error check;
5. add a Kalman innovation/Mahalanobis gate for catastrophic mapped-pose outliers;
6. export rejected frames for inspection.

Items 1-6 are repository-specific robustness around the Swift estimator. Do
not describe them as exact claims from the paper.

## Acceptance criteria for the estimator phase

Do not begin policy training until all of these are demonstrated:

1. **Map reconstruction**: oracle corners + known `T_wg` recover `T_wb` to the
   Stage2A numerical tolerance.
2. **Uncertainty propagation**: for equal pixel sigma, the far-gate positional
   covariance is clearly larger than the near-gate covariance.
3. **Kalman weighting**: small `R` strongly corrects VIO drift; large `R`
   leaves the state mostly on VIO.
4. **Orientation ownership**: fused orientation is exactly VIO orientation;
   gate IPPE rotation is diagnostic only.
5. **Drift correction**: over a rollout, fused translation error should be
   better than raw VIO drift while avoiding the frame-to-frame jumps of raw
   gate PnP.
6. **Dropout behavior**: detector dropouts or rejected measurements continue on
   VIO without state discontinuity.

## After estimator acceptance: reproduce the Swift 31-D actor observation

Only after the estimator phase is validated, construct the paper-style actor
observation:

```text
15-D estimated platform state
12-D relative 3D positions of the next gate's four corners
 4-D previous action
--------------------------------
31 dimensions
```

The critic may retain privileged simulator truth during training.

At that point add the perception-aware reward that encourages the camera
optical axis to remain directed toward the next gate. Do not silently reuse
the current 3-D `target_pos_b` contract and call it a Swift reproduction.

## RL scaling note

The current OpenCV IPPE implementation and 20 CPU perturbation solves are the
**reference/validation backend**. Do not run them independently in thousands
of RL environments.

Once the single-environment estimator is validated, build a batched training
surrogate from empirical detector/PnP residuals, or a batched Torch backend,
while keeping the same measurement/fusion contract. The single-vehicle
reference remains the correctness oracle.

## Branch discipline

All work described here belongs on `swift1`.

Do not modify or force-update:

```text
feature/stage2a-perfect-pnp-framework
```

which remains the frozen Stage2A/Stage2B baseline at
`2286419abc3dc7e9c818e2d7b1aafec059644ce9`.
