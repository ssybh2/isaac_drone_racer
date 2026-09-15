# OpenVINS Baseline Recovery Design

## Status

Approved direction: recover a trustworthy OpenVINS baseline before adding learned residual correction or gate-based absolute correction. The current evidence shows severe position/velocity divergence with much smaller but real roll/pitch errors. GUI versus headless behavior is equivalent, timestamp-aligning truth to the estimator timestamp does not remove the position divergence, and disabling ZUPT does not immediately cure the failure.

## Goal

Determine whether the dominant failure originates in initialization, IMU propagation, or visual correction, using controlled diagnostics that do not change estimator physics until the failing layer is identified.

## Scope

This phase implements four things:

1. identify the exact local OpenVINS source repository and commit used by `~/openvins_ws`;
2. add a diagnostic-only ROS2 ground-truth initialization path that calls OpenVINS `VioManager::initialize_with_gt()` once, with Isaac truth used only at initialization time;
3. add a truly static/UZH-style configuration ablation that removes dynamic initialization and keeps estimator changes isolated;
4. compare automatic initialization versus GT initialization with the same post-initialization hover and translation experiments.

Learned residual estimation and gate-map absolute correction are explicitly deferred to later phases. They remain the intended architecture after the raw VIO baseline is understood.

## Current evidence

The matched hover and post-init translation tests show that the estimator diverges even without deliberate translation. Timestamp alignment reduces the apparent orientation error but barely changes the position/velocity error, so the approximately 97 ms odometry age is not the primary cause of the divergence. In the post-init translation run, the simulator truth stays near the commanded trajectory while OpenVINS position diverges by tens to more than one hundred metres.

The upstream OpenVINS implementation exposes `VioManager::initialize_with_gt(Eigen::Matrix<double,17,1>)`, where the state is ordered as `[time, q_GtoI, p_IinG, v_IinG, b_gyro, b_accel]`. The upstream simulator already uses this entry point, making it suitable for a diagnostic experiment rather than a custom estimator modification.

## Architecture

### 1. Source identity guard

A shell helper inspects `~/openvins_ws/src`, reports each Git remote and HEAD SHA, locates the package containing `ov_msckf`, and verifies that the source contains the expected `initialize_with_gt` API before any external-source patch is applied.

No source patch is applied blindly to an unknown OpenVINS revision.

### 2. Diagnostic GT initialization message

The Isaac-side bridge publishes one diagnostic `nav_msgs/Odometry` message on `/ov_msckf/initialize_gt` while OpenVINS is still uninitialized. The message uses normal ROS pose semantics:

- timestamp: current simulator sensor time;
- pose position: IMU/body origin in simulator world frame;
- pose orientation: active IMU/body-to-world quaternion `(x,y,z,w)`;
- twist linear: IMU/body linear velocity in world coordinates;
- biases: assumed zero for this simulator diagnostic because the current simulated sensor model does not inject a known non-zero persistent hardware bias state.

The OpenVINS ROS2 patch converts the ROS body-to-world quaternion into OpenVINS `q_GtoI` by inversion/conjugation, fills the 17-state vector, sets both bias vectors to zero, and calls `initialize_with_gt()` only if the estimator is not already initialized.

The topic is diagnostic-only and is not part of the deployable real-vehicle interface.

### 3. Experiment runner behavior

`run_openvins_fault_isolation.py` gains an opt-in `--gt_initialize_openvins` flag. When enabled it republishes the current truth initialization message until an OpenVINS estimate is observed or a short initialization timeout is reached. Existing behavior is unchanged when the flag is omitted.

The JSON summary records whether GT initialization was requested and how long it took before the first OpenVINS estimate was observed.

### 4. True-static/UZH-style ablation

A separate OpenVINS YAML variant is added for initialization diagnosis. Relative to the current strict-static debug configuration it changes one conceptual variable set:

- `init_dyn_use: false`;
- `try_zupt: false` for the clean initializer comparison;
- preserves the current camera model, IMU noise baseline, MSCKF representation, and fixed simulator extrinsics.

This is intentionally not a wholesale copy of the UZH-FPV outdoor configuration: the UZH setup is stereo and has dataset-specific masks/calibration, while this simulator is mono with known synthetic intrinsics/extrinsics. The ablation borrows the initialization strategy, not unrelated dataset details.

## Decision tree

### GT initialization is stable

If GT initialization changes 10-20 s post-init drift from metres/tens of metres to a small bounded error while the same sensors and visual update path are used, the primary problem is initializer-derived attitude/velocity/bias state. The next work item is to correct the normal initialization path rather than propagation.

### GT initialization still diverges

If GT initialization produces similar drift, initialization is not the dominant cause. The next phase instruments:

- OpenVINS propagated attitude, velocity, accelerometer bias and gyro bias;
- MSCKF accepted/rejected feature/update counts;
- camera/IMU measurement timestamps used by each visual update;
- expected specific force from Isaac truth versus the IMU sample actually sent to OpenVINS.

Only after this evidence identifies a mismatch do we modify propagation, calibration, or visual-update logic.

## Later architecture after baseline recovery

Once the raw VIO baseline is healthy enough to characterize rather than catastrophically diverging, two additional correction layers are planned:

1. **Learned residual estimator**: a TCN or similar temporal model consumes IMU, thrust/action history, OpenVINS state/covariance, and visual-health statistics. Isaac truth is used only as the training label. The model predicts relative displacement/state residual plus uncertainty, which enters a secondary EKF as a measurement rather than directly replacing state.
2. **Gate absolute correction**: known gate world geometry plus camera PnP provides intermittent absolute pose constraints. These correct long-term VIO/map alignment drift independently of the learned residual model.

The deployable policy never receives Isaac truth.

## Acceptance criteria for this phase

- The exact local OpenVINS Git remote and HEAD can be printed with one command before patching.
- The Isaac-side GT initializer conversion has pure tests for quaternion inversion, position/velocity preservation, zero biases, and invalid quaternion rejection.
- The external OpenVINS patch refuses repeated initialization once `VioManager::initialized()` is true.
- Existing OpenVINS diagnostic behavior is unchanged unless `--gt_initialize_openvins` is provided.
- Pure GitHub Actions checks pass.
- On the project machine, one automatic-init hover and one GT-init hover are run with identical sensor/config settings and compared with timestamp-aligned metrics.
- No claim that the VIO bug is fixed is made until a full Isaac/OpenVINS rerun verifies the original failure mode.
