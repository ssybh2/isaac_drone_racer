# SWIFT1 OpenVINS + perception integration handoff

Branch: `swift1`

This document is the local-Codex handoff for the next estimator stage of the Swift-2023 reproduction. The goal is a single-vehicle reference pipeline, not large-scale PPO yet.

## External dependency

Use OpenVINS as an external ROS2 process. Do not vendor its GPLv3 source into this repository.

Pinned upstream revision:

```text
repository: https://github.com/rpng/open_vins.git
commit:     69488123ed9362dd44b6f28e7f4680abbff1442b
```

On Ubuntu 22.04 / ROS2 Humble:

```bash
mkdir -p ~/openvins_ws/src
cd ~/openvins_ws/src
git clone https://github.com/rpng/open_vins.git
cd open_vins
git checkout 69488123ed9362dd44b6f28e7f4680abbff1442b
cd ~/openvins_ws
source /opt/ros/humble/setup.bash
colcon build --symlink-install
source install/setup.bash
```

Launch with the repository config:

```bash
source /opt/ros/humble/setup.bash
source ~/openvins_ws/install/setup.bash

ros2 launch ov_msckf subscribe.launch.py \
  config_path:=$HOME/isaac_projects/isaac_drone_racer/config/openvins/swift_sim/estimator_config.yaml \
  use_stereo:=false \
  max_cameras:=1
```

Expected bridge topics:

```text
/swift/camera/image_raw     sensor_msgs/Image, RGB8, 256x256, target 30 Hz
/swift/imu                  sensor_msgs/Imu, target 200 Hz
/ov_msckf/odomimu           nav_msgs/Odometry from OpenVINS
```

`estimation/openvins_bridge.py` provides the repository side of this contract.

## Frame contract

OpenVINS publishes odometry in its own estimator/global frame `V` with child frame `imu`. The track is known in Isaac world frame `W`.

After OpenVINS initializes, obtain one reference vehicle pose in the known track frame and construct:

```python
alignment = OpenVinsFrameAlignment.from_reference_pose(
    openvins_sample,
    reference_position_w_b=start_position_w,
    reference_orientation_w_b_wxyz=start_quaternion_wxyz,
)
```

Simulator validation may use Isaac truth only for this one-time frame alignment. Hardware should use the surveyed start-pad pose. Do not keep using simulator truth after alignment.

The current Isaac IMU is mounted on the drone body, so `I == B` for the simulation diagnostic.

## Camera/IMU calibration

OpenVINS uses `T_cam_imu`, which maps IMU -> optical camera. The repository's Stage2 calibration stores `T_bc`, camera -> body. Since `I == B`:

```text
T_cam_imu = inverse(T_bc)
```

Do not accidentally pass `T_bc` directly.

The supplied OpenVINS camera file uses the validated 256x256 Stage2 pinhole calibration:

```text
fx = fy = 293.19970703125
cx = cy = 128
```

and the calibrated mount inverse:

```text
T_cam_imu =
[ 0 -1  0  0.00
  0  0 -1  0.05
  1  0  0 -0.14
  0  0  0  1.00 ]
```

The IMU-noise numbers currently use the OpenVINS UZH-FPV baseline only as an initial simulator configuration. Replace them with measured hardware IMU noise before real-flight experiments.

## Isaac integration task

Use `DroneRacerSwiftPerceptionEnvCfg`. It intentionally runs one environment and re-enables the 256x256 pinhole camera plus Isaac IMU.

The environment loop still needs to call `OpenVinsRos2Bridge` at source rates:

```text
physics loop:
  publish IMU when OpenVinsSensorRateGate.imu_due(t)       -> 200 Hz

rendered camera frame:
  publish RGB when OpenVinsSensorRateGate.camera_due(t)    -> 30 Hz

every control cycle:
  bridge.spin_once()
  latest OpenVinsOdomSample
  -> OpenVinsFrameAlignment.to_world()
  -> VioWorldEstimate
  -> SwiftPerceptionFusion
```

Do not call OpenVINS once per thousands of PPO environments. This is the single-vehicle correctness/reference path.

## R-CNN partial-corner fix

`TorchvisionGateCornerDetector` no longer forces `visible=np.ones(4)`.

Visibility is now conservative:

```text
finite pixel
AND inside image
AND keypoint confidence heuristic above threshold
AND complete ordered quad is convex / non-degenerate
```

Important: torchvision `keypoints_scores` are heatmap peak logits, not calibrated visibility probabilities. The code uses `sigmoid(logit) * instance_score` only as a monotonic gating heuristic.

A partial or rejected observation therefore fails `CornerObservation.complete`, so `GatePoseMeasurementBuilder` does not run IPPE/Kalman correction for that frame. OpenVINS continues as the VIO backbone.

Still required before closed-loop racing:

1. tune the keypoint confidence threshold on held-out complete/partial data;
2. add nominal IPPE reprojection rejection;
3. add Kalman innovation/Mahalanobis rejection;
4. export rejected images and reasons;
5. consider a dedicated learned visibility head if partial-gate recall remains poor, because Keypoint R-CNN does not provide a calibrated visibility head.

## 2-px corner error handling

Normal corner noise is handled at the Swift estimator boundary:

```text
nominal corners -> IPPE -> mapped body position
        +
20 perturbed corner sets -> IPPE x20 -> position covariance R
        +
OpenVINS VIO -> translational drift Kalman filter
```

Far/small gates naturally produce larger `R`, so the filter trusts OpenVINS. Near/clear gates naturally produce smaller `R`, so mapped gate observations correct more of the VIO translation drift.

Catastrophic/partial detections are not treated as Gaussian 2-px noise; they must be rejected before IPPE.

## Perception-aware reward

`mdp.swift_perception_awareness` implements the Swift 2023 term:

```text
r_perc = lambda_2 * exp(lambda_3 * delta_cam^4)
```

where `delta_cam` is the angle between the calibrated camera optical axis and the center of the next gate.

Use the paper values in the Swift reward config:

```text
lambda_2 = 0.02
lambda_3 = -10.0
```

The MDP function returns only `exp(lambda_3 * delta_cam^4)`; `SwiftPerceptionRewardsCfg` applies the IsaacLab RewardTerm weight `0.02`.

## Local tests

```bash
cd ~/isaac_projects/isaac_drone_racer
PYTHONPATH=. pytest -q \
  tests/perception/test_torchvision_visibility.py \
  tests/estimation/test_openvins_bridge.py \
  tests/perception/test_swift_gate_measurement.py \
  tests/estimation/test_swift_vio_drift.py
```

Then validate ROS transport:

```bash
ros2 topic hz /swift/imu
ros2 topic hz /swift/camera/image_raw
ros2 topic echo /ov_msckf/odomimu --once
```

Acceptance criteria before the 31-D PPO stage:

- OpenVINS initializes repeatably from Isaac RGB+IMU;
- aligned OpenVINS state is smooth in track world coordinates;
- partial detections do not enter IPPE;
- 20-sample covariance grows for far/small gates;
- Kalman fusion improves translational drift without replacing VIO attitude;
- camera-aware reward is available for the future Swift training config.
