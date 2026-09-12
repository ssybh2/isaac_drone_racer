# Swift1 estimator safety fixes — 2026-09-12

This patch closes the estimator/runtime issues found during the `swift1` review.

## Fixed

1. **OpenVINS twist frame semantics** — `odomimu.twist.twist.linear` is treated as `v_IinI` and rotated through the published IMU orientation before V->W alignment.
2. **Camera/VIO timestamp alignment** — gate updates use a short VIO history and interpolate the VIO pose at the image timestamp. Out-of-sequence measurements older than the drift-filter state fail closed instead of being fused at the current time.
3. **30 Hz scheduler drift** — source-rate gating now uses ideal deadlines. A 30 Hz camera sampled from a 100 Hz render clock produces a 30/40 ms quantized pattern with a 30 Hz long-run average rather than degrading to 25 Hz.
4. **Double camera throttling** — OpenVINS `track_frequency` is raised to 100 Hz because the repository already gates the source stream to 30 Hz.
5. **Gate outlier rejection** — nominal IPPE reprojection has a finite 5 px default limit; the drift Kalman filter applies a 3-D Mahalanobis/NIS gate and uses a Joseph covariance update.
6. **Isaac/OpenVINS runtime wiring** — `Isaac-Drone-Racer-Swift-OpenVINS-v0` publishes Isaac IMU at 200 Hz and rendered RGB at a phase-preserved 30 Hz, consumes `/ov_msckf/odomimu`, and performs one-time timestamp-matched simulator alignment.
7. **200 Hz odometry backlog** — the ROS bridge now drains queued OpenVINS odometry callbacks before each 100 Hz control/fusion update. Without this, a single `spin_once()` per control cycle could service at most half of the propagated odometry callbacks and gradually consume stale state.
8. **Oracle gate identity leak** — learned detections are unlabeled by default and are associated to the closest gate in the known track layout using the timestamp-aligned VIO pose, matching the Swift method. Isaac `next_gate_idx` is available only behind the explicit `swift_use_oracle_gate_index=True` diagnostic ablation switch.
9. **Asynchronous Kalman process noise** — Swift's `sigma_pos=0.05` and `sigma_vel=0.1` remain the covariance added over one nominal 100 Hz interval, but process noise is scaled by elapsed time. Splitting a 10 ms interval around a camera-time update no longer injects the full process covariance twice, and repeated prediction at the same timestamp adds no noise.
10. **Camera calibration drift guard** — the validated 256x256 pinhole contract (`fx=fy=293.19970703125`, `cx=cy=128`) is centralized in `perception/stage2_calibration.py`. The OpenVINS diagnostic checks Isaac's runtime resolution and intrinsic matrix before publishing the first frame and fails closed on mismatch. Regression tests also check that `kalibr_imucam_chain.yaml` stays synchronized with both the intrinsics and `T_cam_imu = inverse(T_bc)` convention.
11. **Rejected-frame corpus** — the optional detector/fusion runtime can persist every consumed rejected observation as the matching RGB PNG plus JSON containing the corner coordinates, visibility/confidence, rejection reason, NIS/Mahalanobis value, associated gate index, and nominal reprojection RMSE. This provides the data needed to tune visibility, IPPE and innovation thresholds instead of guessing from aggregate metrics.
12. **Stationary OpenVINS startup** — the diagnostic sends a stationary hover command. Upstream OpenVINS sets `wait_for_jerk=true` when no ZUPT updater exists, which can leave a genuinely stationary simulation waiting indefinitely for an acceleration jerk. The simulator config now enables ZUPT only during the beginning/static initialization phase (`try_zupt=true`, `zupt_only_at_beginning=true`), allowing static initialization and disabling zero-velocity updates after motion begins.
13. **Calibrated learned visibility guard** — validation showed that Keypoint R-CNN heatmap confidence alone cannot reject partially visible gates safely (56.9% partial false accepts at the safest tested threshold). The compact detector's dedicated visibility head now guards the R-CNN coordinates. On the held-out seed-2 validation split, threshold `0.75` gives 0.89% partial false accepts and 97.8% complete-sample recall. The same accepted set gives a robust ordinary corner noise estimate of `0.8472 px`, which is now used by the 20-sample IPPE covariance propagation path.
14. **Measured static-initialization tolerance** — a deterministic attitude/position hold keeps the simulated vehicle within 1.6 cm over three seconds, while OpenVINS reports 8–13 px aggregate KLT disparity from the rendered scene and small attitude corrections. The startup-only `init_max_disparity` is therefore calibrated to `15 px`; the static initializer still requires low IMU variance, and beginning-only ZUPT keeps its stricter post-initialization disparity gate.
15. **Fully-visible gate diagnostic start** — gate actor origins are at floor level, not at the opening center. The deterministic diagnostic pose now derives its height from the authoritative calibrated gate geometry (`center_g.z=1.0668 m`) plus gate 0's world z. It also starts 4 m before the gate because the outer frame is cropped at 3 m in the 256x256 image, which correctly causes the calibrated all-corners-visible guard to reject the observation.
16. **Delayed-camera FIFO** — the measured OpenVINS output latency is about 0.10 s, longer than the 30 Hz camera interval. Camera observations now wait in a two-second timestamp-ordered FIFO instead of overwriting one pending slot. Frames are fused only after VIO brackets their source time; frames older than the first aligned VIO sample are discarded because causal interpolation is impossible.

## Diagnostic run

Start OpenVINS using `config/openvins/swift_sim/estimator_config.yaml`, then launch:

```bash
python3 scripts/estimation/run_openvins_diagnostic.py --headless --steps 5000

# Exercise learned corners -> IPPE -> timestamp-aligned Kalman fusion and
# retain rejected frames for inspection/tuning.
python3 scripts/estimation/run_openvins_diagnostic.py --headless --steps 5000 \
  --detector_checkpoint artifacts/stage2_next_steps_20260911/checkpoints/torchvision_keypointrcnn_best.pt \
  --visibility_checkpoint artifacts/stage2_next_steps_20260911/checkpoints/gate_keypoint_net_best.pt \
  --rejection_dump_dir outputs/swift_rejections \
  --rejection_dump_limit 200 \
  --output outputs/swift_openvins_diagnostic.json

# Controlled association ablation only; normal operation should omit this.
python3 scripts/estimation/run_openvins_diagnostic.py --headless --steps 5000 \
  --detector_checkpoint artifacts/stage2_next_steps_20260911/checkpoints/torchvision_keypointrcnn_best.pt \
  --oracle_gate_index
```

Check transport rates separately:

```bash
ros2 topic hz /swift/imu
ros2 topic hz /swift/camera/image_raw
ros2 topic hz /ov_msckf/odomimu
ros2 topic echo /ov_msckf/odomimu --once
```

During the diagnostic, `OpenVINS/drained_odom_callbacks` should normally remain small (about 0-3 callbacks per 100 Hz control cycle after startup) and `OpenVINS/age_s` should stay bounded rather than growing monotonically. A persistent drain count at the configured maximum indicates the ROS consumer is not keeping up. `OpenVINS/camera_contract_validated` must become `1.0` after the first published RGB frame.

The detector/fusion path uses VIO/map association by default. Set `--oracle_gate_index` only for a controlled ablation that intentionally supplies task-level gate identity. When `--rejection_dump_dir` is enabled, rejected images and JSON metadata are written as paired files and the runtime exposes `SwiftFusion/rejection_dump_count`.

The diagnostic environment is not a PPO training task. If Isaac resets/teleports the vehicle, restart `ov_msckf`; the environment exposes `OpenVINS/restart_required` in logs.

## Validated local result

The 2026-09-12 RTX 4090 run in
`artifacts/swift1_openvins_diagnostic/hybrid_1000_steps.json` completed all
1,000 control steps and initialized OpenVINS at step 385. Of 181 processed
camera observations, 51 complete observations entered the Kalman update and
all 130 partial observations failed closed before IPPE. Fused translation RMSE
was `1.036 m`, versus `1.555 m` for raw VIO. The fused second-difference median
was `0.374 m`, versus `0.880 m` for raw mapped gate poses; far-gate covariance
trace was also larger than near-gate covariance trace. The maximum fused/VIO
orientation difference was exactly zero, confirming that gate pose never
replaced VIO attitude.

The absolute error in this stationary stress case remains dominated by
OpenVINS translation drift after its beginning-only ZUPT phase. This run is an
estimator/perception acceptance diagnostic, not a claim of closed-loop racing
performance and not authorization to start PPO training.

## Regression tests

```bash
PYTHONPATH=. pytest -q \
  tests/estimation/test_openvins_config.py \
  tests/estimation/test_openvins_bridge.py \
  tests/estimation/test_vio_time_buffer.py \
  tests/estimation/test_swift_vio_drift.py \
  tests/estimation/test_swift_fusion.py \
  tests/perception/test_stage2_calibration.py \
  tests/perception/test_swift_gate_measurement.py
```

Isaac/ROS integration still requires the runtime machine because CI does not provide Isaac Sim, RTX sensors, or ROS2 OpenVINS.
