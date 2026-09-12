# Swift1 estimator safety fixes — 2026-09-12

This patch closes the estimator/runtime issues found during the `swift1` review.

## Fixed

1. **OpenVINS twist frame semantics** — `odomimu.twist.twist.linear` is treated as `v_IinI` and rotated through the published IMU orientation before V->W alignment.
2. **Camera/VIO timestamp alignment** — gate updates use a short VIO history and interpolate the VIO pose at the image timestamp. Out-of-sequence measurements older than the drift-filter state fail closed instead of being fused at the current time.
3. **30 Hz scheduler drift** — source-rate gating now uses ideal deadlines. A 30 Hz camera sampled from a 100 Hz render clock produces a 30/40 ms quantized pattern with a 30 Hz long-run average rather than degrading to 25 Hz.
4. **Double camera throttling** — OpenVINS `track_frequency` is raised to 100 Hz because the repository already gates the source stream to 30 Hz.
5. **Gate outlier rejection** — nominal IPPE reprojection has a finite 5 px default limit; the drift Kalman filter applies a 3-D Mahalanobis/NIS gate and uses a Joseph covariance update.
6. **Isaac/OpenVINS runtime wiring** — `Isaac-Drone-Racer-Swift-OpenVINS-v0` publishes Isaac IMU at 200 Hz and rendered RGB at a phase-preserved 30 Hz, consumes `/ov_msckf/odomimu`, and performs one-time timestamp-matched simulator alignment.

## Diagnostic run

Start OpenVINS using `config/openvins/swift_sim/estimator_config.yaml`, then launch:

```bash
python3 scripts/estimation/run_openvins_diagnostic.py --headless --steps 5000

# Optional: also exercise learned corners -> IPPE -> time-aligned Kalman fusion
python3 scripts/estimation/run_openvins_diagnostic.py --headless --steps 5000 \
  --detector_checkpoint artifacts/stage2_next_steps_20260911/checkpoints/torchvision_keypointrcnn_best.pt
```

Check transport rates separately:

```bash
ros2 topic hz /swift/imu
ros2 topic hz /swift/camera/image_raw
ros2 topic echo /ov_msckf/odomimu --once
```

The diagnostic environment is not a PPO training task. If Isaac resets/teleports the vehicle, restart `ov_msckf`; the environment exposes `OpenVINS/restart_required` in logs.

## Pure-Python regression tests

```bash
PYTHONPATH=. pytest -q \
  tests/estimation/test_openvins_bridge.py \
  tests/estimation/test_vio_time_buffer.py \
  tests/estimation/test_swift_vio_drift.py \
  tests/estimation/test_swift_fusion.py \
  tests/perception/test_swift_gate_measurement.py
```

Isaac/ROS integration still requires the runtime machine because CI does not provide Isaac Sim, RTX sensors, or ROS2 OpenVINS.
