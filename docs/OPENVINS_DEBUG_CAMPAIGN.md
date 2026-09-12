# OpenVINS fault-isolation campaign (`debug-opvs`)

This branch isolates the translation divergence seen in the 2026-09-12 5000-step run. In that run OpenVINS initialized successfully and ROS age/backlog stayed bounded, but the estimator position diverged by hundreds of metres while Isaac truth stayed near the start. The purpose of this branch is to determine whether the failure is dominated by startup bias estimation, ZUPT gating, low-parallax hover, or a more fundamental visual-inertial calibration problem.

## What changed

- `scripts/estimation/run_openvins_fault_isolation.py` records truth position/velocity/quaternion/RPY/body rates, raw Isaac IMU acceleration/gyro, OpenVINS position/velocity/quaternion, position/velocity/orientation error, ROS odometry age and drain count to CSV plus a compact JSON summary.
- The runner has three controlled motion profiles: `hover`, `translate_x`, and `lissajous`.
- `scripts/estimation/capture_openvins_topics.sh` records native-rate `/swift/imu` and `/ov_msckf/odomimu` to a ROS2 bag while measuring IMU/camera/odom topic rates and saving one human-readable sample from IMU and odometry.
- Three OpenVINS A/B configs avoid changing the `swift1` baseline config:
  - `estimator_config_debug_gated_zupt.yaml`: keeps the permissive 15 px startup disparity but changes beginning-only ZUPT from chi-square multiplier 0 to 1.
  - `estimator_config_debug_strict_static.yaml`: additionally tightens startup disparity to 3 px. It is expected to refuse static initialization if the nominal hover really has 10-12 px KLT motion.
  - `estimator_config_debug_dynamic.yaml`: disables startup ZUPT and relies on OpenVINS dynamic initialization; pair with `translate_x` or `lissajous`.

## Important environment contract

Use the repository's existing Isaac Sim 4.5 environment directly:

```bash
cd ~/isaac_projects/isaac_drone_racer
./.conda-env/bin/python -c 'import isaacsim, isaaclab; print(isaacsim.__file__); print(isaaclab.__file__)'
```

Do not use `IsaacLab-v2.1.0/isaaclab.sh` while its `_isaac_sim` symlink points at an Isaac Sim 5.1 standalone install.

Before each experiment, restart `ov_msckf`. An Isaac reset/teleport invalidates the external estimator trajectory.

## Experiment A — gated ZUPT, hover

Terminal 1:

```bash
cd ~/isaac_projects/isaac_drone_racer
source /opt/ros/humble/setup.bash
source ~/openvins_ws/install/setup.bash
ros2 launch ov_msckf subscribe.launch.py \
  config_path:=$(pwd)/config/openvins/swift_sim/estimator_config_debug_gated_zupt.yaml \
  use_stereo:=false max_cameras:=1 \
  2>&1 | tee artifacts/openvins_fault_isolation/gated_zupt_openvins.log
```

Terminal 2:

```bash
cd ~/isaac_projects/isaac_drone_racer
source /opt/ros/humble/setup.bash
source ~/openvins_ws/install/setup.bash
export PYTHONPATH="$(pwd):${PYTHONPATH}"
./.conda-env/bin/python scripts/estimation/run_openvins_fault_isolation.py \
  --headless --steps 3000 --profile hover \
  --output_dir artifacts/openvins_fault_isolation/gated_zupt_hover
```

Terminal 3, after Isaac starts publishing:

```bash
cd ~/isaac_projects/isaac_drone_racer
source /opt/ros/humble/setup.bash
bash scripts/estimation/capture_openvins_topics.sh \
  artifacts/openvins_fault_isolation/gated_zupt_hover/ros_capture 30
```

Interpretation: if this is much better than the `swift1` baseline, the forced/ungated beginning-only ZUPT was materially corrupting initialization.

## Experiment B — strict static initializer

Restart OpenVINS using:

```bash
config/openvins/swift_sim/estimator_config_debug_strict_static.yaml
```

Run the same `--profile hover` command. If OpenVINS refuses to initialize because KLT disparity remains near 10-12 px, that is useful evidence: the old 15 px threshold was classifying a visibly moving scene as static. Do not loosen the threshold just to obtain an initialization event.

## Experiment C — dynamic initialization with translation

Terminal 1 uses:

```bash
config/openvins/swift_sim/estimator_config_debug_dynamic.yaml
```

Terminal 2:

```bash
./.conda-env/bin/python scripts/estimation/run_openvins_fault_isolation.py \
  --headless --steps 4000 --profile translate_x \
  --motion_start_s 4 \
  --translation_m 1.5 \
  --translation_duration_s 8 \
  --output_dir artifacts/openvins_fault_isolation/dynamic_translate_x
```

If dynamic initialization plus deliberate parallax is stable while hover initialization diverges, the primary failure is startup observability/bias estimation rather than camera-IMU geometry.

## Experiment D — richer parallax

If experiment C is stable, run:

```bash
./.conda-env/bin/python scripts/estimation/run_openvins_fault_isolation.py \
  --headless --steps 5000 --profile lissajous \
  --motion_start_s 4 \
  --lissajous_amplitude_m 0.6 \
  --lissajous_frequency_hz 0.08 \
  --output_dir artifacts/openvins_fault_isolation/dynamic_lissajous
```

This is not a racing controller. It exists only to create repeatable translational parallax while logging estimator error.

## Decision tree

- `hover + gated ZUPT` stable, old baseline unstable: beginning-only ZUPT gating was a primary cause.
- strict static config refuses initialization: the 15 px threshold was too permissive for a claimed static start.
- dynamic translation stable while hover diverges: low-parallax/static initialization is the dominant issue.
- dynamic translation also diverges with low orientation error but growing velocity error: focus on accelerometer model/bias/timing.
- dynamic translation diverges with growing orientation error first: focus on gyro bias, camera-IMU timing/extrinsics and visual update acceptance.
- topic age grows or `drained_callbacks` saturates: revisit ROS transport before estimator tuning.

## Files to keep

For every run, keep the `*_summary.json`, `*_trace.csv`, OpenVINS console log, and the `ros_capture` directory. The CSV is intentionally the primary artifact: it lets us see whether orientation error, velocity error, or position error starts growing first instead of inferring the cause from final position RMSE.
