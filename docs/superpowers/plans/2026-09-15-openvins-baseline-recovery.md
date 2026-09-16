# OpenVINS Baseline Recovery Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a controlled OpenVINS GT-initialization diagnostic and source-identity guard so initialization can be isolated from propagation/visual-update failure.

**Architecture:** Keep the estimator external. Add a pure Isaac-side conversion contract, a ROS2 diagnostic publisher, a guarded external OpenVINS patch, and one true-static config variant. The existing runner enables GT initialization only through an explicit flag and preserves all existing behavior otherwise.

**Tech Stack:** Python 3.10, NumPy, ROS2 Humble (`rclpy`, `nav_msgs`), Bash, OpenVINS C++/Eigen, pytest, GitHub Actions.

**Spec:** `docs/superpowers/specs/2026-09-15-openvins-baseline-recovery-design.md`

## Global Constraints

- Work only on `debug/openvins-imu-convention-isolation`; do not modify `master` or `swift1`.
- Isaac truth is allowed only for diagnostic initialization and offline evaluation; it must not enter the deployable policy observation path.
- Do not change gravity sign, camera extrinsics, IMU axes, or OpenVINS propagation equations in this phase.
- Existing behavior must remain unchanged unless `--gt_initialize_openvins` is explicitly supplied.
- External OpenVINS patching must be guarded by source/API checks; never blindly modify an unknown workspace.
- Full Isaac/OpenVINS verification is required before claiming the VIO failure fixed.

---

### Task 1: Pure GT-initialization conversion contract

**Files:**
- Create: `estimation/openvins_gt_init.py`
- Create: `tests/estimation/test_openvins_gt_init.py`
- Modify: `.github/workflows/debug-opvs-pure-tests.yaml`

**Interfaces:**
- Produces: `OpenVinsGtInitialization` dataclass.
- Produces: `build_openvins_gt_initialization(timestamp_s, position_w_i, orientation_w_i_wxyz, linear_velocity_w_i, gyro_bias_i=None, accel_bias_i=None) -> OpenVinsGtInitialization`.
- The returned OpenVINS quaternion is scalar-first `q_GtoI`; input quaternion is scalar-first active `q_WI`/body-to-world.

- [ ] **Step 1: Write the failing pure test**

Test identity orientation, 90-degree yaw inversion, position/velocity preservation, default zero biases, normalization, and invalid zero quaternion rejection. Load the file directly with `importlib.util.spec_from_file_location` so the heavy `estimation/__init__.py` is not executed in pure CI.

- [ ] **Step 2: Run the new test and verify RED**

Run `pytest -q tests/estimation/test_openvins_gt_init.py`; expected failure is missing `estimation/openvins_gt_init.py`.

- [ ] **Step 3: Implement the minimal pure module**

Normalize the input quaternion, conjugate it to obtain `q_GtoI`, validate finite values and shapes, and store copied NumPy vectors.

- [ ] **Step 4: Extend pure CI and verify GREEN**

Compile the new module and run the new test alongside existing OpenVINS pure-contract tests.

### Task 2: Diagnostic ROS publisher and runner flag

**Files:**
- Modify: `estimation/openvins_bridge.py`
- Modify: `scripts/estimation/run_openvins_fault_isolation.py`

**Interfaces:**
- `OpenVinsRos2Bridge.publish_gt_initialization(...)` publishes `nav_msgs/msg/Odometry` on `/ov_msckf/initialize_gt` using normal ROS pose semantics (body/IMU-to-world quaternion) and world-frame linear velocity.
- Runner flag: `--gt_initialize_openvins`.

- [ ] **Step 1: Reuse Task 1 conversion tests as the frame-contract oracle**

Do not duplicate quaternion math inside the runner. The bridge message remains ROS pose semantics; the external OpenVINS callback performs the inversion before calling `initialize_with_gt`.

- [ ] **Step 2: Add the publisher behind an opt-in method**

Create the diagnostic publisher during bridge construction. `publish_gt_initialization` validates finite arrays, fills timestamp/pose/twist, and publishes exactly one message per call.

- [ ] **Step 3: Add runner flag and repeat-until-observed behavior**

When the flag is set and no OpenVINS estimate exists, publish current truth on each control step for at most 2 simulated seconds. Record request state and elapsed time in the JSON summary. With the flag absent, execute the existing path exactly.

- [ ] **Step 4: Compile-check bridge and runner in CI**

No ROS runtime test is claimed in GitHub Actions; py_compile is the pure CI boundary.

### Task 3: Guarded OpenVINS workspace inspection and source patch

**Files:**
- Create: `scripts/estimation/inspect_openvins_workspace.sh`
- Create: `patches/openvins/ros2_gt_initialization.patch`
- Create: `scripts/estimation/apply_openvins_gt_init_patch.sh`
- Create: `patches/openvins/README.md`

**Interfaces:**
- Inspector accepts optional workspace root, default `~/openvins_ws`, and prints package path, Git root, remote(s), HEAD SHA, dirty status, and presence of `VioManager::initialize_with_gt`.
- Apply script accepts an OpenVINS Git root and uses `git apply --check` before modifying anything.
- Patch adds a ROS2 `nav_msgs/Odometry` subscriber `/ov_msckf/initialize_gt`. It ignores messages after initialization, converts ROS `q_ItoG` to OpenVINS `q_GtoI` by quaternion conjugation, copies world position/velocity, sets biases zero, and calls `initialize_with_gt`.

- [ ] **Step 1: Write shell scripts with fail-closed guards**

Require a Git worktree and expected OpenVINS files/API before applying. If checks fail, exit non-zero without touching the workspace.

- [ ] **Step 2: Add patch against the public upstream layout**

Target `ov_msckf/src/ros/ROS2Visualizer.h` and `.cpp`. Keep the patch diagnostic-only and independent of the estimator core algorithms.

- [ ] **Step 3: Validate shell syntax in CI**

Run `bash -n` on both helpers.

### Task 4: True-static initialization ablation

**Files:**
- Create: `config/openvins/swift_sim/estimator_config_debug_true_static.yaml`
- Modify: `tests/estimation/test_openvins_config.py`

**Interfaces:**
- Mono simulator configuration based on current strict-static config.
- `init_dyn_use: false` disables the dynamic initializer.
- `try_zupt: true` is intentionally retained because upstream `VioManager::try_to_initialize()` sets `wait_for_jerk = (updaterZUPT == nullptr)`; without the ZUPT updater a stationary experiment can wait for motion instead of initializing at standstill.
- `zupt_only_at_beginning: true` keeps the updater limited to startup.
- Preserve known simulator intrinsics/extrinsics and current IMU noise baseline.

- [ ] **Step 1: Add config assertions first**

Test that the file exists, disables dynamic initialization, keeps beginning-only ZUPT, keeps FEJ/RK4, mono camera count, and references the same IMU/camera chain files.

- [ ] **Step 2: Verify RED**

The test should fail because the config file does not yet exist.

- [ ] **Step 3: Add the minimal configuration**

Copy the current strict-static configuration and change only `init_dyn_use` to `false`, with comments documenting why ZUPT remains present at startup.

- [ ] **Step 4: Verify config tests GREEN**

Run the relevant pure tests.

### Task 5: Experiment documentation and verification

**Files:**
- Create: `docs/OPENVINS_GT_INITIALIZATION_EXPERIMENT.md`
- Modify: `.github/workflows/debug-opvs-pure-tests.yaml`

**Interfaces:**
- Documents exact terminal commands for automatic-init and GT-init matched hover runs, workspace inspection, guarded patch application, rebuild, and timestamp-aligned comparison.

- [ ] **Step 1: Document the experiment contract**

Require restart/rebuild of OpenVINS after patching and restart estimator between A/B runs. Define interpretation: GT stable => initializer fault; GT still diverges => propagation/visual-update/calibration fault.

- [ ] **Step 2: Run/verify pure CI**

Expected checks: Python compile, shell syntax, config tests, schedule tests, GT-init conversion tests.

- [ ] **Step 3: Project-machine verification**

Run one baseline automatic-init hover and one GT-init hover with identical post-init conditions. Use `analyze_openvins_time_alignment.py` for comparison. Do not claim success until the original position/velocity divergence is reproduced or removed under this controlled A/B.
