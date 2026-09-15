# Hybrid Learned Motion-Constrained VIO Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Insert a supervised learned relative-motion constraint between raw OpenVINS and the existing Swift gate-map drift correction.

**Architecture:** A UZH-inspired TCN predicts 0.5 s relative displacement and diagonal uncertainty from gyro+thrust history. A NumPy fixed-anchor drift EKF converts the difference between VIO displacement and learned displacement into a relative drift measurement, corrects OpenVINS translation/velocity, and passes that corrected state to the existing gate fusion.

**Tech Stack:** Python 3.10, NumPy, optional PyTorch, Isaac Lab, ROS2/OpenVINS.

**Spec:** `docs/superpowers/specs/2026-09-15-hybrid-learned-motion-vio-design.md`

## Global Constraints

- GT initialization A/B is skipped.
- Truth may be used for supervised labels only, never deployed inference.
- Learned correction is disabled unless a checkpoint is explicitly configured.
- OpenVINS attitude is retained; learned correction changes translation/velocity only.
- Existing gate-map fusion remains the global absolute correction layer.
- Pure estimator imports/tests must not require torch.

---

### Task 1: Relative learned-drift filter

**Files:**
- Create: `tests/estimation/test_learned_vio_drift.py`
- Create: `estimation/learned_vio_drift.py`

**Interfaces:**
- Consumes: `VioWorldEstimate` from `estimation.swift_vio_drift`.
- Produces: `LearnedDisplacementMeasurement`, `LearnedVioDriftFilter.step(vio, measurement=None)` and diagnostics.

- [ ] **Step 1: Write failing tests** for no-update pass-through, known relative drift correction, NIS rejection, covariance PSD/symmetry, orientation preservation and reset.
- [ ] **Step 2: Run** `PYTHONPATH=. pytest -q tests/estimation/test_learned_vio_drift.py` and verify failure because the module does not exist.
- [ ] **Step 3: Implement** the 9-state fixed-anchor EKF with `H=[-I,I,0]`, Joseph update and re-anchoring after accepted learned measurements.
- [ ] **Step 4: Re-run** the test file and verify all tests pass.

### Task 2: Learned motion window and optional TCN inference

**Files:**
- Create: `tests/estimation/test_learned_motion.py`
- Create: `estimation/learned_motion.py`

**Interfaces:**
- Consumes: timestamped world-frame gyro/thrust samples.
- Produces: `LearnedMotionBuffer`, predictor protocol, lazy `TorchTcnDisplacementPredictor`, six-output TCN when torch is present.

- [ ] **Step 1: Write failing tests** for monotonic timestamps, exact 0.5 s non-overlapping windows, reset, and import-without-torch behavior.
- [ ] **Step 2: Run** the test and verify module-missing failure.
- [ ] **Step 3: Implement** pure NumPy buffering/protocol plus lazy torch model/checkpoint loader.
- [ ] **Step 4: Re-run** and verify tests pass without installing torch.

### Task 3: Hybrid corrector orchestration

**Files:**
- Create: `tests/estimation/test_hybrid_vio_corrector.py`
- Create: `estimation/hybrid_vio_corrector.py`

**Interfaces:**
- Consumes: raw `VioWorldEstimate`, world-frame gyro, world-frame thrust, predictor.
- Produces: raw and learned-corrected estimates plus learned update diagnostics.

- [ ] **Step 1: Write failing tests** showing no correction before a full window and correction after a deterministic displacement prediction.
- [ ] **Step 2: Verify RED** with pytest.
- [ ] **Step 3: Implement** window scheduling, measurement construction and filter update.
- [ ] **Step 4: Verify GREEN** with pytest.

### Task 4: Isaac/OpenVINS runtime integration

**Files:**
- Modify: `tasks/drone_racer/drone_racer_swift_perception_env_cfg.py`
- Modify: `tasks/drone_racer/swift_openvins_env.py`

**Interfaces:**
- New config: checkpoint path/device/window/update gate parameters.
- Runtime: `openvins_raw_vio_estimate`, `openvins_learned_vio_estimate`, existing `openvins_vio_estimate` remains downstream selected estimate.

- [ ] **Step 1: Add configuration contract checks** through pure source-level/runtime-safe tests where possible.
- [ ] **Step 2: Add lazy checkpoint initialization** only when the path is configured.
- [ ] **Step 3: Read applied collective thrust from `control_action.processed_actions[0]`, rotate thrust and gyro with the raw VIO orientation, feed the hybrid corrector, and pass corrected VIO to Swift fusion.
- [ ] **Step 4: Reset learned state on simulator reset and log learned update/acceptance metrics.

### Task 5: Training trace and trainer

**Files:**
- Modify: `scripts/estimation/run_openvins_fault_isolation.py`
- Create: `estimation/learned_motion_dataset.py`
- Create: `scripts/estimation/train_learned_motion.py`
- Create: `tests/estimation/test_learned_motion_dataset.py`

**Interfaces:**
- Trace adds processed collective/body thrust.
- Dataset converts complete traces to 100 Hz, 0.5 s gyro+thrust windows and truth `dp` labels.
- Trainer saves a checkpoint containing model state plus window/sample-rate metadata.

- [ ] **Step 1: Write failing dataset tests** using synthetic CSV rows and complete-trace splitting.
- [ ] **Step 2: Verify RED**.
- [ ] **Step 3: Implement dataset utilities and extend trace schema.
- [ ] **Step 4: Implement torch training CLI with Gaussian NLL displacement loss.
- [ ] **Step 5: Verify dataset tests and Python compilation.

### Task 6: CI and documentation

**Files:**
- Modify: `.github/workflows/debug-opvs-pure-tests.yaml`
- Modify: `estimation/__init__.py`
- Create: `docs/HYBRID_LEARNED_MOTION_VIO.md`

- [ ] **Step 1: Add new pure modules to compile checks and NumPy-only tests to CI.
- [ ] **Step 2: Export stable public estimator symbols without importing torch eagerly.
- [ ] **Step 3: Document data collection, training, checkpoint configuration and evaluation workflow.
- [ ] **Step 4: Run the reconstructed pure test suite locally and inspect GitHub Actions after push.
