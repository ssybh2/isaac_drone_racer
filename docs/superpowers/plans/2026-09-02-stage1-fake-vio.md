# Stage 1 Fake VIO Implementation Plan

> **Execution note:** Implement this plan directly, task by task. No `superpowers` skill may be used for this work. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a hardware-portable Stage 1 state-estimation path that replaces drone-state ground truth in the policy with independently timed Fake VIO and Fake IMU estimates while preserving the Stage 0 20-value policy interface and oracle gate-relative target.

**Architecture:** Pure PyTorch providers generate asynchronous phenomenological VIO and gyro errors, and a state-estimate assembler applies the explicit `T_WV` alignment before a policy adapter emits the existing 13 drone-state values. A Stage 1 Isaac Lab environment subclass samples Fake IMU at the 400 Hz physics rate, samples Fake VIO on its lower-rate clock, publishes at the 100 Hz policy rate, and leaves Stage 0 task registration untouched.

**Tech Stack:** Python 3.10, PyTorch, Isaac Sim 4.5.0, Isaac Lab v2.1.0, Gymnasium, SKRL 1.4.2+, pytest.

**Spec:** `docs/superpowers/specs/2026-09-02-stage1-fake-vio-design.md`

## Global Constraints

- `Isaac-Drone-Racer-v0` and `Isaac-Drone-Racer-Play-v0` remain behaviorally unchanged.
- Stage 1 policy observations remain 20 values in this order: estimated position 3, estimated quaternion 4, estimated body linear velocity 3, estimated body angular velocity 3, ground-truth `target_pos_b` 3, last action 4.
- Only Fake VIO and Fake IMU provider adapters may read drone-state ground truth.
- Ground-truth `target_pos_b` remains enabled and every Stage 1 report labels the result as oracle gate-relative guidance.
- `T_AB` maps coordinates from frame B to frame A; `T_WB = T_WV * T_VB`; Stage 1 uses identity `T_WV`.
- VIO and IMU retain separate timestamp, age, validity, rate, latency, noise, bias, and dropout state.
- The implementation stays tensorized on the environment device and supports 4,096 environments without per-environment Python loops or CPU synchronization in the update path.
- Stage 0 checkpoint directories are read-only; all Stage 1 outputs use new run directories.
- Every production behavior follows RED, GREEN, REFACTOR and is committed separately.

---

### Task 1: State contracts and frame alignment

**Files:**
- Create: `estimation/__init__.py`
- Create: `estimation/state_estimate.py`
- Create: `estimation/frame_math.py`
- Test: `tests/estimation/test_state_estimate.py`
- Test: `tests/estimation/test_frame_math.py`

**Interfaces:**
- Produces: `SourceStatus`, `VioEstimate`, `ImuEstimate`, `StateEstimate`, `GroundTruthState` tensor dataclasses.
- Produces: `compose_transform_w_b(position_w_v, orientation_w_v, position_v_b, orientation_v_b) -> tuple[Tensor, Tensor]` and `rotate_world_to_body(orientation_w_b, vector_w) -> Tensor`.

- [x] **Step 1: Write state-shape and quaternion validation tests**

```python
def test_state_estimate_rejects_wrong_position_shape():
    with pytest.raises(ValueError, match="position_w_b"):
        make_state_estimate(position_w_b=torch.zeros(2, 4))

def test_state_estimate_rejects_non_finite_values():
    state = make_state_estimate()
    state.position_w_b[0, 0] = torch.nan
    with pytest.raises(ValueError, match="finite"):
        state.validate()
```

- [x] **Step 2: Run the contract tests and verify RED**

Run: `.conda-env/bin/python -m pytest -q tests/estimation/test_state_estimate.py`

Expected: collection fails because `tasks.drone_racer.estimation` does not exist.

- [x] **Step 3: Implement typed tensor contracts and validation**

Use dataclasses with batched tensors. `SourceStatus` contains `timestamp_s`, `age_s`, and `valid`; `VioEstimate` contains `position_v_b`, `orientation_v_b`, `linear_velocity_v_b`, and `status`; `ImuEstimate` contains `angular_velocity_b` and `status`; `StateEstimate` contains publish time, `T_WV`, world-aligned state, and both statuses. `validate()` checks batch shapes, common device, finite floating values, boolean validity, and unit quaternions within `1e-4`.

- [x] **Step 4: Run contract tests and verify GREEN**

Run: `.conda-env/bin/python -m pytest -q tests/estimation/test_state_estimate.py`

Expected: all tests pass.

- [x] **Step 5: Write hand-derived frame-transform tests**

```python
def test_t_wb_equals_t_wv_times_t_vb_for_yaw_alignment():
    p_w_b, q_w_b = compose_transform_w_b(
        position_w_v=torch.tensor([[10.0, 0.0, 0.0]]),
        orientation_w_v=yaw_quaternion_90_deg,
        position_v_b=torch.tensor([[1.0, 0.0, 0.0]]),
        orientation_v_b=identity_quaternion,
    )
    torch.testing.assert_close(p_w_b, torch.tensor([[10.0, 1.0, 0.0]]), atol=1e-6, rtol=0)
    torch.testing.assert_close(q_w_b, yaw_quaternion_90_deg, atol=1e-6, rtol=0)
```

- [x] **Step 6: Run frame tests and verify RED**

Run: `.conda-env/bin/python -m pytest -q tests/estimation/test_frame_math.py`

Expected: import fails because `frame_math.py` does not exist.

- [x] **Step 7: Implement normalized quaternion composition and vector rotation**

Implement scalar-first quaternion multiply, conjugate, normalize, rotate, transform composition, and world-to-body rotation using batched PyTorch operations only.

- [x] **Step 8: Run Task 1 tests and commit**

Run: `.conda-env/bin/python -m pytest -q tests/estimation/test_state_estimate.py tests/estimation/test_frame_math.py`

Commit:

```bash
git add estimation tests/estimation/test_state_estimate.py tests/estimation/test_frame_math.py
git commit -m "feat: define Stage 1 state estimate contract"
```

---

### Task 2: Fake VIO phenomenological provider

**Files:**
- Create: `estimation/fake_sensor_cfg.py`
- Create: `estimation/fake_vio.py`
- Test: `tests/estimation/test_fake_vio.py`

**Interfaces:**
- Consumes: `GroundTruthState`, `VioEstimate`, quaternion helpers.
- Produces: `Range(low: float, high: float)`, `FakeVioCfg`, and `FakeVio(num_envs, device, cfg, seed)` with `reset(env_ids, ground_truth, timestamp_s)` and `update(ground_truth, timestamp_s) -> VioEstimate`.

- [x] **Step 1: Write clean-profile, seeded-noise, and quaternion tests**

```python
def test_clean_vio_matches_ground_truth():
    vio = FakeVio(2, "cpu", FakeVioCfg.clean(), seed=7)
    out = vio.reset(torch.arange(2), ground_truth_fixture(), timestamp_s=0.0)
    torch.testing.assert_close(out.position_v_b, ground_truth_fixture().position_w_b)
    torch.testing.assert_close(out.orientation_v_b, ground_truth_fixture().orientation_w_b)

def test_vio_noise_is_seed_reproducible():
    a = FakeVio(2, "cpu", noisy_cfg, seed=19)
    b = FakeVio(2, "cpu", noisy_cfg, seed=19)
    torch.testing.assert_close(a.reset(ids, gt, 0.0).position_v_b, b.reset(ids, gt, 0.0).position_v_b)
```

- [x] **Step 2: Run the VIO tests and verify RED**

Run: `.conda-env/bin/python -m pytest -q tests/estimation/test_fake_vio.py`

Expected: import fails because `fake_vio.py` does not exist.

- [x] **Step 3: Implement clean output, sampled episode parameters, and quaternion perturbation**

Sample profile ranges into `(num_envs, field_dim)` tensors at reset. Compose small-angle roll/pitch/yaw error as a quaternion and normalize the result. Never add noise directly to quaternion components.

- [x] **Step 4: Run focused tests and verify GREEN**

Run: `.conda-env/bin/python -m pytest -q tests/estimation/test_fake_vio.py -k 'clean or seed or quaternion'`

Expected: selected tests pass.

- [x] **Step 5: Add failing tests for update rate, latency, drift, dropout, and reset isolation**

Use literal one-dimensional trajectories so a two-step latency must return the value from exactly two source samples earlier. Force dropout probability to `1.0` and assert hold-last plus increasing age. Reset only environment 0 and assert environment 1 drift/history tensors are unchanged.

- [x] **Step 6: Run temporal tests and verify RED**

Run: `.conda-env/bin/python -m pytest -q tests/estimation/test_fake_vio.py -k 'rate or latency or drift or dropout or isolation'`

Expected: assertions fail because temporal behaviors are absent.

- [x] **Step 7: Implement tensorized temporal behavior**

Maintain source clock, ring-buffer samples/timestamps, bias random walks, position/velocity/orientation drift, dropout burst counter, last-delivered sample, age, and validity per environment. Reject update periods below the ingestion period and latency larger than ring capacity. No `.cpu()`, `.item()`, NumPy conversion, or environment loop is allowed in `update`.

- [x] **Step 8: Run Task 2 tests and commit**

Run: `.conda-env/bin/python -m pytest -q tests/estimation/test_fake_vio.py`

Commit:

```bash
git add estimation/fake_sensor_cfg.py estimation/fake_vio.py tests/estimation/test_fake_vio.py
git commit -m "feat: add tensorized Fake VIO provider"
```

---

### Task 3: Fake IMU gyro provider

**Files:**
- Create: `estimation/fake_imu.py`
- Test: `tests/estimation/test_fake_imu.py`

**Interfaces:**
- Consumes: `GroundTruthState.angular_velocity_b`, `FakeImuCfg`, `ImuEstimate`.
- Produces: `FakeImu(num_envs, device, cfg, seed)` with `reset(env_ids, angular_velocity_b, timestamp_s)` and `update(angular_velocity_b, timestamp_s) -> ImuEstimate`.

- [x] **Step 1: Write failing tests for clean gyro and independent bias**

Assert clean output equals a literal gyro tensor, a fixed bias shifts only the selected axis, and two identical seeds produce identical white noise.

- [x] **Step 2: Run tests and verify RED**

Run: `.conda-env/bin/python -m pytest -q tests/estimation/test_fake_imu.py`

Expected: import fails because `fake_imu.py` does not exist.

- [x] **Step 3: Implement gyro noise, episode bias, and bias random walk**

Keep all buffers `(num_envs, 3)` on the configured device. Apply white noise per emitted sample and bias random walk proportional to `sqrt(dt)`.

- [x] **Step 4: Add and verify RED tests for physics-rate timing, latency, dropout, and selective reset**

Feed samples at timestamps `0.0000`, `0.0025`, `0.0050`, and `0.0075`; use a 5 ms latency and assert the last delivery at 7.5 ms contains the 2.5 ms measurement. Force dropout to verify hold-last without affecting VIO state.

- [x] **Step 5: Implement independent source clock and history**

Use the same observable delivery semantics as Fake VIO but separate configuration, RNG, timestamps, buffers, and validity.

- [x] **Step 6: Run Task 3 tests and commit**

Run: `.conda-env/bin/python -m pytest -q tests/estimation/test_fake_imu.py`

Commit:

```bash
git add estimation/fake_imu.py tests/estimation/test_fake_imu.py
git commit -m "feat: add independent Fake IMU gyro provider"
```

---

### Task 4: State assembler and policy adapter

**Files:**
- Create: `estimation/state_estimate_assembler.py`
- Create: `tasks/drone_racer/mdp/stage1_observations.py`
- Modify: `tasks/drone_racer/mdp/__init__.py`
- Test: `tests/estimation/test_state_estimate_assembler.py`
- Test: `tests/test_stage1_observations.py`

**Interfaces:**
- Consumes: most recent `VioEstimate`, `ImuEstimate`, and `T_WV`.
- Produces: `StateEstimateAssembler.publish(vio, imu, publish_timestamp_s) -> StateEstimate`.
- Produces: `estimated_drone_state(env) -> Tensor` with shape `(num_envs, 13)` and exact Stage 0 ordering.

- [x] **Step 1: Write failing identity and non-identity alignment tests**

Hand-check identity `T_WV` and a 90-degree yaw/translation case. Assert VIO and IMU timestamps, ages, and validity survive assembly unchanged.

- [x] **Step 2: Run assembler tests and verify RED**

Run: `.conda-env/bin/python -m pytest -q tests/estimation/test_state_estimate_assembler.py`

- [x] **Step 3: Implement assembly and world alignment**

Apply `T_WV` to pose and linear velocity. Copy gyro in body coordinates without rotation. Compute source ages from publish time minus each source timestamp and reject negative age.

- [x] **Step 4: Write failing 13-value adapter test**

Use a real `StateEstimate` fixture and an environment stub containing only `stage1_state_estimate`; assert the literal flattened output. The stub intentionally has no `scene` attribute so any ground-truth access fails.

- [x] **Step 5: Run adapter test and verify RED**

Run: `.conda-env/bin/python -m pytest -q tests/test_stage1_observations.py`

- [x] **Step 6: Implement the read-only adapter**

Rotate `linear_velocity_w_b` into B using `orientation_w_b`, concatenate position, quaternion, body linear velocity, and body gyro, then clone the result. The function does not update providers and does not accept `asset_cfg`.

- [x] **Step 7: Run Task 4 tests and commit**

Run: `.conda-env/bin/python -m pytest -q tests/estimation/test_state_estimate_assembler.py tests/test_stage1_observations.py`

Commit:

```bash
git add estimation/state_estimate_assembler.py tasks/drone_racer/mdp/stage1_observations.py tasks/drone_racer/mdp/__init__.py tests/estimation/test_state_estimate_assembler.py tests/test_stage1_observations.py
git commit -m "feat: assemble Stage 1 policy state"
```

---

### Task 5: Stage 1 pipeline and Isaac Lab lifecycle

**Files:**
- Create: `estimation/pipeline.py`
- Create: `tasks/drone_racer/stage1_env.py`
- Test: `tests/estimation/test_pipeline.py`
- Test: `tests/test_stage1_env_lifecycle.py`

**Interfaces:**
- Produces: `Stage1StatePipeline` with `reset`, `ingest_imu`, `ingest_vio`, and `publish` methods.
- Produces: `Stage1DroneRacerEnv(ManagerBasedRLEnv)` that supplies `stage1_state_estimate` before observation computation.

- [x] **Step 1: Write failing pure pipeline tests**

Assert IMU can update four times while VIO updates once in 10 ms, publishing retains the latest independent timestamps, reset affects selected environment IDs only, and clean pipeline output matches ground truth.

- [x] **Step 2: Run pipeline tests and verify RED**

Run: `.conda-env/bin/python -m pytest -q tests/estimation/test_pipeline.py`

- [x] **Step 3: Implement provider orchestration**

`Stage1StatePipeline` owns Fake VIO, Fake IMU, assembler, identity `T_WV`, and current estimate. It accepts ground-truth tensors explicitly; it has no Isaac imports.

- [x] **Step 4: Add a failing runtime lifecycle test**

Launch one real Stage 1 environment with clean providers and inspect provider diagnostics after one action. Assert `imu_ingest_count` increases by `env.cfg.decimation`, `vio_delivery_count` follows the configured VIO period, the returned observation carries the most recently published timestamp, and the reward and termination managers expose the same term names as Stage 0.

- [x] **Step 5: Run lifecycle guard and verify RED**

Run the test through the Isaac Sim Python launcher: `.conda-env/bin/python tests/test_stage1_env_lifecycle.py --headless`

- [x] **Step 6: Implement the pinned Stage 1 environment subclass**

Copy the v2.1.0 `ManagerBasedRLEnv.step` control flow into the subclass and make only these insertions: ingest Fake IMU immediately after each `scene.update(self.physics_dt)`; ingest Fake VIO once after the physics loop when its clock is due; reset affected provider rows after automatic environment resets; publish immediately before final observation computation. Override explicit `reset` to refresh providers after `sim.forward` and recompute the returned observation. Add a comment with the pinned upstream file and tag.

- [x] **Step 7: Run Task 5 pure tests and commit**

Run: `.conda-env/bin/python -m pytest -q tests/estimation/test_pipeline.py` and `.conda-env/bin/python tests/test_stage1_env_lifecycle.py --headless`.

Commit:

```bash
git add estimation/pipeline.py tasks/drone_racer/stage1_env.py tests/estimation/test_pipeline.py tests/test_stage1_env_lifecycle.py
git commit -m "feat: integrate Stage 1 estimator lifecycle"
```

---

### Task 6: Stage 1 configs, profiles, and Gym registration

**Files:**
- Create: `tasks/drone_racer/drone_racer_stage1_env_cfg.py`
- Modify: `tasks/drone_racer/__init__.py`
- Test: `tests/test_stage1_config.py`

**Interfaces:**
- Produces: `DroneRacerStage1EnvCfg` and `DroneRacerStage1EnvCfg_PLAY`.
- Produces Gym IDs: `Isaac-Drone-Racer-Stage1-v0` and `Isaac-Drone-Racer-Stage1-Play-v0` using entry point `tasks.drone_racer.stage1_env:Stage1DroneRacerEnv`.

- [x] **Step 1: Write failing configuration behavior tests**

Instantiate Stage 0 and Stage 1 configs. Assert Stage 0 observation terms remain unchanged, Stage 1 has `estimated_drone_state + target_pos_b + actions`, concatenated size is 20, camera/Isaac IMU remain disabled, Stage 1 defaults to `clean`, and `target_pos_b` is the existing ground-truth term.

- [x] **Step 2: Run config tests and verify RED**

Run with Isaac Lab paths:

```bash
PYTHONPATH=/home/donglei/isaac_projects/IsaacLab-v2.1.0/source/isaaclab:/home/donglei/isaac_projects/IsaacLab-v2.1.0/source/isaaclab_tasks .conda-env/bin/python -m pytest -q tests/test_stage1_config.py
```

- [x] **Step 3: Implement profile factories and Stage 1 configs**

Inherit the Stage 0 scene/actions/commands/rewards/terminations, replace only policy state observation terms, and carry a `FakeSensorPipelineCfg`. Define `clean`, `mild`, `nominal`, and `stress` factories with the numeric envelopes from the spec. Use `clean` as the default until an explicit profile is supplied.

- [x] **Step 4: Register both new task IDs without editing Stage 0 registrations**

Use the existing SKRL YAML entry point. Keep training at 4,096 environments and playback overrideable to one.

- [x] **Step 5: Run Task 6 tests and commit**

Run the Task 6 test command plus `.conda-env/bin/python -m pytest -q tests/estimation tests/test_stage1_observations.py`.

Commit:

```bash
git add tasks/drone_racer/drone_racer_stage1_env_cfg.py tasks/drone_racer/__init__.py tests/test_stage1_config.py
git commit -m "feat: register Stage 1 Fake VIO tasks"
```

---

### Task 7: Complete checkpoint and clean-path equivalence

**Files:**
- Create: `utils/checkpoint_validation.py`
- Create: `scripts/rl/validate_stage1_equivalence.py`
- Test: `tests/test_checkpoint_validation.py`

**Interfaces:**
- Produces: `required_skrl_modules(checkpoint: dict) -> set[str]` and `assert_complete_stage0_checkpoint(path)`.
- Produces CLI comparing Stage 0 and Stage 1-clean observations, deterministic mean actions, and value outputs from the same complete checkpoint.

- [x] **Step 1: Write failing checkpoint completeness tests**

Load a small temporary PyTorch dictionary with literal module keys. Assert omission of `state_preprocessor` or `value_preprocessor` raises a named error, and the real `best_agent.pt` exposes policy, value, optimizer, state preprocessor, and value preprocessor.

- [x] **Step 2: Run tests and verify RED**

Run: `.conda-env/bin/python -m pytest -q tests/test_checkpoint_validation.py`

- [x] **Step 3: Implement checkpoint validation**

Use trusted local `torch.load(..., map_location="cpu", weights_only=False)`. Verify scaler states contain `current_count`, `running_mean`, and `running_variance`. Never rewrite the checkpoint.

- [x] **Step 4: Implement the integration comparison CLI**

Create Stage 0 and Stage 1-clean environments with one environment and identical seed. Load the same checkpoint through `runner.agent.load`, set eval mode, compare the raw 20-value observation, preprocessed state, deterministic `mean_actions`, and value output using `atol=1e-5`, `rtol=1e-5`. Exit nonzero and print maximum error for any mismatch.

- [x] **Step 5: Run unit tests and a headless two-environment integration check**

Run:

```bash
.conda-env/bin/python -m pytest -q tests/test_checkpoint_validation.py
.conda-env/bin/python scripts/rl/validate_stage1_equivalence.py --headless --num_envs 2 --checkpoint /home/donglei/isaac_projects/isaac_drone_racer/logs/skrl/drone_racer/2026-09-01_22-09-01_ppo_torch/checkpoints/best_agent.pt
```

Expected: every checkpoint module is restored and all four maximum differences are within tolerance.

- [x] **Step 6: Commit Task 7**

```bash
git add utils/checkpoint_validation.py scripts/rl/validate_stage1_equivalence.py tests/test_checkpoint_validation.py
git commit -m "test: verify Stage 1 checkpoint equivalence"
```

---

### Task 8: Evaluation metrics and corruption sweep

**Files:**
- Create: `utils/stage1_metrics.py`
- Create: `scripts/rl/evaluate_stage1.py`
- Test: `tests/test_stage1_metrics.py`

**Interfaces:**
- Produces episode aggregation for completion, passed gates, collisions, flyaways, return, duration, state errors, source age, and dropout statistics.
- Produces CLI that evaluates a checkpoint across `clean,mild,nominal,stress` and fixed seeds, writing JSON and CSV into a new evaluation directory.

- [ ] **Step 1: Write failing hand-derived metric tests**

Feed two literal episode records and assert exact means/rates, including one collision, one completion, VIO dropout fraction, and 95th-percentile source age.

- [ ] **Step 2: Run metric tests and verify RED**

Run: `.conda-env/bin/python -m pytest -q tests/test_stage1_metrics.py`

- [ ] **Step 3: Implement streaming tensor-to-summary metrics**

Only transfer completed aggregate tensors to CPU at reporting boundaries. Include `oracle_gate_relative_guidance: true`, checkpoint path/hash, profile, seed, and code revision in every output record.

- [ ] **Step 4: Implement evaluation CLI and fixed output layout**

Use `logs/stage1_evaluations/<timestamp>/summary.json` and `episodes.csv`. Require an explicit checkpoint and never write into its training run directory.

- [ ] **Step 5: Run Task 8 tests and commit**

Run: `.conda-env/bin/python -m pytest -q tests/test_stage1_metrics.py`

Commit:

```bash
git add utils/stage1_metrics.py scripts/rl/evaluate_stage1.py tests/test_stage1_metrics.py
git commit -m "feat: add Stage 1 robustness evaluation"
```

---

### Task 9: End-to-end smoke, 4,096-env check, and training launch

**Files:**
- Modify: `README.md`
- Create: `docs/stage1.md`
- Modify: `docs/superpowers/plans/2026-09-02-stage1-fake-vio.md`

**Interfaces:**
- Consumes both Stage 1 Gym IDs, existing `scripts/rl/train.py`, existing `scripts/rl/play.py`, and the evaluation CLI.
- Produces documented commands and recorded verification evidence.

- [ ] **Step 1: Run the complete fast test suite**

Run: `.conda-env/bin/python -m pytest -q tests/estimation tests/test_dynamics.py tests/test_gate_assets.py tests/test_stage1_observations.py tests/test_stage1_config.py tests/test_checkpoint_validation.py tests/test_stage1_metrics.py`

Expected: zero failures.

- [x] **Step 2: Run a small headless Stage 1 environment smoke test**

Run Stage 1 playback for a bounded number of steps using the Stage 0 checkpoint, clean profile, two environments, and no UI. Assert observation shape `(2, 20)`, finite actions, and no simulator error.

- [x] **Step 3: Run a small training smoke test into a new run directory**

Run:

```bash
.conda-env/bin/python scripts/rl/train.py --task Isaac-Drone-Racer-Stage1-v0 --headless --num_envs 32 --max_iterations 2 --checkpoint /home/donglei/isaac_projects/isaac_drone_racer/logs/skrl/drone_racer/2026-09-01_22-09-01_ppo_torch/checkpoints/best_agent.pt env.fake_sensors.profile=mild agent.experiment.experiment_name=stage1_smoke
```

Verify a new Stage 1 checkpoint exists and the Stage 0 checkpoint hash is unchanged.

- [ ] **Step 4: Run the 4,096-environment performance smoke**

Launch the Stage 1 task headlessly for a bounded step count with profile `nominal`, capture peak GPU memory and environment-step throughput, and assert there is no out-of-memory error or provider-side CPU synchronization.

- [x] **Step 5: Document commands, limitations, and oracle-target interpretation**

Document task IDs, profiles, warm-start command, evaluation command, frame notation, separate source rates, full-checkpoint requirement, output locations, and the fact that Stage 1 is an upper-bound state-error robustness test under perfect gate-relative guidance.

- [ ] **Step 6: Run documentation checks and final verification**

Run `git diff --check`, the complete fast suite, clean equivalence CLI, 32-environment training smoke, and 4,096-environment performance smoke. Record exact commands and measured results in `docs/stage1.md`.

- [ ] **Step 7: Commit Task 9**

```bash
git add README.md docs/stage1.md docs/superpowers/plans/2026-09-02-stage1-fake-vio.md
git commit -m "docs: add Stage 1 workflow and verification"
```

- [ ] **Step 8: Push the feature branch without merging it**

Run: `git push -u ssybh2 feature/stage1-fake-vio`

Expected: the remote feature branch points to the locally verified final commit; `master` remains the preserved baseline until the user chooses integration.
