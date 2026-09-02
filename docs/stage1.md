# Stage 1: Fake VIO and Fake IMU

Stage 1 is an estimator-error robustness layer between Isaac ground truth and the
unchanged racing policy. It is a phenomenological model: its parameters are
engineering priors for noisy, biased, drifting, delayed, stale, or temporarily
unavailable state estimates. They do not claim to reproduce a specific VIO
implementation.

## Interface and frames

Transform `T_AB` maps coordinates from frame B into frame A. Fake VIO publishes
`T_VB` and linear velocity in its local VIO frame V. The state assembler keeps
the map alignment explicit:

```text
T_WB = T_WV * T_VB
```

Stage 1 initializes `T_WV = I`, but the API retains V and W as separate frames so
OpenVINS can replace Fake VIO without changing the policy adapter.

Fake VIO supplies position, orientation, and linear velocity at its own rate.
Fake IMU supplies body-frame gyro at an independent, normally higher rate. Each
has separate RNG state, episode bias, random walk, latency history, dropout,
timestamp, age, and freshness flag. The Isaac lifecycle adapter ingests gyro on
all four 400 Hz physics substeps, ingests VIO on its source clock, and publishes
one snapshot at the 100 Hz policy rate.

The 20 policy values retain the Stage 0 order:

```text
position_w_b (3), orientation_w_b scalar-first (4),
linear_velocity_b (3), angular_velocity_b (3),
target_pos_b_ground_truth (3), last_action (4)
```

Only the fake sensor boundary may read Isaac's drone-state truth. The policy
adapter reads `StateEstimate` only. Physics, rewards, terminations, labels, and
estimator-error evaluation may continue to use truth.

## Important interpretation

`target_pos_b` remains oracle ground truth in Stage 1. Results are therefore an
upper-bound test of robustness to drone state-estimation errors under perfect
gate-relative guidance. They do not demonstrate a complete perception-based
autonomy system. Stage 2 will replace gate truth.

## Tasks and profiles

- `Isaac-Drone-Racer-Stage1-v0`: 4,096-environment training configuration.
- `Isaac-Drone-Racer-Stage1-Play-v0`: playback configuration.
- `clean`: zero corruption and Stage 0 compatibility validation.
- `mild`: curriculum entry with low noise and short latency.
- `nominal`: the documented Stage 1 engineering envelope.
- `stress`: held-out evaluation only; do not train gradients on this profile.

## Validation

Fast tensor and checkpoint tests:

```bash
python3 -m pytest -q \
  tests/estimation \
  tests/test_stage1_observations.py \
  tests/test_checkpoint_validation.py \
  tests/test_stage1_metrics.py
```

Real Isaac configuration and lifecycle tests:

```bash
OMNI_KIT_ACCEPT_EULA=YES python3 tests/test_stage1_config.py --headless
OMNI_KIT_ACCEPT_EULA=YES python3 tests/test_stage1_env_lifecycle.py --headless
```

Clean observation, scaler, action, and value equivalence:

```bash
OMNI_KIT_ACCEPT_EULA=YES python3 scripts/rl/validate_stage1_equivalence.py \
  --headless --num_envs 2 \
  --checkpoint /absolute/path/to/stage0_best_agent.pt
```

The command fails if either RunningStandardScaler is missing or any clean-path
comparison exceeds `atol=1e-5`, `rtol=1e-5`.

## Training and playback

Begin the curriculum by warm-starting all 4,096 environments on `mild`:

```bash
OMNI_KIT_ACCEPT_EULA=YES python3 scripts/rl/train.py \
  --task Isaac-Drone-Racer-Stage1-v0 \
  --headless --num_envs 4096 \
  --fake_sensor_profile mild \
  --checkpoint /absolute/path/to/stage0_best_agent.pt
```

Training output is isolated under
`logs/skrl/drone_racer/<timestamp>_ppo_torch_stage1_<profile>/`. The source Stage
0 checkpoint is read only and is never overwritten. After mild validation,
warm-start nominal training from the selected mild checkpoint rather than from a
model-only export.

Playback can select the same profiles:

```bash
python3 scripts/rl/play.py \
  --task Isaac-Drone-Racer-Stage1-Play-v0 --num_envs 1 \
  --fake_sensor_profile nominal \
  --checkpoint /absolute/path/to/stage1_best_agent.pt
```

## Verification evidence (2026-09-02)

- Pure tests: 57 passed.
- Real two-environment lifecycle: 4 IMU ingestions and 1 VIO ingestion per 10 ms policy step; observation shape `(2, 20)`.
- Clean equivalence with the Stage 0 best checkpoint: maximum raw observation,
  preprocessed observation, deterministic mean-action, and value error all `0`.
- 32-environment mild warm-start: 48 training steps completed with PPO updates.
- 4,096-environment mild performance smoke: 48 training steps completed without
  OOM; observed training-loop throughput about 13.7 steps/s including the first
  optimization warm-up, with steady rollout sections above 30 steps/s.
- The repository's pre-existing `tests/test_dynamics.py` imports a removed
  `build_allocation_matrix` symbol and is not a Stage 1 regression.
