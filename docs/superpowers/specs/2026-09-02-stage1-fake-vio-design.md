# Stage 1 Fake VIO Design

## Status

Approved direction: preserve the Stage 0 ground-truth baseline, introduce a
hardware-portable state-estimate boundary, measure the unchanged Stage 0 policy
under estimation errors, and then fine-tune a copied checkpoint with a noise
curriculum. Camera, simulated IMU, gate detection, and OpenVINS are outside this
stage.

## Goals

Stage 1 must:

1. Keep `Isaac-Drone-Racer-v0` behavior unchanged as the Stage 0 upper bound.
2. Add a separate Stage 1 task whose policy receives corrupted state estimates.
3. Keep the policy observation shape and ordering at 20 values so the Stage 0
   checkpoint can be evaluated and fine-tuned without changing the network.
4. Model common VIO/IMU imperfections: white noise, bias, correlated drift,
   latency, and dropout.
5. Establish one timestamped estimator output contract that a later OpenVINS
   adapter can implement without changing the policy-facing observation code.
6. Guarantee that simulator ground truth cannot bypass the fake estimator and
   enter the Stage 1 policy observation.
7. Preserve ground truth for physics, rewards, termination, fake-sensor
   generation, and estimator-error evaluation only.

## Non-goals

Stage 1 does not:

- enable the virtual camera or virtual IMU sensor;
- implement gate projection, PnP/IPPE, a gate detector, or map fusion;
- replace ground-truth gate targeting or ground-truth reward/termination logic;
- implement OpenVINS or ROS 2 transport;
- change the action space, dynamics, rewards, track, or baseline policy network;
- overwrite or resume training inside the Stage 0 run directory.

## Approaches Considered

### Selected: provider contract plus policy adapter

A state-estimate provider owns temporal error state and emits a typed,
timestamped estimate. A policy observation adapter converts that estimate to the
existing policy convention. Stage 1 uses a fake provider; Stage 3 can replace it
with an OpenVINS provider.

This adds a small amount of structure now, but it isolates simulator-only logic,
supports 4,096 parallel environments, and avoids redesigning the policy boundary
when moving to real sensors.

### Rejected: add corruption independently to each observation term

Isaac Lab observation corruptors would be concise, but independent corruption
cannot correctly represent shared latency, time-correlated drift, burst dropout,
or one coherent pose/velocity estimate. It would also give Stage 3 no reusable
estimator boundary.

### Rejected: adopt ROS 2 messages inside the RL loop now

This would resemble the eventual vehicle deployment, but serialization and
middleware in a 4,096-environment GPU training loop would be unnecessary and
slow. Instead, Stage 1 uses a tensor-native contract with the same semantics as
the future scalar/ROS adapter.

## Coordinate and Time Contract

The contract uses explicit frame names rather than ambiguous `position` or
`velocity` labels:

- `timestamp_s`: monotonic sample time in seconds, derived from simulation time
  in Stage 1 and sensor time on hardware;
- `position_w_b`: body origin expressed in the world/map frame, metres;
- `orientation_w_b`: active rotation from body frame to world/map frame,
  quaternion in Isaac order `(w, x, y, z)`;
- `linear_velocity_w_b`: body linear velocity expressed in world/map frame,
  metres per second;
- `angular_velocity_b`: body angular velocity expressed in body frame, radians
  per second;
- `valid`: whether this update contains a newly delivered estimate;
- `age_s`: time since the most recently delivered estimate;
- optional diagonal uncertainty fields for logging and future fusion, not policy
  inputs in Stage 1.

The policy adapter rotates `linear_velocity_w_b` into the body frame and emits
the unchanged observation order:

```text
position_w_b(3), orientation_w_b(4), linear_velocity_b(3),
angular_velocity_b(3), target_pos_b_ground_truth(3), last_action(4)
```

During Stage 1, `target_pos_b` remains ground truth by design. Its replacement is
reserved for Stage 5.

On an estimator dropout, the provider holds the last delivered estimate, sets
`valid=false`, and increases `age_s`. This matches a practical estimator consumer
and avoids silently substituting current ground truth. `valid` and `age_s` are
logged but are not added to the policy in Stage 1, preserving the 20-dimensional
checkpoint interface.

## Components

### State estimate type

A tensor dataclass stores batched fields on the Isaac environment device. It has
shape validation, explicit frame semantics, and no dependency on Isaac asset
objects. This keeps the type reusable by fake, replay, and future OpenVINS
providers.

### Fake VIO provider

The provider is the only Stage 1 component allowed to read vehicle ground truth.
It maintains independent per-environment temporal state:

- pose, velocity, and angular-rate white noise;
- episode-constant bias sampled at reset;
- bias random walk;
- position, velocity, roll/pitch, and yaw drift random walks;
- an integer-step history ring buffer for latency;
- independent and burst dropout state;
- last delivered estimate and age.

All random sampling uses the environment device and an explicitly seeded
generator. Resetting selected environment IDs clears their history, drift,
dropout, and age without touching other environments.

Quaternion corruption is applied as a sampled small rotation composed with the
ground-truth quaternion and normalized afterwards. Quaternion components are
never corrupted independently.

### Policy observation adapter

Stage 1 observation terms fetch only the latest `StateEstimate`. They must not
hold an asset handle or read `asset.data.root_*`. Gate target and last action use
the existing Stage 0 terms because both remain in scope for Stage 1.

### Stage 1 environment configurations

New training and playback configurations inherit unchanged dynamics, actions,
rewards, terminations, and track configuration. They replace only the four state
observation terms and attach/reset/update the fake provider.

Separate Gym IDs make accidental baseline changes visible:

- `Isaac-Drone-Racer-Stage1-v0`
- `Isaac-Drone-Racer-Stage1-Play-v0`

Stage 0 IDs keep their current behavior.

### Configuration profiles

Every error source is configurable. Named profiles are preferred over scattered
Hydra overrides:

- `clean`: zero corruption; validates interface equivalence;
- `mild`: low noise and short latency for curriculum entry;
- `nominal`: target Stage 1 operating distribution;
- `stress`: held-out errors beyond the nominal training distribution.

Initial magnitudes are engineering priors, not claims about a specific physical
sensor. They will later be replaced with empirical distributions collected from
the chosen camera/IMU/VIO stack. Nominal ranges are sampled per episode so one
policy sees a family of estimators rather than one fixed noise level.

The initial nominal envelope is:

| Error | Nominal episode range |
| --- | --- |
| Position white noise sigma | 0.005–0.03 m |
| Orientation white noise sigma | 0.1–1.0 deg |
| Linear velocity white noise sigma | 0.01–0.10 m/s |
| Angular velocity white noise sigma | 0.001–0.01 rad/s |
| Position drift density | 0.001–0.02 m/sqrt(s) |
| Yaw drift density | 0.01–0.30 deg/sqrt(s) |
| Velocity drift density | 0.002–0.03 m/s/sqrt(s) |
| Delivery latency | 0–40 ms, quantized to policy steps |
| Independent dropout | 0–3% per delivered update |
| Burst dropout | disabled to 100 ms bursts |

The stress profile expands these ranges and is never used for gradient updates.

## Update Order and Data Flow

At each policy step:

1. Isaac advances physics and exposes ground truth.
2. The fake provider samples the configured errors and writes the new state into
   its latency history.
3. The provider selects the delayed sample, applies dropout delivery semantics,
   and publishes one `StateEstimate`.
4. Stage 1 observation terms read the published estimate.
5. The observation adapter converts world-frame linear velocity to body frame.
6. Existing ground-truth gate targeting supplies `target_pos_b`.
7. The unchanged 20-dimensional observation is passed to the policy.
8. Ground-truth reward, termination, and estimator metrics are computed outside
   the policy observation path.

No observation call advances estimator state. Estimator updates occur exactly
once per control step, preventing manager evaluation order from changing noise
or latency behavior.

## Ground Truth Firewall

The Stage 1 policy state terms live in a separate module and accept an estimator
provider name, not a robot asset name. Tests enforce that these functions do not
reference `root_pos_w`, `root_quat_w`, `root_lin_vel_*`, or `root_ang_vel_*`.

Ground truth remains legal only in:

- physics and dynamics;
- reward and termination calculations;
- the fake provider input;
- gate target information retained for Stage 1;
- estimator-error metrics and plots.

Run metadata records the task ID, checkpoint source, noise profile, sampled range
configuration, seed, and code revision so a result cannot be confused with the
Stage 0 baseline.

## Training and Evaluation

Stage 0 artifacts are read-only. The Stage 1 workflow is:

1. Evaluate the unchanged Stage 0 best checkpoint on the Stage 1 `clean` profile.
   Its behavior must match the Stage 0 task within deterministic tolerance.
2. Evaluate the same checkpoint without training on `mild`, `nominal`, and
   `stress` profiles to measure its existing robustness.
3. Copy/load the Stage 0 best checkpoint into a new Stage 1 run and fine-tune
   through `clean -> mild -> nominal` curriculum levels.
4. Select checkpoints using nominal validation performance, not training return
   alone.
5. Evaluate the selected checkpoint on clean, nominal, stress, and fixed held-out
   seeds.
6. After the warm-start path is stable, train one Stage 1 policy from scratch as
   a comparison; it is not allowed to replace the preserved baseline.

The first smoke runs use a small environment count. Full training retains 4,096
environments unless measured GPU memory or throughput requires a documented
change.

## Metrics and Acceptance Criteria

The evaluation report records:

- course completion rate;
- correctly passed gates per episode;
- collision and flyaway rates;
- episode return and completion time;
- position, orientation, linear-velocity, and angular-rate estimation errors;
- estimate age and dropout/burst statistics;
- performance grouped by latency and sampled error magnitude.

Stage 1 implementation is accepted when:

1. Existing Stage 0 tests and task behavior remain unchanged.
2. The `clean` provider reproduces all four Stage 0 state observation terms to
   numerical tolerance for multiple seeds and resets.
3. Unit tests prove deterministic seeded noise, quaternion normalization,
   per-environment reset isolation, exact step latency, hold-last dropout, and
   drift accumulation.
4. A leakage test proves Stage 1 policy state terms read only the estimate
   provider.
5. A headless Stage 1 smoke evaluation loads the Stage 0 checkpoint without an
   observation-shape mismatch.
6. A small Stage 1 training smoke run creates a separate run directory and a
   loadable checkpoint.
7. A 4,096-environment performance smoke test completes without out-of-memory
   failure and without CPU synchronization in the estimator update path.
8. Baseline, corrupted-before-training, and fine-tuned evaluation results are
   saved side by side using fixed seeds.

No minimum completion-rate threshold is declared before the pre-training
corruption sweep establishes a measured baseline. The first evaluation report
will propose thresholds based on Stage 0 degradation and achievable recovery.

## Failure Handling

- Invalid quaternion values fail fast in tests and are normalized at runtime.
- A requested latency larger than the allocated history raises a configuration
  error at environment construction.
- Dropout before the first valid delivery publishes the oldest initialized
  delayed state rather than zeros or current ground truth.
- Non-monotonic timestamps raise an error in debug/test mode and increment a
  diagnostic counter in production mode.
- NaN/Inf estimates are counted, logged, and terminate the affected training
  episode; they are never replaced with ground truth.

## Planned File Boundaries

Expected implementation areas are:

```text
tasks/drone_racer/estimation/
  state_estimate.py
  fake_vio.py
  fake_vio_cfg.py
tasks/drone_racer/mdp/stage1_observations.py
tasks/drone_racer/drone_racer_stage1_env_cfg.py
tasks/drone_racer/__init__.py
tests/estimation/
tests/test_stage1_observations.py
```

Exact filenames may be adjusted to follow test-discovered Isaac Lab lifecycle
constraints, but the provider, adapter, environment, and test responsibilities
must remain separate.

## Future Hardware Compatibility

Stage 3 will implement an OpenVINS adapter that produces the same semantic state
contract. Middleware stays outside the control policy: a hardware process may
receive ROS 2 sensor messages and publish an estimate, while the policy adapter
consumes the in-process representation. Frame conversion, camera-IMU extrinsics,
and timestamp-domain conversion happen at the provider boundary and are tested
independently.

The tensor batch dimension is a simulation optimization, not part of the
semantic interface. A one-vehicle instance is therefore the same contract with
batch size one.
