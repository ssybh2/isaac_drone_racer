# Stage 1 Fake VIO Design

## Status

Approved direction: preserve the Stage 0 ground-truth baseline, introduce a
hardware-portable state-estimate boundary, measure the unchanged Stage 0 policy
under estimation errors, and then fine-tune a copied checkpoint with a noise
curriculum. Camera rendering, the Isaac virtual IMU sensor, gate detection, and
OpenVINS are outside this stage; a tensor-native Fake IMU gyro source is included.

## Goals

Stage 1 must:

1. Keep `Isaac-Drone-Racer-v0` behavior unchanged as the Stage 0 upper bound.
2. Add a separate Stage 1 task whose policy receives corrupted state estimates.
3. Keep the policy observation shape and ordering at 20 values so the Stage 0
   checkpoint can be evaluated and fine-tuned without changing the network.
4. Model common VIO and IMU imperfections with independent update rates,
   timestamps, white noise, bias, correlated drift, latency, and dropout.
5. Establish one timestamped estimator output contract that a later OpenVINS
   adapter can implement without changing the policy-facing observation code.
6. Guarantee that simulator ground truth cannot bypass the fake estimator and
   enter the Stage 1 policy observation.
7. Preserve ground truth for physics, rewards, termination, Fake VIO/Fake IMU
   generation, oracle gate-relative targeting, and estimator-error evaluation
   only.

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

## Estimator Model Scope

Fake VIO and Fake IMU are **phenomenological estimator error models**. They do
not reproduce OpenVINS internals. They expose the noisy, biased, drifting,
delayed, stale, and temporarily unavailable outputs that a policy consumer must
handle.

Initial parameter ranges are engineering priors and do not represent a specific
VIO or IMU. After Stage 3 integrates OpenVINS, synchronized
`VIO estimate - Mocap / Isaac ground truth` residuals will be used to fit updated
noise, bias, drift, latency, and dropout distributions.

## Coordinate, Alignment, and Time Contract

`T_AB` denotes the rigid transform that maps coordinates expressed in frame B
into frame A. Frames are:

- `W`: known track/map world frame;
- `V`: local frame established when VIO initializes;
- `B`: drone body frame.

Real VIO produces `T_VB`; it does not naturally produce a pose in the known
track frame. The provider boundary therefore explicitly retains the alignment
transform:

```text
T_WB = T_WV * T_VB
```

Stage 1 sets `T_WV = I`, making the Fake VIO frame coincide with Isaac track
world while preserving the frame distinction in the API. Stage 3 initializes
and maintains `T_WV` at the OpenVINS adapter boundary. Stage 4 may correct this
alignment using mapped gate observations. Because visual-inertial odometry is
metric, `T_WV` is an SE(3) transform rather than a free-scale transform.

The contract uses explicit frame names rather than ambiguous `position` or
`velocity` labels:

- `publish_timestamp_s`: monotonic policy snapshot time in seconds;
- `vio_timestamp_s`, `vio_age_s`, `vio_valid`: acquisition time, age at publish,
  and freshness of the VIO sample;
- `imu_timestamp_s`, `imu_age_s`, `imu_valid`: acquisition time, age at publish,
  and freshness of the gyro sample;
- `transform_w_v`: alignment `T_WV`, represented by translation and quaternion;
- `position_w_b`: body origin expressed in the world/map frame, metres;
- `orientation_w_b`: active rotation from body frame to world/map frame,
  quaternion in Isaac order `(w, x, y, z)`;
- `linear_velocity_w_b`: body linear velocity expressed in world/map frame,
  metres per second;
- `angular_velocity_b`: body angular velocity expressed in body frame, radians
  per second;
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

On a source dropout, the assembler holds that source's last delivered sample,
sets its source-specific `valid=false`, and increases its source-specific
`age_s`. This matches a practical asynchronous estimator consumer and avoids
silently substituting current ground truth. Validity and age are logged but are
not added to the policy in Stage 1, preserving the 20-dimensional checkpoint
interface.

## Components

### State estimate type

A tensor dataclass stores batched fields on the Isaac environment device. It has
shape validation, explicit frame semantics, and no dependency on Isaac asset
objects. This keeps the type reusable by fake, replay, and future OpenVINS
providers.

### Fake VIO provider

The provider is the only Stage 1 component allowed to read vehicle ground truth.
It publishes `T_VB` and linear velocity at its own configurable update rate and
maintains independent per-environment temporal state:

- pose and velocity white noise;
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

### Fake IMU provider

The Fake IMU provider is the only other Stage 1 component allowed to read
vehicle ground truth. It publishes gyroscope angular velocity
`angular_velocity_b` (strictly `omega_IB_B`) at an independently configurable
rate. It has its own white noise, episode bias, bias random walk, latency history,
dropout state, timestamp, validity, and age. It does not inherit VIO latency or
dropout.

The initial implementation models the gyro path needed by the existing policy.
Accelerometer output is reserved for Stage 3 because acceleration is not a
Stage 0 policy input.

### State estimate assembler

The assembler samples the most recently delivered Fake VIO and Fake IMU values,
applies `T_WV`, and publishes a policy snapshot without discarding the two source
timestamps, ages, or validity flags. It never reads simulator ground truth. In
Stage 1, `T_WV` is identity.

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

Every error source and source update rate is configurable. Named profiles are
preferred over scattered Hydra overrides:

- `clean`: zero corruption; validates interface equivalence;
- `mild`: low noise and short latency for curriculum entry;
- `nominal`: target Stage 1 operating distribution;
- `stress`: held-out errors beyond the nominal training distribution.

Initial magnitudes are engineering priors, not claims about a specific physical
sensor. They will later be replaced with empirical distributions collected from
the chosen camera/IMU/VIO stack. Nominal ranges are sampled per episode so one
policy sees a family of estimators rather than one fixed noise level.

The initial nominal envelopes are separate:

| Fake VIO error | Nominal episode range |
| --- | --- |
| Position white noise sigma | 0.005–0.03 m |
| Orientation white noise sigma | 0.1–1.0 deg |
| Linear velocity white noise sigma | 0.01–0.10 m/s |
| Position drift density | 0.001–0.02 m/sqrt(s) |
| Yaw drift density | 0.01–0.30 deg/sqrt(s) |
| Velocity drift density | 0.002–0.03 m/s/sqrt(s) |
| Update rate | configurable, initially 30–100 Hz |
| Delivery latency | 0–40 ms, quantized to policy steps |
| Independent dropout | 0–3% per delivered update |
| Burst dropout | disabled to 100 ms bursts |

| Fake IMU gyro error | Nominal episode range |
| --- | --- |
| Angular velocity white noise sigma | 0.001–0.01 rad/s |
| Gyro bias | configurable per-axis episode range |
| Gyro bias random walk | configurable per-axis density |
| Update rate | configurable, initially 200–400 Hz |
| Delivery latency | 0–5 ms, quantized to simulation steps |
| Independent dropout | 0–0.5% per delivered update |

The stress profile expands these ranges and is never used for gradient updates.

## Update Order and Data Flow

At each policy step:

1. Isaac advances physics and exposes ground truth.
2. Fake VIO updates only when its independent sample clock is due, then writes
   pose and velocity into its latency history.
3. Fake IMU updates only when its independent sample clock is due, then writes
   gyro data into its latency history.
4. Each source independently selects a delayed sample and applies its own
   dropout delivery semantics.
5. The assembler applies `T_WV` and publishes one `StateEstimate` snapshot while
   retaining both source timestamps, ages, and validity flags.
6. Stage 1 observation terms read the published estimate.
7. The observation adapter converts world-frame linear velocity to body frame.
8. Existing ground-truth gate targeting supplies `target_pos_b`.
9. The unchanged 20-dimensional observation is passed to the policy.
10. Ground-truth reward, termination, and estimator metrics are computed outside
   the policy observation path.

No observation call advances estimator state. Estimator updates occur exactly
once per control step, preventing manager evaluation order from changing noise
or latency behavior.

The Stage 1 boundary is:

```text
Isaac ground truth (provider access only)
       |                         |
       v                         v
Fake VIO                    Fake IMU
T_VB, velocity_V            omega_IB_B
VIO time/age/valid           IMU time/age/valid
       |                         |
       v                         |
T_WV alignment                  |
       |                         |
       +------------+------------+
                    v
              StateEstimate
                    v
              Policy Adapter
                    v
position_w_b, orientation_w_b, linear_velocity_b,
angular_velocity_b, target_pos_b_ground_truth, last_action
                    v
                  Policy
```

## Ground Truth Firewall

The Stage 1 policy state terms live in a separate module and accept a state
estimate source name, not a robot asset name. Tests enforce that these functions do not
reference `root_pos_w`, `root_quat_w`, `root_lin_vel_*`, or `root_ang_vel_*`.

Ground truth remains legal only in:

- physics and dynamics;
- reward and termination calculations;
- Fake VIO and Fake IMU provider inputs;
- gate target information retained for Stage 1;
- estimator-error metrics and plots.

Run metadata records the task ID, checkpoint source, noise profile, sampled range
configuration, seed, and code revision so a result cannot be confused with the
Stage 0 baseline.

Stage 1 deliberately retains ground-truth `target_pos_b`. Its results are
therefore an upper-bound robustness test for drone state-estimation error under
**oracle gate-relative guidance**. Good performance under severe VIO drift does
not demonstrate equivalent robustness in a complete autonomous system, because
the policy still receives a perfect relative gate vector. Direct gate-relative
ground truth begins to be removed in Stage 2 and is fully replaced by estimated
pose plus known map in Stage 5.

## Training and Evaluation

Stage 0 artifacts are read-only. The Stage 1 workflow is:

1. Evaluate the unchanged Stage 0 best checkpoint on the Stage 1 `clean` profile.
   Its behavior must match the Stage 0 task within deterministic tolerance.
2. Evaluate the same checkpoint without training on `mild`, `nominal`, and
   `stress` profiles to measure its existing robustness.
3. Copy/load the complete Stage 0 best agent checkpoint into a new Stage 1 run
   and fine-tune through `clean -> mild -> nominal` curriculum levels. Loading
   must restore policy weights, value weights, optimizer state when resuming,
   state `RunningStandardScaler` statistics, value `RunningStandardScaler`
   statistics, and any other registered SKRL checkpoint modules.
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
2. The `clean` providers reproduce all four Stage 0 state observation terms to
   numerical tolerance for multiple seeds and resets.
3. Unit tests prove deterministic seeded noise, quaternion normalization,
   per-environment reset isolation, exact step latency, hold-last dropout, and
   drift accumulation.
4. A leakage test proves Stage 1 policy state terms read only the estimate
   provider.
5. A headless Stage 1 smoke evaluation loads the complete Stage 0 checkpoint,
   including both `RunningStandardScaler` states, without an observation-shape
   mismatch.
6. Given identical observations, seed, environment state, deterministic
   inference mode, and the complete Stage 0 checkpoint, Stage 0 and Stage 1 clean
   paths produce numerically equivalent policy mean actions and value outputs.
7. A small Stage 1 training smoke run creates a separate run directory and a
   loadable checkpoint.
8. A 4,096-environment performance smoke test completes without out-of-memory
   failure and without CPU synchronization in the estimator update path.
9. Baseline, corrupted-before-training, and fine-tuned evaluation results are
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
estimation/
  state_estimate.py
  fake_vio.py
  fake_imu.py
  fake_sensor_cfg.py
  state_estimate_assembler.py
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

Stage 3 will implement an OpenVINS adapter and real/simulated IMU adapter that
produce the same semantic state contract. Middleware stays outside the control policy: a hardware process may
receive ROS 2 sensor messages and publish an estimate, while the policy adapter
consumes the in-process representation. Frame conversion, camera-IMU extrinsics,
`T_WV` alignment, and timestamp-domain conversion happen at the provider boundary
and are tested independently.

The tensor batch dimension is a simulation optimization, not part of the
semantic interface. A one-vehicle instance is therefore the same contract with
batch size one.
