# Learned Inertial V6.2 — UZH-style stochastic cloning

Date: 2026-09-20

Branch:

```text
feature/v6.2-uzh-stochastic-cloning
```

Base:

```text
feature/v6.1-coupled-motion-training
```

## Motivation

V6.1 isolated a remaining estimator-side failure mode: millimetre-scale learned
or Oracle innovations could still produce metre-scale current-position
corrections when the relative-motion update used
`freeze_clones_attitude_bias`.

That structure is not the update used by Cioffi et al. / UZH IMO. In the paper,
the full filter state contains a current IMU state plus past cloned states, the
learned measurement is a relative displacement between two states, and the
standard full Kalman gain updates the correlated state. The public UZH
implementation further stores cloned rotation, velocity, and position.

V6.2 removes the runtime dependence on the legacy "historical clone frozen,
current state corrected" structure.

## What changed

### 1. Clone state: 6D -> 9D

V6.1 clone error state:

```text
[dv, dp]
```

V6.2 clone error state:

```text
[dtheta, dv, dp]
```

Each nominal clone now stores:

```text
[R_w_b, v_w_b, p_w_b]
```

Clone augmentation is an exact stochastic copy of the current kinematic state.
The clone covariance and current/clone cross-covariance come from the full
current covariance through the augmentation Jacobian.

### 2. Relative measurement uses two clones

V6.1 runtime factor used:

```text
historical start clone  <->  evolving current state
```

V6.2 runtime factor uses:

```text
historical start clone  <->  endpoint clone
```

For the clean V6.1 gravity-compensated endpoint-body target, the prediction is

```text
h = R_j^T [ (p_j - p_i) - v_i * dt - 0.5 * g * dt^2 ]
```

where both `i` and `j` are timestamped stochastic clones.

The Jacobian therefore acts on:

```text
end-clone attitude
start-clone velocity
start-clone position
end-clone position
```

and has no direct current-position measurement block.

### 3. Endpoint clone is created before the update

The V6.2 scheduler is:

```text
propagate to endpoint
        |
        v
clone endpoint [R,v,p]
        |
        v
form start-clone <-> end-clone learned factor
        |
        v
full covariance-consistent Kalman update
        |
        v
marginalize used start clone
```

This is intentionally different from the V6.1 scheduler fix. In V6.1,
clone-before-update was unsafe because the clone rows were frozen. In V6.2 the
endpoint clone participates in the same full update, so it stays synchronized
with the evolving endpoint state through exact cross-covariance.

### 4. Full Kalman gain is mandatory for V6.2 runtime learned fusion

The new two-clone update methods always call the EKF update with:

```text
gain_mode = full
```

The learned-inertial environment also rejects `freeze_*` runtime gain modes.

The old masked-gain code remains only as a legacy diagnostic path so previous
V6.1 failure modes and regression tests can still be reproduced.

### 5. Gauge/observability diagnostics

The estimator now exposes a four-column local-error basis for:

```text
global yaw
global translation X
global translation Y
global translation Z
```

and provides the UZH/TLIO-style information diagnostic:

```text
diag(N^T pinv(P) N)
```

Every Kalman update also records:

```text
measurement_unobservable_projection_norm
measurement_translation_nullspace_norm
unobservable_information_diag
```

For a pure relative two-clone factor, the translation-nullspace projection
should be approximately zero.

## What intentionally did not change

V6.2 does **not** revert the clean V6.1 network to the original paper's
world-frame network input/output representation.

The current learned target remains the deployment-oriented endpoint-body,
gravity-compensated residual. The purpose of V6.2 is to import the UZH
**filter structure** — stochastic [R,v,p] clones, a two-endpoint relative
factor, full cross-covariance, full Kalman update, and fixed-lag
marginalization — without reintroducing estimator-attitude leakage into the
network input.

The Joseph covariance update is also retained. It is numerically safer than
the algebraically equivalent simplified covariance form and is not the source
of the V6.1 structural deviation.

## Primary code paths

```text
estimation/learned_inertial_odometry.py
tasks/drone_racer/learned_inertial_racing_env.py
tasks/drone_racer/drone_racer_learned_inertial_env_cfg.py
scripts/estimation/evaluate_learned_inertial_mode.py
tests/estimation/test_learned_inertial_odometry.py
```

## Local validation

From the repository root:

```bash
git fetch myfork
git checkout feature/v6.2-uzh-stochastic-cloning
git pull --ff-only myfork feature/v6.2-uzh-stochastic-cloning

./.conda-env/bin/python -m pytest -q \
  tests/estimation/test_learned_inertial_odometry.py

./.conda-env/bin/python -m pytest -q \
  tests/estimation/test_learned_motion.py \
  tests/estimation/test_learned_motion_dataset.py \
  tests/estimation/test_learned_inertial_odometry.py
```

The V6.2-specific tests verify:

- 9D [R,v,p] clone augmentation and covariance dimensions;
- two-clone relative Jacobians;
- finite-difference correctness of the endpoint-body gravity-compensated
  two-clone factor;
- global-translation and global-yaw nullspace projection;
- full-gain fusion even if a legacy mask is configured directly on the core
  estimator;
- synchronization of the endpoint clone and evolving endpoint state after the
  joint update.

## First estimator experiment

Do not retrain the TCN yet. First run exact Oracle measurements through the new
filter structure.

Recommended comparison:

```text
A: IMU-only
V6.1: Oracle x30, freeze_clones_attitude_bias, scheduler-fixed
V6.2: Oracle x30, uzh_two_clone_full
```

Run the existing 30 s five-seed Lissajous replay set. The first acceptance
criteria are structural rather than only RMSE:

```text
measurement_translation_nullspace_norm ~ 0
endpoint clone stays synchronized with endpoint state
no stale-clone correction echo
no masked-gain current-position-only correction path
full clone/current correction remains covariance-consistent
```

Then compare position and velocity RMSE against the V6.1 baselines.

## Next steps after the Oracle test

If Oracle is stable:

1. run clean V6.1 network measurements through the V6.2 filter;
2. compare legacy covariance floor versus V6.1-native uncertainty;
3. run longer 30–60 s coupled/racing-like multi-seed tests;
4. inspect yaw observability diagnostics under long aggressive motion;
5. only if required, add an FEJ/observability-constrained linearization layer;
6. then reintroduce mapped gate PnP as the intermittent absolute position
   anchor.

Gate PnP should not be used to hide an unresolved inertial-core inconsistency.
