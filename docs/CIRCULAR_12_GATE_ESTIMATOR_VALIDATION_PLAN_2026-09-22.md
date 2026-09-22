# 2026-09-22 Circular-12 GT Upper-Bound Plan for Estimator Replacement

## Purpose

The immediate objective is not to optimize perception-aware racing reward.  It
is to create a racing task in which the production camera has a reasonable
geometric opportunity to observe gate corners, train a strong GT policy on that
task, and then switch only the 31-D policy observation source from simulator GT
to the project's IMU + learned-inertial + SC-EKF + mapped-gate estimator.

The experiment is therefore:

```text
same track
same policy architecture
same CTBR action
same next-gate map interface

GT state --------------------> 31-D policy  (upper bound)
                                   |
                                   | after GT + visibility pass
                                   v
estimated p/v/R -------------> 31-D policy  (target experiment)
```

## Why the previous Easy-7 loop was a poor estimator-isolation track

The validated Stage2 pinhole camera uses:

```text
resolution: 256 x 256
fx = fy = 293.1997 px
cx = cy = 128 px

horizontal FOV ~= 47.17 deg
half-FOV ~= 23.58 deg
```

The previous Easy-7 loop had radius 8 m and seven gates.  Consecutive polygon
segments changed heading by:

```text
360 / 7 = 51.43 deg
```

This is larger than the entire ~47.2 deg horizontal FOV.  V3 then added a
continuous camera-angle penalty to a trajectory that was already geometrically
difficult for the camera, and PPO found a reward-optimal but non-racing policy.

V3 is therefore retired as an estimator-validation direction.

## New track geometry

New config:

```text
CIRCULAR_12_GATE_TRACK_CONFIG
```

Geometry:

```text
gate count     = 12
circle radius  = 12 m
circle centre  = (0, 12)
gate height    = unchanged
gate yaw       = local circle tangent
yaw step       = 30 deg
```

Key positions:

```text
Gate 1   (  0.000,  0.000)
Gate 4   ( 12.000, 12.000)
Gate 7   (  0.000, 24.000)
Gate 10  (-12.000, 12.000)
```

Neighbour spacing:

```text
2 * 12 * sin(15 deg) ~= 6.21 m
```

This remains close to the previous Easy-7 spacing of ~6.94 m, so the task does
not become a dense sequence of nearly overlapping gates.

With tangent-aligned gate normals, the next gate centre is approximately:

```text
30 / 2 = 15 deg
```

off the current tangent direction at a gate crossing.  This is comfortably
inside the camera half-FOV of ~23.58 deg.

Using the actual 1.524 m gate opening, calibrated camera offset and radius-12
geometry, both horizontal sides of the next opening are approximately within
the camera's horizontal image limits at the nominal level gate-crossing pose.
High-speed roll/pitch can still cause vertical loss, so real GT projection
statistics remain the acceptance test.

## New GT task

```text
Isaac-Drone-Racer-Swift-CTBR-GT-Circular12-v0
```

The task deliberately keeps the successful GT racing reward:

```text
termination
progress
gate_passed
weak lookat_next
```

There is no V1/V2/V3 camera reward.

The policy contract remains exactly 31-D:

```text
GT p_w, v_w, R_wb                15D
known-map next-gate corners      12D
previous CTBR action              4D
                                ----
                                 31D
```

The PPO profile remains matched to the successful GT baseline:

```text
4096 env
rollout 24
5 PPO epochs
4 minibatches
lr 1e-4
gamma .99
lambda .95
entropy .005
3 x 256 ELU shared model
50000 trainer timesteps
```

Logging:

```text
logs/skrl/swift_ctbr_gt_circular12
```

## Acceptance order

Do not connect estimator state to the actor until both of the following are
true.

### 1. GT racing upper bound

The GT actor must recover stable multi-lap racing on the Circular-12 track.

Do not select checkpoints only by total reward.  Gate-pass performance and
completion are mandatory first-stage filters.

### 2. Camera geometry / corner observability

Collect production-camera frames from the successful GT policy and measure:

```text
active next gate visible corners
ANY mapped gate visible corners
speed / body rate
horizontal / vertical gate bearing
distance to gate
```

The current estimator path requires at least two visible corners for a mapped
direct-reprojection update.

Only after the GT policy can race and the corner statistics are adequate should
the policy input source be switched.

## Estimator replacement experiment

The intended later comparison is strict:

```text
A. GT upper bound
   p, v, R from simulator GT
   next-gate geometry from the known Circular-12 map

B. Estimator policy
   p_hat, v_hat, R_hat from IMU + learned inertial + SC-EKF + vision
   next-gate geometry from the same known Circular-12 map
```

No change should be made to:

```text
policy network
CTBR action definition
track
gate map
next-gate 12-D representation
checkpoint architecture
```

The purpose is to make observation source the controlled experimental variable.

## Important caveat

Increasing gate count solves the large horizontal next-gate direction jump, but
it does not guarantee visibility under aggressive body roll/pitch because the
camera remains rigidly mounted to the vehicle.

Therefore the final decision is based on measured GT projection statistics,
not geometry alone.
