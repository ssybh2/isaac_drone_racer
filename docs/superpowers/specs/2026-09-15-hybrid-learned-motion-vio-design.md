# Hybrid Learned Motion-Constrained VIO Design

## Status

Approved for implementation on `debug/openvins-imu-convention-isolation`. The GT-initialization A/B campaign is intentionally skipped per project direction.

## Goal

Add a learned short-horizon motion constraint between raw OpenVINS and the existing Swift gate-map correction so that systematic VIO translation drift can be corrected without replacing OpenVINS or directly regressing global pose.

## Architecture

The deployed estimator has three explicit state products:

1. `raw_openvins`: world-aligned OpenVINS pose/velocity/attitude;
2. `learned_corrected`: OpenVINS translation/velocity corrected by a learned relative-motion drift estimator while retaining OpenVINS attitude;
3. `gate_fused`: the existing mapped-gate correction applied to `learned_corrected` when gate measurements are available.

The learned model consumes a fixed-duration history of body angular rate and collective thrust. It predicts relative displacement over the same window plus diagonal displacement variance. Simulator truth is permitted only when constructing supervised training labels; deployed inference never consumes truth.

## Learned measurement

For a non-overlapping window from `t_i` to `t_j`:

- raw VIO displacement: `dp_vio = p_vio(t_j) - p_vio(t_i)`;
- learned physical displacement: `dp_nn`;
- relative drift measurement: `z = dp_vio - dp_nn`.

The drift filter state is `x=[p_anchor_d, p_current_d, v_current_d]`. During propagation, the anchor is fixed while current position integrates drift velocity. The learned update uses:

`h(x) = p_current_d - p_anchor_d`

with Jacobian `H=[-I, I, 0]`. After an accepted update the current drift position is cloned into a new anchor and the next non-overlapping learned window begins. This preserves the relative-measurement structure without directly overwriting VIO pose.

## Model contract

The first trainable model is a UZH-inspired causal TCN:

- input: six channels `[gyro_x, gyro_y, gyro_z, thrust_x, thrust_y, thrust_z]`;
- nominal sample rate: 100 Hz;
- nominal window: 0.5 s;
- output: six scalars `[dp_x, dp_y, dp_z, log_var_x, log_var_y, log_var_z]`;
- loss: Gaussian negative log likelihood on relative displacement with a small variance floor.

Runtime torch imports are lazy so the pure estimation tests remain runnable with NumPy only. The estimator accepts an injected predictor interface, allowing deterministic tests without a checkpoint.

## Input frame

V0 uses world-frame gyro and thrust, following the public UZH implementation. Body-frame measurements are rotated using the world-aligned OpenVINS attitude at each accepted VIO sample. A later body-frame-displacement ablation can remove this dependence; it is not required for the first implementation.

## Safety and gating

Learned updates are probabilistic measurements, not pose replacements. The filter:

- rejects non-finite or non-positive covariance;
- uses configurable chi-square/NIS gating;
- uses Joseph-form covariance updates;
- does not update until one complete learned window exists;
- preserves OpenVINS attitude exactly;
- remains disabled unless a learned checkpoint is explicitly configured.

The existing gate-map Kalman correction remains the absolute/global anchor and receives `learned_corrected` VIO rather than raw VIO when learned correction is enabled.

## Data collection

The OpenVINS fault-isolation CSV is extended with processed collective thrust and the three translational components of the applied body thrust vector. Existing truth pose, gyro, and VIO fields remain unchanged. A conversion/training script builds windows and truth displacement labels from these traces. Train/validation/test splitting must be by complete trace/trajectory, never by random rows from the same trace.

## Acceptance criteria

- Pure tests prove the relative-drift update math, covariance symmetry/PSD, NIS rejection, orientation preservation, window timing, and reset behavior.
- The learned model module can be imported without torch installed; torch is required only when constructing/training/loading a TCN.
- Runtime remains behaviorally identical when no learned checkpoint is configured.
- With a checkpoint configured, the environment exposes both raw and learned-corrected VIO and sends learned-corrected VIO into existing Swift gate fusion.
- Fault-isolation traces contain thrust fields sufficient to build UZH-style six-channel training windows.
- CI runs the new pure tests.
- No claim of estimator accuracy improvement is made until Isaac/OpenVINS experiments are run on the project machine with a trained checkpoint.
