# Hybrid Learned-Motion VIO Progress — 2026-09-15

Branch: `debug/openvins-imu-convention-isolation`

This note captures the current state of the OpenVINS + learned inertial-motion correction work, the main failure modes isolated so far, the changes already implemented, and the next tuning step.

## 1. Current goal

The target architecture is a hybrid learned motion-constrained VIO pipeline for the Swift drone:

```text
Isaac / vehicle IMU + collective thrust
        ↓
learned TCN short-horizon displacement + uncertainty
        ↓
raw OpenVINS
        ↓
raw VIO frame-jump isolation
        ↓
learned translational drift EKF
        ↓
explicit learned drift-velocity measurement
        ↓
common-mode learned position-residual slew
        ↓
learned-corrected VIO
        ↓
Swift gate PnP absolute correction (future integration)
```

The learned network consumes a 0.5 s, 100 Hz motion window with 6-D input `[gyro_w, thrust_w]` and predicts world-frame displacement plus uncertainty.

## 2. Learned-motion model status

The current checkpoint is:

```text
artifacts/learned_motion/final/model.pt
```

Training / held-out performance from the current dataset:

- best epoch: 75 (1-based)
- validation displacement RMSE: ~0.111 m
- test displacement RMSE: ~0.088 m
- test axis RMSE: ~[0.132, 0.075, 0.007] m

In hybrid OpenVINS experiments, NN 0.5 s displacement error remains on the order of ~0.06–0.08 m RMSE while raw OpenVINS 0.5 s displacement error is typically several metres. The present evidence therefore does **not** indicate that network retraining is the main bottleneck.

## 3. Failure modes isolated and fixed

### 3.1 Position-to-velocity Kalman cross-coupling

Originally, position-related learned / absolute measurements could modify the velocity-drift mean through EKF cross-covariance. This produced non-physical velocity spikes.

Fixes:

- absolute position update is Schmidt-style position-only in the mean update;
- learned relative update has an explicit `relative_position_only` A/B mode;
- velocity rows of the learned relative gain are frozen in position-only mode.

This removed the large correction-induced velocity explosions.

### 3.2 Raw OpenVINS position frame jumps

Raw OpenVINS was observed to make metre-scale adjacent-frame position jumps in ~10 ms while ground truth moved only millimetres.

Implemented raw jump isolation:

```text
observed_dp = p_raw(k) - p_raw(k-1)
expected_dp = 0.5 * (v_raw(k-1) + v_raw(k)) * dt
residual    = observed_dp - expected_dp
```

When `||residual|| > threshold`, the residual is accumulated as a raw-VIO frame-shift compensation and removed only from the internal VIO stream. Original raw VIO is preserved for diagnostics.

Current A/B controls:

```text
--raw_vio_jump_isolation
--raw_vio_jump_threshold_m 0.5
```

This prevents instantaneous raw OpenVINS frame shifts from leaking directly into the learned-corrected output.

### 3.3 Missing velocity-drift estimate

After position-only relative updates, corrected velocity originally remained identical to raw OpenVINS velocity because `v_d_current` was frozen.

Offline diagnosis showed that the drift-rate inferred from the learned displacement residual tracked raw velocity error extremely strongly:

```text
estimated drift-rate mean = 5.219 m/s
raw velocity error mean   = 5.363 m/s
mean |difference|         = 0.461 m/s
correlation               = 0.9826
```

Implemented explicit velocity-drift measurement from each accepted jump-isolated learned window:

```text
Delta p_d = Delta p_vio_isolated - Delta p_nn
v_d_meas  = Delta p_d / Delta t
```

The update is velocity-only: only `v_d_current` may change.

A representative no-Oracle 6000-step run with this enabled produced:

```text
raw velocity RMSE      = 10.066 m/s
learned velocity RMSE  = 0.978 m/s
raw position RMSE      = 135.501 m
learned position RMSE  = 1.410 m
```

Thus the explicit drift-velocity path is a major improvement and should remain in the architecture.

### 3.4 Learned position hard-snap at 0.5 s boundaries

Even after velocity drift was fixed, accepted learned position updates still produced discrete position corrections at learned-window boundaries.

Measured before smoothing:

```text
boundary jump mean = 0.408 m
boundary jump p95  = 1.156 m
boundary jump max  = 1.658 m
```

Offline decomposition showed that output jump and instantaneous learned position injection were nearly identical:

```text
correlation(output jump, position injection) = 0.99373
```

This confirmed that the remaining sawtooth came from instantaneous position-drift mean injection, not from the TCN, raw jump detector, or explicit velocity update.

## 4. Common-mode position residual slew

Implemented optional learned position residual slew:

```text
--learned_position_residual_slew
--learned_position_residual_max_rate_mps <rate>
```

After an accepted learned position update, the measurement-induced common-mode position correction is moved into a pending correction buffer instead of being exposed immediately. The pending correction is then released at a bounded rate.

The release is applied equally to both:

```text
p_d_anchor
p_d_current
```

so the learned relative state `p_d_current - p_d_anchor` is preserved while the absolute/common-mode output correction is made continuous.

Absolute position anchors clear any remaining pending learned correction, preventing stale learned common-mode residuals from being released after a newer absolute observation.

Evaluator diagnostics now include:

```text
learned_position_injection_norm_m
learned_position_release_norm_m
learned_position_pending_norm_m
```

## 5. 4 m/s slew result

6000-step `translate_x`, no Oracle, with:

```text
relative position-only          ON
explicit drift velocity         ON
velocity sigma floor            0.5 m/s
raw jump isolation              ON
raw jump threshold              0.5 m
position residual slew          ON
position residual max rate      4.0 m/s
```

Summary:

```text
raw position RMSE       = 111.649 m
learned position RMSE   = 1.947 m
raw velocity RMSE       = 7.742 m/s
learned velocity RMSE   = 1.008 m/s
learned updates         = 91 / 91 accepted
raw jump detections     = 45
final pending norm      = 0.0 m
```

Boundary smoothness:

```text
all adjacent learned output:
  mean = 0.0210 m
  p95  = 0.0583 m
  p99  = 0.2278 m
  max  = 0.4650 m

accepted learned boundaries:
  mean = 0.0405 m
  p95  = 0.0797 m
  max  = 0.2442 m
```

The observed release limit was exactly ~0.04 m per 10 ms sample, consistent with:

```text
4.0 m/s * 0.01 s = 0.04 m
```

This reduced the previous ~1–2 m learned-window hard snaps to centimetre-scale p95 boundary motion.

## 6. 8 m/s slew result

A second 6000-step run used a residual max rate of 8.0 m/s.

Summary:

```text
raw position RMSE       = 134.728 m
learned position RMSE   = 2.293 m
raw velocity RMSE       = 8.971 m/s
learned velocity RMSE   = 1.049 m/s
learned updates         = 92 / 92 accepted
raw jump detections     = 53
final pending norm      = 0.0 m
```

Boundary smoothness:

```text
accepted learned boundaries:
  mean = 0.0701 m
  p95  = 0.0932 m
  max  = 0.2917 m
```

The observed release limit was ~0.08 m per 10 ms sample, as expected:

```text
8.0 m/s * 0.01 s = 0.08 m
```

8 m/s therefore remains stable and still eliminates metre-scale hard snaps, but it has not yet demonstrated a clear accuracy advantage over 4 m/s.

Important caveat: the 4 m/s and 8 m/s runs are **not strict seed-matched A/B runs**. Raw OpenVINS realizations differed materially between runs (raw drift level and jump count differed), so their position RMSE values should not be interpreted as a controlled direct comparison.

## 7. Current interpretation

The current evidence supports the following conclusions:

1. The TCN displacement predictor is not the dominant current failure mode.
2. Raw OpenVINS contains genuine metre-scale position/reference-frame discontinuities; jump isolation is required.
3. Learned position-to-velocity Kalman cross-coupling was unsafe; position-only relative updates should remain.
4. The learned displacement residual contains a useful velocity-drift signal; explicit velocity-drift estimation reduces velocity error by roughly an order of magnitude in representative runs.
5. The remaining 0.5 s sawtooth was caused by instantaneous learned position injection.
6. Common-mode residual slew solves that hard-snap problem without reintroducing velocity instability.
7. 4 m/s is currently the conservative candidate; 8 m/s is also stable. A 6 m/s experiment is the next useful tuning point.

## 8. Current recommended experimental configuration

For continued no-Oracle tuning:

```bash
--learned_relative_position_only \
--learned_drift_velocity_from_displacement \
--learned_drift_velocity_sigma_floor_mps 0.5 \
--learned_position_residual_slew \
--learned_position_residual_max_rate_mps 4.0 \
--raw_vio_jump_isolation \
--raw_vio_jump_threshold_m 0.5
```

Use 4 m/s as the current conservative baseline. Test 6 m/s next before selecting a final slew rate.

## 9. Next steps

Immediate next step:

- run a fresh OpenVINS 6000-step experiment at `6.0 m/s` slew rate;
- compare boundary p95/max, learned position RMSE, learned velocity RMSE, and pending-release behavior;
- do not change the TCN, jump threshold, or velocity sigma floor during this A/B.

After slew-rate selection:

- add reproducible seed control for stricter A/B runs;
- validate on more aggressive racing-like trajectories, not only `translate_x`;
- integrate the Swift gate PnP path as an intermittent absolute correction source;
- repeat evaluation without Isaac oracle dependencies;
- keep Oracle only as a simulation diagnostic upper bound.

## 10. Verification status

Before this progress note, the branch had passed the dedicated `debug-opvs pure checks` workflow with:

```text
42 passed
```

The current implementation includes regression coverage for:

- learned position-only updates;
- explicit velocity-drift updates;
- raw VIO jump isolation;
- deferred bounded learned position residual release;
- pending correction clearing on absolute position anchor;
- evaluator CSV schema consistency.
