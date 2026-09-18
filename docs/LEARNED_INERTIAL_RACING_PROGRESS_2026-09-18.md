# Learned-Inertial Racing Progress — 2026-09-18

> Branch: `feature/learned-inertial-racing`  
> Goal: build an OpenVINS-free estimator for autonomous drone racing using onboard IMU + thrust history + learned short-horizon motion constraints, then use mapped gate detection/PnP as intermittent absolute pose anchors. Simulator GT may be used for training, diagnostics, rewards, and the known fixed initial pose, but not as a deployed runtime estimator/actor input.

## 1. Starting point

At the beginning of 2026-09-18, the branch already contained:

- a 15-state error-state EKF for attitude, velocity, position, accelerometer bias, and gyro bias;
- fixed-lag velocity+position clones;
- a TCN using 0.5 s / 100 Hz gyro+thrust windows;
- 20 Hz learned prediction and a separately configurable lower fusion rate;
- the V3 target
  [
  r_w=(p_t-p_s)-v_sDelta t
  ]
  to avoid forcing the network to infer arbitrary starting translational velocity;
- Stage2 gate detector/PnP/map infrastructure, not yet ready to be the next integration step.

The main question was: **why does a very accurate TCN still make the EKF worse when fused?**

---

## 2. Realistic IMU corruption was added

The first task was to move away from ideal Isaac IMU validation.

Added deterministic synthetic onboard-IMU corruption:

- accelerometer white noise;
- gyroscope white noise;
- accelerometer bias random walk;
- gyro bias random walk;
- configurable initial biases;
- independent EKF process-noise assumptions;
- deterministic `imu_noise_seed`.

Main test noise:

```text
accel white noise      0.01 m/s^2
gyro white noise       0.001 rad/s
accel bias RW          0.001 m/s^2 / sqrt(s)
gyro bias RW           0.0001 rad/s / sqrt(s)
initial bias sigma     0
```

Representative noisy 10 s translation baseline:

```text
Mode A — IMU only
position RMSE     0.31354 m
velocity RMSE     0.06209 m/s
orientation RMSE  0.06206 deg
```

Relevant commits:

- `49cc5df` configure synthetic IMU noise/bias;
- `f0bc41d` deterministic noise/bias/random-walk injection;
- `67cbfd4` worker CLI;
- `29c3a3d` diagnostics integration.

---

## 3. Covariance tuning was ruled out

V3 shadow mode remained very accurate:

```text
Mode S
state estimate      identical to Mode A
TCN prediction RMSE about 1.19 mm
```

But fused B was worse:

```text
Mode B
position RMSE     about 0.337 m
velocity RMSE     about 0.081 m/s
orientation RMSE  about 0.108 deg
```

At 2 Hz fusion, decreasing the learned covariance multiplier made the result monotonically worse:

```text
cov multiplier   position RMSE
1.00             ~0.337 m
0.50             ~0.386 m
0.25             ~0.431 m
0.10             ~0.535 m
0.04             ~0.674 m
0.01             ~0.963 m
```

Conclusion: the issue was **not simply an overly conservative learned covariance**.

---

## 4. Oracle residual fusion isolated the EKF measurement path

A diagnostic-only oracle mode was added. The TCN still runs, but the EKF receives the exact GT residual for the same window.

Relevant commits:

- `dcac545` oracle config flag;
- `91c23dc` oracle runtime path;
- `8bb6e7b` evaluator CLI/reporting.

Noisy translation result:

```text
A baseline:
position     0.31354 m
velocity     0.06209 m/s
orientation  0.06206 deg

Oracle V3, covariance x1:
position     0.29392 m
velocity     0.05035 m/s
orientation  0.04329 deg
fusions      19
```

This proved that the V3 fixed-lag measurement equation can improve the EKF when the residual is exact.

A much stronger oracle update (`covariance x0.01`) kept position/velocity similar but worsened orientation to about `0.128 deg`, revealing strong cross-covariance coupling into attitude.

---

## 5. EKF attitude convention was corrected

The propagation Jacobian used a right/local multiplicative attitude error,

[
R_{	ext{true}}=R_{	ext{nominal}}operatorname{Exp}(delta	heta),
]

while the old correction injection used the opposite convention.

Fixed:

- `R = R @ Exp(dtheta)`;
- local/right absolute-orientation residual;
- error-state reset Jacobian after attitude injection;
- regression tests.

Relevant commits:

- `b30ef1e`;
- `57d497f`.

This was a real bug, but V3 B performance stayed essentially unchanged. Therefore it was not the main reason for the learned-fusion degradation.

---

## 6. The dominant V3 problem was identified: EKF attitude leaked into the TCN input

V3 constructed world-frame TCN features by rotating body gyro/thrust using the **current EKF attitude**.

Therefore:

[
z_{	ext{TCN}}=f(R_{	ext{EKF}},omega_b,T_b),
]

but the EKF treated the learned measurement as if it were independent of the state.

A diagnostic flag replaced only this feature rotation with GT attitude.

Relevant commits:

- `9a4ab72` truth-orientation diagnostic flag;
- `07b412a` state-dependence isolation;
- `a331ee0` bias/feature-frame diagnostics.

Results:

```text
S standard
position        0.31354 m
velocity        0.06209 m/s
orientation     0.06206 deg
prediction      1.19 mm

B standard
position        0.33797 m
velocity        0.08171 m/s
orientation     0.10899 deg
prediction      2.37 mm

B with GT attitude only for TCN feature rotation
position        0.27248 m
velocity        0.04641 m/s
orientation     0.04452 deg
prediction      1.02 mm
```

This was the decisive proof that V3 had a harmful **estimator-state -> learned-measurement feedback loop**.

---

## 7. V4: endpoint-body residual

V4 removed world/EKF attitude from the TCN input contract.

Input:

```text
gyro_b
thrust_b
```

Target:

[
z_{B_e}=R_t^T[(p_t-p_s)-v_sDelta t].
]

Attitude dependence was moved into the explicit EKF measurement function/Jacobian.

Relevant commits:

- `00e2c77`, `49c3954`, `c5caa03`, `d063cfe`;
- `1a0e0ce`, `4b4d011`, `4dbff74`;
- `a3877e5`, `a2270bf`.

Offline V4:

```text
overall norm RMSE  0.0585 m
translate_x        0.0137 m
lissajous          0.1103 m
racing_like        0.0659 m
```

Online V4 B:

```text
position RMSE     0.5612 m
velocity RMSE     0.2107 m/s
orientation RMSE  0.4686 deg
prediction RMSE   0.0137 m
```

But V4 oracle endpoint-body fusion was good:

```text
position RMSE     0.29342 m
velocity RMSE     0.05024 m/s
orientation RMSE  0.03983 deg
fusions           19
```

Conclusion: **the endpoint-body EKF measurement is usable; raw body-frame TCN representation was too difficult to learn accurately.**

---

## 8. V5: gyro-aligned endpoint-body features

V5 used only window-internal relative rotation.

Pipeline:

```text
raw gyro_b + thrust_b
    -> gyro-only relative attitude integration
    -> rotate every sample into endpoint body frame
    -> TCN
    -> endpoint-body residual
```

No global attitude and no EKF attitude were needed.

Relevant commits:

- `8e5f4af`;
- `98f9bb1`, `a986273`;
- `9d3d261`, `2e1e0a7`, `ed72390`;
- `5e0be2d`.

V5 did not improve enough:

```text
overall norm RMSE  0.0659 m
translate_x        0.0138 m
lissajous          0.1119 m
racing_like        0.0923 m
```

This led to the key observability insight.

---

## 9. Why V4/V5 were fundamentally limited: gravity direction was unobservable

The target before V6 was

[
r_w=(p_t-p_s)-v_sT.
]

Its dynamics include

[
r_w=
rac12gT^2+
int_0^T(T-	au)R(	au)T_b(	au)d	au.
]

In endpoint-body coordinates, gravity contributes

[
rac12R_t^TgT^2.
]

V5 knows only **relative rotation inside the window**. It does not know absolute roll/pitch relative to gravity at the beginning of the window.

Therefore V4/V5 were asking the network to infer a quantity that was not fully observable from their inputs.

This explains why:

- V3 with global attitude was very accurate;
- V4/V5 without global attitude became centimetre-scale;
- more epochs or more network capacity were not the correct fix.

---

## 10. V6: gravity-compensated endpoint-body residual

V6 removed the known gravity-motion term:

[
z_{B_e}
=
R_t^T
left[
(p_t-p_s)-v_sT-rac12gT^2
ight].
]

Runtime:

```text
gyro_b + thrust_b
    -> gyro-only relative attitude integration
    -> endpoint-frame aligned features
    -> TCN
    -> gravity-compensated endpoint-body residual
    -> fixed-lag EKF
```

The TCN no longer needs GT orientation or EKF world attitude.

Relevant commits:

- `60293d8` register V6 mode;
- `bee3d45` gravity-compensated dataset target;
- `b59d0a3` training support;
- `fd54aaa` EKF measurement/Jacobian/update;
- `60e0d18` runtime support;
- `330a2d3` online evaluator;
- `b34cb01`, `8c08c77` tests.

### V6 offline result

This was the main modeling success of the day:

```text
overall coord RMSE  0.0026 m
overall norm RMSE   0.0045 m

translate_x         0.0010 m
tx_scurve 2.5 m     0.0011 m
tx_scurve 4.5 m     0.0009 m
vertical_down       0.0010 m
yaw_45              0.0009 m
lissajous           0.0073 m
racing_like         0.0068 m
```

Comparison:

```text
V4 overall norm RMSE   58.5 mm
V5 overall norm RMSE   65.9 mm
V6 overall norm RMSE    4.5 mm
```

This strongly validates the gravity-observability diagnosis.

### V6 short-run noisy online result

10 s noisy translation:

```text
A noisy
position        0.31354 m
velocity        0.06209 m/s
orientation     0.06206 deg

V6 S
state           identical to A
prediction      0.998 mm

V6 B
position        0.31495 m
velocity        0.05818 m/s
orientation     0.05320 deg
prediction      0.998 mm
fusions         19
```

Most important:

```text
V6 S prediction RMSE = 0.000997547 m
V6 B prediction RMSE = 0.000997549 m
```

The TCN prediction no longer degrades after fusion. The V3 state-to-network feedback loop has been removed.

---

## 11. 30 s / five-seed test exposed the current blocker

The next test was a 30 s Lissajous trajectory with five IMU-noise seeds.

This is an estimator stress trajectory, **not the final 7-gate racing trajectory**.

Five-seed mean:

```text
IMU-only A
position RMSE       2.8080 m
velocity RMSE       0.4031 m/s
orientation RMSE    0.3201 deg
tail20 position     5.6788 m
final position      7.9663 m

V6 fused B
position RMSE      82.6436 m
velocity RMSE       9.0124 m/s
orientation RMSE    3.8783 deg
tail20 position   179.9733 m
final position    256.6012 m
```

Across all five seeds:

```text
B better than A
position      0 / 5
velocity      0 / 5
orientation   0 / 5
tail position 0 / 5
final position 0 / 5
```

Yet V6 TCN prediction stayed around only `2.4–2.6 mm`.

Therefore:

> **The V6 learned representation is accurate, but repeated long-run EKF fusion is unstable.**

This is now the main blocker.

---

## 12. Current working hypothesis

The likely positive-feedback mechanism is:

```text
small learned residual error
    -> learned update modifies attitude / IMU biases through cross-covariance
    -> small attitude/bias error corrupts inertial propagation
    -> gravity is projected into the wrong direction
    -> velocity and position error grow
    -> later learned updates act through the same covariance structure
    -> long-run divergence
```

The 30 s result is structural, not random: the same failure occurred in all five seeds.

The next decisive question is:

> Does an exact V6 oracle residual also diverge over 30 s?

If yes, the problem is in the fixed-lag EKF measurement/clone/cross-covariance structure itself.

If no, then the exact measurement model can work, but the remaining millimetre-level learned error is being amplified into sensitive attitude/bias modes.

---

## 13. Diagnostics added at the end of the day

Long-run metrics were added:

- final position/velocity/orientation error;
- tail-20% RMSE for position/velocity/orientation.

Commit:

- `ca120d0`.

Then per-update EKF diagnostics were added:

- NIS;
- `dx_theta`;
- `dx_velocity`;
- `dx_position`;
- `dx_accel_bias`;
- `dx_gyro_bias`;
- accel-bias final/max norm;
- gyro-bias final/max norm.

Commits:

- `b1b19be`;
- `6a26eb6`.

At the time of this summary, the seed-0 30 s replay needed for the network-vs-oracle diagnostic has been generated successfully.

---

## 14. Current project stage

### Solved / strong progress

- OpenVINS is no longer required by the learned-inertial runtime path.
- Realistic deterministic IMU corruption exists.
- V3 start-velocity residual formulation is implemented.
- 20 Hz prediction / 2 Hz non-overlapping fusion exists.
- Oracle diagnostics can isolate network error from EKF error.
- EKF right-multiplicative attitude convention is corrected.
- V3 estimator-attitude leakage into TCN input was experimentally identified.
- V4 endpoint-body measurement/Jacobian was oracle-validated.
- V5 showed that relative gyro alignment alone is insufficient.
- Gravity observability was identified as the missing variable.
- V6 solves the learned representation problem:
  - millimetre-scale simple-trajectory prediction;
  - single-digit-millimetre Lissajous/racing-like offline prediction;
  - about 1 mm online noisy-translation prediction;
  - no V3-style state-to-network feedback.

### Not solved

- V6 learned updates are not safe over 30 s.
- Long-run B diverges catastrophically while the TCN remains accurate.
- We do not yet know whether the dominant cause is:
  1. the fixed-lag EKF learned-measurement/cross-covariance structure itself; or
  2. amplification of the remaining millimetre-level learned error into attitude/bias states.
- Gate PnP absolute fusion should not be used to hide this failure.
- RL estimator-in-the-loop training should not start yet.

---

## 15. Immediate next experiment

Using the already generated seed-0 30 s replay, run two otherwise identical Mode-B cases:

1. normal V6 network residual;
2. V6 oracle residual.

Inspect:

```text
position / velocity / orientation RMSE
final and tail20 errors
NIS mean/max
dx_theta mean/max
dx_velocity mean/max
dx_position mean/max
dx_accel_bias mean/max
dx_gyro_bias mean/max
accel-bias final/max norm
gyro-bias final/max norm
```

### Branch A: oracle also diverges

Stop tuning the TCN.

Investigate:

- clone/state observability;
- fixed-lag cross-covariance;
- repeated exact-residual synthetic tests;
- Kalman gain blocks;
- whether the learned relative-motion measurement should directly correct attitude/bias states;
- Schmidt-state or constrained-update formulations.

### Branch B: oracle remains stable but network diverges

The fixed-lag measurement works in principle, but millimetre-scale learned errors are being amplified.

Then implement and compare:

- translation-only / velocity-position-only learned corrections;
- frozen attitude/bias correction blocks;
- Schmidt-style learned update;
- long-run NIS/correction diagnostics.

Only after this 30–60 s stability issue is solved should Gate absolute fusion become the next main task.

---

## 16. Roadmap to full drone racing

```text
A. V6 30 s network-vs-oracle diagnosis
    ↓
B. Make learned-inertial fusion long-run stable
    ↓
C. Multi-seed + Lissajous + racing-like + 30–60 s validation
    ↓
D. Freeze learned-inertial core
    ↓
E. Validate mapped Gate detector/PnP absolute-position fusion
    ↓
F. Validate Gate orientation update, covariance, frame conventions, latency
    ↓
G. Multi-gate long-run estimator on the actual 7-gate course
    ↓
H. Actor observations use estimator/map state only
    ↓
I. Estimator-in-the-loop RL training
    ↓
J. Full autonomous drone-racing evaluation and sim-to-real robustness
```

The real Isaac racing environment contains seven gates. The Lissajous trajectory used today is only a safe estimator stress test around the fixed start; it is not the final racing flight path.

---

## 17. Bottom line

The biggest achievement on 2026-09-18 was not simply training a better network.

Two structural issues were identified and removed:

1. **V3:** EKF/world attitude leaked into the TCN input, making the learned measurement implicitly state-dependent and creating a feedback loop.
2. **V4/V5:** after removing global attitude, the learned target still contained a gravity-direction term that was not observable from relative gyro+thrust alone.

**V6 fixes both and produces a deployment-compatible, millimetre-accurate learned residual.**

The remaining blocker is now sharply localized:

> **The learned residual is accurate, but repeated long-run EKF fusion is unstable.**

So the project has moved from **“learn the right inertial representation”** to **“make the fixed-lag learned EKF update statistically and dynamically stable over racing time horizons.”**
