# Learned-Inertial Drone Racing Progress — 2026-09-20

## 1. Executive summary

As of 2026-09-20, the project has moved from estimator-architecture research into the final integration stage before reinforcement learning.

The main objective remains:

> Build an OpenVINS-free racing stack in which onboard IMU/thrust signals and a learned inertial model provide short-horizon motion information, a stochastic-cloning EKF maintains the full vehicle state, mapped gate observations provide absolute drift correction, and the final racing policy acts only on deployment-available estimated state rather than simulator ground truth.

The current production candidate is the V6.4/V6.5 stack:

```text
onboard IMU + thrust
        |
        v
Body-Delta-v TCN
        |
        v
stochastic-cloning EKF
        ^
        |
detected gate corner pixels
        |
known static gate map
        |
direct pixel reprojection
```

The most important result achieved today is that replacing standalone planar PnP pose recovery with direct mapped-corner reprojection changed the estimator from roughly 0.3 m position RMSE to roughly 0.05 m position RMSE and also made partial 2/3-corner observations usable.

The estimator is no longer the primary blocker to RL. The remaining work is mainly deployment-faithful RL integration: remove indirect GT dependence from target progression, freeze the validated estimator configuration into an RL task, validate one estimated-state closed loop, and then begin PPO training.

V6.6 adds timestamp-aware delayed visual updates using stochastic pose clones. The 50/100/200 ms compensated-latency experiments have now completed successfully: all three cases returned to the same approximately 5-6 cm position-RMSE regime as the neutral baseline. This validates the delayed-measurement stochastic-clone design. These experiments remain robustness/engineering hardening rather than a required RL gate for the intended hardware, because the target hardware pipeline is not expected to incur such large uncompensated visual latency.

---

## 2. What changed from the original direction

Earlier work used OpenVINS/VIO as the principal state estimator. Small convention/model errors accumulated through integration and produced large position errors. The project therefore moved toward an OpenVINS-free learned-inertial architecture inspired by UZH learned inertial odometry.

The development path became:

```text
OpenVINS/VIO debugging
    ->
learned inertial TCN
    ->
stochastic-cloning EKF
    ->
mapped gate absolute correction
    ->
PnP diagnosis
    ->
direct gate-corner reprojection
    ->
robustness validation
    ->
RL integration
```

The project is now at the last arrow.

---

## 3. Learned inertial estimator achievements

### 3.1 Body-Delta-v learned factor

The current learned factor predicts endpoint-body-frame gravity-compensated velocity change:

```text
z_b = R_j^T [ (v_j - v_i) - g * dt ]
```

The deployed network input is based on IMU gyro and thrust/force history, aligned to the endpoint body frame. Simulator truth is used only to construct offline supervision labels and evaluation metrics, not as a runtime estimator input.

Current checkpoint:

```text
artifacts/imo_tcn/model_v6_2_body_delta_velocity_balanced.pt
```

Validated runtime network error remains approximately:

```text
Body-Delta-v norm RMSE ~= 0.00343 m/s
```

The learned prediction is evaluated at 20 Hz and conservatively fused at 2 Hz to reduce process/measurement double counting because the learned measurement reuses the same IMU/thrust history as propagation.

The covariance floor is intentionally conservative:

```text
used sigma ~= 0.0625 m/s
```

No TCN retraining is currently justified by the estimator results.

### 3.2 UZH-style stochastic cloning

The EKF now maintains current state plus stochastic [R, v, p] clones with full cross-covariance.

Important properties already implemented and validated:

- right-multiplicative attitude error,
- stochastic cloning with cross-covariance,
- full Kalman gain,
- clone marginalization,
- two-clone learned factors,
- gauge-consistency diagnostics,
- current-state and clone-state updates through covariance coupling.

This structure is the basis both for learned inertial fusion and the V6.6 delayed-vision work.

---

## 4. V6.3: diagnosis of the gate/PnP problem

V6.3 established that the poor gate correction was not caused by a broken camera model, gate geometry, camera-to-body extrinsic, or frame convention.

With exact simulator gate corners sent through the same PnP path, errors were essentially numerical zero:

```text
gate translation error ~ numerical zero
body translation error ~ 1e-7 m
body orientation error ~ 1e-5 deg
```

The real detector itself was also substantially better than initially suspected:

```text
valid detector corner RMSE ~= 1.09 px
accepted detector corner RMSE ~= 1.03 px
```

However, the same approximately 1 px corner error produced a large planar-PnP orientation error:

```text
PnP orientation RMSE ~= 6.7 deg
PnP-derived body position RMSE ~= 0.55 m
```

A key decomposition showed:

```text
FULL PnP body position RMSE       ~= 0.548 m
GT attitude + PnP translation     ~= 0.100 m
EKF attitude + PnP translation    ~= 0.113 m
```

This isolated the dominant failure mechanism:

> Small image-space errors on a planar gate produce poorly conditioned PnP orientation; at 4-6 m range the resulting angular error creates a large translational lever-arm error.

This was the main structural reason to abandon PnP pose as the EKF visual measurement.

---

## 5. V6.4: direct gate-corner reprojection

V6.4 replaced:

```text
detector corners -> planar PnP -> 6DoF gate/body pose -> EKF
```

with:

```text
detector corners + known gate map -> direct pixel residual -> EKF
```

For a mapped gate corner P_w:

```text
q_b = R_wb^T (P_w - p_wb)
p_c = R_bc^T (q_b - t_bc)
uv  = project(K p_c)
```

The filter directly fuses:

```text
r = z_pixel - h(x)
```

The analytic Jacobian with respect to current attitude and position has been finite-difference tested.

The visual factor supports 2, 3, or 4 semantic corners. Therefore a complete four-corner observation is no longer required.

Robustification includes:

- per-corner normalized innovation,
- Huber weighting,
- full joint normalized NIS gate,
- pixel-space mapped-gate association,
- full Kalman gain.

### 5.1 V6.4 30-second seed-0 result

Representative result:

```text
position RMSE        = 0.0522 m
position tail20      = 0.0417 m
position max         = 0.1235 m
position final       = 0.0405 m

velocity RMSE        = 0.0275 m/s
orientation RMSE     = 0.696 deg

gate acceptance      = 536 / 750 = 71.47%
association match    = 97.66%
```

The same run had:

```text
4-corner accepted = 329
3-corner accepted = 35
2-corner accepted = 172
```

The direct reprojection path therefore recovered a large amount of useful visual information that the PnP path could not use.

### 5.2 Improvement over V6.3 PnP

Approximate comparison:

| Metric | V6.3 PnP Mode C | V6.4 direct reprojection |
|---|---:|---:|
| Position RMSE | ~0.31 m | ~0.05 m |
| Position tail20 | ~0.54-0.56 m | ~0.04 m |
| Velocity RMSE | ~0.15 m/s | ~0.027 m/s |
| Orientation RMSE | ~0.80-0.85 deg | ~0.65-0.70 deg |
| Gate acceptance | ~42% | ~71.5% |

The approximately 84% reduction in position RMSE is a structural measurement-model improvement, not a small hyperparameter gain.

---

## 6. V6.4 five-seed robustness result

Five-seed evaluation under the same replay trajectory with different IMU-noise realizations produced:

```text
position RMSE:
mean = 0.049345 m
std  = 0.005211 m
min  = 0.042954 m
max  = 0.058528 m

position tail20:
mean = 0.041369 m
max  = 0.056534 m

position max error:
mean = 0.109243 m
worst = 0.122542 m

velocity RMSE:
mean = 0.027219 m/s
std  = 0.001750 m/s

orientation RMSE:
mean = 0.647294 deg
std  = 0.071193 deg

gate acceptance:
71.6% for all five runs

association match:
mean = 97.56%

normalized NIS mean:
mean = 1.100

Huber weight mean:
mean = 0.9796
```

Total accepted partial-corner updates across five runs:

```text
4 corners = 1645
3 corners = 175
2 corners = 865
```

Thus 1040 accepted updates, about 38.7% of all accepted visual updates, used incomplete gate observations.

Interpretation:

- the estimator is stable across the tested inertial-noise seeds;
- pixel covariance calibration is broadly consistent;
- Huber is acting mainly as an outlier safety mechanism rather than compensating for a generally bad detector;
- the current gate association is good enough to continue, although mismatches should remain monitored during racing trajectories.

---

## 7. V6.5 perception robustness results

V6.5 added controlled visual corruption without changing simulator truth, replay actions, or the estimator architecture.

Tested perturbations:

- deterministic camera blackout,
- extra pixel noise,
- corner erasure infrastructure,
- uncompensated visual latency infrastructure.

### 7.1 Camera blackout

Seed 2 results:

| Case | Position RMSE | Tail | Max | Velocity RMSE | Orientation RMSE |
|---|---:|---:|---:|---:|---:|
| neutral | 0.0578 m | 0.0552 m | 0.1178 m | 0.0296 m/s | 0.766 deg |
| 0.25 s blackout | 0.0569 m | 0.0533 m | 0.1196 m | 0.0296 m/s | 0.749 deg |
| 0.50 s blackout | 0.0565 m | 0.0525 m | 0.1192 m | 0.0295 m/s | 0.746 deg |
| 1.00 s blackout | 0.0575 m | 0.0529 m | 0.1186 m | 0.0297 m/s | 0.756 deg |

Conclusion:

> Up to 1 second of complete visual loss did not materially damage the estimator on this trajectory.

This is an important validation of the IMU + Body-Delta-v propagation path.

### 7.2 Additional pixel noise

| Extra pixel noise | Position RMSE | Velocity RMSE | Orientation RMSE | NIS/dof mean |
|---|---:|---:|---:|---:|
| 0.5 px | 0.0510 m | 0.0288 m/s | 0.674 deg | 1.392 |
| 1.0 px | 0.0485 m | 0.0314 m/s | 0.646 deg | 2.140 |
| 2.0 px | 0.0520 m | 0.0447 m/s | 0.676 deg | 4.048 |

The extra noise was injected while the EKF still assumed the nominal 0.85 px measurement sigma. Therefore the rising NIS is expected.

Conclusion:

> The current direct reprojection estimator tolerates materially worse pixel measurements without position divergence.

There is no reason to rerun burst/noise tests during the current V6.6 latency experiment.

---

## 8. Latency finding and V6.6 status

### 8.1 What V6.5 uncompensated latency showed

V6.5 deliberately fused delayed pixels against the current state without out-of-sequence compensation.

Results were catastrophic:

```text
50 ms:
position RMSE ~= 9.55 m

100 ms:
position RMSE ~= 512 m

200 ms:
position RMSE ~= 824 m
```

This result should not be interpreted as a weakness of the nominal direct-reprojection factor itself. The measurement model in this stress experiment was intentionally inconsistent:

```text
old image z(t - tau)
       fused against
current pose x(t)
```

For a moving drone, this generates a false pixel innovation.

### 8.2 V6.6 implementation

Branch:

```text
feature/v6.6-latency-compensated-reprojection
```

V6.6 adds timestamp-aware delayed visual updates:

```text
capture image at t_capture
        |
        v
create/retain stochastic [R,v,p] clone at t_capture
        |
        v
measurement arrives later
        |
        v
evaluate gate reprojection residual at capture-time clone
        |
        v
full Kalman update
        |
        v
correct current state through clone/current cross-covariance
```

Association is also evaluated using the capture-time clone rather than the current pose.

Unit tests cover:

- delayed-clone reprojection Jacobian,
- finite-difference agreement,
- delayed clone measurement correcting current state through cross-covariance.

The latest CI passed after fixing the test-only missing `pytest` import.

### 8.3 V6.6 compensated-latency result

The three key compensated-latency cases completed successfully:

| Case | Position RMSE | Tail20 | Position max | Velocity RMSE | Orientation RMSE | Gate acceptance | Association match | NIS/dof |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 50 ms configured | 0.05677 m | 0.05139 m | 0.11968 m | 0.03042 m/s | 0.74595 deg | 71.33% | 97.66% | 1.106 |
| 100 ms configured | 0.05822 m | 0.05439 m | 0.12184 m | 0.03122 m/s | 0.76352 deg | 71.20% | 97.65% | 1.104 |
| 200 ms configured | 0.05681 m | 0.05263 m | 0.11773 m | 0.03192 m/s | 0.73832 deg | 70.93% | 97.64% | 1.111 |

Observed mean measurement ages were approximately 0.08 s, 0.12 s, and 0.20 s respectively because measurements are released on the discrete camera/evaluator schedule.

All three runs used:

```text
latency_compensation_enabled = True
max_position_clones = 32
```

The results are effectively back at the neutral V6.5 baseline:

```text
neutral position RMSE ~= 0.0578 m
neutral position max  ~= 0.1178 m
neutral orientation   ~= 0.766 deg
```

This is a strong validation that the catastrophic V6.5 latency result was caused by timestamp/state mismatch rather than by the direct-reprojection factor itself.

The compensated update:

```text
z(t_capture)
   ->
reprojection at clone x(t_capture)
   ->
full stochastic-clone covariance update
   ->
current x(t_now)
```

restores consistency even for deliberately exaggerated delay.

Do **not** rerun burst-dropout or pixel-noise experiments; V6.5 already established those results.

For the intended real hardware, these 50-200 ms tests remain robustness hardening rather than an RL entry criterion. Their purpose is now complete: the delayed-measurement mechanism has been shown to work.

---

## 9. Current unresolved issues

### 9.1 Gate-target progression still has indirect simulator-GT dependence

This is currently the most important RL integration issue.

The policy observation implementation itself is already deployment-faithful:

```text
learned_inertial_drone_state()
    -> estimator p, q, v
    -> onboard IMU angular velocity

learned_target_pos_b()
    -> known mapped gate position
    -> estimator pose
```

However, `GateTargetingCommand` still updates the current/next gate index using simulator truth:

```text
robot.data.root_pos_w
    ->
passed_gate_plane
    ->
gate_passed / gate_missed
    ->
next_gate_idx
```

Therefore the actor does not directly receive GT pose, but its target identity can still be indirectly advanced by simulator truth.

Before formal deployment-faithful RL, this must be split into:

```text
mission/actor gate progression:
    estimator/map based

training evaluation/reward gate truth:
    simulator GT allowed
```

GT may remain in reward, termination, logging, and evaluation. It must not determine the actor-visible target state.

### 9.2 Validated estimator settings are not yet frozen into a dedicated RL configuration

The learned-inertial task is registered:

```text
Isaac-Drone-Racer-Learned-Inertial-v0
```

and the repository already contains:

```text
scripts/rl/train.py
tasks/drone_racer/agents/skrl_cfg.yaml
```

However, the generic learned-inertial environment config still does not by itself guarantee the exact validated V6.4/V6.5 production settings and checkpoints.

A dedicated RL integration configuration/profile should freeze:

- Body-Delta-v checkpoint,
- learned update rate 20 Hz,
- learned fusion rate 2 Hz,
- conservative learned covariance,
- direct gate reprojection,
- 0.85 px nominal pixel sigma,
- 2-corner minimum,
- current Huber/NIS settings,
- fixed known initial pose contract,
- no oracle/debug fusion,
- no GT actor observation.

The goal is that one RL training command always launches the validated estimator stack without manual evaluator-only overrides.

### 9.3 Closed-loop control has not yet been validated using only estimated state

The existing estimator evaluations replay a known action sequence. They prove estimation accuracy but do not yet prove that the complete online loop works:

```text
sensors
 -> estimator
 -> actor-style observation
 -> controller
 -> action
 -> vehicle motion
 -> sensors
```

Before long PPO training, one deterministic/scripted controller should drive the vehicle toward mapped gates using only estimated state.

This test is not intended to solve racing optimally. It is an integration check that:

- observation signs/frames are correct,
- gate target progression is correct,
- action semantics are correct,
- estimator updates remain stable under closed-loop motion,
- at least several gates can be traversed without GT state being fed to control.

---

## 10. What is already ready for RL

The repository already contains the RL infrastructure:

```text
scripts/rl/train.py
scripts/rl/play.py
tasks/drone_racer/agents/skrl_cfg.yaml
```

The learned-inertial Gym task is already registered:

```text
Isaac-Drone-Racer-Learned-Inertial-v0
```

The current SKRL configuration already defines PPO with a 256-256-256 ELU network, 24-step rollouts, 5 learning epochs, 4 mini-batches, gamma 0.99, lambda 0.95, and 1e-4 learning rate.

Therefore the project does not need a new RL framework. It needs the existing RL path connected to the now-validated estimator in a deployment-faithful way.

---

## 11. Minimum remaining path before formal RL

The shortest defensible path is:

### Step 1 — Freeze the estimator baseline

The V6.6 compensated-latency rerun is complete and passed. Treat the estimator architecture as frozen unless RL integration exposes a genuine estimator bug.

### Step 2 — Create the RL-integration branch from the accepted estimator baseline

Recommended purpose:

```text
GT-free mission state + frozen V6.5/V6.6 estimator configuration
```

The estimator itself should be treated as frozen unless this integration work exposes a genuine bug.

### Step 3 — Remove indirect GT dependence from actor target progression

Refactor gate progression so that the actor-visible `next_gate_idx` / target is advanced from deployment-available estimator/map state.

Keep a separate simulator-truth gate-pass signal for reward/evaluation.

### Step 4 — Freeze a production RL task configuration

Make the learned-inertial PPO task automatically instantiate:

```text
IMU
+ Body-Delta-v TCN
+ stochastic-cloning EKF
+ mapped direct gate reprojection
+ deployment-faithful actor observations
```

No manual estimator flags should be required for ordinary training.

### Step 5 — Run GT-free observation audit

Verify that the actor input path contains no runtime read of:

```text
root_pos_w
root_quat_w
root_lin_vel_w
root_state_w
```

or equivalent simulator truth.

Allowed simulator-truth uses remain:

- reward,
- termination,
- training diagnostics,
- evaluation metrics,
- one-time fixed known initialization when explicitly part of the task definition.

### Step 6 — Estimated-state scripted closed-loop smoke test

Use a simple deterministic controller to fly through multiple gates using the same observations that the future policy will receive.

Pass condition:

- target progression works,
- controller receives no GT pose,
- estimator remains bounded,
- several gates can be traversed.

### Step 7 — Start PPO

Once the above steps pass, formal RL can begin.

Recommended progression:

```text
single/simple gate behavior
    ->
multi-gate nominal track
    ->
higher speed
    ->
sensor/perception randomization
    ->
full racing policy
```

Estimator changes should be avoided during normal PPO tuning unless a new closed-loop failure clearly traces back to estimation.

---

## 12. Current project status

```text
OpenVINS-free architecture                 DONE
Body-Delta-v TCN                           DONE
stochastic-cloning EKF                     DONE
direct gate reprojection                   DONE
2/3/4-corner fusion                        DONE
30 s direct-reprojection validation        DONE
5-seed inertial-noise robustness           DONE
1 s complete visual blackout               DONE
+0.5 / +1 / +2 px visual-noise tests       DONE
V6.6 delayed-measurement mechanism         DONE
V6.6 50/100/200 ms rerun                   DONE

GT-free actor state                        DONE
GT-free target-position computation        DONE
GT-free gate target progression            TODO
frozen RL estimator/task config            TODO
estimated-state closed-loop                TODO

formal PPO/RL                              NEXT AFTER ABOVE
```

---

## 13. Main conclusions from 2026-09-20

1. The dominant V6.3 failure was planar-PnP orientation conditioning, not bad gate geometry or a fundamentally poor detector.
2. Direct mapped-corner reprojection solved the main position-estimation problem.
3. Position accuracy improved from approximately 0.31 m RMSE to approximately 0.05 m RMSE.
4. Partial 2/3-corner frames now contribute useful absolute updates instead of being discarded.
5. Five-seed testing shows good robustness to the tested IMU-noise realizations.
6. One second of total visual blackout caused almost no degradation on the tested trajectory.
7. Additional 2 px corner noise did not cause position divergence.
8. Uncompensated delayed pixels are dangerous because they violate the measurement timestamp/state relationship; V6.6 clone-based delayed visual updates restore the 50/100/200 ms cases to approximately 5-6 cm position RMSE.
9. The delayed-measurement stochastic-clone implementation is now experimentally validated.
10. For the intended hardware, the artificial 50-200 ms latency stress is not treated as a blocker to RL.
11. The estimator architecture is sufficiently mature to stop being the main research focus.
12. The remaining pre-RL work is interface/integration work: GT-free gate progression, frozen RL configuration, and one estimated-state closed-loop test.
13. Once these three integration tasks pass, the project should move directly into PPO rather than continuing to optimize estimator RMSE.

---

## 14. Target final architecture

The intended final racing stack is:

```text
                  +----------------------+
IMU --------------> EKF propagation     |
                  |                      |
thrust + gyro ----> Body-Delta-v TCN ----> stochastic-cloning EKF
                  |                      |          ^
                  +----------------------+          |
                                                    |
camera -> gate corners -----------------------------+
        + known surveyed gate map
        + direct pixel reprojection

stochastic-cloning EKF
        |
        v
estimated p, q, v + onboard omega + mapped target
        |
        v
RL policy
        |
        v
motor/control action
```

Simulator truth is outside the deployment actor/estimator path and is reserved for training reward, termination, diagnostics, and evaluation.

That is the architecture the next RL-integration work should preserve.
