# Learned-Inertial Racing Progress — 2026-09-19

> Branch: `feature/v6.1-coupled-motion-training`  
> End-of-day goal state: the learned-motion representation is no longer the dominant blocker. A real fixed-lag endpoint-clone feedback artifact was isolated and fixed. With exact Oracle relative-motion measurements, the corrected estimator improves velocity RMSE over IMU-only in all 5 seeds, but position still shows residual seed sensitivity. The next problem is now sharply localized to the interaction between a relative measurement, the constrained Kalman gain, gauge/observability, and covariance structure.
>
> Ultimate goal: build an OpenVINS-free estimator for autonomous drone racing using only onboard IMU + thrust history + learned short-horizon motion constraints, then use mapped gate detector/PnP observations as intermittent absolute pose anchors. Simulator GT is allowed only for supervised training, diagnostics, rewards, and the known initial pose; it must not be a deployed estimator or actor input.

---

## 1. Where 2026-09-19 started

The previous day ended with V6:

[
z_{B_e}
=
R_t^T
left[
(p_t-p_s)-v_sT-rac12 gT^2
ight]
]

and the deployment-compatible runtime:

```text
gyro_b + thrust_b
    -> gyro-only relative attitude integration
    -> endpoint-frame aligned features
    -> TCN
    -> gravity-compensated endpoint-body residual
    -> fixed-lag error-state EKF
```

V6 had solved the major representation problem:

- no GT/global attitude is needed by the TCN at deployment;
- no EKF world attitude leaks into the TCN input;
- the known gravity term is removed from the learned target;
- short-window prediction is millimetre-scale.

However, 30 s / 5-seed Lissajous fusion still showed that accurate short-window prediction did **not** automatically give stable long-run fusion.

The starting 30 s baseline was:

```text
IMU-only A, five-seed mean

position RMSE       2.80803 m
position tail20     5.67884 m
final position      7.96635 m

velocity RMSE       0.40305 m/s
velocity tail20     0.77884 m/s
final velocity      1.00144 m/s

orientation RMSE    0.32010 deg
```

Here, **A means Mode A = IMU-only**. It is the baseline against which every learned-inertial configuration is compared.

---

## 2. The constrained learned update had already removed the catastrophic V6 failure

Before the V6.1 data work, the learned-update gain ablation had identified a much safer mode:

```text
freeze_clones_attitude_bias
```

This mode:

- freezes current attitude correction;
- freezes accelerometer- and gyro-bias correction;
- freezes all clone velocity/position correction;
- allows only current velocity and current position to be corrected.

This prevented the old catastrophic full-state feedback into attitude/bias, but did not yet make network fusion consistently better than A.

Oracle testing had already shown that exact relative measurements could improve the filter:

```text
Oracle x30, old scheduler, five-seed mean

position RMSE       2.22062 m
velocity RMSE       0.32180 m/s

position wins vs A  4/5
velocity wins vs A  4/5
```

This established an important fact:

> A correct relative-motion measurement can help the estimator. The project should not abandon the fixed-lag learned-inertial concept merely because the network case is worse.

---

## 3. V6.1 changed the problem from model capacity to trajectory-distribution coverage

The main V6 residual problem was trajectory-dependent millimetre-scale bias/generalization error.

A V6.1 data/training branch was built around two changes:

1. broader coupled-motion trajectory coverage;
2. trace-balanced training rather than letting long traces dominate by raw window count.

The V6.1 manifest contains 34 traces:

```text
20 train
 7 validation
 7 test
```

Trajectory families include:

- circle;
- Lissajous;
- racing-like coupled motion;
- translate+yaw;
- vertical;
- simple translation / yaw support cases.

The held-out distribution was deliberately constructed to stay inside the training support while changing amplitudes, frequencies, phases, yaw combinations, durations, and random seeds.

The data-support audit passed:

```text
held-out p95 target support remained inside configured train support
on all axes
```

This meant the first V6.1 candidate could be trained without expanding the manifest again.

---

## 4. First V6.1 candidate: large held-out improvement

The first trace-balanced V6.1 training produced:

```text
checkpoint:
artifacts/imo_tcn/model_v6_1_coupled_balanced.pt

best epoch            155
selection metric      NLL
best val NLL          -5.8947746740

train windows         60500
val windows           22650
test windows          26650
```

Held-out metrics:

```text
Validation
norm RMSE       1.680 mm
axis RMSE       [0.964, 1.097, 0.831] mm
axis bias       [-0.064, +0.162, +0.228] mm

Test
norm RMSE       1.849 mm
axis RMSE       [1.067, 1.219, 0.891] mm
axis bias       [-0.098, +0.166, +0.258] mm
```

Compared on the **same V6.1 held-out dataset**, the old V6 achieved approximately:

```text
old V6 validation norm RMSE   6.3 mm
old V6 test norm RMSE         6.2 mm

V6.1 validation norm RMSE     1.7 mm
V6.1 test norm RMSE           1.8 mm
```

Therefore V6.1 reduced held-out norm RMSE by roughly 70%+.

The improvement was especially clear on the difficult coupled trajectories.

Representative test examples:

```text
old V6 -> V6.1

Lissajous #1     8.5 -> 1.8 mm
Lissajous #2     6.6 -> 1.7 mm
Racing-like #1   5.2 -> 1.6 mm
Racing-like #2   8.6 -> 3.0 mm
```

Conclusion:

> The old V6 long-run problem had a real trajectory-distribution/generalization component. V6.1 substantially reduced it without increasing model complexity.

---

## 5. One bad training trace was found and removed from the useful training interval

The first V6.1 aggregate training evaluation contained one obvious outlier:

```text
train_translate_yaw_4m_7s_y50.csv

norm RMSE about 109.4 mm
```

Most other training traces were around 0.8–2.0 mm.

A trace-health audit showed the file was not corrupted:

```text
rows                    3000
duration                29.99 s
dt                      exactly 0.01 s
quaternion norm         ~1
action saturation       0
position step jump max  0.0385 m
speed max               3.85 m/s
```

But near the end of the 30 s trace, the target became much more aggressive:

```text
around t = 28.2 s
0.5 s residual norm max = 2.145 m
```

The trace was physically continuous but had entered a controller-runaway / no-longer-representative tail.

The manifest was changed so this training trace keeps only the useful controlled section:

```text
3000 steps -> 2000 steps
```

Relevant commit:

- `b9918ed` — `fix: shorten unstable V6.1 translate-yaw training trace`

A clean V6.1 checkpoint was then retrained:

```text
artifacts/imo_tcn/model_v6_1_coupled_balanced_clean.pt
```

Clean held-out result:

```text
Validation
norm RMSE    1.7 mm
NLL          -5.9174

Test
norm RMSE    1.8 mm
NLL          -5.9064
```

The clean retraining therefore preserved the strong held-out generalization.

---

## 6. V6.1 30 s learned fusion improved greatly but did not yet beat A consistently

The clean V6.1 checkpoint was fused with:

```text
trajectory             Lissajous
duration               30 s
seeds                  0,1,2,3,4
fusion rate            2 Hz
gain mode              freeze_clones_attitude_bias
legacy covariance      x30
```

Comparison:

```text
                       A          old V6 x30     clean V6.1 x30
position RMSE          2.80803    4.19911        3.03504 m
position tail20        5.67884    8.03700        5.78768 m
position final         7.96635   10.23310        7.52118 m

velocity RMSE          0.40305    0.48017        0.37713 m/s
velocity tail20        0.77884    0.86507        0.68285 m/s
velocity final         1.00144    0.99191        0.84985 m/s
```

V6 -> V6.1 improvement:

```text
position RMSE       +27.72%
position tail20     +27.99%
position final      +26.50%
velocity RMSE       +21.46%
```

Wins of V6.1 x30 vs A:

```text
position RMSE       2/5
position tail20     2/5
position final      3/5
velocity RMSE       3/5
velocity final      3/5
```

This was a major change from old V6:

> V6.1 moved learned fusion from clearly harmful to roughly break-even in position and beneficial in several velocity/final-state metrics.

But it still was not stable enough to freeze the inertial core.

---

## 7. What the covariance multiplier actually means

The experiment used:

```text
x1, x3, x10, x30, x100
```

These are **measurement covariance multipliers**, not model versions and not multipliers on the displacement prediction.

For an EKF update:

[
K = PH^T(HPH^T + R)^{-1}
]

the multiplier changes (R).

For example:

[
R_{	ext{used}} = 30 R
]

and because (R=sigma^2),

[
sigma_{	ext{used}} = sqrt{30},sigma.
]

Therefore larger multipliers make the EKF less trusting of the learned/oracle measurement.

---

## 8. The legacy V6 safety floor is still active

The current estimator still contains an old V6 protection mechanism:

```python
learned_sigma_floor_xyz_m = (0.10, 0.10, 0.01)
learned_covariance_scale = 1.25
```

Runtime protection is effectively:

```python
predicted_sigma = sqrt(diag(network_covariance))
sigma_used = max(predicted_sigma, floor)
R = diag((1.25 * sigma_used)**2)
R *= covariance_multiplier
```

For clean V6.1:

```text
raw predicted sigma ~2.68 mm
actual fusion-schedule prediction RMSE ~1.15–1.27 mm
```

But because the legacy floor dominates, x30 uses approximately:

```text
used sigma:
X/Y  684.653 mm
Z     68.465 mm
```

Thus the current runtime is not yet using V6.1's native learned uncertainty.

This floor was originally reasonable because older learned models could report millimetre sigma while making much larger errors. For clean V6.1, it is likely over-conservative.

However, **the floor should not be removed yet**, because a deeper fixed-lag structural artifact was found later in the day. Removing the floor before fixing that artifact would have increased the gain of an already problematic feedback loop.

---

## 9. Network covariance sweep showed that a single global weight was not the answer

Clean V6.1 was swept over:

```text
x1, x3, x10, x30, x100
```

Five-seed position RMSE means:

```text
A        2.80803 m
x1       3.40442 m
x3       3.15407 m
x10      3.57518 m
x30      3.03504 m
x100     2.94613 m
```

Velocity showed more benefit, especially at x3:

```text
                       A          V6.1 x3
velocity RMSE          0.40305    0.35577 m/s
velocity tail20        0.77884    0.57358 m/s
velocity final         1.00144    0.71411 m/s
```

At x3:

```text
velocity final wins vs A = 5/5
```

But position was strongly seed-sensitive:

```text
seed 0   A 2.495 -> x3 2.198 m
seed 1   A 2.434 -> x3 7.140 m
seed 2   A 4.382 -> x3 1.861 m
seed 3   A 2.516 -> x3 1.811 m
seed 4   A 2.213 -> x3 2.760 m
```

There was no monotonic relationship between covariance multiplier and position quality.

Conclusion:

> The remaining problem could not be reduced to “find a better scalar multiplier.”

---

## 10. Fusion-schedule network error was already very consistent across seeds

The actual clean V6.1 prediction errors at the 59 fused windows were inspected.

Five seeds:

```text
seed 0 norm RMSE   1.1771 mm
seed 1             1.2202 mm
seed 2             1.1564 mm
seed 3             1.1572 mm
seed 4             1.2733 mm
```

Representative axis RMSE:

```text
x about 0.82–0.97 mm
y about 0.59–0.61 mm
z about 0.54–0.57 mm
```

Raw network sigma:

```text
~[2.68, 2.68, 2.68] mm
```

Lag-1 temporal correlation was modest, roughly 0.07–0.23 depending on axis/seed.

Therefore the network quality was **very similar across seeds**, while long-run position outcomes differed by metres.

This strongly shifted attention away from raw network prediction accuracy and toward EKF/fixed-lag state-covariance interaction.

---

## 11. Oracle covariance sweep isolated the estimator from the network

A 25-run Oracle experiment was performed:

```text
5 seeds x covariance {1,3,10,30,100}
```

Oracle replaces the network measurement mean with the exact simulator-GT kinematic residual while leaving the same fixed-lag EKF path.

This means:

> If Oracle fusion behaves strangely, that behavior cannot primarily be blamed on TCN mean prediction.

Before the scheduler fix, the five-seed Oracle results were:

```text
Cov x     position RMSE       velocity RMSE      position wins vs A
1         2.431 +/- 0.904 m   0.366 +/- 0.071    3/5
3         2.938 +/- 0.961 m   0.350 +/- 0.083    1/5
10        2.411 +/- 0.632 m   0.358 +/- 0.080    3/5
30        2.221 +/- 1.176 m   0.322 +/- 0.131    4/5
100       2.565 +/- 1.019 m   0.382 +/- 0.135    3/5
```

Even exact Oracle measurements showed strong non-monotonic covariance/seed behavior.

This proved that network mean error was **not** the only remaining issue.

---

## 12. Read-only trace diagnosis found a real fixed-lag endpoint-clone feedback artifact

The 25 Oracle runs were analyzed event-by-event without changing code.

Experimental integrity checks passed:

- 3000 trace rows per run;
- 590 learned predictions;
- 59 actual fusions;
- fused windows exactly 0.5 s apart;
- source = `oracle_kinematic_residual_body_end`;
- replay GT position matches A;
- theta, accel bias, gyro bias, clone velocity, and clone position corrections are all zero under `freeze_clones_attitude_bias`;
- orientation RMSE is unchanged from A.

The key discovery was the scheduler ordering.

Old behavior:

```text
propagate current state to endpoint t
    ↓
clone current endpoint state
    ↓
perform learned update
    ↓
current p/v changes
    ↓
new endpoint clone remains frozen at pre-update p/v
    ↓
0.5 s later this stale clone becomes the next window start
```

Because `freeze_clones_attitude_bias` freezes all clone gain rows, the just-created endpoint clone did not receive the current-state correction.

The resulting next innovation was almost exactly:

[

u_{k+1}^{	ext{echo}}
approx
-R_{	ext{end}}^T
left(
Delta p_k + 0.5,Delta v_k
ight).
]

Across the 25 Oracle traces, this stale-clone term explained approximately:

```text
99.9736%–99.99795%
```

of the next innovation variance.

After subtracting the echo term, the residual norm RMSE was only about:

```text
2.82–3.09 mm
```

This was the strongest estimator-level diagnosis of the day.

### Concrete old seed-0 x30 example

At:

```text
t = 25.01 s
Oracle innovation norm = 0.189 m
approx position correction = 3.194 m
```

That correction was approximately:

```text
dxP_world = [0.650, 3.127, -0.001] m
dxV_world = [0.087, 0.378, 0.005] m/s
```

The predicted stale-clone echo at the next fusion was:

```text
[-0.377, -3.156, -1.173] m
```

Observed next innovation:

```text
[-0.373, -3.153, -1.171] m
```

The agreement was within only a few millimetres.

Therefore a large portion of the old “Oracle instability” was not fresh propagation error; it was deterministic feedback caused by endpoint clone timing.

---

## 13. Fixed-lag scheduler repair

The scheduler was changed so that the endpoint clone is created **after** the learned update at the same timestamp.

New ordering:

```text
propagate to t
    ↓
use historical start clone + current endpoint state
to evaluate/fuse learned relative measurement
    ↓
finish state injection + Joseph covariance update
    ↓
marginalize used start clone
    ↓
clone corrected current endpoint state and corrected covariance
```

This is important because `clone_current_position()` does not just copy nominal (v,p). It also augments covariance using the corrected (P):

```python
cross = J @ P
P_clone = J @ P @ J.T
```

Therefore the fix maintains nominal-state and covariance consistency.

Relevant commits:

- `6a7f094` — `fix: clone corrected endpoint after learned update`
- `e0fcfb2` — `test: cover learned endpoint clone update echo`

The regression test explicitly compares:

- correct update-then-clone behavior;
- old clone-then-update behavior;
- the expected next-window (Delta p + 0.5Delta v) echo.

No network, covariance-floor, process-noise, measurement Jacobian, or gain-mode defaults were changed by this fix.

---

## 14. Post-fix trace audit proved that the deterministic echo disappeared

Two high-value Oracle x30 cases were rerun first:

```text
seed 0
seed 4
```

Scheduling remained intact:

```text
rows             3000
fusion count     59
fusion interval  0.5 s
source           oracle_kinematic_residual_body_end
```

The old echo explained essentially all of the next innovation:

```text
old R^2

seed 0  0.999970
seed 4  0.999930
```

After the fix, applying the same old echo formula gave huge negative (R^2), i.e. it no longer predicts the new innovations at all.

More directly:

```text
new actual next innovations are only a few millimetres,
while the old stale-clone formula predicts multi-metre echoes.
```

Representative post-fix events:

```text
seed 4, t=26.51 s
current innovation        ~5.006 mm
approx dxP                ~3.352 m
next actual innovation    ~5.465 mm
old-style predicted echo  ~3.544 m

seed 0, t=26.51 s
current innovation        ~4.349 mm
approx dxP                ~2.844 m
next actual innovation    ~4.885 mm
old-style predicted echo  ~3.008 m
```

Therefore:

> The stale endpoint-clone deterministic feedback mechanism was genuinely removed. The improvement is causal, not merely an RMSE coincidence.

---

## 15. Five-seed Oracle x30 after the clone-timing fix

The corrected scheduler was then run on all five seeds at Oracle x30.

### Aggregate

```text
position RMSE

A                  2.80803 +/- 0.88802 m
Oracle OLD x30     2.22062 +/- 1.17632 m
Oracle FIX x30     2.20721 +/- 0.80200 m

wins vs A:
OLD 4/5
FIX 4/5
```

The mean changed only slightly, but the standard deviation reduced substantially:

```text
1.176 -> 0.802 m
```

This is an important stability improvement.

### Position tail/final

```text
position tail20

A                  5.67884 m
Oracle OLD x30     4.27245 m
Oracle FIX x30     4.30079 m

position final

A                  7.96635 m
Oracle OLD x30     5.71554 m
Oracle FIX x30     5.73230 m
```

### Velocity

```text
velocity RMSE

A                  0.40305 +/- 0.14412 m/s
Oracle OLD x30     0.32180 +/- 0.13093
Oracle FIX x30     0.33107 +/- 0.11560

wins vs A:
OLD 4/5
FIX 5/5
```

Other corrected velocity metrics:

```text
velocity tail20    0.61841 m/s
velocity final     0.77257 m/s
```

### Corrected position RMSE per seed

```text
seed 0   A 2.495 | FIX 2.650
seed 1   A 2.434 | FIX 1.844
seed 2   A 4.382 | FIX 3.038
seed 3   A 2.516 | FIX 2.507
seed 4   A 2.213 | FIX 0.997
```

Interpretation:

> With exact relative measurements and the stale-clone bug removed, velocity RMSE improves over IMU-only in all 5 seeds. Position improves in 4/5 seeds, but the relative measurement is still not uniformly safe as a direct current-position correction.

---

## 16. The remaining problem is now much more specific

After the clone fix, actual Oracle fusion innovations are typically only a few millimetres:

```text
roughly 3–6 mm in the inspected late-run events
```

Yet the constrained update can still produce:

```text
multi-metre current-position corrections
```

and very large position Kalman-gain norms.

This is no longer explainable by the stale-clone echo.

The key structural observation is that the learned/oracle measurement is **relative**:

[
h(x)
=
R_t^T
left[
(p_t-p_s)-v_sT-rac12 gT^2
ight].
]

It is invariant to a common global translation:

[
p_t ightarrow p_t + c,qquad
p_s ightarrow p_s + c.
]

It also does not provide an absolute world-position anchor.

However, `freeze_clones_attitude_bias` freezes the historical clone while allowing current position to move.

Therefore the constrained update can effectively turn a relative factor into a large correction on current position alone.

This creates a new leading hypothesis:

> The remaining position instability is caused by the interaction between relative-measurement gauge/observability, current↔clone cross-covariance, and the constrained Kalman gain that permits current position to move while all clone kinematic states are frozen.

This hypothesis is consistent with all current evidence:

- velocity benefits are already strong and stable;
- position is the remaining seed-sensitive state;
- perfect Oracle measurements do not produce 5/5 position wins;
- post-fix innovations are millimetre-scale while current-position corrections can be metre-scale.

---

## 17. What has now been ruled out or strongly de-prioritized

### 17.1 Raw V6.1 mean prediction accuracy is no longer the primary blocker

Evidence:

- held-out norm RMSE ~1.7–1.8 mm;
- fusion-schedule norm RMSE ~1.15–1.27 mm;
- measurement quality is similar across seeds;
- Oracle, which removes network mean error entirely, still shows residual position sensitivity.

### 17.2 The old catastrophic attitude/bias feedback is no longer the current failure mode

Under `freeze_clones_attitude_bias`:

- attitude correction = 0;
- accel-bias correction = 0;
- gyro-bias correction = 0;
- orientation RMSE is identical to A.

### 17.3 Fused-window overlap is not the current issue

The actual fused windows occur every 0.5 s:

```text
0.51, 1.01, 1.51, ...
```

so the 0.5 s fused residual windows are non-overlapping.

The TCN still predicts at 20 Hz for diagnostics, but only every tenth prediction is fused at 2 Hz.

### 17.4 The stale endpoint-clone echo is fixed

The old next-innovation echo explained ~99.97–99.998% of innovation variance.

After the scheduler repair, the same model has no predictive power.

This issue is considered solved.

### 17.5 A single covariance multiplier is not the right next target

Network and Oracle covariance sweeps are non-monotonic.

Therefore searching for a scalar “best x5” before resolving gain/gauge behavior would be parameter tuning around a structural issue.

---

## 18. The legacy covariance floor is still suspicious, but should be handled later

Clean V6.1's raw uncertainty:

```text
raw sigma ~2.68 mm
actual error ~0.5–1.3 mm
```

Current legacy floor:

```text
100 mm / 100 mm / 10 mm
```

Therefore the safety floor is clearly not calibrated to the current network.

It remains likely that a future V6.1-native covariance configuration should remove or greatly reduce this floor.

However, the correct order is:

```text
1. solve relative-factor gain/gauge behavior
2. verify Oracle stability
3. verify Network stability
4. then recalibrate/remove legacy floor
```

Otherwise a change in floor would be confounded with a change in state-update structure.

---

## 19. Immediate next experiment

The next high-value ablation should be a new learned Kalman-gain mode:

```text
current_velocity_only
```

Semantics:

```text
allow:
  current velocity

freeze:
  current attitude
  current position
  accel bias
  gyro bias
  all clone velocity
  all clone position
```

Reason:

- exact relative-motion constraints already improve velocity in 5/5 seeds;
- position is the state still showing problematic direct corrections;
- the relative factor does not provide an absolute-position anchor;
- corrected velocity can naturally improve position through later propagation without directly injecting metre-scale current-position jumps.

### First test

Run:

```text
Oracle x30
5 seeds
current_velocity_only
```

Compare against:

```text
A
Oracle FIX x30 freeze_clones_attitude_bias
```

Success would look like:

- velocity RMSE remains 5/5 better than A;
- position becomes at least as stable as the current Oracle configuration;
- position correction jumps disappear by construction;
- position improves through propagation rather than direct relative-factor position injection.

### Then

If Oracle current-velocity-only is healthy:

1. run clean V6.1 network with the same gain mode;
2. repeat 5-seed 30 s Lissajous;
3. move to 30–60 s racing-like/multi-seed validation;
4. only then revisit the legacy covariance floor.

---

## 20. Recommended sequence from here

```text
A. Add current_velocity_only learned gain mode
      ↓
B. Oracle x30, 5 seeds
      ↓
C. If healthy, clean V6.1 Network, 5 seeds
      ↓
D. Compare legacy floor vs V6.1-native uncertainty
      ↓
E. 30–60 s Lissajous + racing-like multi-seed
      ↓
F. Freeze learned-inertial core
      ↓
G. Gate PnP absolute-position fusion
      ↓
H. Gate orientation / covariance / frame / latency validation
      ↓
I. Actual 7-gate long-run estimator
      ↓
J. Actor observations from estimator/map only
      ↓
K. Estimator-in-the-loop RL
      ↓
L. Full autonomous racing + sim-to-real robustness
```

Gate PnP should **not** be used early to hide an unresolved inertial-core inconsistency.

---

## 21. Current project status by subsystem

### Learned representation

Status: **strong / near-ready**

- V6 gravity-compensated endpoint-body residual is deployment-compatible.
- V6.1 coupled-motion training reduced held-out error substantially.
- clean V6.1 is millimetre-scale on difficult coupled trajectories.
- no EKF/global-attitude leakage remains in the network input.

### Fixed-lag scheduler

Status: **major bug fixed**

- stale pre-update endpoint clone was found;
- its correction echo was measured quantitatively;
- scheduler changed to update-then-clone;
- old deterministic echo disappeared in replay diagnostics.

### Learned EKF gain structure

Status: **current main blocker**

- velocity correction is clearly useful;
- direct current-position correction is still questionable;
- relative factor + frozen clones can generate large current-position gain/correction despite millimetre Oracle innovations.

### Learned covariance

Status: **not calibrated for V6.1 yet**

- raw network sigma is a few millimetres;
- runtime legacy floor is tens to hundreds of millimetres before multiplier;
- do not recalibrate until the gain/gauge issue is isolated.

### Gate PnP

Status: **intentionally deferred**

- infrastructure exists;
- should become the intermittent absolute anchor after the inertial core is stable;
- must not mask an unresolved relative-fusion problem.

### RL / racing policy

Status: **not ready to start**

The actor must eventually consume estimator/map state only. Estimator-in-the-loop RL should wait until the inertial + gate estimator is trustworthy.

---

## 22. Important artifacts

### Checkpoints

```text
old V6:
artifacts/imo_tcn/model_v6_gravity_compensated.pt

clean V6.1:
artifacts/imo_tcn/model_v6_1_coupled_balanced_clean.pt
```

### V6.1 dataset

```text
config/learned_inertial/imo_v6_1_manifest.json
artifacts/imo_dataset_v6_1/
```

### Old 30 s replay set

```text
artifacts/learned_inertial_diagnostics/
v6_30s_lissajous_multiseed/seed_<N>/replay.npz
```

### Oracle covariance sweep before scheduler fix

```text
artifacts/learned_inertial_diagnostics/
v6_1_oracle_covariance_sweep/
```

### Oracle regression after scheduler fix

```text
artifacts/learned_inertial_diagnostics/
v6_1_clone_timing_fix/
```

### Relevant end-of-day commits

```text
b9918ed  fix: shorten unstable V6.1 translate-yaw training trace
6a7f094  fix: clone corrected endpoint after learned update
e0fcfb2  test: cover learned endpoint clone update echo
```

Earlier V6.1 training/data commits on this branch include:

```text
4abe940  trace-balanced training
36d2b19  V6.1 manifest
a182cb8  dataset target-coverage audit
5330c8a  manifest tests
eea9151  V6.1 documentation
ebfcf3f  training weight refactor
093dae7  sampling test
a2f99b7  avoid duplicate V6.1 audit window generation
30cd0b4  training preprocessing progress output
```

---

## 23. What was accomplished on 2026-09-19

The main achievement was **not** simply that V6.1 reduced network RMSE.

The day produced a sequence of increasingly precise isolations:

1. V6.1 showed that broader coupled-motion data and trace-balanced training materially improve learned generalization.
2. A bad training trace tail was found and removed without changing the core architecture.
3. Network fusion improved from clearly harmful V6 behavior to near break-even/beneficial velocity behavior.
4. Covariance sweep showed that scalar measurement weighting alone cannot explain the remaining position instability.
5. Fusion-schedule prediction errors showed that network quality is very similar across seeds.
6. Oracle covariance sweep proved that exact measurement means still exhibit estimator-side seed/covariance sensitivity.
7. Event-level trace analysis discovered a concrete stale endpoint-clone feedback artifact.
8. The scheduler was fixed to clone the corrected endpoint after learned update.
9. The deterministic echo disappeared quantitatively.
10. Five-seed corrected Oracle x30 now gives:
   - position RMSE better than A in 4/5 seeds;
   - velocity RMSE better than A in 5/5 seeds;
   - lower position RMSE variance than the old scheduler.

The remaining research question is therefore much narrower than it was at the start of the day.

---

## 24. End-of-day diagnosis

At the end of 2026-09-19, the best-supported interpretation is:

```text
V6/V6.1 representation problem:
largely solved

trajectory-distribution generalization:
substantially improved

old full-state attitude/bias feedback:
contained

stale endpoint-clone feedback:
found and fixed

current primary blocker:
relative-factor gauge / covariance / constrained-gain interaction,
especially direct current-position correction
```

The exact Oracle experiment is now especially informative:

> If an exact 0.5 s relative-motion residual consistently improves velocity but not every seed's position, the next task is not to train a larger neural network. It is to make the relative measurement act on the estimator state in a way that respects what that measurement actually observes.

---

## 25. Ultimate target architecture

The intended final system remains:

```text
                 onboard IMU + thrust
                         │
            ┌────────────┴────────────┐
            │                         │
            ↓                         ↓
      EKF propagation          V6.x learned
                               short-horizon
                               motion residual
            │                         │
            └────────────┬────────────┘
                         ↓
                fixed-lag estimator
                         ↑
                         │
           mapped Gate detector / PnP
           intermittent absolute anchors
                         │
                         ↓
              estimator/map state only
                         │
                         ↓
                 racing RL policy
```

Deployment constraints:

- no OpenVINS dependency;
- no simulator GT;
- no GT orientation in TCN features;
- no GT pose fed to the actor;
- onboard IMU/thrust provide continuous inertial information;
- learned motion reduces short-horizon inertial drift;
- gate PnP bounds long-term global drift;
- the racing policy consumes estimator/map state, not privileged truth.

The target is not merely a low offline TCN RMSE. The target is a **stable, deployable, estimator-in-the-loop autonomous drone-racing system that remains accurate between gates and recovers global position when gates are observed.**

---

## 26. First task for the next session

Do **not** begin by retraining the network or sweeping more covariance multipliers.

Start from:

```text
branch:
feature/v6.1-coupled-motion-training

latest estimator-fix commits:
6a7f094
e0fcfb2
```

Then implement and test:

```text
current_velocity_only
```

as a learned-update gain ablation.

The first decisive experiment should be:

```text
30 s Lissajous
5 seeds
Oracle
covariance x30
current_velocity_only
```

Compare against:

```text
A
Oracle FIX x30 freeze_clones_attitude_bias
```

Only after that result is understood should the project decide whether to:

- keep/revise the constrained gain structure;
- reintroduce clean V6.1 network measurements;
- remove/recalibrate the old V6 covariance floor.

---

## 27. Bottom line

2026-09-19 moved the project from:

> “The network is accurate, but learned fusion is mysteriously unstable.”

to:

> “The network is now accurate enough to stop being the main suspect. A concrete fixed-lag clone-timing feedback bug was found and removed. Exact relative measurements now improve velocity in every tested seed, while the remaining instability is concentrated in how a relative factor is allowed to correct current position through the constrained covariance/gain structure.”

That is a substantially better research position.

The next work should therefore focus on **state-update observability and gain structure**, not on another round of network architecture or dataset expansion.
