# Learned-Inertial Observability Discovery — 2026-09-20

> Branch: `feature/v6.1-coupled-motion-training`
>
> Purpose: document the external literature/code comparison performed on 2026-09-20 after the V6.1 clone-timing repair. The central question is why a millimetre-scale relative-motion innovation can still create metre-scale global-position corrections, and how closely related learned-inertial / VIO systems avoid this failure mode.

---

## 1. Executive summary

The main discovery is that the current remaining failure mode is not primarily a learned-motion accuracy problem.

V6.1 has already reduced the short-horizon learned residual error to roughly millimetre scale, and the stale endpoint-clone feedback bug was isolated and fixed on 2026-09-19. Nevertheless, exact Oracle relative-motion measurements can still produce very large current-position corrections.

The closest related systems and theory point to a common interpretation:

> A relative measurement must not inject information into gauge directions that it does not observe. In particular, a relative displacement constraint does not provide an absolute global-position anchor.

The most important implementation difference found on 2026-09-20 is:

- UZH Learned Inertial Odometry and TLIO formulate the learned displacement as a constraint between **two cloned states**.
- Their measurement Jacobian acts on both ends of the relative factor.
- The stochastic-cloning EKF then applies a covariance-consistent joint update.
- Our current diagnostic mode `freeze_clones_attitude_bias` instead freezes the historical clone kinematic states while allowing current position/velocity to move.
- This changes how a relative factor is allowed to express its correction and is a plausible explanation for the remaining large current-position gain/correction.

This does **not** mean that every clone-before-update schedule is wrong. UZH/TLIO clone before the update as part of normal stochastic cloning. Their endpoint clone participates in the same update. Our former failure was specifically:

```text
clone endpoint
    ↓
update current state
    ↓
endpoint clone is frozen and remains pre-update
    ↓
next relative window uses stale clone
    ↓
deterministic correction echo
```

That stale-clone mechanism has already been repaired in this project.

The new problem is narrower:

```text
relative factor
    +
current↔clone cross covariance
    +
constrained Kalman gain
    +
gauge / observability structure
    ↓
millimetre innovation can cause metre-scale current-position correction
```

The next work should therefore focus on **observability, gauge consistency, and state-update structure**, not on another network-capacity expansion.

---

## 2. Current project state before this discovery

The relevant 2026-09-19 status is documented in:

- `docs/LEARNED_INERTIAL_RACING_PROGRESS_2026-09-19.md`
- `docs/LEARNED_INERTIAL_V6_1_TRAINING.md`

Key facts entering the 2026-09-20 literature review:

### 2.1 V6.1 representation/generalization

Clean V6.1 held-out performance is approximately:

```text
Validation norm RMSE    ~1.7 mm
Test norm RMSE          ~1.8 mm
```

Actual fused-window prediction error over the 30 s Lissajous runs is approximately:

```text
seed 0   1.1771 mm
seed 1   1.2202 mm
seed 2   1.1564 mm
seed 3   1.1572 mm
seed 4   1.2733 mm
```

This makes raw network mean accuracy a lower-priority suspect.

### 2.2 Endpoint-clone timing bug

The old scheduler created the endpoint clone before applying the learned update while the constrained gain prevented the clone from receiving the same kinematic correction.

The resulting next-window innovation was almost exactly explained by the stale-clone correction echo.

That issue was fixed by creating the next endpoint clone from the corrected state/covariance after the learned update.

Relevant commits:

```text
6a7f094  fix: clone corrected endpoint after learned update
e0fcfb2  test: cover learned endpoint clone update echo
```

### 2.3 Remaining Oracle behavior

After the clone-timing repair, exact Oracle relative-motion measurements improve velocity RMSE over IMU-only in all tested seeds, but position remains seed-sensitive.

Representative aggregate result:

```text
                         IMU-only A      Oracle FIX x30
position RMSE              2.808 m          2.207 m
velocity RMSE              0.403 m/s        0.331 m/s

position wins vs A          -               4/5
velocity wins vs A          -               5/5
```

The key remaining symptom is:

```text
Oracle innovation:       typically a few millimetres
current-position dx:     can still become multiple metres
```

This symptom motivated the 2026-09-20 external comparison.

---

## 3. Closest direct reference: UZH Learned Inertial Odometry for Autonomous Drone Racing

### 3.1 Reference

**Paper**

Giovanni Cioffi, Leonard Bauersfeld, Elia Kaufmann, Davide Scaramuzza,
“Learned Inertial Odometry for Autonomous Drone Racing,”
IEEE Robotics and Automation Letters, 2023.

- PDF: https://rpg.ifi.uzh.ch/docs/RAL2023_Cioffi.pdf
- DOI: https://doi.org/10.1109/LRA.2023.3252342
- Project/publication page: https://giovanni-cioffi.netlify.app/publication/imo_ral23/

**Code**

- https://github.com/uzh-rpg/learned_inertial_model_odometry

Important implementation files:

- `src/filter/python/src/scekf.py`
- `src/filter/python/src/filter_runner.py`
- `src/filter/python/src/meas_source_network.py`

### 3.2 What UZH does

The UZH system combines:

```text
IMU propagation
    +
learned short-horizon displacement
    +
stochastic-cloning EKF
```

The learned displacement is treated as a **relative state measurement** between two stored past states.

In the public implementation, the learned update uses two clone timestamps:

```python
begin_idx = self.state.si_timestamps_us.index(t_begin_us)
end_idx   = self.state.si_timestamps_us.index(t_end_us)

pred = self.state.si_ps[end_idx] - self.state.si_ps[begin_idx]
```

The corresponding position blocks of the measurement Jacobian are:

```python
H[:, p_begin] = -I
H[:, p_end]   = +I
```

Conceptually:

[
z = p_j - p_i + n
]

and therefore:

[
H =
egin{bmatrix}
-I & +I
end{bmatrix}.
]

The update then uses the standard joint covariance:

[
K = PH^T(HPH^T + R)^{-1}
]

and applies the resulting correction to the correlated stochastic-cloning state.

### 3.3 Why this matters for global translation

For any common world translation (c),

[
p_i' = p_i + c,qquad p_j' = p_j + c,
]

the relative displacement is unchanged:

[
p_j' - p_i' = p_j - p_i.
]

The common-translation direction is therefore a null direction of the relative factor.

If

[
N_t =
egin{bmatrix}
I\
I
end{bmatrix},
]

then:

[
HN_t = -I + I = 0.
]

This captures the physical meaning of the measurement:

> The factor can constrain motion between the two states, but it cannot determine an absolute global translation.

### 3.4 Critical difference from our current constrained-gain mode

Our current V6.1 diagnostics often use:

```text
freeze_clones_attitude_bias
```

which approximately means:

```text
current velocity             allowed
current position             allowed

current attitude             frozen
accelerometer bias           frozen
gyro bias                    frozen

historical clone velocity    frozen
historical clone position    frozen
```

The raw relative measurement may still be mathematically translation-invariant, but the **allowed correction subspace is no longer symmetric between the two ends**.

The historical side cannot move, while the current side can.

This can force a relative correction to be expressed disproportionately through the current state via the cross covariance.

That is a plausible mechanism for the observed symptom:

```text
few-mm innovation
    ↓
large current-position Kalman-gain block
    ↓
metre-scale current-position correction
```

### 3.5 Important clarification: UZH also augments before updating

It would be incorrect to conclude:

> “clone-before-update is always wrong.”

UZH performs propagation/state augmentation and then uses the cloned endpoints in the learned update.

The key difference is that the endpoint clone is **part of the measurement state and participates in the joint correction**.

Our old 2026-09-19 bug was specifically:

```text
clone before update
+
freeze clone kinematics
+
update current state
```

which left the new clone stale.

Therefore:

```text
UZH/TLIO:
clone → relative update involving clone → consistent joint correction

old project behavior:
clone → freeze clone → update current only → stale endpoint echo
```

These are not equivalent.

### 3.6 Additional clue in the UZH code

The UZH/TLIO-derived filter maintains an explicit `unobs_shift` state and includes a diagnostic equivalent to:

[
N^T P^{-1} N.
]

This is strong evidence that the filter design is explicitly aware of unobservable/gauge directions.

That style of diagnostic should be added to this project.

---

## 4. TLIO: the upstream stochastic-cloning learned-IO design

### 4.1 Reference

**Paper**

Wenxin Liu, David Caruso, Eddy Ilg, Jing Dong, Anastasios I. Mourikis,
Kostas Daniilidis, Vijay Kumar, Jakob Engel,
“TLIO: Tight Learned Inertial Odometry.”

- arXiv: https://arxiv.org/abs/2007.01867

**Code**

- https://github.com/CathIAS/TLIO

Key implementation:

- `src/tracker/scekf.py`
- `src/tracker/imu_tracker_runner.py`

### 4.2 What TLIO contributes to this diagnosis

TLIO predicts a 3D displacement and uncertainty from an IMU segment and **tightly fuses** that relative measurement into a stochastic-cloning EKF.

Its relative update connects two cloned poses rather than treating one side as a frozen history record and the other as the only movable position.

The public implementation includes a measurement of the form, conceptually:

[
R_i^T(p_j-p_i),
]

with Jacobian blocks on:

- the beginning clone orientation,
- the beginning clone position,
- the ending clone position.

It then applies a standard Kalman update to the full correlated state.

This is important because it demonstrates that the UZH architecture is not an isolated implementation choice. It follows a broader learned-inertial stochastic-cloning pattern.

### 4.3 Relation to our V6.1 residual

Our current V6/V6.1 measurement is more nonlinear:

[
z_{B_e}
=
R_t^T
left[
(p_t-p_s)
-
v_sT
-
rac12gT^2
ight].
]

It depends on at least:

[
R_t,quad p_t,quad p_s,quad v_s.
]

Therefore, compared with the simpler UZH displacement factor, our Jacobian/observability structure is more delicate.

A final estimator should respect the relative nature of all of these dependencies rather than allowing a constrained gain to collapse most of the correction into current global position.

---

## 5. OpenVINS and FEJ: why a numerically valid EKF can still become inconsistent

### 5.1 References

**OpenVINS repository**

- https://github.com/rpng/open_vins

**OpenVINS FEJ / observability documentation**

- https://docs.openvins.com/fej.html

**Related consistency paper**

Mingyang Li, Anastasios I. Mourikis,
“High-precision, consistent EKF-based visual-inertial odometry,” IJRR, 2013.

- DOI: https://doi.org/10.1177/0278364913481251
- Technical report: https://intra.ece.ucr.edu/~mourikis/tech_reports/VIO.pdf

### 5.2 The relevant failure mode

A standard EKF repeatedly linearizes a nonlinear model around changing estimates.

In VIO/VINS, this can change the nullspace of the linearized system and make physically unobservable directions appear observable.

The resulting failure chain is:

```text
incorrect linearized observability
        ↓
spurious information gain
        ↓
covariance becomes overconfident / structurally wrong
        ↓
Kalman gain becomes wrong
        ↓
state correction becomes inconsistent
```

OpenVINS documents the standard VINS gauge directions as including global translation and global yaw.

The exact set for this project depends on what measurements/anchors are active, but the general rule applies:

> A learned relative-motion factor must not create absolute information along a direction that leaves that factor unchanged.

### 5.3 FEJ: First-Estimate Jacobians

FEJ evaluates selected Jacobians at consistent first-estimate linearization points rather than repeatedly relinearizing every historical quantity at newly corrected values.

The purpose is to preserve the correct unobservable subspace across time.

This is especially relevant to V6.1 because our measurement contains:

[
R_t^T.
]

Endpoint attitude therefore affects the measurement Jacobian.

If the residual is linearized with one attitude/covariance structure while the constrained gain prevents attitude from moving, correction can be redistributed into other state blocks.

This motivates an explicit FEJ study for the learned factor.

---

## 6. Observability-Constrained VINS / OC-EKF

### 6.1 References

Dimitrios G. Kottas, Joel A. Hesch, Sean L. Bowman, Stergios I. Roumeliotis,
“On the Consistency of Vision-Aided Inertial Navigation.”

- DOI: https://doi.org/10.1007/978-3-319-00065-7_22
- Author/lab publication list: https://mars.cs.umn.edu/publications.php

Related technical report:

Joel A. Hesch, Dimitrios G. Kottas, Sean L. Bowman, Stergios I. Roumeliotis,
“Observability-Constrained Vision-aided Inertial Navigation,”
University of Minnesota MARS Lab Technical Report 2012-001.

The core result is that ordinary linearized EKF models can alter the number/structure of unobservable directions and admit spurious information.

### 6.2 Their solution

OC-style estimators explicitly enforce the correct observability structure.

Two general strategies appear in this literature:

1. choose linearization points so the desired nullspace is preserved;
2. explicitly project/modify Jacobians or state-transition matrices so the unobservable directions remain in the nullspace.

For this project, a minimum relative-factor requirement is:

[
|H N_t| approx 0
]

for the common-translation gauge direction whenever no absolute position anchor is included in that measurement.

But checking (H N_t) alone is not sufficient if the project subsequently modifies the Kalman gain.

The **effective update** must also be checked for information injection into gauge directions.

---

## 7. Schmidt-EKF: the correct theory if historical states are intentionally frozen

### 7.1 Reference

Patrick Geneva, James Maley, Guoquan Huang,
“An Efficient Schmidt-EKF for 3D Visual-Inertial SLAM,” CVPR 2019.

- Paper: https://openaccess.thecvf.com/content_CVPR_2019/papers/Geneva_An_Efficient_Schmidt-EKF_for_3D_Visual-Inertial_SLAM_CVPR_2019_paper.pdf
- arXiv: https://arxiv.org/abs/1903.08636
- Related OpenVINS ecosystem: https://github.com/rpng/open_vins

### 7.2 Why this matters to `freeze_clones_*`

A Schmidt-Kalman filter separates:

```text
active states
+
nuisance / Schmidt states
```

and allows the nuisance-state mean to remain unchanged while still accounting for its statistical correlation with active states.

This is conceptually close to:

```text
freeze historical clone
update current state
```

but there is an important difference:

> Schmidt filtering is not equivalent to computing an ordinary full-state Kalman gain and then arbitrarily zeroing selected gain rows without re-deriving the corresponding covariance update.

If this project wants a final architecture where historical clones are intentionally frozen, the update should be derived as an explicit Schmidt partition rather than treated only as a gain-mask heuristic.

This is especially important because the current failure is dominated by current↔clone cross covariance.

---

## 8. Invariant EKF: a more fundamental symmetry-preserving alternative

### 8.1 References

Axel Barrau, Silvère Bonnabel,
“The Invariant Extended Kalman Filter as a Stable Observer.”

- arXiv: https://arxiv.org/abs/1410.1465
- DOI: https://doi.org/10.1109/TAC.2016.2594085

A practical aided-inertial implementation:

- https://github.com/RossHartley/invariant-ekf

### 8.2 Relevance

Invariant-EKF approaches define estimation error using system symmetry/Lie-group structure so that important error dynamics and gauge properties are less dependent on the current trajectory/linearization point.

This is a larger redesign than the immediate V6.1 work requires.

It should therefore be treated as:

```text
long-term robust estimator architecture option
```

rather than the first patch.

The immediate high-value work remains:

- proper stochastic cloning,
- observability diagnostics,
- FEJ / OC-style constraints,
- or a correctly derived Schmidt update.

---

## 9. AirIO: evidence that the V6/V6.1 representation direction should not be abandoned

### 9.1 Reference

Yuheng Qiu, Can Xu, Yutian Chen, Shibo Zhao, Junyi Geng, Sebastian Scherer,
“AirIO: Learning Inertial Odometry with Enhanced IMU Feature Observability,”
IEEE RA-L, 2025.

- arXiv: https://arxiv.org/abs/2501.15659
- DOI: https://doi.org/10.1109/LRA.2025.3581130
- Project: https://air-io.github.io/
- Code: https://github.com/Air-IO/Air-IO

### 9.2 Relevance to this project

AirIO reports that transforming UAV IMU features into a global frame can destroy useful motion information for aggressive flight, and instead emphasizes body-frame representation and explicit attitude information.

This supports the direction already taken in V6/V6.1:

```text
gyro_b + thrust_b
    ↓
relative attitude integration
    ↓
endpoint/body-aligned features
    ↓
learned short-horizon motion
```

Therefore the 2026-09-20 literature review does **not** suggest returning to an older global-frame network representation.

The current bottleneck is much more likely estimator fusion consistency than network representation.

---

## 10. Exact ways this project currently deviates from the closest references

### 10.1 UZH/TLIO: factor between two clones

Reference design:

```text
clone i ───── relative learned factor ───── clone j
   │                                         │
   └──────── full covariance coupling ───────┘
```

Our current diagnostic design:

```text
historical start clone ─ relative factor ─ current endpoint
         │                                      │
      frozen                              p/v allowed
```

This asymmetry is the largest newly identified architectural deviation.

### 10.2 Gain masking rather than a formally derived constrained filter

Current project modes such as:

- `freeze_clones`
- `freeze_clones_attitude_bias`
- `freeze_position`
- etc.

are valuable ablations and have been extremely useful for fault isolation.

However, a manually masked (K) is not automatically equivalent to:

- an ordinary full-state EKF,
- a Schmidt-EKF,
- an observability-constrained EKF,
- or an invariant EKF.

The final estimator should move from “gain-mask debugging mode” to a formally justified state/update formulation.

### 10.3 V6.1 measurement is more nonlinear than the UZH learned displacement

UZH public code uses a simpler relative-displacement structure.

V6.1 uses:

[
R_t^T
left[
(p_t-p_s)-v_sT-rac12gT^2
ight].
]

The residual depends on endpoint orientation and start velocity as well as the endpoint positions.

This makes:

- linearization-point consistency,
- attitude coupling,
- clone structure,
- and nullspace checks

more important.

### 10.4 Legacy covariance floor still dominates V6.1 uncertainty

V6.1 predicts a few-millimetre uncertainty, while runtime still includes the old V6 safety floor:

```text
X/Y: 100 mm
Z:    10 mm
```

This is badly mismatched to the current model, but it should **not** be fixed before the state-update structure is isolated.

Changing both update structure and covariance calibration simultaneously would confound the diagnosis.

---

## 11. How the architectural deviation maps to the current observed failure

The current evidence can be organized as follows.

### Observation A: network error is small and seed-consistent

```text
~1.15–1.27 mm fused-window norm RMSE across five seeds
```

Therefore large seed-to-seed position differences are unlikely to be caused primarily by TCN mean error.

### Observation B: exact Oracle measurements still show position sensitivity

Therefore the failure survives even when the learned mean is removed.

This moves the problem into:

```text
state representation
measurement Jacobian
covariance
Kalman gain
gauge / observability
scheduler / marginalization
```

### Observation C: stale endpoint-clone echo was real and is now fixed

Therefore the old deterministic feedback loop is no longer the leading explanation.

### Observation D: post-fix innovations can be millimetre-scale while position corrections are metre-scale

For:

[
delta x = K
u,
]

if (
u) is only a few millimetres but (delta p) is metres, then the relevant block of (K) is extremely large.

The main question becomes:

[
K_p = P_{p,cdot}H^TS^{-1}
]

and specifically how the relative factor interacts with:

```text
current-position covariance
current↔clone cross covariance
frozen clone gain rows
endpoint-attitude coupling
gauge directions
```

### Observation E: velocity behaves much better than position

Post-fix Oracle velocity improves over IMU-only in 5/5 seeds.

This suggests that the learned relative-motion information itself is useful.

The unstable part is likely the direct global-position correction path rather than the existence of the learned measurement.

---

## 12. Interpretation of `current_velocity_only`

The previously proposed next ablation remains highly valuable:

```text
current_velocity_only
```

Suggested semantics:

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

But after the 2026-09-20 literature comparison, its role should be understood as:

> **diagnostic ablation, not necessarily final estimator architecture.**

If Oracle + `current_velocity_only` preserves the 5/5 velocity benefit and eliminates position jumps, it strongly localizes the problem to the direct position-correction channel.

The more principled final solution is likely to restore a relative constraint between correlated states rather than permanently banning all position information from the learned factor.

---

## 13. New diagnostics that should be added before more tuning

### 13.1 Translation nullspace test

Construct a global-translation gauge basis (N_t) for all relevant position states.

For a pure relative factor, verify:

[
|HN_t| ll 1.
]

This should become a unit/regression test.

### 13.2 Effective-update gauge information test

Because this project modifies the Kalman gain, checking (H N_t = 0) is not enough.

Track gauge-direction information before/after an update, e.g.:

[
mathcal I_N = N^T P^{-1} N.
]

A relative-only learned factor should not suddenly inject large absolute translation information.

### 13.3 Gain-block logging

For every learned/Oracle fusion, log:

```text
||innovation||
||K_position||
||K_velocity||
||K_attitude||
||dx_position||
||dx_velocity||
||dx_attitude||
condition(S)
smallest/largest eigenvalue(S)
```

Also record the dominant current↔clone covariance blocks contributing to (PH^T).

### 13.4 Compare full K vs masked K

For the same Oracle event, save:

```text
K_full
K_masked
delta_x_full
delta_x_masked
```

This directly reveals how the gain mask redistributes the relative measurement correction.

### 13.5 Nullspace before and after marginalization

Because this is a fixed-lag system, observability checks should be repeated:

```text
before update
after update
after marginalization
after next clone augmentation
```

This prevents a locally correct measurement from being corrupted by clone/marginalization bookkeeping.

---

## 14. Highest-value next experiments

### Experiment A — current-velocity-only Oracle ablation

```text
trajectory:        30 s Lissajous
seeds:             0,1,2,3,4
source:            Oracle
cov multiplier:    x30
gain mode:         current_velocity_only
```

Compare against:

- IMU-only A,
- Oracle FIX x30 `freeze_clones_attitude_bias`.

Desired diagnostic outcome:

- velocity remains better than A across seeds;
- direct position jumps disappear by construction;
- position improves only through propagation of corrected velocity.

### Experiment B — UZH/TLIO-style two-clone Oracle factor

Implement a clean experimental mode where the learned/Oracle factor connects two cloned endpoints.

Core requirement:

```text
start clone
    ↕
relative factor
    ↕
end clone
```

Both endpoints should be represented in the state/covariance and participate in the mathematically defined update.

Do **not** combine this first test with V6.1 covariance-floor changes.

Start with Oracle.

### Experiment C — compare gauge behavior

For the same seeds/events, compare:

```text
1. current freeze_clones_attitude_bias
2. current_velocity_only
3. two-clone full relative update
```

Metrics should include not only trajectory RMSE but:

- (H N),
- (N^TP^{-1}N),
- Kalman gain norms,
- correction norms,
- covariance eigenvalues/conditioning.

### Experiment D — orientation/FEJ ablation

Because the V6.1 residual is endpoint-body aligned, test a clone state that explicitly preserves the orientation linearization reference.

Candidate clone content:

[
[R, v, p]
]

or at minimum:

- clone orientation/reference needed by the measurement,
- clone position,
- start velocity term required by the residual.

Compare:

```text
standard Jacobian
vs
FEJ-style Jacobian
```

under Oracle measurements first.

### Experiment E — observability-constrained projection

If the full two-clone factor still gains spurious translation/yaw information, add an OC-style projection/constraint so the measurement/update respects the intended nullspace.

Only after these structural experiments should covariance tuning resume.

---

## 15. Recommended implementation order

```text
A. implement current_velocity_only
      ↓
B. add gauge / gain diagnostics
      ↓
C. Oracle x30 five-seed diagnostic
      ↓
D. implement UZH/TLIO-style two-clone Oracle factor
      ↓
E. compare nullspace + correction behavior
      ↓
F. add orientation clone / FEJ if needed
      ↓
G. add OC-style constraint if needed
      ↓
H. reintroduce clean V6.1 network measurements
      ↓
I. recalibrate learned covariance / remove legacy V6 floor
      ↓
J. 30–60 s Lissajous + racing-like multi-seed
      ↓
K. freeze learned-inertial core
      ↓
L. add Gate PnP absolute anchors
      ↓
M. estimator-in-the-loop RL
```

---

## 16. What should *not* be done next

The 2026-09-20 evidence argues against spending the next iteration on:

- a larger TCN;
- extra hidden layers;
- another broad dataset expansion before estimator isolation;
- blind covariance multiplier sweeps;
- immediately removing the covariance floor;
- adding Gate PnP early merely to make global position look stable;
- treating `current_velocity_only` as the final architecture without testing a proper relative two-clone factor.

Gate PnP will eventually provide an absolute anchor, but it should not be used to hide an unresolved inertial-core consistency issue.

---

## 17. Refined target estimator architecture

The desired final architecture remains:

```text
                     IMU + thrust
                         │
              ┌──────────┴──────────┐
              │                     │
              ↓                     ↓
       inertial propagation     V6.x learned
                                short-horizon
                                relative motion
              │                     │
              └──────────┬──────────┘
                         ↓
             gauge-consistent fixed-lag
                    estimator
                         ↑
                         │
                Gate detector / PnP
              intermittent absolute anchor
                         │
                         ↓
                 estimator/map state
                         │
                         ↓
                    racing policy
```

Deployment constraints remain:

- no simulator GT in the deployed estimator;
- no GT pose/orientation in actor observations;
- no OpenVINS runtime dependency in the intended final learned-inertial core;
- IMU/thrust provide continuous propagation and learned short-horizon constraints;
- gate observations provide intermittent absolute corrections;
- the racing policy consumes estimator/map state only.

---

## 18. Working hypothesis after the 2026-09-20 discovery

The strongest current hypothesis is:

> V6.1 learned motion is accurate enough that the remaining dominant failure is caused by the way a relative factor is mapped through the fixed-lag covariance and constrained Kalman gain, especially when historical clone kinematics are frozen while current global position is allowed to move.

More explicitly:

```text
learned/oracle relative measurement
            ↓
correctly small innovation
            ↓
relative-factor Jacobian H
            ↓
P contains strong current↔clone correlations
            ↓
manual constrained-gain structure
            ↓
relative correction becomes concentrated in current p
            ↓
large direct world-position correction
            ↓
seed-sensitive long-run position
```

This is closely related to the classical VIO/VINS consistency problem:

> Linearization/update choices can inject spurious information into unobservable gauge directions, producing incorrect covariance and therefore incorrect Kalman gain.

---

## 19. Bottom line

The 2026-09-20 external review changes the interpretation of the project in an important way.

Before this review, the leading immediate idea was:

> “Maybe the final fix is to allow only current velocity to update.”

After comparing UZH, TLIO, OpenVINS, observability-constrained VINS, Schmidt-EKF, invariant-EKF, and AirIO, the better interpretation is:

> `current_velocity_only` is an excellent diagnostic, but the final estimator should probably preserve the learned measurement as a true relative factor between appropriately cloned/correlated states, while explicitly respecting the gauge/observability structure.

The most relevant external precedent is therefore:

```text
UZH / TLIO
    ↓
stochastic cloning
    ↓
relative factor between two clones
    ↓
joint covariance-consistent correction
```

combined with the consistency tools from:

```text
OpenVINS / FEJ
OC-VINS
Schmidt-EKF
Invariant-EKF
```

The project is no longer primarily searching for a better learned-motion model.

It is now solving a state-estimation consistency problem.

---

## 20. Reference index

### Learned inertial odometry / UAV

1. **Cioffi et al. — Learned Inertial Odometry for Autonomous Drone Racing (RA-L 2023)**
   - Paper: https://rpg.ifi.uzh.ch/docs/RAL2023_Cioffi.pdf
   - DOI: https://doi.org/10.1109/LRA.2023.3252342
   - Code: https://github.com/uzh-rpg/learned_inertial_model_odometry

2. **Liu et al. — TLIO: Tight Learned Inertial Odometry**
   - Paper: https://arxiv.org/abs/2007.01867
   - Code: https://github.com/CathIAS/TLIO

3. **Qiu et al. — AirIO: Learning Inertial Odometry with Enhanced IMU Feature Observability (RA-L 2025)**
   - Paper: https://arxiv.org/abs/2501.15659
   - DOI: https://doi.org/10.1109/LRA.2025.3581130
   - Project: https://air-io.github.io/
   - Code: https://github.com/Air-IO/Air-IO

### EKF consistency / observability

4. **OpenVINS — First-Estimate Jacobian Estimators**
   - FEJ docs: https://docs.openvins.com/fej.html
   - Code: https://github.com/rpng/open_vins

5. **Li & Mourikis — High-precision, consistent EKF-based visual-inertial odometry**
   - DOI: https://doi.org/10.1177/0278364913481251
   - Technical report: https://intra.ece.ucr.edu/~mourikis/tech_reports/VIO.pdf

6. **Kottas, Hesch, Bowman, Roumeliotis — On the Consistency of Vision-Aided Inertial Navigation**
   - DOI: https://doi.org/10.1007/978-3-319-00065-7_22
   - Lab publications: https://mars.cs.umn.edu/publications.php

7. **Geneva, Maley, Huang — An Efficient Schmidt-EKF for 3D Visual-Inertial SLAM**
   - Paper: https://openaccess.thecvf.com/content_CVPR_2019/papers/Geneva_An_Efficient_Schmidt-EKF_for_3D_Visual-Inertial_SLAM_CVPR_2019_paper.pdf
   - arXiv: https://arxiv.org/abs/1903.08636
   - Related OpenVINS codebase: https://github.com/rpng/open_vins

8. **Barrau & Bonnabel — The Invariant Extended Kalman Filter as a Stable Observer**
   - Paper: https://arxiv.org/abs/1410.1465
   - DOI: https://doi.org/10.1109/TAC.2016.2594085
   - Practical aided-inertial implementation: https://github.com/RossHartley/invariant-ekf

---

## 21. First task after this document

Do not start by retraining V6.1.

Implement:

```text
current_velocity_only
```

and simultaneously add:

```text
translation-nullspace diagnostic
gauge-information diagnostic
Kalman-gain block logging
full-K vs masked-K comparison
```

Then run:

```text
30 s Lissajous
5 seeds
Oracle
x30
```

After that result, implement a clean **UZH/TLIO-style two-clone Oracle relative factor** and compare the estimator-level behavior before changing learned covariance calibration.
