# 2026-09-24 — Circular12 No-GT Closed-Loop Milestone

Branch at validation time: `feature/circular12-imitation-rl-estimator-curriculum`  
Code baseline before this documentation-only commit: `c788d0cef6468794deb6a710de20351c5c6c07d8`  
Frozen snapshot branch: `freeze/2026-09-24-nogt-closedloop-success`

## 1. Executive summary

2026-09-24 produced the first complete 14 m/s Circular12 closed-loop run in which
the racing actor no longer depends on simulator GT for either platform state or
mission-gate progression.

The successful runtime path is:

```text
RGB camera
  -> Color20 Gate-ID + semantic-corner detector
  -> known Circular12 gate map
  -> direct pixel reprojection
  -> stochastic-cloning EKF visual update
                         ^
IMU accel + gyro -------|---- inertial propagation
                         |
                         v
                 p_est / v_est / R_est
                         |
              estimator-driven gate crossing
                         |
                 estimator next_gate_idx
                         |
             mapped next-gate corners - p_est
                         |
                 31D BC observation
                         |
              bc_14_gateclearance_r3.pt
                         |
                       CTBR
                         |
             body-rate PID -> mixer -> motors
```

Final one-episode no-GT-control-path result:

```text
gates                                   44
termination                             timeout
full_lap_rate                           1.0
inversion_events_total                  0
gross_attitude_excursion_events_total   0
zero_tumble_gate                        true
body_rate_p95_mean_radps                1.182401
radius_rmse_mean_m                      0.037435
height_rmse_mean_m                      0.036088
speed_mean_mps                          14.001733

estimator_position_rmse_mean_m          0.025492
estimator_velocity_rmse_mean_mps        0.017452
estimator_orientation_rmse_mean_deg     0.079765
estimator_position_final_error_mean_m   0.015637
estimator_velocity_final_error_mean_mps 0.007490
estimator_orientation_final_error_deg   0.039565

gate attempts                           500
gate updates                            500
gate rejects                            0
gate acceptance                         100%

controller <-> expert action MAE        0.002719
observation z-max p95                   2.228644
observation z-max max                   4.095253
observation clip fraction               0
```

This is the milestone to preserve.

---

## 2. Frozen controller model

Primary controller checkpoint:

```text
artifacts/imitation/bc_14_gateclearance_r3.pt
```

The checkpoint is tracked through Git LFS.

Git LFS object:

```text
sha256:cbd8e35d3d9664e3f0bb9039d570cf9885c850de3b59e2a946966a90b54cf1e4
size: 569162 bytes
```

Metadata:

```text
samples:  177766
episodes: 130
speed:    14.0 m/s
best epoch: 116
architecture contract: 31 -> 256 -> 256 -> 256 -> tanh(4)
```

Training datasets:

```text
artifacts/imitation/expert_14_v1.npz
artifacts/imitation/dagger_14_r1_beta090.npz
artifacts/imitation/dagger_14_r2_safety.npz
artifacts/imitation/dagger_14_r3_gateclearance.npz
```

Offline test metrics stored with the checkpoint:

```text
MAE              0.00266586
RMSE             0.01284974
sign agreement   0.99990785
```

Do not overwrite this checkpoint. Any future controller training must use a new
checkpoint filename.

---

## 3. Qualified BC baseline before estimator takeover

The frozen BC controller was already validated for 20 GT-state episodes:

```text
episodes                   20
gates mean                 44.55
gates min / max            44 / 45
full lap rate              1.0
inversion events           0
gross attitude events      0
zero tumble                true
body-rate p95 mean         1.19033 rad/s
radius RMSE                0.04254 m
height RMSE                0.03985 m
speed mean                 13.98767 m/s
BC <-> Expert action MAE   0.002746
observation clip fraction  0
```

This remains the rollback control baseline.

---

## 4. The critical estimator failure discovered today

The first coordinated-14 GT-shadow estimator run initially looked catastrophic:

```text
position RMSE      ~184.65 m
velocity RMSE      ~26.16 m/s
orientation RMSE   ~117.84 deg
gate updates       2 / 500
gate rejects       498 / 500
acceptance         0.4%
```

Reject distribution:

```text
gate_identity_fallback_pixel_gate   267
no_projectable_mapped_gate          216
reprojection_nis_gate                 9
pixel_association_gate                6

association stage                   489
reprojection-update stage             9
```

At first this looked like an association lockout problem.

An oracle active-gate association experiment was then added to isolate
association from reprojection/EKF behavior. The oracle run still failed:

```text
position RMSE        161.44 m
velocity RMSE         13.97 m/s
orientation RMSE       0.117 deg
final position error 279.41 m
gate updates            0 / 500
```

The combination of nearly correct attitude and an almost exactly 14 m/s
velocity error exposed the real cause.

---

## 5. Root cause: Isaac Lab 2.1 reset-induced IMU acceleration spike

Isaac Lab 2.1 computes IMU linear acceleration using a finite difference of
link velocity. Its IMU reset path does not reset the internal previous linear
velocity buffer.

The coordinated Circular12 reset instantaneously initializes the drone to about
14 m/s. Therefore the first post-reset IMU sample effectively contained:

```text
(previous velocity ~ 0 -> current velocity ~ 14 m/s) / dt
```

The estimator integrated that reset discontinuity as physical acceleration,
causing approximately one full 14 m/s false delta-v. Over a 20 s episode this
naturally generated roughly 280 m of position error.

This matched the observed failure almost exactly.

Fix:

```text
commit a7b6c6b
fix(estimator): discard reset-induced first IMU spike
```

Implementation:

1. read the first post-reset Isaac IMU sample so the simulator sensor can update
   its internal finite-difference history;
2. do not propagate that sample into the estimator;
3. synchronize only the estimator timestamp;
4. begin ordinary IMU propagation from the next sample.

This does not read runtime simulator GT and does not change the physical
estimator equations.

---

## 6. Post-fix estimator shadow result

Immediately after the reset fix, the ordinary production association path
recovered without the oracle:

```text
gates                                   44
zero tumble                             true

position RMSE                           0.026545 m
velocity RMSE                           0.019807 m/s
orientation RMSE                        0.106995 deg
final position error                    0.016508 m
final velocity error                    0.004974 m/s
final orientation error                 0.068528 deg

gate attempts                           500
gate updates                            500
gate rejects                            0
gate acceptance                         100%
```

Therefore the earlier association failure was a downstream symptom of the IMU
reset artifact, not evidence that the production association design was
fundamentally broken.

The oracle-association task remains diagnostic-only and is not part of the
successful production path.

---

## 7. Stage E: estimator p/v/R replaced GT state without retraining BC

Task:

```text
Isaac-Drone-Racer-Learned-Inertial-Swift-CTBR-Circular12-Imitation-EstStateTruthMission-Coordinated14-Color20-v0
```

The same frozen BC checkpoint was used. No controller retraining was performed.

Result:

```text
gates                                   44
full lap                                1.0
zero tumble                             true
body-rate p95                           1.18230 rad/s
radius RMSE                             0.04091 m
height RMSE                             0.02799 m
speed                                   14.00933 m/s

estimator position RMSE                 0.03936 m
estimator velocity RMSE                 0.02983 m/s
estimator orientation RMSE              0.12381 deg
gate updates                            500 / 500
controller <-> expert action MAE        0.002882
observation clip fraction               0
```

This proved that the 31D BC observation could replace GT `p/v/R` with
estimated `p/v/R` without material loss of flight quality.

Stage E still used a truth-only mission gate index, so it was an isolation
experiment rather than the final no-GT control path.

---

## 8. Stage F: final no-truth-index closed loop

A strict Stage F A/B task was added:

```text
Isaac-Drone-Racer-Learned-Inertial-Swift-CTBR-Circular12-Imitation-EstimatorMission-Legacy-Coordinated14-Color20-v0
```

Important detail: the task preserves the historical gate-crossing semantics
used during BC training. This avoids changing two variables at once.

The only control-path change from Stage E is:

```text
truth-only current gate index
        ->
estimator-driven mission progression
```

Final result:

```text
gates                                   44
cause                                   timeout
full_lap_rate                           1.0
inversion events                        0
gross attitude excursion events        0
zero_tumble                             true

body_rate_p95                           1.182401 rad/s
radius_rmse                             0.037435 m
height_rmse                             0.036088 m
speed_mean                              14.001733 m/s

estimator_position_rmse                 0.025492 m
estimator_velocity_rmse                 0.017452 m/s
estimator_orientation_rmse              0.079765 deg

estimator_final_position_error          0.015637 m
estimator_final_velocity_error          0.007490 m/s
estimator_final_orientation_error       0.039565 deg

gate_attempts                           500
gate_updates                            500
gate_rejects                            0
gate_acceptance_rate                    1.0

controller_expert_action_mae            0.002719
obs_zmax_p95                            2.228644
obs_zmax_max                            4.095253
obs_clip_fraction                       0
```

This is the first validated no-GT actor/control-path milestone.

---

## 9. What "no-GT control path" means here

The actor no longer receives simulator GT for:

- world position;
- world velocity;
- attitude;
- current mission gate index.

The actor-visible 31D observation is:

```text
estimated p_w                     3
estimated v_w                     3
estimated R_wb                    9
mapped next-gate corners - p_est 12
previous action                   4
                                ----
                                  31
```

Simulator truth remains in the environment only for evaluation and diagnostics:

- estimator RMSE calculation;
- independent truth gate-pass counting;
- inversion / gross-attitude metrics;
- expert-action comparison;
- optional debug diagnostics.

It does not determine the BC action or mission progression in the successful
Stage F run.

---

## 10. Vision path used in the successful run

Camera:

```text
Stage2 reference camera
pitch up: 20 deg
```

Detector configured by the task:

```text
artifacts/racing_vision/
circular12_color20_h207_gateid_kprcnn_v1/
torchvision_keypointrcnn_multigate_best.pt
```

Runtime outputs:

- gate instance detections;
- semantic gate-corner pixels;
- Gate ID and Gate-ID confidence.

Production thresholds:

```text
detection threshold                        0.35
keypoint confidence threshold              0.35
gate identity minimum confidence           0.50
minimum visible corners                    2
association max RMSE                       80 px
gate-ID preferred max RMSE                 15 px
gate-ID preference margin                   4 px
Huber delta                                 2.5 sigma
max normalized NIS                         25
```

The successful estimator does not first recover a planar PnP pose. It directly
uses the pixel measurement:

```text
observed 2D semantic corners
        -
projection of known 3D map corners through current EKF pose
        =
pixel residual
        ->
full SC-EKF update
```

This is a tightly coupled mapped-gate reprojection factor.

Important repository-freeze caveat:

The detector checkpoint path is referenced by code but the detector binary is
not present in the GitHub tree of this branch at the time of this summary. It
must be preserved separately on the workstation or added to an artifact/LFS
archive before the local machine is cleaned or the artifact directory is
reorganized.

---

## 11. IMU / estimator architecture in the successful run

Runtime state:

```text
R_wb
v_w
p_w
accelerometer bias
gyroscope bias
+ stochastic clones
```

Propagation:

```text
omega = gyro_measurement - gyro_bias

R_next = R * Exp(omega * dt)

specific_force = accel_measurement - accel_bias
a_world = gravity + R * specific_force

v_next = v + a_world * dt
p_next = p + v * dt + 0.5 * a_world * dt^2

P_next = F P F^T + G Q G^T
```

Vision periodically corrects the inertial trajectory with known-map gate
reprojection residuals.

---

## 12. What is deliberately NOT active in the frozen successful baseline

### OpenVINS

OpenVINS is not part of the successful runtime loop. It remains an earlier
comparison / possible fallback research path.

### Learned V7 motion fusion

The successful Stage F task has:

```text
learned_apply_displacement_updates = False
```

Therefore the current success is not evidence that the learned TCN motion
measurement improves racing localization.

The learned-motion infrastructure remains available but must stay OFF in the
frozen baseline until an explicit shadow/dropout experiment proves benefit.

### PPO continuation

The no-GT success uses the frozen BC controller. PPO is not required for this
milestone.

The earlier conservative PPO work did not materially outperform this BC
baseline, while older long PPO continuation could collapse. Do not use PPO as a
reason to modify the frozen controller.

---

## 13. Reproduction command for the milestone

After checking out the frozen snapshot and restoring the local Color20 detector
checkpoint:

```bash
./.conda-env/bin/python \
  scripts/imitation/evaluate_circular12_flight_quality.py \
  --task Isaac-Drone-Racer-Learned-Inertial-Swift-CTBR-Circular12-Imitation-EstimatorMission-Legacy-Coordinated14-Color20-v0 \
  --controller bc \
  --checkpoint artifacts/imitation/bc_14_gateclearance_r3.pt \
  --episodes 1 \
  --target-speed-mps 14 \
  --output-dir artifacts/imitation/bc14_no_truth_index_closedloop_1ep \
  --seed 4 \
  --fail-on-tumble \
  --device cuda:0 \
  --enable_cameras \
  --headless
```

Expected nominal signature:

```text
gates ~44
timeout
zero_tumble = true
estimator position RMSE = centimetre scale
estimator velocity RMSE = centimetres/second scale
estimator attitude RMSE = sub-degree
gate acceptance = ~100%
```

---

## 14. FPV visualization command/status

The repository contains:

```text
scripts/imitation/record_circular12_bc_fpv.py
```

and the same successful task can be run with camera recording to generate an
FPV clip. The detector can then be visualized by running the frozen Color20
model over the exact recorded frames and drawing Gate ID + semantic corners.

These videos are diagnostics only and must not change the frozen control path.

---

## 15. What remains unproven

Do not over-claim this milestone.

Still unproven:

1. multiple randomized initial conditions;
2. multiple IMU noise/bias seeds;
3. deliberate detector dropouts;
4. multi-second vision blackout;
5. large pixel noise / keypoint dropout;
6. camera latency stress;
7. speeds above 14 m/s;
8. real camera / real IMU timing and calibration;
9. sim-to-real dynamics;
10. real-flight validation;
11. whether V7 learned inertial improves gate-vision blackout behavior;
12. whether OpenVINS is useful as a fallback independent visual odometry source.

The current result is specifically:

> 14 m/s Circular12, coordinated known start, simulated RGB + IMU, Color20
> mapped-gate direct reprojection + SC-EKF, estimator-driven mission progression,
> frozen BC controller, one fully successful no-GT actor/control-path episode.

---

## 16. Next-session priority

Do not change the frozen baseline first.

Recommended order:

1. reproduce the frozen one-episode result;
2. archive/hash the Color20 detector checkpoint;
3. run 5-10 seeds with controlled IMU noise/bias variation;
4. add small start-position / velocity / attitude perturbations;
5. run detector pixel-noise and corner-drop stress;
6. run a deliberate vision-blackout experiment;
7. compare during blackout:
   - IMU only;
   - IMU + V7 learned inertial;
   - optional OpenVINS fallback;
8. only after robustness qualification, consider 16 m/s and then 17.7 m/s.

---

## 17. Frozen milestone rule

Treat the following as a protected baseline:

```text
Controller:
  artifacts/imitation/bc_14_gateclearance_r3.pt

Task:
  Isaac-Drone-Racer-Learned-Inertial-Swift-CTBR-Circular12-
  Imitation-EstimatorMission-Legacy-Coordinated14-Color20-v0

Estimator:
  IMU propagation
  + stochastic-cloning EKF
  + Color20 known-map direct pixel reprojection

Learned motion:
  OFF

OpenVINS:
  OFF

Speed:
  14 m/s

Track:
  Circular12

Start:
  deterministic coordinated known start

Result:
  44 gates / full lap / zero tumble / 500-of-500 vision updates
```

Future experiments should branch from this baseline and must not overwrite the
frozen BC checkpoint or silently change the task semantics.

---

## 18. Milestone statement

On 2026-09-24 the project crossed from a GT-driven racing demonstration into a
working estimator-driven closed loop:

```text
GT state -> BC -> racing
```

was successfully replaced by:

```text
RGB + IMU
    -> mapped-gate visual-inertial estimator
    -> estimator mission state
    -> frozen BC
    -> CTBR
    -> racing
```

while preserving:

```text
44 gates
full lap
zero tumble
~14 m/s
centimetre-scale state error
100% accepted visual updates
```

This is the 2026-09-24 frozen reference point.
