# Swift CTBR Racing + Estimator Status and Racing-Vision Plan — 2026-09-21

## 1. Current status

The project has reached a new integration boundary:

- the GT Swift-style CTBR policy can produce genuine high-speed multi-lap racing;
- the V6.4-V6.6 learned-inertial estimator architecture remains the accepted estimator baseline;
- GTShadow has now exposed that the estimator/perception operating envelope does not match the new high-speed racing distribution.

The current successful GT checkpoint is:

```text
logs/skrl/swift_ctbr_gt_racing/
2026-09-21_20-47-02_ppo_torch_easy7_4096env_roll24_256x3_gt/
checkpoints/best_agent.pt
```

A representative successful GT racing episode completes:

```text
37 truth gates
2000 steps
20 s timeout
```

This is sufficient to generate real racing-distribution sensor data.

## 2. What was already solved before RL

The estimator work from V6.4-V6.6 is still present in the current branch.

### V6.4 direct gate reprojection

Representative 30 s result:

```text
position RMSE      ~0.052 m
velocity RMSE      ~0.028 m/s
orientation RMSE   ~0.70 deg
gate acceptance    ~71.5%
association match  ~97.7%
```

The direct mapped-corner reprojection factor accepts 2/3/4-corner observations and replaced the poorly conditioned planar-PnP pose measurement.

### V6.5 robustness

The estimator was validated against:

- up to 1 s complete visual blackout;
- +0.5 / +1 / +2 px additional corner noise;
- multiple IMU-noise seeds.

These tests remained in the centimetre-level position-error regime on the validated replay trajectory.

### V6.6 delayed visual measurements

Capture-time stochastic pose clones fixed the severe delayed-measurement inconsistency.

Compensated 50/100/200 ms visual-latency tests returned to approximately:

```text
position RMSE ~0.057 m
```

The current `feature/swift-ctbr-rl` branch contains these fixes. No later branch inspection found an estimator fix that was left behind.

## 3. Important pre-RL warning that is now relevant again

The first estimated-state closed-loop smoke test already showed:

```text
visual updates stopped after ~1.45 s
dominant reject reason = insufficient_visible_corners
position error then grew
```

The cause was camera-heading geometry. The next gate could leave the camera field of view.

After changing the controller to point the body/camera toward the mapped gate through-point, the 12 s closed-loop test achieved:

```text
accepted visual updates = 179
last accepted update     = 11.97 s
max position error       = 0.316 m
final position error     = 0.030 m
final velocity error     = 0.016 m/s
mission gates            = 2
truth gates              = 2
```

Therefore the estimator can work closed-loop when gate visibility is maintained.

The same discovery also found a learned-motion distribution shift:

```text
replay Body-Delta-v RMSE      ~0.0034 m/s
closed-loop Body-Delta-v RMSE ~0.109 m/s
fused-factor RMSE             ~0.147 m/s
```

This was accepted temporarily because frequent visual corrections kept the full state bounded.

## 4. What GTShadow now proves

GTShadow isolates estimator failure from policy feedback:

```text
GT state + GT gate target -> frozen GT policy -> CTBR -> drone
                                   |
                                   + estimator runs only in shadow
```

A successful high-speed shadow trajectory showed:

```text
truth gates       = 37
duration          = 20 s
mission gates     = 0
visual updates    = 2 / 500
position RMSE     ~120 m
```

Across the 10-episode benchmark:

```text
visual attempts                     = 806
accepted visual updates             = 19
rejected                            = 787
insufficient_visible_corners        = 718
pixel_association_gate              = 34
no_projectable_mapped_gate          = 8
reprojection_nis_gate               = 27
```

Approximately 91% of visual rejects occur before association/NIS, at the visibility stage.

## 5. Why this is not a V6.6 regression

The production estimator parameters in `feature/rl-integration-estimated-state` and `feature/swift-ctbr-rl` remain aligned:

- same V6.2 Body-Delta-v checkpoint;
- 20 Hz learned update;
- 2 Hz learned fusion;
- same nominal IMU noise/bias random walk;
- same Stage2 detector and visibility checkpoints;
- direct reprojection;
- 0.85 px pixel sigma;
- minimum 2 visible corners;
- 80 px association gate;
- normalized NIS threshold 25.

The new failure is an operating-distribution mismatch.

## 6. Distribution mismatch identified

### 6.1 Learned inertial data is too mild for current racing

The V6.1/V6.2 `racing_like` traces are analytic, low-frequency trajectories. Typical aggressive entries use approximately:

```text
amplitude  ~1.4-2.0 m
frequency  ~0.12-0.14 Hz
roll/pitch target clamp ~ +/-0.25 rad
```

Their motion is only around the low-single-digit m/s regime.

The new GT racing policy performs repeated high-speed gate-to-gate flight with much larger:

- translational speed;
- body rates;
- roll/pitch excursions;
- thrust transients;
- CTBR saturation;
- coupled manoeuvres.

Therefore the learned-motion model must be explicitly audited on GT PPO racing traces before deciding whether V7 retraining is required.

### 6.2 Vision training data is also too static

The existing Stage2 dataset collector places the robot at randomized poses and explicitly writes zero root velocity before rendering.

The dataset includes useful pose/lighting/image randomization, including synthetic blur, but it does not reproduce the actual temporal and geometric distribution of high-speed racing:

- rapid gate motion across the image;
- partial crops during aggressive turns;
- large roll/pitch/body-rate states;
- racing-specific approach/exit viewpoints;
- repeated gate transitions;
- true policy-induced camera trajectories.

### 6.3 Perception-control co-design became weaker

The earlier Swift-style reward contains an explicit perception-aware term.

The later high-performance GT racing reward prioritizes:

```text
progress
gate_passed
lookat_next_gate
```

The weak gate-centre look-at objective does not guarantee that at least two semantic gate corners remain inside the 256x256 image or that image-plane motion remains detector-friendly.

The current GT policy is therefore capable of racing quickly while being indifferent to whether the estimator can obtain visual corrections.

## 7. Decision

Do not replace the successful 37-gate GT policy.

Use it as a data generator and warm start.

The immediate next phase is:

```text
successful high-speed GT racing
        |
        +--> collect real racing camera frames + exact GT corner labels
        |
        +--> collect racing inertial traces
        |
        v
Racing Vision Dataset + Racing Motion Dataset
        |
        +--> retrain / fine-tune a more aggressive visual model
        +--> audit V6.2 Body-Delta-v on racing data
        |
        v
restore perception-aware GT racing
        |
        v
GTShadow re-evaluation
```

## 8. Racing Vision Dataset plan

Create a dedicated pure-GT racing camera task:

- frozen successful GT CTBR policy controls the drone;
- production Stage2 camera is enabled;
- no estimator/perception feedback affects control;
- collect at the production visual cadence;
- simulator GT projects exact gate corners;
- record dynamics metadata per frame;
- keep complete successful racing episodes or episodes above a configurable gate threshold.

Per-frame metadata should include:

- episode / step / timestamp;
- active gate index;
- GT gate count;
- GT position / velocity / attitude;
- body angular velocity;
- CTBR action;
- vehicle speed;
- visible corner count;
- camera intrinsics/extrinsics;
- exact projected semantic gate corners.

The first dataset should prioritize successful 30-37+ gate episodes.

## 9. Aggressive visual-model retraining plan

Start from the existing Keypoint R-CNN pipeline rather than changing architecture immediately.

The racing retraining profile should:

- train at 256x256 rather than the old 128 internal minimum;
- mix the legacy static Stage2 dataset with the new racing dataset;
- oversample racing/partial-corner samples;
- use stronger brightness/contrast/noise augmentation;
- use stronger random directional motion blur;
- validate on held-out complete racing episodes rather than random frames;
- report detection availability in addition to corner RMSE.

Primary metrics must become:

```text
P(instance detected | GT >= 2 corners visible)
P(>=2 usable corners | GT >= 2 corners visible)
corner RMSE on visible corners
availability vs speed
availability vs body-rate magnitude
availability vs image-edge proximity
availability vs visible-corner count
```

The main target is not merely lower average pixel RMSE. It is sustained usable-corner availability during high-speed racing.

## 10. Perception-aware policy follow-up

After the racing visual dataset collection path is working, fine-tune the current 37-gate GT policy with a stronger image-space observability objective.

The reward should directly use GT-projected gate corners during training and reward:

- at least 2 visible corners;
- four-corner visibility when feasible;
- image-border margin;
- low gate image-plane velocity.

GT may be used for this training-only reward. It must not enter deployment actor observations.

## 11. Acceptance criteria before reconnecting estimated state to the actor

The next estimator-shadow milestone is:

```text
GT racing          >= 30 gates and preferably 37
visual corrections continuous through the episode
mission gate index tracks truth
position / velocity / attitude remain bounded for 20 s
```

Only after GTShadow passes this operating regime should the frozen GT racing policy be driven by estimated state again.

## 12. Immediate next implementation

1. Create a new branch from this checkpoint.
2. Add a camera-enabled pure-GT racing data task.
3. Add a dataset collector that keeps successful high-speed episodes.
4. Reuse exact Stage2 GT projection for labels.
5. Add dynamics metadata and dataset manifest statistics.
6. Add an aggressive racing Keypoint R-CNN training script/config.
7. Run a short smoke collection.
8. Collect several successful 30-37+ gate episodes.
9. Fine-tune/retrain the racing vision model.
10. Evaluate racing availability before changing EKF thresholds.
