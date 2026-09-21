# 2026-09-21 Progress Summary — GT Racing, Camera Observability, and Racing Vision

## 1. Goal

The long-term goal remains a high-speed drone-racing stack that does **not** depend on simulator ground truth at runtime:

```text
high-speed racing policy
        +
IMU propagation
        +
learned inertial model
        +
EKF / stochastic cloning
        +
camera gate detection
        +
known global gate map
        ↓
closed-loop high-speed autonomous racing
```

Today the main focus was deliberately narrowed to one prerequisite:

> First obtain a strong GT racing policy that can still race at high speed while keeping the next gate observable in the production camera.

This is necessary before retraining the vision stack or reconnecting estimator feedback.

---

## 2. Starting point

The existing successful GT racing checkpoint was:

```text
logs/skrl/swift_ctbr_gt_racing/
2026-09-21_20-47-02_ppo_torch_easy7_4096env_roll24_256x3_gt/
checkpoints/best_agent.pt
```

Its training contract is:

```text
task                  Isaac-Drone-Racer-Swift-CTBR-GT-Racing-v0
track                 Easy-7
num_envs              4096
episode length        20 s
observation           31-D GT observation
action                4-D bounded CTBR
network               shared 256 x 256 x 256 ELU
rollout               24
PPO epochs            5
minibatches           4
learning rate         1e-4
gamma                 0.99
lambda                0.95
entropy               0.005
reward shaper         0.6
trainer timesteps     50000
```

The original high-speed GT policy already had strong racing performance, but camera observability had not been explicitly optimized.

---

## 3. Racing-vision collector reset bug was fixed

A serious dataset-collection bug appeared after the first collision:

```text
successful episodes
successful episodes
collision
then every following episode immediately terminated at step 1 as collision
```

An explicit environment reset alone did not fix it.

The root cause was traced to stale contact-sensor state during repeated episodes. The collector was changed so the collision sensor is refreshed every physics step:

```text
collision_sensor.history_length = 1
```

Together with an explicit reset after every episode, the collector now recovers correctly after collisions.

Validation example:

```text
attempt 01  KEEP 36 gates
attempt 02  KEEP 37 gates
attempt 03  KEEP 37 gates
attempt 04  KEEP 37 gates
attempt 05  DROP collision
attempt 06  KEEP 36 gates
```

This means the racing-vision dataset collector is now usable for long multi-episode collection.

---

## 4. Baseline high-speed camera-observability measurement

A 5-successful-episode smoke dataset was collected using the original GT racing policy.

Total:

```text
2495 frames
25 Hz camera sampling
256 x 256
```

Active-gate GT corner visibility:

```text
0 visible corners : 2099
1 visible corner  : 173
2 visible corners : 118
3 visible corners : 49
4 visible corners : 56
```

Therefore:

```text
>=2 visible corners = 223 / 2495
                     = 8.94%
```

This was the first important quantitative result of the day.

### Meaning

The estimator currently requires at least two visible gate corners for a direct reprojection update.

However, with the original GT racing policy:

> Only about 9% of racing frames contain at least two visible corners of the active gate.

Therefore the previous GTShadow result of almost no visual corrections cannot be explained only by detector failure.

A major part of the problem is upstream:

> The racing policy frequently does not keep the active gate inside the camera FOV.

---

## 5. Real racing dynamics are much more aggressive than the old training data

The new racing dataset also quantified the actual motion regime.

Typical original-policy racing values:

```text
mean speed             ~13.1 to 13.4 m/s
max speed              ~15 to 16 m/s
mean body-rate norm    ~6.5 rad/s
max body-rate norm     ~9 to 10 rad/s
```

This confirms that both old perception training data and old learned-inertial training data were substantially milder than the real high-speed PPO racing distribution.

The old Stage2 vision dataset was largely based on static randomized poses plus synthetic blur.

The old learned-inertial `racing_like` data used smooth analytic trajectories with much lower motion aggressiveness.

Therefore both subsystems still require later racing-distribution adaptation.

---

## 6. Perception-aware GT policy V1

A new GT training task was added:

```text
Isaac-Drone-Racer-Swift-CTBR-GT-PerceptionAware-v0
```

All PPO specifications were kept identical to the successful GT baseline.

Only one image-space perception-aware reward was added.

The reward analytically projects the calibrated four gate corners into the production 256 x 256 camera and rewards:

```text
gate center near image center
+
more visible corners
+
larger border margin
+
explicit >=2 visible-corner bonus
```

No rendered camera image and no detector are needed during 4096-env PPO training.

The original GT checkpoint was used as the warm start rather than training from zero.

### V1 result

Five successful 20-second episodes:

```text
gates passed:
39
39
39
40
39
```

The policy actually became faster.

However camera observability became worse:

```text
OLD >=2 visible corners : 8.94%
V1  >=2 visible corners : 6.89%
```

Conclusion:

> V1 did not teach the intended perception behavior. The policy mainly learned to increase racing performance.

---

## 7. Perception-aware GT policy V2

V2 changed the perception objective from a mostly positive bonus into a stronger penalty-oriented objective.

The idea was:

```text
good visibility       -> near zero extra penalty
poor edge margin      -> penalty
<2 visible corners    -> explicit extra penalty
```

The objective was intentionally designed so the policy could not simply hover and accumulate positive perception reward.

New task:

```text
Isaac-Drone-Racer-Swift-CTBR-GT-PerceptionAwareV2-v0
```

Again, training used the original successful GT checkpoint as the warm start.

### V2 result

Five accepted episodes were extremely strong in racing performance:

```text
40
39
40
40
40 gates
```

Typical motion:

```text
mean speed             ~14.6 m/s
max speed              ~17.5 m/s
mean body-rate norm    ~6.4 rad/s
max body-rate norm     ~9.7 rad/s
```

But active-gate observability still barely improved:

```text
OLD >=2 visible corners : 223 / 2495 = 8.94%
V1  >=2 visible corners : 172 / 2495 = 6.89%
V2  >=2 visible corners : 239 / 2495 = 9.58%
```

So V2 improved only:

```text
+0.64 percentage points over OLD
```

while significantly increasing racing speed.

### Current interpretation

V2's `<2 visible corners` penalty is mostly binary.

When the gate is completely outside the image, the policy sees almost the same penalty regardless of whether it should rotate left, right, up, or down.

Therefore the reward gives weak directional information once the gate has already left the FOV.

This is likely why PPO learned to compensate by improving gate progress / gate-pass rate instead of learning a fundamentally different camera-facing trajectory.

---

## 8. Current main problem

At the end of 2026-09-21, the primary blocker is:

> We have a very strong high-speed GT racing policy, but the active gate is still usable by the camera in only about 9–10% of sampled frames.

This is now more important than retraining the detector.

Even a perfect detector cannot provide visual updates when the gate is geometrically outside the image.

The current failure chain is:

```text
high-speed policy
        ↓
aggressive turns / body orientation
        ↓
active gate leaves camera FOV
        ↓
<2 GT-visible corners
        ↓
vision cannot provide a usable reprojection
        ↓
EKF receives too few visual corrections
        ↓
learned-inertial / IMU drift eventually dominates
```

---

## 9. V3 implementation prepared on the remote branch

A V3 implementation has already been added to the remote branch, but the local workstation has **not yet been synchronized** with these latest changes.

Current remote branch:

```text
feature/racing-vision-retraining
```

Remote HEAD at the end of the day:

```text
7cce3344d42c3168ff0d7ca80baf5af97eeac5c5
```

New task:

```text
Isaac-Drone-Racer-Swift-CTBR-GT-PerceptionAwareV3-v0
```

### V3 idea

Instead of only asking whether the gate is inside the image, V3 adds a dense continuous camera-direction penalty:

```text
camera_angle_l2
```

It measures the squared 3-D angle between:

```text
camera optical axis
and
next-gate center direction
```

This signal remains meaningful even when the gate is completely outside the image.

Therefore PPO always gets directional information:

```text
gate slightly off-axis   -> small penalty
gate far from boresight  -> larger penalty
gate behind / badly misaligned -> very large penalty
```

V3 keeps a lighter image-space visibility constraint as well.

Current V3 reward terms:

```text
camera_angle_l2       weight = -8.0
camera_observability weight = +2.0
```

All PPO hyperparameters remain identical to the successful GT baseline.

---

## 10. Next session — first action

The local workstation has not yet pulled the V3 code.

The first action next session should therefore be:

```bash
cd ~/isaac_projects/isaac_drone_racer

git fetch ssybh2 --prune
git switch feature/racing-vision-retraining
git pull --ff-only ssybh2 feature/racing-vision-retraining

git rev-parse --short HEAD
```

Expected:

```text
7cce3344
```

Then validate:

```bash
./.conda-env/bin/python -m pytest -q   tests/rl/test_swift_ctbr_static.py   tests/perception/test_racing_vision_static.py

./.conda-env/bin/python -m compileall -q   tasks/drone_racer   scripts/rl   scripts/perception   perception
```

---

## 11. Next session — V3 training

If tests pass, train from the original successful GT checkpoint, not from V1 or V2:

```bash
export BEST="logs/skrl/swift_ctbr_gt_racing/2026-09-21_20-47-02_ppo_torch_easy7_4096env_roll24_256x3_gt/checkpoints/best_agent.pt"

./.conda-env/bin/python   scripts/rl/train.py   --task Isaac-Drone-Racer-Swift-CTBR-GT-PerceptionAwareV3-v0   --checkpoint "$BEST"   --post_load_learning_rate 1.0e-4   --num_envs 4096   --seed 1   --device cuda:0   --headless
```

TensorBoard:

```bash
./.conda-env/bin/tensorboard   --logdir logs/skrl/swift_ctbr_gt_perception_v3   --host 0.0.0.0   --port 6006
```

---

## 12. V3 acceptance criteria

V3 should not be judged by PPO reward alone.

It must satisfy both:

### Racing performance

```text
>=30 gates / 20 s minimum
preferably still around 35–40 gates / 20 s
```

### Camera observability

Baseline:

```text
OLD = 8.94% frames with >=2 GT-visible active-gate corners
```

A useful improvement should be clearly larger than statistical noise.

Desired direction:

```text
10%     -> not enough
15%     -> modest improvement
20%+    -> meaningful
30%+    -> strong improvement
```

The exact final target can be refined after V3 results.

---

## 13. If V3 still fails

If V3 also remains near 10% visibility, stop blindly tuning reward weights.

The next investigation should be structural:

1. verify the production camera mounting orientation and calibrated optical axis;
2. quantify the actual horizontal / vertical FOV relative to the Easy-7 gate geometry;
3. inspect gate-bearing angle distributions during high-speed trajectories;
4. measure how often the next gate is physically impossible to keep inside the current FOV at 14–17 m/s;
5. compare active-gate visibility with **any mapped gate visibility**;
6. consider detecting / associating any visible mapped gate instead of only the mission gate;
7. consider camera mounting / wider-FOV changes if the geometry itself is the limiting factor.

This is important because the current camera is relatively narrow for aggressive racing.

If the geometry makes continuous next-gate visibility impossible, no reward design can fully solve the problem.

---

## 14. Work intentionally postponed

Do **not** move to the following tasks until the camera-observability question is understood:

```text
Racing Vision detector retraining
V7 learned-inertial retraining
full GTShadow estimator re-test
estimated-state policy feedback
full no-GT racing
```

The current priority is to solve the upstream observability problem first.

---

## 15. Current branch status

Branch:

```text
feature/racing-vision-retraining
```

Important work already present on this branch includes:

```text
racing-vision dataset collector
collision/reset recovery fix
aggressive racing-vision training tools
production-hybrid evaluator
GT perception-aware V1
GT perception-aware V2
GT perception-aware V3 implementation
daily design / experiment documentation
```

Remote HEAD at end of session:

```text
7cce3344d42c3168ff0d7ca80baf5af97eeac5c5
```

---

## 16. End-of-day conclusion

The most important result from 2026-09-21 is not the 39–40 gate speed increase.

It is the diagnosis that:

> The current high-speed racing policy itself does not maintain sufficient camera observability.

The original GT policy, V1, and V2 all leave the active gate with >=2 visible corners in only about 7–10% of sampled frames.

Therefore the next step is not simply “train a better detector.”

The next step is:

> Make the GT racing trajectory itself camera-aware enough that the vision system is given usable geometry often enough to support the estimator.

V3 is the next controlled experiment for that question.
