# 2026-09-22 Progress Summary — Circular-12 GT Racing, Anti-Spin Fine-Tuning, FPV Tumble Diagnosis

## 1. Project objective

The end goal remains a high-speed autonomous drone-racing stack whose deployed
policy does **not** depend on simulator ground truth:

```text
IMU propagation
    +
learned inertial model
    +
SC-EKF
    +
gate detector
    +
known global gate map
    ↓
estimated p / v / R
    ↓
frozen CTBR racing policy
    ↓
high-speed autonomous racing
```

The current experiment is still deliberately staged.

Before reconnecting the estimator to control, the GT upper-bound policy itself
must be physically reasonable and compatible with a body-fixed FPV camera.

Today exposed an important upstream problem:

> The original Circular-12 GT policy can race very fast, but it learned a
> sustained high-angular-rate / tumbling behavior.  Camera observability work
> must not be finalized around that pathological policy.

The main work today therefore moved from camera retraining back to the GT
racing policy.

---

## 2. Circular-12 GT baseline before today's anti-spin work

The successful Circular-12 GT checkpoint is:

```text
logs/skrl/swift_ctbr_gt_circular12/
2026-09-22_15-45-53_ppo_torch_circular12_r12_4096env_roll24_256x3_gt/
checkpoints/best_agent.pt
```

Policy contract:

```text
observation:
    GT p_w                        3
    GT v_w                        3
    GT R_wb                       9
    current next-gate corners    12
    previous action               4
                                  --
                                  31 D

action:
    bounded CTBR [collective, p_cmd, q_cmd, r_cmd]
```

The original CTBR rate authority was:

```text
roll rate  : +/-10 rad/s
pitch rate : +/-10 rad/s
yaw rate   : +/- 6 rad/s
```

The racing reward inherited the original baseline form:

```text
termination       -500
progress           +20
gate pass / miss  +400 / -400
look-at-next       +0.1
angular velocity   -0.0001
```

The angular-rate penalty was therefore tiny compared with the gate-pass reward.

### Original Circular-12 racing performance

20-episode GT evaluation was approximately:

```text
mean gates             51.3
median gates           57
full-lap rate          0.90
timeout                18 / 20
collision               1 / 20
flyaway                 1 / 20

mean speed             17.806 m/s
median speed           18.437 m/s
p95 speed              19.520 m/s
max speed              20.272 m/s

mean body-rate norm     6.270 rad/s
p95 body-rate norm      7.386 rad/s
max body-rate norm      9.851 rad/s

mean |action|           0.618
action saturation      0.362
```

This policy was very effective at passing gates, but the angular-rate regime was
extreme.

---

## 3. The attitude audit exposed the real problem

The original 30 accepted Circular-12 trajectories contained 59,970 100 Hz
samples.

Euler-angle statistics were:

```text
ROLL
signed mean            -1.741 deg
mean |roll|             55.522 deg
median |roll|           62.050 deg
p95 |roll|              76.960 deg
p99 |roll|              83.536 deg
max |roll|             174.872 deg

PITCH
signed mean            -0.549 deg
mean |pitch|            40.083 deg
median |pitch|          41.281 deg
p95 |pitch|             72.172 deg
p99 |pitch|             76.322 deg
max |pitch|             87.827 deg
```

The yaw Euler angle spans the expected +/-180 deg range and is not by itself a
useful tumble diagnostic because of wrapping.

The original interpretation that the camera should simply be pitched upward by
roughly the mean absolute pitch was therefore too simplistic.

The key discovery was made by manually reviewing body-fixed FPV footage:

> The vehicle visibly performs a wind-tumble / rolling-forward motion rather
> than merely holding a large, coordinated bank angle.

This interpretation is also supported by the very large true body-rate norm
from the original policy.

Important caveat:

- Large Euler roll/pitch values alone do not prove tumbling.
- Circular flight at roughly 18-19 m/s on a radius-12 m track physically
  requires a very large bank angle.
- The problem is the **continued rotation / full-roll behavior**, not the
  existence of a 60-75 deg bank angle itself.

---

## 4. Camera +40 deg smoke test and why it is not final

Before the tumble issue was fully understood, a +40 deg upward camera mount was
implemented and smoke-tested.

Old 0 deg dataset:

```text
artifacts/racing_estimator/circular12_v1/vision/train/manifest.json

samples                         9980
ANY mapped gate >=2 corners     28.176%
active >=2 corners               8.88% approximately
ANY mapped 4 corners            10.39%
```

+40 deg smoke:

```text
artifacts/racing_estimator/circular12_pitch40_smoke/vision/train/manifest.json

samples                          499
ANY mapped gate >=2 corners      50.501%
active >=2 corners                3.61%
ANY mapped 4 corners             38.48%
ANY mapped >=3 corners           40.68%
```

So +40 deg clearly improved **any mapped gate** visibility on the old trajectory.

However, that trajectory came from the tumbling GT policy.

Therefore:

> The +40 deg result is a useful geometric observation, but it is **not** a
> final production camera decision.

Do not collect the full 30-episode +40 deg dataset yet.

The final camera angle must be selected only after the GT policy's tumbling
behavior has been removed.

---

## 5. Why the original reward permitted tumbling

The CTBR policy directly commands body rates.

The original maximum commanded rates were:

```text
p_max = 10 rad/s
q_max = 10 rad/s
r_max =  6 rad/s
```

while the racing angular-velocity penalty was only:

```text
weight = -0.0001
```

The policy therefore had a very cheap way to exploit aggressive body rotation
while collecting very large gate-pass rewards.

A second structural issue is now also clear:

```text
lookat_next_gate
```

aligns the vehicle **body +X axis** with the next-gate direction.

A rotation around body +X leaves body +X unchanged.

Therefore a slow or fast roll about the forward axis can preserve a good
look-at reward while still rotating the camera horizon through a full cycle.

This explains why simply strengthening the look-at term cannot guarantee the
absence of wind-tumble behavior.

---

## 6. Stable anti-spin GT task added

A separate task was added instead of modifying the successful original GT
baseline in place:

```text
Isaac-Drone-Racer-Swift-CTBR-GT-Circular12-Stable-v0
```

The Stable-v1 changes are:

```text
body_rate_max_radps:
    old    (10, 10, 6)
    stable ( 6,  6, 3)

angular velocity:
    old    -0.0001
    stable -0.02

look-at-next:
    old    +0.1
    stable +0.5

body-rate command L2:
    stable -0.01

CTBR command delta L2:
    stable -0.002
```

No level-attitude penalty was added.

This was intentional because high-speed radius-12 circular flight needs a large
bank angle.

---

## 7. Stable-v1 fine-tuning

The Stable policy was not trained from scratch.

It was warm-started from the successful original Circular-12 GT checkpoint.

Training command used:

```bash
./.conda-env/bin/python \
  scripts/rl/train.py \
  --task Isaac-Drone-Racer-Swift-CTBR-GT-Circular12-Stable-v0 \
  --checkpoint "$C12_OLD" \
  --num_envs 4096 \
  --max_iterations 1000 \
  --post_load_learning_rate 5e-5 \
  --post_load_max_action_std 0.20 \
  --entropy_loss_scale 0.001 \
  --adaptive_lr_max 1e-4 \
  --seed 1 \
  --device cuda:0 \
  --headless
```

Training completed:

```text
24000 / 24000 vector steps
1000 PPO iterations
wall time ~12 min
```

The loaded action standard deviations were already below 0.20, so the requested
std cap did not change them:

```text
before / after:
0.15439
0.09126
0.09455
0.0699998
```

Stable-v1 checkpoint:

```text
logs/skrl/swift_ctbr_gt_circular12_stable/
2026-09-22_22-20-55_ppo_torch_circular12_r12_4096env_roll24_256x3_gt_stable_antispin/
checkpoints/best_agent.pt
```

---

## 8. Stable-v1 motion / attitude audit

Five collected trajectories produced 7,996 samples.

Measured values:

```text
speed
mean                    18.793 m/s
median                  19.582 m/s
p95                     20.706 m/s
max                     21.391 m/s

body-rate norm
mean                     3.424 rad/s
median                   3.358 rad/s
p90                      4.115 rad/s
p95                      4.479 rad/s
p99                      5.635 rad/s
max                      6.371 rad/s

|body p|
mean                     1.211 rad/s
p95                      2.804 rad/s
max                      5.400 rad/s

|body q|
mean                     1.221 rad/s
p95                      2.817 rad/s
max                      4.941 rad/s

|body r|
mean                     2.726 rad/s
p95                      3.004 rad/s
max                      3.116 rad/s

body tilt
mean                    72.040 deg
median                  72.741 deg
p95                     79.099 deg
p99                     81.512 deg
max                     96.792 deg

rate > 3 rad/s          80.54%
rate > 4 rad/s          12.77%
rate > 5 rad/s           2.50%

tilt > 80 deg            3.06%
inverted > 90 deg        0.19%
```

### Interpretation

The anti-spin fine-tune clearly reduced angular-rate magnitude:

```text
old mean body-rate       ~6.27 rad/s
Stable-v1 mean            3.42 rad/s

old p95 body-rate         ~7.39 rad/s
Stable-v1 p95              4.48 rad/s
```

At the same time, speed increased rather than collapsed:

```text
old mean speed           ~17.81 m/s
Stable-v1 mean            18.79 m/s
```

The approximately 72 deg mean body tilt is not automatically pathological.

For a radius-12 m circle at 18.79 m/s:

```text
a_c = v^2 / r ~= 29.4 m/s^2
required coordinated bank ~= atan(a_c / g) ~= 71.6 deg
```

So the steady large bank is physically consistent with high-speed circular
flight.

---

## 9. Stable-v1 20-episode racing evaluation

Stable-v1 random-start evaluation:

```text
episodes                 20
mean gates               45.2
median gates             60
min                       0
max                      60

histogram:
0 gates                   2
1 gate                    1
2 gates                   1
5 gates                   1
59 gates                  4
60 gates                 11

full-lap completion       0.75

termination:
timeout                   15
collision                  4
flyaway                    1

mean duration             15.395 s
median duration           20.0 s

mean |action|              0.6366
action saturation          0.3647
```

The successful 15 / 20 episodes were extremely strong:

```text
4 x 59 gates
11 x 60 gates

successful-episode mean ~= 59.73 gates
```

This indicates that Stable-v1 has excellent **steady-state racing** once it
enters the normal racing orbit.

The current random-start weakness is concentrated in early transient failures:

```text
0 gates / flyaway
2 gates / collision
5 gates / collision
0 gates / collision
1 gate / collision
```

So Stable-v1 is not yet a complete replacement for the old baseline in terms of
random-start robustness.

---

## 10. Most important end-of-day observation: wind-tumble still exists

The Stable-v1 checkpoint was rendered with a **0 deg body-fixed camera** so that
FPV directly reflects vehicle attitude.

Manual FPV inspection shows:

> Wind-tumble behavior is still present.

However:

> The tumble frequency is substantially lower than with the original policy,
> and the period is much longer.

This is the most important unresolved issue at the end of 2026-09-22.

Therefore the correct conclusion is **not** that anti-spin is solved.

The correct conclusion is:

```text
original policy:
    high-frequency / severe tumble

Stable-v1:
    angular-rate magnitude strongly reduced
    steady-state racing preserved
    BUT
    long-period tumble events still occur
```

This also explains why the aggregate inversion fraction can be very small while
FPV still reveals obvious occasional complete roll events.

A rare full roll can occupy only a small percentage of all 100 Hz samples.

---

## 11. Why Stable-v1 did not fully eliminate tumbling

Stable-v1 mainly penalizes:

```text
angular-rate magnitude
body-rate command magnitude
command changes
```

This reduces the **speed** of unwanted rotation.

It does not explicitly constrain the vehicle's **roll phase / orientation around
the forward axis**.

A slow complete roll can therefore become much cheaper than the original fast
roll while still allowing the policy to:

```text
keep body +X roughly toward the next gate
keep high progress
pass gates
retain large gate reward
```

This matches the observed FPV result:

> The wind-tumble remains, but its period becomes long.

This is now the leading hypothesis for Stable-v1.

---

## 12. Required next policy change: anti-tumble, not simply more anti-rate

The next iteration should be treated as **Stable-v2 / anti-tumble**, not just a
larger angular-rate penalty.

The key requirement is:

> Prevent complete roll/tumble cycles while still allowing the approximately
> 70 deg coordinated bank required by the track.

Do **not** add a generic “keep the drone level” reward.

That would conflict directly with the physics of 18-19 m/s radius-12 racing.

A better design is to add an orientation constraint that distinguishes:

```text
valid coordinated bank
from
unnecessary roll phase / inversion
```

Two candidate mechanisms should be tested next:

### A. Safety barrier on excessive tilt / inversion

Use the body +Z axis relative to world +Z:

```text
c = z_body dot z_world
tilt = acos(c)
```

Normal racing is around:

```text
tilt ~= 72 deg
```

Current Stable-v1 p99 is:

```text
81.5 deg
```

A soft barrier beginning around 82-85 deg can therefore leave normal bank
largely untouched while making a full roll expensive.

A stronger termination condition can be considered near / beyond 90 deg.

This is the simplest anti-tumble experiment.

### B. Coordinated-turn attitude reference

A more principled solution is to build a privileged GT reference attitude from
the known track:

```text
desired forward direction:
    local track tangent / velocity direction

desired inward acceleration:
    centripetal direction toward the Circular-12 centre

desired body +Z:
    normalized(gravity compensation + required centripetal acceleration)
```

Then penalize attitude error relative to that coordinated-turn frame.

This preserves the required 70 deg bank while removing the free roll degree of
freedom around the forward axis.

This is the preferred long-term formulation if the simple tilt barrier is not
sufficient.

---

## 13. Acceptance criteria for the next GT policy

Do not continue to vision training until the GT policy passes **all** of these
checks.

Suggested acceptance targets:

```text
RACING
random-start full-lap rate          >= 0.85-0.90
median gates                         >= 59
successful episodes                  ~59-60 gates / 20 s
mean speed                           >= 17 m/s

ANGULAR MOTION
mean body-rate                       <= ~3.5 rad/s
p95 body-rate                        <= ~4.5 rad/s
inverted fraction                    approximately 0

FPV
zero complete wind-tumble events
across several successful 20 s episodes

CONTROL
action saturation should not worsen materially
```

FPV is a mandatory acceptance test.

Aggregate angle statistics alone are not sufficient.

---

## 14. Camera work is paused until anti-tumble passes

Do not yet:

```text
collect the full 30-episode +40 deg dataset
train the final multi-gate detector
freeze the +40 deg camera mount
reconnect vision corrections to final policy evaluation
```

Once the GT policy is physically acceptable, run a camera-angle sweep on the
**new stable trajectory**.

Suggested mount sweep:

```text
0 deg
+10 deg
+20 deg
+30 deg
+40 deg
+50 deg
```

For each angle measure:

```text
ANY mapped gate >=2 corners
ANY mapped gate >=3 corners
ANY mapped gate 4 corners
active gate >=2 corners
longest visual blackout
distribution of consecutive unusable frames
```

The final camera angle should be chosen from these trajectory-conditioned
visibility measurements, not from mean Euler pitch.

---

## 15. Estimator status remains unchanged

Today's discovery does **not** invalidate the estimator architecture.

The current estimator direction remains:

```text
100 Hz IMU propagation
+
learned inertial model in shadow / robustness role
+
SC-EKF
+
known map
+
multi-gate detector
+
direct mapped-corner reprojection
+
capture-time clone compensation
```

The previous nominal audit already showed that under current nominal simulator
noise, raw IMU propagation over the tested window can outperform the V7 learned
delta-v prediction:

```text
IMU-only:
position RMSE            ~0.726 m
velocity RMSE            ~0.101 m/s
orientation RMSE         ~0.026 deg

V7 network:
online delta-v norm RMSE ~0.116 m/s

same-window IMU delta-v:
norm RMSE                ~0.062 m/s
```

Therefore learned-motion fusion remains OFF in the nominal production
comparison and the network remains shadow / robustness support for now.

The current blocker is still upstream policy/camera behavior, not the basic EKF
measurement model.

---

## 16. Remote branch state at end of session

Branch:

```text
feature/racing-vision-retraining
```

Before this summary was added, the remote branch contained the Stable-v1 task,
the Stable PPO profile, and a Stable KnownStart diagnostic task.

The user workstation had **not yet pulled the latest KnownStart additions** at
the time the session ended.

Therefore the first operation next session should be to synchronize the branch
before running new code.

The Stable KnownStart task already present remotely is:

```text
Isaac-Drone-Racer-Swift-CTBR-GT-Circular12-Stable-KnownStart-v0
```

It is useful for separating random-start transient failures from steady-state
policy quality, but it has not yet been run locally.

---

## 17. Next-session execution order

Use this order next session.

### Step 1 — sync branch

```bash
cd ~/isaac_projects/isaac_drone_racer

git fetch ssybh2 --prune
git switch feature/racing-vision-retraining
git pull --ff-only ssybh2 feature/racing-vision-retraining

git rev-parse --short HEAD
```

### Step 2 — preserve Stable-v1 as a frozen diagnostic checkpoint

Do not overwrite:

```text
logs/skrl/swift_ctbr_gt_circular12_stable/
2026-09-22_22-20-55_ppo_torch_circular12_r12_4096env_roll24_256x3_gt_stable_antispin/
checkpoints/best_agent.pt
```

This checkpoint is useful because it demonstrates:

```text
anti-rate shaping substantially reduces angular velocity
while preserving very fast steady-state racing
```

even though it does not yet eliminate long-period tumble.

### Step 3 — quantify tumble events more directly

Before another long training run, add / run diagnostics that measure:

```text
body +Z dot world +Z
tilt excursions above 80 / 82 / 85 / 90 deg
quaternion incremental rotation
body-rate components
duration between tumble events
number of complete FPV roll cycles per episode
```

Do not rely only on Euler roll / pitch.

### Step 4 — implement Stable-v2 anti-tumble constraint

First try:

```text
soft high-tilt barrier around 82-85 deg
+
strong inversion penalty / termination near 90 deg
```

while retaining enough authority for the normal ~72 deg bank.

If this harms racing or does not remove the slow roll, move to the
coordinated-turn attitude reference.

### Step 5 — fine-tune from Stable-v1

Warm-start Stable-v2 from the current Stable-v1 best checkpoint rather than
returning to the original tumbling checkpoint.

### Step 6 — validate in this order

```text
1. 20 random-start GT episodes
2. motion / body-rate audit
3. several 0-deg FPV videos
4. KnownStart benchmark
```

FPV must show no complete tumble cycle.

### Step 7 — only then resume perception work

After the GT policy passes:

```text
camera mount sweep
    ->
new 30-episode racing vision dataset
    ->
multi-instance gate detector retraining
    ->
GT-shadow estimator
    ->
closed-loop estimator-state policy
```

---

## 18. End-of-day state

The project is **not** back at the beginning.

Today's result narrowed the problem substantially.

We now know:

```text
1. Circular-12 GT racing itself can be extremely strong.

2. The original GT policy's high-rate tumbling was a real control-policy issue,
   not merely an Euler-angle plotting artifact.

3. The first Stable anti-spin fine-tune cut mean body-rate from about
   6.27 -> 3.42 rad/s while increasing mean speed to about 18.79 m/s.

4. Stable-v1 can sustain 59-60 gates / 20 s when it enters the normal racing
   orbit.

5. Stable-v1 random-start robustness is lower than the old baseline
   (75% full-lap vs roughly 90%).

6. Most importantly, manual 0-deg FPV inspection proves that long-period
   wind-tumble behavior still remains.

7. Therefore the next policy problem is no longer simply "reduce angular
   velocity".  It is "remove the free tumble / roll phase while preserving the
   physically required ~70 deg coordinated bank".

8. Camera angle and final vision-dataset work must wait until this policy issue
   is solved.
```

The immediate next milestone is therefore:

> **Stable-v2: zero complete tumble events in FPV, strong random-start racing,
> and retained high-speed coordinated bank.**
