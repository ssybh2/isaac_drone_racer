# Learned-Inertial RL Discovery — 2026-09-21

## 1. Executive summary

As of the end of 2026-09-20 / beginning of 2026-09-21, the project has formally moved from estimator research into reinforcement-learning integration and policy adaptation.

The current end-to-end deployment-faithful stack is:

```text
IMU + thrust
    |
    v
Body-Delta-v TCN
    |
    v
stochastic-cloning EKF
    ^
    |
mapped gate corner pixels
    |
direct pixel reprojection
    |
known static gate map

EKF estimated state
    |
    + current mapped target gate
    |
    v
20-D actor observation
    |
    v
PPO policy
    |
    v
4 motor actions
```

The estimator itself is no longer the main blocker. V6.4-V6.6 established approximately 5-6 cm replay position RMSE, robustness to temporary visual loss and additional pixel noise, and correct timestamp-aware delayed-measurement compensation.

The current problem is now policy adaptation: the Stage 1.5 PPO policy can be loaded and fine-tuned on the real learned-inertial + camera stack, but Phase 1 training did not yet produce a reliable full-lap racing policy. The main unresolved question is why episodes usually terminate after only about 2-4 seconds and only a small number of gates.

The next required step is not more blind PPO training. It is an independent deterministic evaluation of the Phase 1 `best_agent.pt` and `agent_25600.pt` checkpoints using the newly added learned-inertial policy evaluator, so that collision, flyaway, timeout, actual truth-gate count, estimator accuracy, and visual-update continuity can be separated.

At the time this document is written, the local repository has already been synchronized, but the planned **20-episode deterministic BEST evaluation has not yet been run**.

---

## 2. Estimator work that is now considered complete

### 2.1 Body-Delta-v learned inertial model

The production learned factor is the endpoint-body-frame gravity-compensated velocity increment:

```text
z_b = R_j^T [ (v_j - v_i) - g * dt ]
```

Current checkpoint:

```text
artifacts/imo_tcn/model_v6_2_body_delta_velocity_balanced.pt
```

Validated replay runtime network accuracy was approximately:

```text
Body-Delta-v norm RMSE ~= 0.00343 m/s
```

The learned model is evaluated at 20 Hz and fused at 2 Hz with conservative uncertainty.

### 2.2 Stochastic-cloning EKF

The filter now contains:

- current 15-state inertial error state,
- stochastic [R, v, p] clones,
- current/clone cross-covariance,
- right-multiplicative attitude error,
- full Kalman gain,
- clone marginalization,
- learned two-clone factors,
- mapped-gate visual factors.

This structure supports both the learned inertial measurement and timestamp-aware delayed visual measurements.

### 2.3 Direct gate-corner reprojection

The old PnP pose measurement was replaced by direct mapped-corner pixel residuals:

```text
detected pixel
    versus
project(known gate corner, estimated body pose, calibrated camera)
```

This solved the dominant V6.3 planar-PnP orientation-lever-arm failure.

Representative improvement:

```text
V6.3 PnP position RMSE        ~0.31 m
V6.4 direct reprojection      ~0.05 m
```

The direct factor supports 2, 3, or 4 visible semantic corners.

### 2.4 V6.5 robustness

Validated stress tests included:

```text
1.0 s complete visual blackout       PASS
+0.5 px corner noise                 PASS
+1.0 px corner noise                 PASS
+2.0 px corner noise                 PASS
```

The 1-second blackout produced almost no position degradation on the tested replay.

### 2.5 V6.6 delayed visual measurements

The deliberately inconsistent V6.5 latency test showed catastrophic failure when delayed pixels were fused against the current pose:

```text
50 ms uncompensated      ~9.55 m position RMSE
100 ms uncompensated     ~512 m
200 ms uncompensated     ~824 m
```

V6.6 added capture-time stochastic pose clones and evaluates delayed pixel residuals at the correct camera timestamp.

Compensated results returned to the nominal regime:

| Configured delay | Position RMSE | Position max | Velocity RMSE | Orientation RMSE |
|---|---:|---:|---:|---:|
| 50 ms | 0.0568 m | 0.1197 m | 0.0304 m/s | 0.746 deg |
| 100 ms | 0.0582 m | 0.1218 m | 0.0312 m/s | 0.764 deg |
| 200 ms | 0.0568 m | 0.1177 m | 0.0319 m/s | 0.738 deg |

Therefore the estimator architecture is considered frozen unless later RL experiments expose a genuine estimator defect.

---

## 3. GT-free RL integration completed

Branch:

```text
feature/rl-integration-estimated-state
```

The actor-visible path has been cleaned so that simulator truth does not drive policy state or mission progression.

### 3.1 Policy state

The learned-inertial actor receives:

```text
p_w estimate                 3
q_wb estimate                4
v_b estimate                 3
onboard/noisy gyro           3
target_pos_b                 3
previous action              4
--------------------------------
total                       20
```

The 20-D contract is intentionally compatible with the earlier Stage 1 policy.

### 3.2 Target position

`learned_target_pos_b()` uses:

```text
known gate map
+
estimated body pose
```

and does not use simulator root pose.

### 3.3 Gate mission progression

A dedicated `EstimatedStateGateTargetingCommand` now advances the actor-visible `next_gate_idx` using estimator-based gate-plane crossing.

The truth path is separate:

```text
estimated mission progression
    -> actor next_gate_idx

simulator truth progression
    -> evaluation/reward gt_next_gate_idx
```

Truth does not control the actor target.

### 3.4 Fixed RL estimator configuration

The task

```text
Isaac-Drone-Racer-Learned-Inertial-RL-v0
```

uses a frozen validated configuration containing:

- V6.2 Body-Delta-v checkpoint,
- 20 Hz learned prediction,
- 2 Hz learned fusion,
- direct gate reprojection,
- 0.85 px nominal pixel sigma,
- minimum 2 corners,
- Huber robustification,
- NIS consistency gate,
- detector checkpoint,
- visibility checkpoint,
- nominal IMU corruption,
- no oracle estimator updates,
- no GT actor observation.

The RL launcher also automatically enables the onboard camera and currently enforces `num_envs=1` because the validated detector/estimator runtime is single-stream.

---

## 4. Closed-loop discovery before PPO

The first estimated-state closed-loop smoke initially failed badly.

Observed behavior:

```text
visual updates stopped after ~1.45 s
position error then grew continuously
gate updates stayed fixed while gate rejects kept growing
```

Root-cause diagnostics showed:

```text
accepted gate indices = gate 0 only
dominant reject reason = insufficient_visible_corners
```

The learned TCN continued to operate, so the first failure was not a dead estimator.

### 4.1 Camera-heading geometry problem

The original scripted controller aligned yaw with gate normal instead of pointing the camera toward the next gate.

For the gate-0 to gate-1 transition:

```text
next-gate bearing ~26.6 deg
camera horizontal half-FOV ~23.6 deg
```

The next gate could therefore leave the image immediately after the first transition.

The smoke controller was changed to point body +X / camera optical axis toward the mapped gate through-point using only estimated state and the known map.

### 4.2 Closed-loop result after camera-aware heading

The corrected 12-second test produced:

```text
accepted visual updates = 179
accepted gate 0 = 36
accepted gate 1 = 82
accepted gate 2 = 61

last accepted visual timestamp = 11.97 s

maximum position error = 0.316 m
final position error   = 0.030 m
final velocity error   = 0.016 m/s

mission gate passes = 2
truth gate passes   = 2
```

This demonstrated that the learned-inertial estimator can operate in a true closed loop when perception visibility is maintained.

### 4.3 Closed-loop TCN distribution shift

A new issue was also discovered:

```text
replay Body-Delta-v RMSE          ~0.0034 m/s
closed-loop Body-Delta-v RMSE     ~0.109 m/s
fused-factor RMSE                 ~0.147 m/s
fused-factor max error            ~0.703 m/s
```

The Z component dominated this degradation.

This is a genuine closed-loop distribution shift, but it is not currently the dominant blocker because direct visual reprojection keeps the full state bounded and repeatedly pulls the estimate back to centimetre-scale error.

Planned later work is to collect more aggressive/coupled policy-generated motion and retrain the learned inertial network. This is not being treated as a reason to stop the first RL stage.

---

## 5. Formal PPO entry

A prior Stage 1.5 policy was selected as the warm start:

```text
artifacts/stage1_recovery_20260905/checkpoints/best_agent.pt
```

This is compatible with the learned-inertial task because both use the same 20-D actor contract and the same 4-D motor action interface.

The first PPO smoke used:

```text
loaded Stage 1.5 best_agent.pt
post-load learning rate = 3e-5
max action std          = 0.5
entropy scale           = 0.001
adaptive LR max         = 1e-4
single environment
full camera + detector + TCN + EKF
```

The smoke completed:

```text
2560 / 2560 steps
no traceback
no CUDA OOM
checkpoint produced
```

Checkpoint:

```text
logs/skrl/learned_inertial_rl_smoke/
2026-09-21_00-41-54_ppo_torch/
checkpoints/agent_2560.pt
```

This established that the real learned-inertial environment can participate in PPO gradient updates.

---

## 6. Phase 1 PPO result

Phase 1 then continued for:

```text
100 iterations
256 rollout steps / iteration
= 25,600 learned-inertial environment steps
```

The run completed normally.

No traceback or CUDA OOM was observed.

Local training monitoring reported approximately:

| Training interval | Return median | Episode length median | gate_passed reward mean | progress reward median |
|---|---:|---:|---:|---:|
| 0-5120 | 3.875 | 272 steps | 0.364 | 0.045 |
| 10240-15360 | 3.813 | 230 steps | 0.400 | 0.041 |
| 20480-25600 | 4.328 | 280 steps | 0.369 | 0.123 |

Important interpretation:

- training remained numerically stable;
- progress reward increased late in training;
- gate-passed reward did not improve in the same way;
- typical episode length stayed only about 230-280 steps, i.e. 2.3-2.8 seconds;
- occasional high returns appeared, but were not sustained;
- final training return dropped back again.

Therefore more training steps alone are not currently justified.

---

## 7. First deterministic BEST-vs-FINAL evidence

Two Phase 1 checkpoints are now important:

```text
BEST:
logs/skrl/learned_inertial_rl_phase1/
2026-09-21_00-57-41_ppo_torch/
checkpoints/best_agent.pt

FINAL:
logs/skrl/learned_inertial_rl_phase1/
2026-09-21_00-57-41_ppo_torch/
checkpoints/agent_25600.pt
```

Using the existing `play.py --log 20` path, 20 deterministic episodes were run for each.

The legacy logger only records:

```text
a1 a2 a3 a4
w1 w2 w3 w4
time
```

so these files cannot yet tell whether an episode ended from collision, flyaway, or timeout.

However, episode length alone already shows a clear difference:

| Checkpoint | Mean episode length | Median | Min | Max |
|---|---:|---:|---:|---:|
| Phase 1 BEST | 377.0 steps | 308.5 | 268 | 793 |
| Phase 1 FINAL | 255.95 steps | 232.0 | 219 | 439 |

This is currently the strongest policy-selection evidence.

Interpretation:

> Continuing training to 25,600 steps did not monotonically improve the policy. The selected `best_agent.pt` survives substantially longer than the final checkpoint and should be treated as the preferred Phase 1 candidate until independent evaluation says otherwise.

Do not continue training from `agent_25600.pt` by default.

---

## 8. Current unresolved problem

The dominant unresolved RL question is:

> Why do deterministic episodes usually terminate after only a few hundred steps, and how many true gates are actually passed before termination?

The current TensorBoard and legacy play logs cannot distinguish:

```text
collision
flyaway
timeout
wrong gate progression
estimator failure
perception loss
policy trajectory failure
```

The current evidence does **not** justify blaming the estimator:

- the camera-aware scripted closed loop was stable;
- direct reprojection remained active across multiple gates;
- final estimator errors returned to centimetre scale;
- PPO itself remained numerically stable.

The most likely remaining bottleneck is policy behavior / survival / racing curriculum, but that has not yet been proven.

---

## 9. New evaluator added

A dedicated deterministic evaluator has now been added:

```text
scripts/rl/evaluate_learned_inertial_policy.py
```

It records per episode:

```text
return
duration
truth gates passed
mission gates passed
mission next gate
truth next gate
collision
flyaway
timeout
position RMSE
velocity RMSE
position max error
velocity max error
gate attempts
gate updates
gate rejects
learned updates
learned fusions
learned skips
```

It also produces a summary containing:

```text
full_lap_completion_rate
truth gate-count histogram
termination counts
episode duration statistics
estimator error statistics
perception update statistics
learned-motion update statistics
```

Supporting environment diagnostics were also added so terminal information is captured **before IsaacLab auto-reset**.

---

## 10. Exact current stopping point

At the time this discovery document is written:

```text
repository synchronized locally                         DONE

Phase 1 BEST checkpoint exists                         DONE
Phase 1 FINAL checkpoint exists                        DONE

legacy 20-episode deterministic BEST playback          DONE
legacy 20-episode deterministic FINAL playback         DONE

episode-length comparison                              DONE

new dedicated learned-inertial evaluator implemented   DONE
GitHub CI for evaluator                                 DONE

new evaluator: BEST, 20 deterministic episodes         NOT RUN YET
new evaluator: FINAL, 20 deterministic episodes        NOT RUN YET
```

This is the correct place to stop for today.

---

## 11. Next session: immediate commands

The next session should begin with the dedicated BEST evaluation, not more PPO training.

### 11.1 BEST

```bash
cd ~/isaac_projects/isaac_drone_racer

export PHASE1="logs/skrl/learned_inertial_rl_phase1/2026-09-21_00-57-41_ppo_torch"
export BEST="$PHASE1/checkpoints/best_agent.pt"
export FINAL="$PHASE1/checkpoints/agent_25600.pt"

rm -rf artifacts/rl_integration/phase1_policy_eval_best

./.conda-env/bin/python \
  scripts/rl/evaluate_learned_inertial_policy.py \
  --task Isaac-Drone-Racer-Learned-Inertial-RL-v0 \
  --checkpoint "$BEST" \
  --episodes 20 \
  --seed 1 \
  --output-dir artifacts/rl_integration/phase1_policy_eval_best \
  --device cuda:0 \
  --headless
```

### 11.2 FINAL

Only after BEST completes:

```bash
rm -rf artifacts/rl_integration/phase1_policy_eval_final

./.conda-env/bin/python \
  scripts/rl/evaluate_learned_inertial_policy.py \
  --task Isaac-Drone-Racer-Learned-Inertial-RL-v0 \
  --checkpoint "$FINAL" \
  --episodes 20 \
  --seed 1 \
  --output-dir artifacts/rl_integration/phase1_policy_eval_final \
  --device cuda:0 \
  --headless
```

The key outputs will be:

```text
artifacts/rl_integration/phase1_policy_eval_best/summary.json
artifacts/rl_integration/phase1_policy_eval_best/episodes.csv

artifacts/rl_integration/phase1_policy_eval_final/summary.json
artifacts/rl_integration/phase1_policy_eval_final/episodes.csv
```

---

## 12. Decision tree for Phase 2

The next training strategy should depend on the new evaluator.

### Case A — collisions dominate

If most failures are collisions while estimator error is small:

```text
estimator is good
policy trajectory/speed is bad
```

Then Phase 2 should focus on:

- lower initial speed / easier curriculum,
- more survival before racing speed,
- possibly stronger near-gate trajectory shaping,
- controlled exploration variance,
- possibly a collision curriculum rather than more raw steps.

### Case B — flyaway dominates

If flyaway dominates:

- inspect target progression consistency,
- inspect visual update continuity,
- inspect estimator error immediately before failure,
- determine whether the policy leaves the visual/estimator operating envelope.

Do not assume it is an estimator issue without evidence.

### Case C — estimator remains accurate but gates remain low

If:

```text
position RMSE stays small
velocity RMSE stays small
visual updates continue
truth gates passed stays around 1-2
```

then the remaining problem is predominantly RL policy adaptation / reward / curriculum.

This would justify concentrating on the policy rather than learned-inertial estimation.

### Case D — BEST is clearly better than FINAL

This is currently expected from episode length.

Then:

```text
continue from BEST
not from agent_25600.pt
```

and use a lower-risk Phase 2 continuation rather than simply increasing training duration.

---

## 13. Full-lap objective

The simulator track currently contains seven gates.

The next meaningful RL metric is therefore no longer merely return or episode length. It is:

```text
truth gates passed per episode
full 7/7 lap completion rate
failure gate distribution
collision/flyaway distribution
```

The policy architecture itself is not restricted to seven gates. It receives only the current target gate relative position, so a later known map may contain many more gates without increasing actor observation dimension.

The development order should be:

```text
reliably complete current 7-gate lap
    ->
randomize geometry / spacing / turns
    ->
train across multiple tracks
    ->
extend to longer known gate maps
    ->
real multi-gate race track
```

---

## 14. Current project state

```text
OpenVINS-free estimator                    DONE
Body-Delta-v TCN                           DONE
stochastic-cloning EKF                     DONE
direct gate reprojection                   DONE
partial-corner visual fusion               DONE
V6.5 visual robustness                     DONE
V6.6 latency compensation                  DONE

GT-free actor observation                  DONE
GT-free mission progression                DONE
truth-only evaluation progression          DONE
frozen learned-inertial RL task            DONE
estimated-state closed-loop smoke          PASS

PPO smoke 2,560 steps                      PASS
PPO Phase 1 25,600 steps                   DONE
numerical PPO stability                    PASS

policy full-lap reliability                NOT SOLVED
failure-cause identification               IN PROGRESS
BEST-vs-FINAL formal evaluation            NEXT
Phase 2 PPO curriculum                     AFTER EVALUATION
```

---

## 15. Main discovery of 2026-09-21

The project crossed an important boundary today.

Before this stage, most failures were estimator-model problems.

Now the estimator can remain stable in closed loop with camera-aware behavior, and a real PPO policy can train against the complete learned-inertial + camera stack.

The new bottleneck is policy learning quality:

```text
not:
"can the estimator work?"

not:
"can PPO run?"

but:
"can the policy survive long enough and learn reliable multi-gate racing?"
```

The answer should now be pursued with deterministic episode-level diagnostics rather than more blind gradient steps.

The immediate next action is therefore:

> Evaluate Phase 1 `best_agent.pt` over 20 deterministic episodes with the dedicated learned-inertial evaluator, then evaluate `agent_25600.pt`, compare gate-count histograms and termination causes, and only then design Phase 2.
