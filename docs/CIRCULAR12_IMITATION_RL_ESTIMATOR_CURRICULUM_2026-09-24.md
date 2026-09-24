# Circular-12 Imitation -> RL -> Estimator Curriculum

Date: 2026-09-24  
Branch: `feature/circular12-imitation-rl-estimator-curriculum`

## Goal

Replace the pathological wind-tumble Circular-12 GT PPO trajectory source with
an imitation-first pipeline based on the high-quality 20-degree-camera,
2.07-m prescribed circular reference, then fine-tune with PPO and replace
simulator truth with the production estimator in controlled stages.

The actor contract remains unchanged throughout:

```text
[p_w(3), v_w(3), R_wb(9), next-gate corners relative world(12), previous action(4)]
= 31 dimensions
        ->
256 x 256 x 256 ELU
        ->
normalized CTBR [collective, p_cmd, q_cmd, r_cmd]
        ->
SwiftCTBRAction -> body-rate PID -> mixer -> motors -> physics
```

## Route 1 - Extract the prescribed reference

`imitation/circular12_reference.py`

The old prescribed experiment directly wrote pose and velocity. The new
reference module only computes:

- radius 12 m;
- height 2.07 m;
- requested speed;
- coordinated attitude;
- translational acceleration;
- structured body rate;
- required collective acceleration.

It never writes simulator state.

## Route 2 - Physics-valid GT CTBR expert

`imitation/circular12_expert.py`

The expert uses simulator GT only for supervision/diagnostics and converts
tracking error into the same normalized CTBR action consumed by the learned
policy. No pose or velocity is written.

Default imitation authority is:

```text
p/q/r max = 4 / 4 / 2 rad/s
```

This matches the demonstration and imitation-PPO tasks.

### Audit the expert in free flight

```bash
./.conda-env/bin/python   scripts/imitation/evaluate_circular12_flight_quality.py   --episodes 10   --target-speed-mps 14   --output-dir artifacts/imitation/expert_audit_14   --device cuda:0   --headless
```

Do not continue to BC until the expert itself flies without inversion/tumble
and tracks radius/height/speed acceptably.

## Route 3 - Collect demonstrations and train BC

Collect a first 14 m/s curriculum dataset:

```bash
./.conda-env/bin/python   scripts/imitation/collect_circular12_expert_dataset.py   --episodes 30   --target-speed-mps 14   --output artifacts/imitation/expert_14.npz   --device cuda:0   --headless
```

Train:

```bash
python scripts/imitation/train_circular12_bc.py   --dataset artifacts/imitation/expert_14.npz   --output artifacts/imitation/bc_14.pt   --epochs 120   --device cuda:0
```

The BC network is deliberately identical to the PPO actor topology:

```text
31 -> 256 -> 256 -> 256 -> tanh(4)
```

## Route 4 - DAgger recovery training

Run the student on-policy while the GT expert labels every visited state:

```bash
./.conda-env/bin/python   scripts/imitation/collect_circular12_dagger_dataset.py   --student artifacts/imitation/bc_14.pt   --episodes 20   --target-speed-mps 14   --expert-prob 0.25   --output artifacts/imitation/dagger_14_r1.npz   --device cuda:0   --headless
```

Retrain on the aggregated data:

```bash
python scripts/imitation/train_circular12_bc.py   --dataset artifacts/imitation/expert_14.npz   --dataset artifacts/imitation/dagger_14_r1.npz   --output artifacts/imitation/bc_dagger_14_r1.pt   --epochs 120   --device cuda:0
```

Evaluate the student:

```bash
./.conda-env/bin/python scripts/imitation/evaluate_circular12_flight_quality.py \
  --controller bc \
  --checkpoint artifacts/imitation/bc_dagger_14_r1.pt \
  --episodes 20 \
  --target-speed-mps 14 \
  --output-dir artifacts/imitation/bc_dagger_14_r1_audit \
  --device cuda:0 \
  --headless \
  --fail-on-tumble
```

Repeat DAgger until inversion entries are zero and recovery is reliable.

Then repeat the curriculum at higher speeds rather than mixing incompatible
hidden target speeds in one dataset:

```text
14 m/s -> 16 m/s -> 17.7127 m/s
```

## Route 5 - Conservative PPO fine-tuning

Task:

`Isaac-Drone-Racer-Swift-CTBR-GT-Circular12-ImitationFineTune-v0`

Warm-start PPO directly from the BC/DAgger actor:

```bash
./.conda-env/bin/python scripts/rl/train.py   --task Isaac-Drone-Racer-Swift-CTBR-GT-Circular12-ImitationFineTune-v0   --imitation_bc_checkpoint artifacts/imitation/bc_dagger_17p7.pt   --imitation_bc_action_std 0.05   --num_envs 4096   --max_iterations 1000   --seed 1   --device cuda:0   --headless
```

The PPO reward keeps racing terms and adds:

- decaying expert-action anchor;
- coordinated attitude;
- coordinated body-rate;
- radius;
- height;
- speed;
- command magnitude/smoothness.

The expert anchor decays rather than disappearing abruptly.

## Route 6 - Robustness before estimator takeover

### Stage A - pure GT actor

Qualify the imitation/PPO policy first.

### Stage B - estimator shadow + Color20

Task:

`Isaac-Drone-Racer-Learned-Inertial-Swift-CTBR-Circular12-Imitation-GTShadow-Color20-v0`

Actor remains on GT while IMU + SC-EKF + Color20 run in shadow. V7 is also
evaluated online in shadow, but learned-motion fusion remains disabled.

### Stage C - GT plus estimator-like residual noise

Task:

`Isaac-Drone-Racer-Swift-CTBR-GT-Circular12-ImitationResidualNoise-v0`

Default synthetic residuals are deliberately configurable:

```text
position std = 0.08 m
velocity std = 0.06 m/s
attitude std = 0.5 deg
```

Replace these with measured estimator residuals once Stage B data are available.

Stage C continuation command (use the qualified Stage A PPO checkpoint):

```bash
./.conda-env/bin/python scripts/rl/train.py \
  --task Isaac-Drone-Racer-Swift-CTBR-GT-Circular12-ImitationResidualNoise-v0 \
  --checkpoint "$IMITATION_PPO" \
  --num_envs 4096 \
  --max_iterations 1000 \
  --seed 1 \
  --device cuda:0 \
  --headless
```

### Stage D - continuous GT -> estimator blend

Tasks:

```text
Blend25
Blend50
Blend75
```

Position and velocity use linear interpolation. Attitude uses quaternion SLERP.

### Stage E - 100% estimator state, truth mission

Task:

`Isaac-Drone-Racer-Learned-Inertial-Swift-CTBR-Circular12-Imitation-EstStateTruthMission-Color20-v0`

This stage uses the production estimator platform-state function directly;
it does not compute an alpha=1 blend and therefore does not read simulator
root pose for actor p/v/R. Before estimator initialization it returns the
existing fail-closed zero estimator state rather than GT. GT is used only for
the truth mission gate index in this diagnostic stage. Partial 25/50/75%
stages remain explicit GT/estimator blends.

### Stage F - estimator state + estimator mission

Task:

`Isaac-Drone-Racer-Learned-Inertial-Swift-CTBR-Circular12-Imitation-EstimatorMission-Color20-v0`

This is the first actor-visible no-GT mission state.

## Route 7 - Learned-motion reintroduction only after shadow qualification

Default final Color20 estimator mission task keeps learned displacement fusion
OFF.

Optional task:

`Isaac-Drone-Racer-Learned-Inertial-Swift-CTBR-Circular12-Imitation-EstimatorMission-Color20-V7-v0`

Only use it after V7 beats or meaningfully complements IMU-only propagation on
the new stable imitation-policy trajectory distribution.

## Hard acceptance gates

Do not advance from imitation to estimator qualification until:

```text
body-fixed FPV complete tumble events    = 0
inversion entries                        = 0
inverted fraction                        ~0
radius / height errors                   bounded
body rates                               structured and low
20 s free flight                         reliable
random perturbation recovery             reliable
gate completion                          comparable to the reference goal
```

The free-flight evaluator reports inversion entries, inverted fraction,
radius/height/speed error, body-rate statistics, coordinated-attitude error,
action saturation and gate performance.

## Current implementation status

Implemented in this branch:

1. prescribed-reference extraction;
2. GT CTBR expert;
3. expert demonstration collector;
4. behavior cloning;
5. DAgger collector;
6. BC -> PPO actor/scaler initialization;
7. PPO expert/coordinated-flight shaping;
8. Stage B Color20 GT shadow;
9. Stage C noisy-GT robustness;
10. Stage D 25/50/75% estimator blend;
11. Stage E 100% estimator-state / truth-mission isolation;
12. Stage F estimator-state + estimator-mission task;
13. optional V7 learned-motion final stage;
14. free-flight expert/student quality audit.

Important: GitHub CI has not executed this branch yet. The code still needs to
be pulled onto the Isaac Sim 4.5 / Isaac Lab 2.1 workstation and exercised in
the order above.
