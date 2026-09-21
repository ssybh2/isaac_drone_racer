# GT Perception-Aware Racing Training — 2026-09-21

## Goal

Keep the successful 36-37 gate GT CTBR racing behavior while teaching the
policy to keep the active gate usable in the production camera image.

The new task is:

```text
Isaac-Drone-Racer-Swift-CTBR-GT-PerceptionAware-v0
```

It preserves the successful baseline contract:

```text
4096 environments
20 s episodes
31-D GT observation
bounded 4-D CTBR action
shared 256x256x256 ELU policy/value model
rollout = 24
PPO epochs = 5
minibatches = 4
learning rate = 1e-4
gamma = 0.99
lambda = 0.95
entropy = 0.005
reward shaper = 0.6
trainer timesteps = 50000
Easy-7 track
same random-start distribution
```

The only behavioral change is an additional training-only image-space
perception reward.

## Image-space reward

The reward analytically projects the four calibrated gate-opening corners into
the production Stage2 camera:

```text
256 x 256
fx = fy = 293.1997 px
cx = cy = 128 px
camera offset = (0.14, 0.0, 0.05) m
camera forward = body +X
```

No camera rendering and no detector are used during 4096-env PPO training.

The score combines:

- smooth gate-center centering, including a recovery signal just outside FOV;
- fraction of four semantic corners inside the image;
- margin from image borders;
- explicit bonus when at least two corners are visible.

The >=2-corner bonus directly matches the production direct-reprojection
minimum visual-measurement requirement.

The new term has reward weight:

```text
camera_visibility = 2.0
```

while the successful racing rewards remain unchanged:

```text
termination   = -500
ang_vel_l2    = -0.0001
progress      = +20
gate_passed   = +400 / -400
lookat_next   = +0.1
```

## Why warm-start instead of starting from zero

The existing checkpoint already knows how to race at approximately:

```text
36-37 gates / 20 s
mean speed ~13 m/s
max speed ~15-16 m/s
```

The immediate objective is not to relearn racing. It is to modify the existing
racing behavior so the gate remains observable.

Therefore use the successful checkpoint as the initialization and reset the
continuation learning rate to the original 1e-4.

## Validation

Sync and test first:

```bash
cd ~/isaac_projects/isaac_drone_racer

git fetch ssybh2 --prune
git switch feature/racing-vision-retraining
git pull --ff-only ssybh2 feature/racing-vision-retraining

./.conda-env/bin/python -m pytest -q   tests/rl/test_swift_ctbr_static.py   tests/perception/test_racing_vision_static.py

./.conda-env/bin/python -m compileall -q   tasks/drone_racer   scripts/rl   scripts/perception   perception
```

## Training

```bash
export BEST="logs/skrl/swift_ctbr_gt_racing/2026-09-21_20-47-02_ppo_torch_easy7_4096env_roll24_256x3_gt/checkpoints/best_agent.pt"

./.conda-env/bin/python   scripts/rl/train.py   --task Isaac-Drone-Racer-Swift-CTBR-GT-PerceptionAware-v0   --checkpoint "$BEST"   --post_load_learning_rate 1.0e-4   --num_envs 4096   --seed 1   --device cuda:0   --headless
```

Logs are written under:

```text
logs/skrl/swift_ctbr_gt_perception/
```

The PPO budget remains 50,000 vector steps, exactly matching the successful GT
racing training profile.

## First evaluation after training

Evaluate the best perception-aware checkpoint with the normal racing evaluator:

```bash
export PA_BEST="<new best_agent.pt>"

./.conda-env/bin/python   scripts/rl/evaluate_swift_ctbr_policy0.py   --task Isaac-Drone-Racer-Swift-CTBR-GT-PerceptionAware-v0   --checkpoint "$PA_BEST"   --episodes 20   --seed 1   --output-dir artifacts/swift_ctbr/gt_perception_aware_eval   --device cuda:0   --headless
```

The first requirement is to retain strong racing performance.

Then collect a small camera-enabled racing dataset with the new policy and
compare the GT visibility histogram against the old policy:

```text
OLD policy:
active gate >=2 GT-visible corners ~9% in the current smoke dataset
```

The perception-aware policy should materially increase that fraction without
destroying 30-37+ gate racing.

## Acceptance target

Do not judge the new policy only by reward.

Require both:

```text
racing:
  still repeatedly reaches 30-37+ gates / 20 s

observability:
  active-gate >=2 GT-visible-corner fraction rises substantially
  from the current ~9% baseline
```

After that, use the new policy to generate the main Racing Vision Dataset and
then retrain the detector/visibility models on the harder image distribution.
