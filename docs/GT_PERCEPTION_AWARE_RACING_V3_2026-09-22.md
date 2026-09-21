# GT Perception-Aware Racing V3 — 2026-09-22

V2 preserved and improved racing speed but did not materially improve active-gate camera visibility.

Measured five-episode camera smoke results:

```text
OLD:
  >=2 visible corners = 223 / 2495 = 8.94%

V1:
  >=2 visible corners = 172 / 2495 = 6.89%

V2:
  >=2 visible corners = 239 / 2495 = 9.58%
  accepted episodes  = 39-40 gates / 20 s
  mean speed          ~14.6 m/s
  max speed           ~17.5 m/s
```

V2 therefore improved visibility by only +0.64 percentage points over OLD while learning a faster racing policy.

## Why V2 was weak

When the gate is completely outside the image, the hard `<2 corners` penalty is nearly constant.
The image-center exponential also becomes close to zero far outside the FOV.

Therefore PPO receives little directional information about *which way* the body/camera should rotate to recover the gate.

## V3

V3 adds a dense squared camera-to-next-gate angle penalty:

```text
camera_angle_l2 weight = -8.0
```

This term is active even when the gate is completely outside the image.

V3 also retains a lighter image-space observability penalty:

```text
camera_observability weight = +2.0
output_bias                 = -1.0
insufficient_visible_penalty= 0.50
```

All PPO specifications remain identical to the successful GT racing baseline.

## Train

Always warm-start from the original successful GT racing checkpoint, not V1/V2:

```bash
cd ~/isaac_projects/isaac_drone_racer

export BEST="logs/skrl/swift_ctbr_gt_racing/2026-09-21_20-47-02_ppo_torch_easy7_4096env_roll24_256x3_gt/checkpoints/best_agent.pt"

./.conda-env/bin/python   scripts/rl/train.py   --task Isaac-Drone-Racer-Swift-CTBR-GT-PerceptionAwareV3-v0   --checkpoint "$BEST"   --post_load_learning_rate 1.0e-4   --num_envs 4096   --seed 1   --device cuda:0   --headless
```

TensorBoard:

```bash
./.conda-env/bin/tensorboard   --logdir logs/skrl/swift_ctbr_gt_perception_v3   --host 0.0.0.0   --port 6006
```

Acceptance still requires both strong racing and a material visibility improvement over the 8.94% OLD baseline.
