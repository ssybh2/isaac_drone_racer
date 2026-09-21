# GT Perception-Aware Racing V2 — 2026-09-21

The first perception-aware run did **not** improve camera observability.

Measured on five successful 20 s episodes:

```text
old GT racing policy:
  mean gates        ~36.6
  active gate >=2 GT-visible corners ~8.9%

perception-aware V1:
  mean gates        ~39.2
  active gate >=2 GT-visible corners ~6.9%
```

V1 therefore learned to race faster but did not learn the intended camera-visibility behavior.

## V2 change

All PPO/racing specifications remain identical to the successful GT baseline.

Only the perception term changes.

V2 uses a **penalty-only** objective:

- perfect visibility approaches zero penalty;
- poor centering / poor border margin is penalized;
- fewer than two visible gate corners receives an additional explicit penalty;
- there is no positive perception reward that can be accumulated by hovering.

Configuration:

```text
camera_observability weight = 5.0
output_bias                 = -1.0
insufficient_visible_penalty= 0.50

component weights:
center       0.20
coverage     0.25
margin       0.15
>=2 usable   0.40
```

New task:

```text
Isaac-Drone-Racer-Swift-CTBR-GT-PerceptionAwareV2-v0
```

Train from the **original successful GT racing checkpoint**, not from V1, so the comparison stays clean.

```bash
export BEST="logs/skrl/swift_ctbr_gt_racing/2026-09-21_20-47-02_ppo_torch_easy7_4096env_roll24_256x3_gt/checkpoints/best_agent.pt"

./.conda-env/bin/python   scripts/rl/train.py   --task Isaac-Drone-Racer-Swift-CTBR-GT-PerceptionAwareV2-v0   --checkpoint "$BEST"   --post_load_learning_rate 1.0e-4   --num_envs 4096   --seed 1   --device cuda:0   --headless
```

Logs:

```text
logs/skrl/swift_ctbr_gt_perception_v2/
```

Acceptance requires **both**:

```text
racing remains >=30 gates and preferably near 35-40 gates
active-gate >=2 GT-visible-corner fraction rises materially above ~8.9%
```
