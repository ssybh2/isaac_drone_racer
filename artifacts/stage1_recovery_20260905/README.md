# Stage 1.5 Fake VIO recovery results

Training completed September 5, 2026 at 12:08 Asia/Shanghai; both evaluation
sweeps completed at 12:31. Results were verified September 10. The preferred
checkpoint for clean/mild/nominal operation is `checkpoints/best_agent.pt`.
Stress performance remains poor; these results do not establish real VIO or
camera-based autonomy. Gate-relative guidance remains oracle ground truth.

| Profile | Before recovery | Best agent | Last agent (50,400 steps) |
| --- | ---: | ---: | ---: |
| clean | 172/384 (44.8%) | 313/384 (81.5%) | 282/384 (73.4%) |
| mild | 248/384 (64.6%) | 320/384 (83.3%) | 294/384 (76.6%) |
| nominal | 302/384 (78.6%) | 327/384 (85.2%) | 307/384 (79.9%) |
| stress | 21/384 (5.5%) | 23/384 (6.0%) | 35/384 (9.1%) |

Success means the first seven gates passed from a randomized training start,
using deterministic mean actions and no additional push disturbance. All rows
use seeds 11/29/47, 128 episodes per seed, 64 environments, and the corrected
`fixed_per_environment_quotas_v2` protocol. Results from the earlier global
first-finished episode selection are not directly comparable.

Contents:

- `checkpoints/best_agent.pt`: preferred recovered policy, selected during training.
- `checkpoints/agent_50400.pt`: final recovered policy.
- `checkpoints/source_agent_50010.pt`: original mixed policy used to start recovery.
- `evaluations/before/`, `evaluations/best/`, `evaluations/final/`: original
  summary JSON and episode CSV for each model; 1,536 episodes per model.
- `params/`: original saved environment, agent, and continuation configuration.
- `SHA256SUMS`: integrity hashes for the original copied artifacts.

Checkpoints retain the policy, value function, optimizer, state scaler and value
scaler. The repository's existing Git LFS rules apply to `.pt` files. Before using
them, from the repository root:

```bash
git lfs pull
cd artifacts/stage1_recovery_20260905
sha256sum -c SHA256SUMS
cd ../..
```

After installing this repository's Isaac Lab / Isaac Sim Python environment,
evaluate the best policy from the repository root:

```bash
OMNI_KIT_ACCEPT_EULA=YES PYTHONPATH="$PWD" python scripts/rl/evaluate_stage1.py \
  --headless --num_envs 64 --episodes_per_case 128 --seeds 11,29,47 \
  --profiles clean,mild,nominal,stress \
  --checkpoint artifacts/stage1_recovery_20260905/checkpoints/best_agent.pt
```

To repeat the recovery training from the same source checkpoint:

```bash
OMNI_KIT_ACCEPT_EULA=YES PYTHONPATH="$PWD" python scripts/rl/train.py \
  --task Isaac-Drone-Racer-Stage1-v0 --headless --num_envs 4096 \
  --fake_sensor_profile mixed --seed 1 \
  --checkpoint artifacts/stage1_recovery_20260905/checkpoints/source_agent_50010.pt \
  --post_load_learning_rate 0.0001 --entropy_loss_scale 0.001 \
  --adaptive_lr_max 0.0003 --max_iterations 2100 \
  agent.agent.experiment.directory=stage1_recovery_reproduction
```

Each reset selects jointly clean VIO/IMU with probability 25%; other episodes
retain the broad mixed prior. Stress is held out from gradient training. Internal
source clocks use float64; measurements and the 20-dimensional policy interface
remain float32. The run contains 50,400 vector steps, approximately 206 million
transitions across 4,096 environments. Exact numerical reproduction may depend
on simulator, GPU and library versions.

Original JSON/CSV/YAML files intentionally retain the absolute paths and base
Git revision from the development machine. Their checkpoint SHA-256 values map
to the copies here, even where the basename differs. The base revision
`150d107` predates the then-uncommitted recovery fixes; use the source committed
alongside this directory to reproduce recovery, not that base revision alone.
The original rollout logs, intermediate checkpoints and machine-specific
service launcher remain local. See [the audit](../../docs/2026-09-05-fake-vio-audit.md)
for the diagnosis and validation history.
