"""Collect real on-policy transfer data from a legacy hard-clipped policy.

The legacy actor is instantiated with its original semantics:
    raw Gaussian mean -> environment ControlAction clamp[-1, 1]

The resulting dataset contains the exact standardized policy observations,
legacy raw means and environment-executed deterministic motor commands needed
for behavior-preserving transfer into a bounded tanh actor.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--task",
    default="Isaac-Drone-Racer-Learned-Inertial-RL-v0",
)
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--episodes", type=int, default=20)
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--output", type=Path, required=True)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

args_cli.enable_cameras = True
if args_cli.episodes <= 0:
    parser.error("--episodes must be positive")

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from skrl.utils.runner.torch import Runner  # noqa: E402
from isaaclab_rl.skrl import SkrlVecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils import load_cfg_from_registry, parse_env_cfg  # noqa: E402

import tasks  # noqa: F401,E402


def _legacy_agent_cfg(task: str, seed: int) -> dict:
    cfg = load_cfg_from_registry(task, "skrl_cfg_entry_point")
    policy_cfg = cfg["models"]["policy"]
    policy_cfg["clip_actions"] = False
    policy_cfg["clip_mean_actions"] = False
    policy_cfg["clip_log_std"] = True
    policy_cfg["min_log_std"] = -20.0
    policy_cfg["max_log_std"] = 2.0
    policy_cfg["initial_log_std"] = 0.0
    policy_cfg["output"] = "ACTIONS"

    cfg["seed"] = int(seed)
    cfg["trainer"]["close_environment_at_exit"] = False
    cfg["agent"]["experiment"]["write_interval"] = 0
    cfg["agent"]["experiment"]["checkpoint_interval"] = 0
    return cfg


def main() -> None:
    torch.manual_seed(int(args_cli.seed))
    np.random.seed(int(args_cli.seed))

    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=1,
    )
    env_cfg.scene.num_envs = 1
    env_cfg.seed = int(args_cli.seed)

    env = gym.make(args_cli.task, cfg=env_cfg)
    raw_env = env.unwrapped
    wrapped = SkrlVecEnvWrapper(env, ml_framework="torch")

    runner = Runner(
        wrapped,
        _legacy_agent_cfg(args_cli.task, args_cli.seed),
    )

    checkpoint = str(Path(args_cli.checkpoint).expanduser().resolve())
    print(f"[collect] loading legacy checkpoint: {checkpoint}", flush=True)
    runner.agent.load(checkpoint)
    runner.agent.set_running_mode("eval")

    observations: list[np.ndarray] = []
    standardized: list[np.ndarray] = []
    raw_means: list[np.ndarray] = []
    executed_actions: list[np.ndarray] = []
    episode_ids: list[int] = []
    episode_steps: list[int] = []
    truth_gate_counts: list[int] = []
    mission_gate_counts: list[int] = []

    obs, _ = wrapped.reset()
    episode = 0
    step_in_episode = 0

    try:
        while episode < int(args_cli.episodes):
            with torch.inference_mode():
                # Match PPO.act exactly: the checkpoint's RunningStandardScaler
                # transforms the 20D actor observation before the policy trunk.
                std_obs = runner.agent._state_preprocessor(obs)
                mean, _ = runner.agent.policy.compute(
                    {"states": std_obs},
                    role="policy",
                )
                executed = mean.clamp(-1.0, 1.0)

            observations.append(
                obs.detach().cpu().numpy().reshape(-1).astype(np.float32)
            )
            standardized.append(
                std_obs.detach().cpu().numpy().reshape(-1).astype(np.float32)
            )
            raw_means.append(
                mean.detach().cpu().numpy().reshape(-1).astype(np.float32)
            )
            executed_actions.append(
                executed.detach().cpu().numpy().reshape(-1).astype(np.float32)
            )
            episode_ids.append(episode)
            episode_steps.append(step_in_episode)

            obs, _, terminated, truncated, _ = wrapped.step(mean)
            step_in_episode += 1

            done = bool(terminated.reshape(-1)[0].item()) or bool(
                truncated.reshape(-1)[0].item()
            )
            if not done:
                continue

            terminal = raw_env.last_episode_diagnostic
            if terminal is None:
                raise RuntimeError(
                    "environment terminated without last_episode_diagnostic"
                )

            truth_gate_counts.append(int(terminal["truth_gates_passed"]))
            mission_gate_counts.append(int(terminal["mission_gates_passed"]))

            cause = (
                "collision"
                if terminal["collision"]
                else "flyaway"
                if terminal["flyaway"]
                else "timeout"
                if terminal["time_out"]
                else "other"
            )
            print(
                "[collect] "
                f"ep={episode + 1:02d}/{args_cli.episodes} "
                f"steps={step_in_episode} "
                f"truth_gates={terminal['truth_gates_passed']} "
                f"mission_gates={terminal['mission_gates_passed']} "
                f"cause={cause}",
                flush=True,
            )
            episode += 1
            step_in_episode = 0

        output = args_cli.output.expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)

        np.savez_compressed(
            output,
            observation=np.stack(observations, axis=0),
            standardized_observation=np.stack(standardized, axis=0),
            legacy_raw_mean=np.stack(raw_means, axis=0),
            legacy_executed_action=np.stack(executed_actions, axis=0),
            episode_id=np.asarray(episode_ids, dtype=np.int32),
            episode_step=np.asarray(episode_steps, dtype=np.int32),
            truth_gates_per_episode=np.asarray(truth_gate_counts, dtype=np.int32),
            mission_gates_per_episode=np.asarray(
                mission_gate_counts, dtype=np.int32
            ),
            source_checkpoint=np.asarray(checkpoint),
            action_semantics=np.asarray(
                "legacy_raw_mean_environment_hard_clip"
            ),
        )

        raw = np.stack(raw_means, axis=0)
        executed = np.stack(executed_actions, axis=0)
        print("=" * 96)
        print("LEGACY ON-POLICY TRANSFER DATASET")
        print("=" * 96)
        print(f"samples                         : {raw.shape[0]}")
        print(f"episodes                        : {args_cli.episodes}")
        print(
            "truth gates / episode           : "
            f"{truth_gate_counts}"
        )
        print(f"legacy raw RMS                  : {np.sqrt(np.mean(raw**2)):.6f}")
        print(
            "executed saturation fraction    : "
            f"{np.mean(np.abs(executed) > 0.999):.6f}"
        )
        print(f"[collect] wrote: {output}")
    finally:
        wrapped.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
