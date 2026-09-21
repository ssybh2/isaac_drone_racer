"""Runtime smoke for the vectorized Swift CTBR policy-0 training task."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--task",
    default="Isaac-Drone-Racer-Swift-CTBR-Train-v0",
)
parser.add_argument("--num_envs", type=int, default=8)
parser.add_argument("--steps", type=int, default=16)
parser.add_argument("--output-json", type=Path, default=None)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

import tasks  # noqa: F401,E402


def _policy_tensor(observation):
    if isinstance(observation, dict):
        if "policy" not in observation:
            raise RuntimeError(
                f"observation dict has no 'policy' group: {list(observation)}"
            )
        return observation["policy"]
    return observation


def main() -> None:
    cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=int(args_cli.num_envs),
    )
    cfg.scene.num_envs = int(args_cli.num_envs)

    env = gym.make(args_cli.task, cfg=cfg)
    raw_env = env.unwrapped

    try:
        observation, _ = env.reset()
        policy = _policy_tensor(observation)

        action_space = raw_env.single_action_space
        finite_action_space = bool(
            torch.isfinite(
                torch.as_tensor(action_space.low, dtype=torch.float32)
            ).all()
            and torch.isfinite(
                torch.as_tensor(action_space.high, dtype=torch.float32)
            ).all()
        )

        rewards = []
        done_count = 0
        zero_action = torch.zeros(
            (raw_env.num_envs, 4),
            dtype=torch.float32,
            device=raw_env.device,
        )

        for _ in range(int(args_cli.steps)):
            observation, reward, terminated, truncated, _ = env.step(
                zero_action
            )
            policy = _policy_tensor(observation)
            rewards.append(reward.detach().reshape(-1))
            done_count += int(
                torch.logical_or(terminated, truncated).sum().item()
            )

        reward_tensor = torch.cat(rewards) if rewards else torch.zeros(1)
        action_term = raw_env.action_manager.get_term("control_action")

        summary = {
            "task": args_cli.task,
            "num_envs": int(raw_env.num_envs),
            "policy_observation_shape": list(policy.shape),
            "expected_policy_dim": 31,
            "action_dim": int(raw_env.action_manager.total_action_dim),
            "single_action_space_low": action_space.low.tolist(),
            "single_action_space_high": action_space.high.tolist(),
            "finite_action_space": finite_action_space,
            "reward_terms": list(raw_env.reward_manager.active_terms),
            "reward_mean": float(reward_tensor.mean().item()),
            "reward_min": float(reward_tensor.min().item()),
            "reward_max": float(reward_tensor.max().item()),
            "reward_all_finite": bool(torch.isfinite(reward_tensor).all()),
            "done_events_during_smoke": int(done_count),
            "zero_action_collective_accel_mps2": float(
                action_term.ctbr_command[:, 0].mean().item()
            ),
        }

        if policy.shape != (raw_env.num_envs, 31):
            raise RuntimeError(
                "Swift policy observation contract mismatch: "
                f"expected {(raw_env.num_envs, 31)}, got {tuple(policy.shape)}"
            )
        if raw_env.action_manager.total_action_dim != 4:
            raise RuntimeError("Swift CTBR action dimension is not 4")
        if not finite_action_space:
            raise RuntimeError("Swift CTBR Gym action space is not finite")
        if not bool(torch.isfinite(policy).all()):
            raise RuntimeError("Swift policy observation contains NaN/Inf")
        if not summary["reward_all_finite"]:
            raise RuntimeError("Swift reward contains NaN/Inf")

        print("=" * 96)
        print("SWIFT CTBR POLICY-0 TASK SMOKE")
        print("=" * 96)
        print(json.dumps(summary, indent=2))

        if args_cli.output_json is not None:
            output = args_cli.output_json.expanduser().resolve()
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps(summary, indent=2) + "\n")
            print(f"[swift-policy0-smoke] wrote: {output}")
    finally:
        env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
