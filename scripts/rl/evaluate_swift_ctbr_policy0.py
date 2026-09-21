"""Deterministic evaluator for Swift CTBR policy-0 checkpoints."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--task",
    default="Isaac-Drone-Racer-Swift-CTBR-Train-v0",
)
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--episodes", type=int, default=20)
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--output-dir", type=Path, required=True)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from isaaclab_rl.skrl import SkrlVecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils import load_cfg_from_registry, parse_env_cfg  # noqa: E402
from skrl.utils.runner.torch import Runner  # noqa: E402

import tasks  # noqa: F401,E402


def _termination_cause(raw_env) -> str:
    manager = raw_env.termination_manager
    names = list(manager.active_terms)
    row = manager._last_episode_dones[0].detach().cpu().tolist()
    fired = {name for name, value in zip(names, row) if bool(value)}

    if "collision" in fired:
        return "collision"
    if "flyaway" in fired:
        return "flyaway"
    if "time_out" in fired:
        return "timeout"
    return "other"


def main() -> None:
    if args_cli.episodes <= 0:
        raise ValueError("--episodes must be positive")

    cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=1,
    )
    cfg.scene.num_envs = 1
    cfg.seed = int(args_cli.seed)

    env = gym.make(args_cli.task, cfg=cfg)
    raw_env = env.unwrapped
    wrapped = SkrlVecEnvWrapper(env, ml_framework="torch")

    agent_cfg = load_cfg_from_registry(args_cli.task, "skrl_cfg_entry_point")
    agent_cfg["seed"] = int(args_cli.seed)
    agent_cfg["trainer"]["close_environment_at_exit"] = False
    agent_cfg["agent"]["experiment"]["write_interval"] = 0
    agent_cfg["agent"]["experiment"]["checkpoint_interval"] = 0

    runner = Runner(wrapped, agent_cfg)
    checkpoint = str(Path(args_cli.checkpoint).expanduser().resolve())
    print(f"[swift-eval] loading checkpoint: {checkpoint}", flush=True)
    runner.agent.load(checkpoint)
    runner.agent.set_running_mode("eval")

    out_dir = args_cli.output_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    obs, _ = wrapped.reset()
    episode = 0
    ep_steps = 0
    ep_return = 0.0
    ep_gate_passes = 0
    action_abs_sum = 0.0
    action_count = 0
    action_saturation_count = 0
    action_abs_max = 0.0

    command = raw_env.command_manager.get_term("target")
    prev_gate_idx = int(command.next_gate_idx[0].item())

    try:
        while episode < int(args_cli.episodes):
            with torch.inference_mode():
                outputs = runner.agent.act(obs, timestep=0, timesteps=0)
                actions = outputs[-1].get("mean_actions", outputs[0])

            action_abs = actions.abs()
            action_abs_sum += float(action_abs.sum().item())
            action_count += int(action_abs.numel())
            action_saturation_count += int(
                (action_abs > 0.95).sum().item()
            )
            action_abs_max = max(
                action_abs_max,
                float(action_abs.max().item()),
            )

            obs, reward, terminated, truncated, _ = wrapped.step(actions)
            ep_steps += 1
            ep_return += float(reward.reshape(-1)[0].item())

            done = bool(terminated.reshape(-1)[0].item()) or bool(
                truncated.reshape(-1)[0].item()
            )

            current_gate_idx = int(command.next_gate_idx[0].item())
            if not done and current_gate_idx != prev_gate_idx:
                delta = (current_gate_idx - prev_gate_idx) % int(
                    command.num_gates
                )
                ep_gate_passes += int(delta)
            prev_gate_idx = current_gate_idx

            if not done:
                continue

            cause = _termination_cause(raw_env)
            duration_s = ep_steps * float(raw_env.step_dt)

            row = {
                "episode": episode + 1,
                "gates_passed": ep_gate_passes,
                "steps": ep_steps,
                "duration_s": duration_s,
                "return": ep_return,
                "termination": cause,
                "action_abs_mean": (
                    action_abs_sum / max(action_count, 1)
                ),
                "action_abs_max": action_abs_max,
                "action_saturation_fraction": (
                    action_saturation_count / max(action_count, 1)
                ),
            }
            rows.append(row)

            print(
                "[swift-eval] "
                f"ep={episode + 1:02d}/{args_cli.episodes} "
                f"gates={ep_gate_passes} steps={ep_steps} "
                f"cause={cause} return={ep_return:.3f} "
                f"|a|max={action_abs_max:.3f} "
                f"sat={row['action_saturation_fraction']:.3f}",
                flush=True,
            )

            episode += 1
            ep_steps = 0
            ep_return = 0.0
            ep_gate_passes = 0
            action_abs_sum = 0.0
            action_count = 0
            action_saturation_count = 0
            action_abs_max = 0.0
            prev_gate_idx = int(command.next_gate_idx[0].item())

        gates = np.asarray(
            [row["gates_passed"] for row in rows],
            dtype=np.int32,
        )
        returns = np.asarray(
            [row["return"] for row in rows],
            dtype=np.float64,
        )
        durations = np.asarray(
            [row["duration_s"] for row in rows],
            dtype=np.float64,
        )

        histogram = {
            str(int(k)): int(v)
            for k, v in zip(*np.unique(gates, return_counts=True))
        }
        terminations = {}
        for row in rows:
            key = row["termination"]
            terminations[key] = terminations.get(key, 0) + 1

        summary = {
            "checkpoint": checkpoint,
            "episodes": int(args_cli.episodes),
            "num_gates": int(command.num_gates),
            "gates_passed": {
                "mean": float(gates.mean()),
                "median": float(np.median(gates)),
                "min": int(gates.min()),
                "max": int(gates.max()),
                "histogram": histogram,
            },
            "full_lap_completion_rate": float(
                np.mean(gates >= int(command.num_gates))
            ),
            "termination_counts": terminations,
            "duration_s": {
                "mean": float(durations.mean()),
                "median": float(np.median(durations)),
            },
            "return": {
                "mean": float(returns.mean()),
                "median": float(np.median(returns)),
            },
            "action": {
                "abs_mean": float(
                    np.mean([r["action_abs_mean"] for r in rows])
                ),
                "abs_max": float(
                    np.max([r["action_abs_max"] for r in rows])
                ),
                "saturation_fraction_mean": float(
                    np.mean(
                        [r["action_saturation_fraction"] for r in rows]
                    )
                ),
            },
        }

        with (out_dir / "episodes.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

        (out_dir / "summary.json").write_text(
            json.dumps(summary, indent=2) + "\n"
        )

        print("=" * 96)
        print("SWIFT CTBR POLICY-0 EVALUATION SUMMARY")
        print("=" * 96)
        print(json.dumps(summary, indent=2))
        print(f"[swift-eval] wrote: {out_dir / 'episodes.csv'}")
        print(f"[swift-eval] wrote: {out_dir / 'summary.json'}")
    finally:
        wrapped.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
