"""Deterministic hover and body-rate smoke test for the Swift CTBR controller."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--task",
    default="Isaac-Drone-Racer-Swift-CTBR-Control-v0",
)
parser.add_argument("--hover_s", type=float, default=2.0)
parser.add_argument("--warmup_s", type=float, default=0.5)
parser.add_argument("--rate_step_s", type=float, default=0.4)
parser.add_argument("--settle_s", type=float, default=0.6)
parser.add_argument("--roll_rate_radps", type=float, default=2.0)
parser.add_argument("--pitch_rate_radps", type=float, default=2.0)
parser.add_argument("--yaw_rate_radps", type=float, default=1.0)
parser.add_argument("--output", type=Path, default=None)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

import tasks  # noqa: F401,E402


def _steps(seconds: float, dt: float) -> int:
    return max(1, int(round(float(seconds) / float(dt))))


def _step(env, action):
    out = env.step(action)
    terminated = bool(out[2].reshape(-1)[0].item())
    truncated = bool(out[3].reshape(-1)[0].item())
    return terminated or truncated


def main() -> None:
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=1)
    env_cfg.scene.num_envs = 1
    env = gym.make(args_cli.task, cfg=env_cfg)
    raw_env = env.unwrapped
    robot = raw_env.scene["robot"]
    action_term = raw_env.action_manager.get_term("control_action")
    dt = float(raw_env.step_dt)
    device = robot.device

    summary: dict[str, object] = {
        "task": args_cli.task,
        "control_dt_s": dt,
        "max_collective_accel_mps2": float(action_term._max_collective_accel),
        "body_rate_max_radps": list(action_term.cfg.body_rate_max_radps),
    }

    try:
        env.reset()
        hover_action = torch.zeros((1, 4), device=device)
        z0 = float(robot.data.root_pos_w[0, 2].item())
        z_values = []
        vz_values = []
        rate_values = []
        terminated_early = False
        for _ in range(_steps(args_cli.hover_s, dt)):
            terminated_early = _step(env, hover_action) or terminated_early
            z_values.append(float(robot.data.root_pos_w[0, 2].item()))
            vz_values.append(float(robot.data.root_lin_vel_w[0, 2].item()))
            rate_values.append(
                robot.data.root_ang_vel_b[0].detach().cpu().numpy().copy()
            )

        rate_array = np.asarray(rate_values, dtype=np.float64)
        summary["hover"] = {
            "start_z_m": z0,
            "final_z_m": z_values[-1],
            "z_drift_m": float(z_values[-1] - z0),
            "z_peak_to_peak_m": float(max(z_values) - min(z_values)),
            "vertical_velocity_rms_mps": float(
                np.sqrt(np.mean(np.square(vz_values)))
            ),
            "body_rate_rms_radps": float(np.sqrt(np.mean(rate_array**2))),
            "terminated_early": bool(terminated_early),
        }

        targets = [
            ("roll", 0, float(args_cli.roll_rate_radps)),
            ("pitch", 1, float(args_cli.pitch_rate_radps)),
            ("yaw", 2, float(args_cli.yaw_rate_radps)),
        ]
        rate_results = {}
        max_rates = tuple(float(x) for x in action_term.cfg.body_rate_max_radps)

        for name, axis, target in targets:
            env.reset()
            neutral = torch.zeros((1, 4), device=device)
            for _ in range(_steps(args_cli.warmup_s, dt)):
                _step(env, neutral)

            action = neutral.clone()
            action[0, axis + 1] = float(target / max_rates[axis])
            measured = []
            terminated = False
            for _ in range(_steps(args_cli.rate_step_s, dt)):
                terminated = _step(env, action) or terminated
                measured.append(
                    float(robot.data.root_ang_vel_b[0, axis].item())
                )

            measured_array = np.asarray(measured, dtype=np.float64)
            tail = measured_array[max(0, len(measured_array) // 2):]

            settle = []
            for _ in range(_steps(args_cli.settle_s, dt)):
                terminated = _step(env, neutral) or terminated
                settle.append(
                    float(robot.data.root_ang_vel_b[0, axis].item())
                )

            rate_results[name] = {
                "target_radps": target,
                "command_normalized": float(action[0, axis + 1].item()),
                "tail_mean_radps": float(np.mean(tail)),
                "tracking_rmse_radps": float(
                    np.sqrt(np.mean(np.square(measured_array - target)))
                ),
                "peak_abs_radps": float(np.max(np.abs(measured_array))),
                "settled_abs_rate_radps": float(abs(settle[-1])),
                "terminated_early": bool(terminated),
            }

        summary["rate_steps"] = rate_results

        print("=" * 96)
        print("SWIFT CTBR CONTROL SMOKE")
        print("=" * 96)
        print(json.dumps(summary, indent=2))

        if args_cli.output is not None:
            output = args_cli.output.expanduser().resolve()
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps(summary, indent=2) + "\n")
            print(f"[ctbr] wrote: {output}")
    finally:
        env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
