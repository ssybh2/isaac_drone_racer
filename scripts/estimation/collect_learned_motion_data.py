"""Collect supervised gyro+thrust -> relative-displacement training traces in Isaac Lab."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--steps", type=int, default=5000)
parser.add_argument(
    "--profile",
    choices=("hover", "translate_x", "lissajous", "circle"),
    default="lissajous",
)
parser.add_argument("--amplitude_m", type=float, default=0.8)
parser.add_argument("--frequency_hz", type=float, default=0.10)
parser.add_argument("--translation_m", type=float, default=2.0)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--task", default="Isaac-Drone-Racer-Swift-OpenVINS-v0")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import numpy as np
import torch

from isaaclab_tasks.utils import parse_env_cfg

import tasks  # noqa: F401


def _wrap_angle(angle: float) -> float:
    return float(np.arctan2(np.sin(angle), np.cos(angle)))


def _quat_to_yaw(q) -> float:
    w, x, y, z = [float(v) for v in q]
    return float(np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))


def _target_xy(initial_xy: torch.Tensor, t_s: float) -> torch.Tensor:
    target = initial_xy.clone()
    amp = float(args_cli.amplitude_m)
    omega = 2.0 * np.pi * float(args_cli.frequency_hz)
    if args_cli.profile == "hover":
        return target
    if args_cli.profile == "translate_x":
        alpha = min(1.0, max(0.0, t_s / max(args_cli.steps * 0.01 * 0.7, 1.0e-6)))
        target[0] += float(args_cli.translation_m) * alpha
        return target
    if args_cli.profile == "circle":
        target[0] += amp * float(np.cos(omega * t_s) - 1.0)
        target[1] += amp * float(np.sin(omega * t_s))
        return target
    target[0] += amp * float(np.sin(omega * t_s))
    target[1] += 0.6 * amp * float(np.sin(2.0 * omega * t_s + 0.4))
    return target


def _controller_action(raw_env, initial_xy, target_height_m, target_yaw_rad, t_s):
    robot = raw_env.scene["robot"]
    target_xy = _target_xy(initial_xy, t_s)
    height_error = target_height_m - float(robot.data.root_pos_w[0, 2])
    vertical_velocity = float(robot.data.root_lin_vel_w[0, 2])
    common = 0.18 * height_error - 0.09 * vertical_velocity

    xy_error = target_xy - robot.data.root_pos_w[0, :2]
    desired_accel_xy = xy_error - 1.5 * robot.data.root_lin_vel_w[0, :2]
    desired_roll = float(torch.clamp(-desired_accel_xy[1] / 9.81, -0.16, 0.16))
    desired_pitch = float(torch.clamp(desired_accel_xy[0] / 9.81, -0.16, 0.16))
    q = robot.data.root_quat_w[0]
    rates = robot.data.root_ang_vel_b[0]
    roll = float(torch.atan2(2.0 * (q[0] * q[1] + q[2] * q[3]), 1.0 - 2.0 * (q[1] ** 2 + q[2] ** 2)))
    pitch = float(torch.asin(torch.clamp(2.0 * (q[0] * q[2] - q[3] * q[1]), -1.0, 1.0)))
    yaw = _quat_to_yaw(q)
    roll_u = -0.08 * (roll - desired_roll) - 0.015 * float(rates[0])
    pitch_u = -0.08 * (pitch - desired_pitch) - 0.015 * float(rates[1])
    yaw_error = _wrap_angle(yaw - target_yaw_rad)
    yaw_u = float(np.clip(-0.06 * yaw_error - 0.03 * float(rates[2]), -0.08, 0.08))

    return torch.tensor(
        [
            common + roll_u - pitch_u + yaw_u,
            common - roll_u - pitch_u - yaw_u,
            common - roll_u + pitch_u + yaw_u,
            common + roll_u + pitch_u - yaw_u,
        ],
        dtype=torch.float32,
        device=raw_env.device,
    ).clamp(-1.0, 1.0)


def main() -> None:
    if args_cli.steps < 100:
        raise ValueError("--steps must be at least 100")
    output = args_cli.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=1)
    env_cfg.swift_detector_checkpoint = None
    env_cfg.swift_visibility_checkpoint = None
    env_cfg.swift_use_oracle_gate_index = False
    env_cfg.swift_rejection_dump_dir = None
    if hasattr(env_cfg, "learned_motion_checkpoint"):
        env_cfg.learned_motion_checkpoint = None
    env_cfg.terminations.collision = None
    env_cfg.terminations.flyaway = None
    env_cfg.episode_length_s = max(float(env_cfg.episode_length_s), args_cli.steps * 0.01 + 5.0)

    env = gym.make(args_cli.task, cfg=env_cfg)
    raw_env = env.unwrapped
    env.reset()
    robot = raw_env.scene["robot"]
    initial_xy = robot.data.root_pos_w[0, :2].clone()
    target_height_m = float(robot.data.root_pos_w[0, 2])
    target_yaw_rad = _quat_to_yaw(robot.data.root_quat_w[0])

    fields = [
        "step", "t_s", "profile",
        "truth_px", "truth_py", "truth_pz",
        "truth_vx", "truth_vy", "truth_vz",
        "truth_qw", "truth_qx", "truth_qy", "truth_qz",
        "imu_gx", "imu_gy", "imu_gz",
        "imu_ax", "imu_ay", "imu_az",
        "collective_thrust", "thrust_b_x", "thrust_b_y", "thrust_b_z",
        "action_m1", "action_m2", "action_m3", "action_m4",
    ]

    completed = 0
    try:
        with output.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for step in range(args_cli.steps):
                if not simulation_app.is_running():
                    break
                t_before = float(raw_env._timestamp_s())
                action = _controller_action(
                    raw_env, initial_xy, target_height_m, target_yaw_rad, t_before
                )
                _, _, terminated, truncated, _ = env.step(action.unsqueeze(0))
                completed = step + 1

                truth = raw_env._truth_vio_state()
                imu = raw_env.scene["imu"]
                gyro_b = np.asarray(imu.data.ang_vel_b[0].detach().cpu(), dtype=np.float64)
                accel_b = np.asarray(imu.data.lin_acc_b[0].detach().cpu(), dtype=np.float64)
                control_term = raw_env.action_manager.get_term("control_action")
                processed = np.asarray(
                    control_term.processed_actions[0].detach().cpu(), dtype=np.float64
                ).reshape(-1)
                collective = float(processed[0])
                thrust_b = np.array([0.0, 0.0, collective], dtype=np.float64)
                action_np = np.asarray(action.detach().cpu(), dtype=np.float64)

                writer.writerow(
                    {
                        "step": step,
                        "t_s": float(raw_env._timestamp_s()),
                        "profile": args_cli.profile,
                        "truth_px": float(truth.position_w_b[0]),
                        "truth_py": float(truth.position_w_b[1]),
                        "truth_pz": float(truth.position_w_b[2]),
                        "truth_vx": float(truth.linear_velocity_w_b[0]),
                        "truth_vy": float(truth.linear_velocity_w_b[1]),
                        "truth_vz": float(truth.linear_velocity_w_b[2]),
                        "truth_qw": float(truth.orientation_w_b_wxyz[0]),
                        "truth_qx": float(truth.orientation_w_b_wxyz[1]),
                        "truth_qy": float(truth.orientation_w_b_wxyz[2]),
                        "truth_qz": float(truth.orientation_w_b_wxyz[3]),
                        "imu_gx": float(gyro_b[0]),
                        "imu_gy": float(gyro_b[1]),
                        "imu_gz": float(gyro_b[2]),
                        "imu_ax": float(accel_b[0]),
                        "imu_ay": float(accel_b[1]),
                        "imu_az": float(accel_b[2]),
                        "collective_thrust": collective,
                        "thrust_b_x": float(thrust_b[0]),
                        "thrust_b_y": float(thrust_b[1]),
                        "thrust_b_z": float(thrust_b[2]),
                        "action_m1": float(action_np[0]),
                        "action_m2": float(action_np[1]),
                        "action_m3": float(action_np[2]),
                        "action_m4": float(action_np[3]),
                    }
                )
                if bool(torch.as_tensor(terminated).any()) or bool(torch.as_tensor(truncated).any()):
                    break
    finally:
        metadata = {
            "schema": "isaac_drone_racer.learned_motion_trace.v1",
            "trace_csv": output.name,
            "profile": args_cli.profile,
            "completed_steps": completed,
            "nominal_control_rate_hz": 100.0,
            "amplitude_m": float(args_cli.amplitude_m),
            "frequency_hz": float(args_cli.frequency_hz),
            "translation_m": float(args_cli.translation_m),
        }
        output.with_suffix(output.suffix + ".json").write_text(
            json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
        )
        env.close()
        print(f"[learned-motion] wrote {output}")


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
