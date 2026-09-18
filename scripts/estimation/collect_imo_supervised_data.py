"""Collect supervised IMO training traces from Isaac Lab ground truth.

Ground truth is used only to create training labels and to rotate training
features into world frame later.  Runtime LearnedInertialRacingEnv never uses
GT pose after its fixed known initialization.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--steps", type=int, default=6000)
parser.add_argument(
    "--profile",
    choices=("hover", "translate_x", "circle", "lissajous"),
    default="lissajous",
)
parser.add_argument("--amplitude_m", type=float, default=1.0)
parser.add_argument("--frequency_hz", type=float, default=0.10)
parser.add_argument("--translation_m", type=float, default=3.0)
parser.add_argument("--vehicle_mass_kg", type=float, default=0.6076)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--task", default="Isaac-Drone-Racer-Learned-Inertial-v0")
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


def _quat_to_rpy(q):
    w, x, y, z = [float(v) for v in q]
    roll = np.arctan2(2 * (w*x + y*z), 1 - 2 * (x*x + y*y))
    pitch = np.arcsin(np.clip(2 * (w*y - z*x), -1.0, 1.0))
    yaw = np.arctan2(2 * (w*z + x*y), 1 - 2 * (y*y + z*z))
    return float(roll), float(pitch), float(yaw)


def _target_xy(initial_xy: torch.Tensor, t_s: float) -> torch.Tensor:
    target = initial_xy.clone()
    amp = float(args_cli.amplitude_m)
    omega = 2.0 * np.pi * float(args_cli.frequency_hz)
    if args_cli.profile == "hover":
        return target
    if args_cli.profile == "translate_x":
        duration = max(args_cli.steps * 0.01 * 0.7, 1.0e-6)
        target[0] += float(args_cli.translation_m) * np.clip(t_s / duration, 0.0, 1.0)
        return target
    if args_cli.profile == "circle":
        target[0] += amp * (float(np.cos(omega * t_s)) - 1.0)
        target[1] += amp * float(np.sin(omega * t_s))
        return target
    target[0] += amp * float(np.sin(omega * t_s))
    target[1] += 0.7 * amp * float(np.sin(2.0 * omega * t_s + 0.35))
    return target


def _controller_action(raw_env, initial_xy, target_height_m, target_yaw_rad, t_s):
    robot = raw_env.scene["robot"]
    target_xy = _target_xy(initial_xy, t_s)
    height_error = target_height_m - float(robot.data.root_pos_w[0, 2])
    vertical_velocity = float(robot.data.root_lin_vel_w[0, 2])
    common = 0.18 * height_error - 0.09 * vertical_velocity

    xy_error = target_xy - robot.data.root_pos_w[0, :2]
    desired_accel_xy = xy_error - 1.5 * robot.data.root_lin_vel_w[0, :2]
    desired_roll = float(torch.clamp(-desired_accel_xy[1] / 9.81, -0.20, 0.20))
    desired_pitch = float(torch.clamp(desired_accel_xy[0] / 9.81, -0.20, 0.20))

    q = robot.data.root_quat_w[0]
    rates = robot.data.root_ang_vel_b[0]
    roll, pitch, yaw = _quat_to_rpy(q)
    yaw_error = float(np.arctan2(np.sin(yaw - target_yaw_rad), np.cos(yaw - target_yaw_rad)))
    roll_u = -0.08 * (roll - desired_roll) - 0.015 * float(rates[0])
    pitch_u = -0.08 * (pitch - desired_pitch) - 0.015 * float(rates[1])
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
    if args_cli.vehicle_mass_kg <= 0.0:
        raise ValueError("--vehicle_mass_kg must be positive")
    output = args_cli.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=1)
    env_cfg.learned_motion_checkpoint = None
    env_cfg.swift_detector_checkpoint = None
    env_cfg.swift_visibility_checkpoint = None
    env_cfg.terminations.collision = None
    env_cfg.terminations.flyaway = None
    env_cfg.episode_length_s = max(float(env_cfg.episode_length_s), args_cli.steps * 0.01 + 5.0)

    env = gym.make(args_cli.task, cfg=env_cfg)
    raw_env = env.unwrapped
    env.reset()
    robot = raw_env.scene["robot"]
    initial_xy = robot.data.root_pos_w[0, :2].clone()
    target_height_m = float(robot.data.root_pos_w[0, 2])
    _, _, target_yaw_rad = _quat_to_rpy(robot.data.root_quat_w[0])

    fields = [
        "step", "t_s", "profile",
        "truth_px", "truth_py", "truth_pz",
        "truth_vx", "truth_vy", "truth_vz",
        "truth_qw", "truth_qx", "truth_qy", "truth_qz",
        "imu_gx", "imu_gy", "imu_gz",
        "imu_ax", "imu_ay", "imu_az",
        "collective_thrust_n",
        "thrust_b_x", "thrust_b_y", "thrust_b_z",
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
                action = _controller_action(
                    raw_env, initial_xy, target_height_m, target_yaw_rad, raw_env._timestamp_s()
                )
                _, _, terminated, truncated, _ = env.step(action.unsqueeze(0))
                completed = step + 1

                imu = raw_env.scene["imu"]
                gyro_b = np.asarray(imu.data.ang_vel_b[0].detach().cpu(), dtype=np.float64)
                accel_b = np.asarray(imu.data.lin_acc_b[0].detach().cpu(), dtype=np.float64)
                processed = np.asarray(
                    raw_env.action_manager.get_term("control_action").processed_actions[0]
                    .detach().cpu(),
                    dtype=np.float64,
                )
                collective_n = float(processed[0])
                thrust_b = np.array(
                    [0.0, 0.0, collective_n / float(args_cli.vehicle_mass_kg)],
                    dtype=np.float64,
                )
                p = np.asarray(robot.data.root_pos_w[0].detach().cpu(), dtype=np.float64)
                v = np.asarray(robot.data.root_lin_vel_w[0].detach().cpu(), dtype=np.float64)
                q = np.asarray(robot.data.root_quat_w[0].detach().cpu(), dtype=np.float64)
                a = np.asarray(action.detach().cpu(), dtype=np.float64)

                writer.writerow({
                    "step": step,
                    "t_s": float(raw_env._timestamp_s()),
                    "profile": args_cli.profile,
                    "truth_px": p[0], "truth_py": p[1], "truth_pz": p[2],
                    "truth_vx": v[0], "truth_vy": v[1], "truth_vz": v[2],
                    "truth_qw": q[0], "truth_qx": q[1], "truth_qy": q[2], "truth_qz": q[3],
                    "imu_gx": gyro_b[0], "imu_gy": gyro_b[1], "imu_gz": gyro_b[2],
                    "imu_ax": accel_b[0], "imu_ay": accel_b[1], "imu_az": accel_b[2],
                    "collective_thrust_n": collective_n,
                    "thrust_b_x": thrust_b[0], "thrust_b_y": thrust_b[1], "thrust_b_z": thrust_b[2],
                    "action_m1": a[0], "action_m2": a[1], "action_m3": a[2], "action_m4": a[3],
                })
                if bool(torch.as_tensor(terminated).any()) or bool(torch.as_tensor(truncated).any()):
                    break
    finally:
        metadata = {
            "schema": "isaac_drone_racer.imo_supervised_trace.v1",
            "trace_csv": output.name,
            "profile": args_cli.profile,
            "completed_steps": completed,
            "nominal_control_rate_hz": 100.0,
            "vehicle_mass_kg": float(args_cli.vehicle_mass_kg),
            "thrust_b_units": "m/s^2 (mass-normalized collective thrust)",
            "ground_truth_use": "supervised labels/evaluation only",
        }
        output.with_suffix(output.suffix + ".json").write_text(
            json.dumps(metadata, indent=2) + "\n",
            encoding="utf-8",
        )
        env.close()
        print(f"[IMO data] wrote {output}")


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
