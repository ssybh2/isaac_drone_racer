"""Collect supervised IMO training traces from Isaac Lab ground truth.

Ground truth is used only to create supervised labels and to rotate training
features into world frame later. Runtime LearnedInertialRacingEnv never uses GT
pose after its fixed known initialization.

V2 collection deliberately separates trajectory duration from collection length
so the same displacement can be flown at different speeds/accelerations. This
avoids the V1 failure mode where long 60 s traces produced only very slow
translation examples, while evaluation used much faster motion.
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
    choices=(
        "hover",
        "translate_x",
        "translate_scurve",
        "vertical",
        "yaw_sweep",
        "translate_yaw",
        "circle",
        "lissajous",
        "racing_like",
    ),
    default="lissajous",
)
parser.add_argument("--amplitude_m", type=float, default=1.0)
parser.add_argument("--frequency_hz", type=float, default=0.10)
parser.add_argument("--translation_m", type=float, default=3.0)
parser.add_argument("--vertical_m", type=float, default=1.0)
parser.add_argument(
    "--motion_duration_s",
    type=float,
    default=None,
    help=(
        "Duration of translate/vertical S-curve motion. If omitted, translate_x "
        "keeps the legacy 70%%-of-trace ramp."
    ),
)
parser.add_argument("--yaw_amplitude_deg", type=float, default=45.0)
parser.add_argument("--yaw_frequency_hz", type=float, default=0.10)
parser.add_argument("--phase_rad", type=float, default=0.0)
parser.add_argument("--vehicle_mass_kg", type=float, default=0.6076)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--dataset_split", choices=("train", "val", "test", "unspecified"), default="unspecified")
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--task", default="Isaac-Drone-Racer-Learned-Inertial-v0")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Supervised IMO collection needs IMU + actuator data, not the camera. Do not
# force RTX camera rendering; the environment config also disables the camera
# below. This makes large V2 manifests much faster to collect.
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


def _smoothstep5(u: float) -> float:
    """Quintic S-curve with zero velocity/acceleration at both endpoints."""
    u = float(np.clip(u, 0.0, 1.0))
    return float(10.0 * u**3 - 15.0 * u**4 + 6.0 * u**5)


def _motion_duration_s() -> float:
    if args_cli.motion_duration_s is not None:
        if args_cli.motion_duration_s <= 0.0:
            raise ValueError("--motion_duration_s must be positive")
        return float(args_cli.motion_duration_s)
    # Legacy translate_x behavior: motion occupies 70% of the nominal trace.
    return max(float(args_cli.steps) * 0.01 * 0.7, 1.0e-6)


def _target_state(
    initial_xy: torch.Tensor,
    initial_height_m: float,
    initial_yaw_rad: float,
    t_s: float,
) -> tuple[torch.Tensor, float, float]:
    target_xy = initial_xy.clone()
    target_z = float(initial_height_m)
    target_yaw = float(initial_yaw_rad)
    amp = float(args_cli.amplitude_m)
    omega = 2.0 * np.pi * float(args_cli.frequency_hz)
    yaw_amp = np.deg2rad(float(args_cli.yaw_amplitude_deg))
    yaw_omega = 2.0 * np.pi * float(args_cli.yaw_frequency_hz)
    phase = float(args_cli.phase_rad)

    if args_cli.profile == "hover":
        return target_xy, target_z, target_yaw

    if args_cli.profile == "translate_x":
        duration = _motion_duration_s()
        target_xy[0] += float(args_cli.translation_m) * np.clip(t_s / duration, 0.0, 1.0)
        return target_xy, target_z, target_yaw

    if args_cli.profile == "translate_scurve":
        duration = _motion_duration_s()
        target_xy[0] += float(args_cli.translation_m) * _smoothstep5(t_s / duration)
        return target_xy, target_z, target_yaw

    if args_cli.profile == "vertical":
        duration = _motion_duration_s()
        target_z += float(args_cli.vertical_m) * _smoothstep5(t_s / duration)
        return target_xy, target_z, target_yaw

    if args_cli.profile == "yaw_sweep":
        target_yaw += yaw_amp * float(np.sin(yaw_omega * t_s + phase))
        return target_xy, target_z, target_yaw

    if args_cli.profile == "translate_yaw":
        duration = _motion_duration_s()
        target_xy[0] += float(args_cli.translation_m) * _smoothstep5(t_s / duration)
        target_yaw += yaw_amp * float(np.sin(yaw_omega * t_s + phase))
        return target_xy, target_z, target_yaw

    if args_cli.profile == "circle":
        target_xy[0] += amp * (float(np.cos(omega * t_s + phase)) - float(np.cos(phase)))
        target_xy[1] += amp * (float(np.sin(omega * t_s + phase)) - float(np.sin(phase)))
        return target_xy, target_z, target_yaw

    if args_cli.profile == "lissajous":
        target_xy[0] += amp * float(np.sin(omega * t_s + phase))
        target_xy[1] += 0.7 * amp * float(np.sin(2.0 * omega * t_s + 0.35 + phase))
        return target_xy, target_z, target_yaw

    # Mixed racing-like lateral motion with heading excitation. It is not a
    # gate-following policy; it exists only to cover coupled inertial dynamics.
    target_xy[0] += amp * float(np.sin(omega * t_s + phase))
    target_xy[1] += 0.8 * amp * float(np.sin(1.5 * omega * t_s + 0.45 + phase))
    target_z += 0.25 * amp * float(np.sin(0.5 * omega * t_s + 0.2 + phase))
    target_yaw += yaw_amp * float(np.sin(yaw_omega * t_s + phase))
    return target_xy, target_z, target_yaw


def _controller_action(
    raw_env,
    initial_xy,
    initial_height_m,
    initial_yaw_rad,
    t_s,
):
    robot = raw_env.scene["robot"]
    target_xy, target_height_m, target_yaw_rad = _target_state(
        initial_xy,
        initial_height_m,
        initial_yaw_rad,
        t_s,
    )

    height_error = target_height_m - float(robot.data.root_pos_w[0, 2])
    vertical_velocity = float(robot.data.root_lin_vel_w[0, 2])
    common = 0.18 * height_error - 0.09 * vertical_velocity

    xy_error = target_xy - robot.data.root_pos_w[0, :2]
    desired_accel_w_xy = xy_error - 1.5 * robot.data.root_lin_vel_w[0, :2]

    q = robot.data.root_quat_w[0]
    rates = robot.data.root_ang_vel_b[0]
    roll, pitch, yaw = _quat_to_rpy(q)

    # Convert desired world-frame horizontal acceleration into the current
    # yaw-aligned body frame before mapping it to roll/pitch. The previous V2
    # collector implicitly assumed yaw=0:
    #
    #   desired_roll  = -a_w_y / g
    #   desired_pitch =  a_w_x / g
    #
    # That is only valid while the vehicle heading is fixed. During
    # translate+yaw/racing-like traces it rotated the tilt command into the
    # wrong direction and could drive the vehicle into a runaway trajectory.
    cy = float(np.cos(yaw))
    sy = float(np.sin(yaw))
    accel_body_x = cy * float(desired_accel_w_xy[0]) + sy * float(desired_accel_w_xy[1])
    accel_body_y = -sy * float(desired_accel_w_xy[0]) + cy * float(desired_accel_w_xy[1])
    desired_roll = float(np.clip(-accel_body_y / 9.81, -0.25, 0.25))
    desired_pitch = float(np.clip(accel_body_x / 9.81, -0.25, 0.25))
    yaw_error = float(
        np.arctan2(
            np.sin(yaw - target_yaw_rad),
            np.cos(yaw - target_yaw_rad),
        )
    )
    roll_u = -0.08 * (roll - desired_roll) - 0.015 * float(rates[0])
    pitch_u = -0.08 * (pitch - desired_pitch) - 0.015 * float(rates[1])
    yaw_u = float(np.clip(-0.06 * yaw_error - 0.03 * float(rates[2]), -0.10, 0.10))

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
    if args_cli.steps < 1:
        raise ValueError("--steps must be positive")
    if args_cli.vehicle_mass_kg <= 0.0:
        raise ValueError("--vehicle_mass_kg must be positive")
    output = args_cli.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=1)
    env_cfg.learned_motion_checkpoint = None
    env_cfg.swift_detector_checkpoint = None
    env_cfg.swift_visibility_checkpoint = None
    env_cfg.scene.tiled_camera = None
    env_cfg.terminations.collision = None
    env_cfg.terminations.flyaway = None
    env_cfg.commands.target.randomise_start = None
    env_cfg.events.push_robot = None
    env_cfg.episode_length_s = max(float(env_cfg.episode_length_s), args_cli.steps * 0.01 + 5.0)

    env = gym.make(args_cli.task, cfg=env_cfg)
    raw_env = env.unwrapped
    env.reset(seed=int(args_cli.seed))
    robot = raw_env.scene["robot"]
    initial_xy = robot.data.root_pos_w[0, :2].clone()
    initial_height_m = float(robot.data.root_pos_w[0, 2])
    _, _, initial_yaw_rad = _quat_to_rpy(robot.data.root_quat_w[0])
    start_time_s = float(raw_env._timestamp_s())

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
                t_s = float(raw_env._timestamp_s()) - start_time_s
                action = _controller_action(
                    raw_env,
                    initial_xy,
                    initial_height_m,
                    initial_yaw_rad,
                    t_s,
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
                    "t_s": float(raw_env._timestamp_s()) - start_time_s,
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
            "schema": "isaac_drone_racer.imo_supervised_trace.v2",
            "trace_csv": output.name,
            "profile": args_cli.profile,
            "dataset_split": args_cli.dataset_split,
            "completed_steps": completed,
            "nominal_control_rate_hz": 100.0,
            "vehicle_mass_kg": float(args_cli.vehicle_mass_kg),
            "thrust_b_units": "m/s^2 (mass-normalized collective thrust)",
            "ground_truth_use": "supervised labels/evaluation only",
            "controller_revision": "yaw_compensated_world_xy_v2",
            "trajectory": {
                "amplitude_m": float(args_cli.amplitude_m),
                "frequency_hz": float(args_cli.frequency_hz),
                "translation_m": float(args_cli.translation_m),
                "vertical_m": float(args_cli.vertical_m),
                "motion_duration_s": (
                    None if args_cli.motion_duration_s is None else float(args_cli.motion_duration_s)
                ),
                "yaw_amplitude_deg": float(args_cli.yaw_amplitude_deg),
                "yaw_frequency_hz": float(args_cli.yaw_frequency_hz),
                "phase_rad": float(args_cli.phase_rad),
                "seed": int(args_cli.seed),
            },
        }
        output.with_suffix(output.suffix + ".json").write_text(
            json.dumps(metadata, indent=2) + "\n",
            encoding="utf-8",
        )
        env.close()
        print(
            f"[IMO data] wrote {output} "
            f"profile={args_cli.profile} split={args_cli.dataset_split} "
            f"steps={completed}",
            flush=True,
        )


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
