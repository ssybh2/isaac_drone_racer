"""Run targeted OpenVINS fault-isolation experiments in Isaac Lab.

This script is intentionally detector-free. It isolates the Isaac -> ROS2 ->
OpenVINS path and records truth, raw IMU and OpenVINS state side-by-side so
translation divergence can be attributed to motion, IMU propagation or visual
observability rather than to gate fusion.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Run OpenVINS fault-isolation experiments.")
parser.add_argument("--steps", type=int, default=3000)
parser.add_argument(
    "--profile",
    choices=("hover", "translate_x", "lissajous"),
    default="hover",
    help="Motion profile used after reset.",
)
parser.add_argument("--hover_action", type=float, default=0.0)
parser.add_argument("--motion_start_s", type=float, default=6.0)
parser.add_argument("--translation_m", type=float, default=1.0)
parser.add_argument("--translation_duration_s", type=float, default=10.0)
parser.add_argument("--lissajous_amplitude_m", type=float, default=0.5)
parser.add_argument("--lissajous_frequency_hz", type=float, default=0.08)
parser.add_argument(
    "--yaw_kp",
    type=float,
    default=0.06,
    help="Yaw-angle hold gain in normalized motor-action units per radian.",
)
parser.add_argument(
    "--yaw_kd",
    type=float,
    default=0.03,
    help="Yaw-rate damping gain in normalized motor-action units per rad/s.",
)
parser.add_argument(
    "--yaw_max_correction",
    type=float,
    default=0.08,
    help="Absolute per-motor yaw correction limit in normalized action units.",
)
parser.add_argument(
    "--disable_yaw_hold",
    action="store_true",
    help="Disable the diagnostic yaw hold so the previous uncontrolled behavior can be reproduced.",
)
parser.add_argument(
    "--output_dir",
    type=Path,
    default=Path("artifacts/openvins_fault_isolation"),
)
parser.add_argument(
    "--abort_position_error_m",
    type=float,
    default=250.0,
    help="Stop after initialization once VIO position error exceeds this value; <=0 disables.",
)
parser.add_argument(
    "--task",
    default="Isaac-Drone-Racer-Swift-OpenVINS-v0",
)
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


def _quat_to_rpy(q) -> np.ndarray:
    w, x, y, z = np.asarray(q, dtype=np.float64).reshape(4)
    roll = np.arctan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch = np.arcsin(np.clip(2.0 * (w * y - z * x), -1.0, 1.0))
    yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return np.array([roll, pitch, yaw], dtype=np.float64)


def _wrap_angle(angle_rad: float) -> float:
    return float(np.arctan2(np.sin(angle_rad), np.cos(angle_rad)))


def _quat_angle_error(q_a, q_b) -> float:
    a = np.asarray(q_a, dtype=np.float64).reshape(4)
    b = np.asarray(q_b, dtype=np.float64).reshape(4)
    a /= np.linalg.norm(a)
    b /= np.linalg.norm(b)
    dot = float(np.clip(abs(np.dot(a, b)), 0.0, 1.0))
    return 2.0 * float(np.arccos(dot))


def _scalar_summary(values: list[float]) -> dict:
    if not values:
        return {"samples": 0, "mean": None, "std": None, "rmse": None, "p95": None, "max": None}
    x = np.asarray(values, dtype=np.float64)
    return {
        "samples": int(x.size),
        "mean": float(np.mean(x)),
        "std": float(np.std(x)),
        "rmse": float(np.sqrt(np.mean(np.square(x)))),
        "p95": float(np.percentile(x, 95)),
        "max": float(np.max(x)),
    }


def _vector_summary(values: list[np.ndarray]) -> dict:
    if not values:
        return {"samples": 0, "mean": None, "std": None, "max_abs": None}
    x = np.asarray(values, dtype=np.float64)
    return {
        "samples": int(x.shape[0]),
        "mean": np.mean(x, axis=0).tolist(),
        "std": np.std(x, axis=0).tolist(),
        "max_abs": np.max(np.abs(x), axis=0).tolist(),
    }


def _target_xy(initial_xy: torch.Tensor, t_s: float) -> torch.Tensor:
    target = initial_xy.clone()
    if args_cli.profile == "hover" or t_s < args_cli.motion_start_s:
        return target
    tau = t_s - args_cli.motion_start_s
    if args_cli.profile == "translate_x":
        alpha = min(1.0, max(0.0, tau / max(args_cli.translation_duration_s, 1.0e-6)))
        target[0] += float(args_cli.translation_m) * alpha
        return target
    omega = 2.0 * np.pi * float(args_cli.lissajous_frequency_hz)
    amplitude = float(args_cli.lissajous_amplitude_m)
    target[0] += amplitude * float(np.sin(omega * tau))
    target[1] += 0.5 * amplitude * float(np.sin(2.0 * omega * tau))
    return target


def _controller_action(
    raw_env,
    initial_xy: torch.Tensor,
    target_height_m: float,
    target_yaw_rad: float,
    t_s: float,
) -> tuple[torch.Tensor, float, float]:
    robot = raw_env.scene["robot"]
    target_xy = _target_xy(initial_xy, t_s)
    height_error = target_height_m - float(robot.data.root_pos_w[0, 2])
    vertical_velocity = float(robot.data.root_lin_vel_w[0, 2])
    common = float(args_cli.hover_action) + 0.18 * height_error - 0.09 * vertical_velocity
    xy_error = target_xy - robot.data.root_pos_w[0, :2]
    desired_acceleration_xy = xy_error - 1.5 * robot.data.root_lin_vel_w[0, :2]
    desired_roll = float(torch.clamp(-desired_acceleration_xy[1] / 9.81, -0.1, 0.1))
    desired_pitch = float(torch.clamp(desired_acceleration_xy[0] / 9.81, -0.1, 0.1))
    quaternion = robot.data.root_quat_w[0]
    angular_velocity = robot.data.root_ang_vel_b[0]
    roll = float(
        torch.atan2(
            2.0 * (quaternion[0] * quaternion[1] + quaternion[2] * quaternion[3]),
            1.0 - 2.0 * (quaternion[1] ** 2 + quaternion[2] ** 2),
        )
    )
    pitch = float(
        torch.asin(
            torch.clamp(
                2.0 * (quaternion[0] * quaternion[2] - quaternion[3] * quaternion[1]),
                -1.0,
                1.0,
            )
        )
    )
    yaw = float(
        torch.atan2(
            2.0 * (quaternion[0] * quaternion[3] + quaternion[1] * quaternion[2]),
            1.0 - 2.0 * (quaternion[2] ** 2 + quaternion[3] ** 2),
        )
    )
    roll_correction = -0.08 * (roll - desired_roll) - 0.015 * float(angular_velocity[0])
    pitch_correction = -0.08 * (pitch - desired_pitch) - 0.015 * float(angular_velocity[1])

    yaw_error = _wrap_angle(yaw - target_yaw_rad)
    if args_cli.disable_yaw_hold:
        yaw_correction = 0.0
    else:
        yaw_correction = -float(args_cli.yaw_kp) * yaw_error - float(args_cli.yaw_kd) * float(angular_velocity[2])
        yaw_correction = float(
            np.clip(
                yaw_correction,
                -abs(float(args_cli.yaw_max_correction)),
                abs(float(args_cli.yaw_max_correction)),
            )
        )

    # Allocation yaw signs are [+ - + -]. A positive yaw_correction therefore
    # commands positive body-z torque; the negative feedback above damps yaw
    # error and yaw rate while leaving collective thrust unchanged to first order.
    action = torch.tensor(
        [
            common + roll_correction - pitch_correction + yaw_correction,
            common - roll_correction - pitch_correction - yaw_correction,
            common - roll_correction + pitch_correction + yaw_correction,
            common + roll_correction + pitch_correction - yaw_correction,
        ],
        dtype=torch.float32,
        device=raw_env.device,
    ).clamp(-1.0, 1.0)
    return action, yaw_error, yaw_correction


def _empty_estimate_fields() -> dict:
    names = (
        "vio_px", "vio_py", "vio_pz", "vio_vx", "vio_vy", "vio_vz",
        "vio_qw", "vio_qx", "vio_qy", "vio_qz", "pos_err_x", "pos_err_y",
        "pos_err_z", "pos_err_norm", "vel_err_x", "vel_err_y", "vel_err_z",
        "vel_err_norm", "orientation_err_rad", "odom_age_s", "drained_callbacks",
    )
    return {name: "" for name in names}


def main() -> None:
    if args_cli.steps < 1:
        raise ValueError("--steps must be positive")
    output_dir = args_cli.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    trace_path = output_dir / f"{args_cli.profile}_trace.csv"
    summary_path = output_dir / f"{args_cli.profile}_summary.json"

    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=1)
    env_cfg.swift_detector_checkpoint = None
    env_cfg.swift_visibility_checkpoint = None
    env_cfg.swift_use_oracle_gate_index = False
    env_cfg.swift_rejection_dump_dir = None
    env_cfg.terminations.collision = None
    env_cfg.terminations.flyaway = None
    env_cfg.episode_length_s = max(float(env_cfg.episode_length_s), int(args_cli.steps) * 0.01 + 2.0)

    env = gym.make(args_cli.task, cfg=env_cfg)
    raw_env = env.unwrapped
    env.reset()
    robot = raw_env.scene["robot"]
    target_height_m = float(robot.data.root_pos_w[0, 2])
    initial_xy = robot.data.root_pos_w[0, :2].clone()
    initial_quaternion = np.asarray(robot.data.root_quat_w[0].detach().cpu(), dtype=np.float64)
    target_yaw_rad = float(_quat_to_rpy(initial_quaternion)[2])

    fieldnames = [
        "step", "t_s", "target_x", "target_y", "target_yaw", "yaw_error", "yaw_correction",
        "truth_px", "truth_py", "truth_pz", "truth_vx", "truth_vy", "truth_vz",
        "truth_qw", "truth_qx", "truth_qy", "truth_qz",
        "truth_roll", "truth_pitch", "truth_yaw",
        "truth_wx", "truth_wy", "truth_wz",
        "imu_ax", "imu_ay", "imu_az", "imu_acc_norm",
        "imu_gx", "imu_gy", "imu_gz", "imu_gyro_norm",
        "vio_px", "vio_py", "vio_pz", "vio_vx", "vio_vy", "vio_vz",
        "vio_qw", "vio_qx", "vio_qy", "vio_qz",
        "pos_err_x", "pos_err_y", "pos_err_z", "pos_err_norm",
        "vel_err_x", "vel_err_y", "vel_err_z", "vel_err_norm",
        "orientation_err_rad", "odom_age_s", "drained_callbacks",
    ]

    truth_positions: list[np.ndarray] = []
    truth_velocities: list[np.ndarray] = []
    truth_rpy: list[np.ndarray] = []
    truth_angular_velocities: list[np.ndarray] = []
    imu_accels: list[np.ndarray] = []
    imu_gyros: list[np.ndarray] = []
    preinit_imu_accels: list[np.ndarray] = []
    preinit_imu_gyros: list[np.ndarray] = []
    yaw_errors: list[float] = []
    yaw_corrections: list[float] = []
    pos_error_norms: list[float] = []
    vel_error_norms: list[float] = []
    orientation_errors: list[float] = []
    initialized_step: int | None = None
    error_checkpoints: dict[str, float] = {}
    completed_steps = 0
    stopped_for_divergence = False

    try:
        with trace_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for step in range(int(args_cli.steps)):
                if not simulation_app.is_running():
                    break
                t_before = float(raw_env._timestamp_s())
                action, control_yaw_error, yaw_correction = _controller_action(
                    raw_env,
                    initial_xy,
                    target_height_m,
                    target_yaw_rad,
                    t_before,
                )
                _, _, terminated, truncated, _ = env.step(action.unsqueeze(0))
                completed_steps = step + 1

                t_s = float(raw_env._timestamp_s())
                truth = raw_env._truth_vio_state()
                robot = raw_env.scene["robot"]
                imu = raw_env.scene["imu"]
                truth_rpy_now = _quat_to_rpy(truth.orientation_w_b_wxyz)
                truth_w_b = np.asarray(robot.data.root_ang_vel_b[0].detach().cpu(), dtype=np.float64)
                imu_acc_b = np.asarray(imu.data.lin_acc_b[0].detach().cpu(), dtype=np.float64)
                imu_gyro_b = np.asarray(imu.data.ang_vel_b[0].detach().cpu(), dtype=np.float64)
                target_xy = _target_xy(initial_xy, t_s)

                truth_positions.append(truth.position_w_b.copy())
                truth_velocities.append(truth.linear_velocity_w_b.copy())
                truth_rpy.append(truth_rpy_now)
                truth_angular_velocities.append(truth_w_b)
                imu_accels.append(imu_acc_b)
                imu_gyros.append(imu_gyro_b)
                yaw_errors.append(control_yaw_error)
                yaw_corrections.append(yaw_correction)

                estimate = raw_env.openvins_vio_estimate
                if estimate is None:
                    preinit_imu_accels.append(imu_acc_b)
                    preinit_imu_gyros.append(imu_gyro_b)

                row = {
                    "step": step,
                    "t_s": t_s,
                    "target_x": float(target_xy[0]),
                    "target_y": float(target_xy[1]),
                    "target_yaw": target_yaw_rad,
                    "yaw_error": control_yaw_error,
                    "yaw_correction": yaw_correction,
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
                    "truth_roll": float(truth_rpy_now[0]),
                    "truth_pitch": float(truth_rpy_now[1]),
                    "truth_yaw": float(truth_rpy_now[2]),
                    "truth_wx": float(truth_w_b[0]),
                    "truth_wy": float(truth_w_b[1]),
                    "truth_wz": float(truth_w_b[2]),
                    "imu_ax": float(imu_acc_b[0]),
                    "imu_ay": float(imu_acc_b[1]),
                    "imu_az": float(imu_acc_b[2]),
                    "imu_acc_norm": float(np.linalg.norm(imu_acc_b)),
                    "imu_gx": float(imu_gyro_b[0]),
                    "imu_gy": float(imu_gyro_b[1]),
                    "imu_gz": float(imu_gyro_b[2]),
                    "imu_gyro_norm": float(np.linalg.norm(imu_gyro_b)),
                    **_empty_estimate_fields(),
                }

                if estimate is not None:
                    if initialized_step is None:
                        initialized_step = step
                    pos_error = estimate.position_w_b - truth.position_w_b
                    vel_error = estimate.linear_velocity_w_b - truth.linear_velocity_w_b
                    pos_error_norm = float(np.linalg.norm(pos_error))
                    vel_error_norm = float(np.linalg.norm(vel_error))
                    orientation_error = _quat_angle_error(
                        estimate.orientation_w_b_wxyz, truth.orientation_w_b_wxyz
                    )
                    pos_error_norms.append(pos_error_norm)
                    vel_error_norms.append(vel_error_norm)
                    orientation_errors.append(orientation_error)
                    row.update(
                        {
                            "vio_px": float(estimate.position_w_b[0]),
                            "vio_py": float(estimate.position_w_b[1]),
                            "vio_pz": float(estimate.position_w_b[2]),
                            "vio_vx": float(estimate.linear_velocity_w_b[0]),
                            "vio_vy": float(estimate.linear_velocity_w_b[1]),
                            "vio_vz": float(estimate.linear_velocity_w_b[2]),
                            "vio_qw": float(estimate.orientation_w_b_wxyz[0]),
                            "vio_qx": float(estimate.orientation_w_b_wxyz[1]),
                            "vio_qy": float(estimate.orientation_w_b_wxyz[2]),
                            "vio_qz": float(estimate.orientation_w_b_wxyz[3]),
                            "pos_err_x": float(pos_error[0]),
                            "pos_err_y": float(pos_error[1]),
                            "pos_err_z": float(pos_error[2]),
                            "pos_err_norm": pos_error_norm,
                            "vel_err_x": float(vel_error[0]),
                            "vel_err_y": float(vel_error[1]),
                            "vel_err_z": float(vel_error[2]),
                            "vel_err_norm": vel_error_norm,
                            "orientation_err_rad": orientation_error,
                            "odom_age_s": max(0.0, t_s - float(estimate.timestamp_s)),
                            "drained_callbacks": int(raw_env._openvins_bridge.last_drain_count),
                        }
                    )
                    since_init = (step - initialized_step) * float(raw_env.step_dt)
                    for checkpoint_s in (1.0, 5.0, 10.0, 20.0, 30.0):
                        key = f"{checkpoint_s:g}s"
                        if key not in error_checkpoints and since_init >= checkpoint_s:
                            error_checkpoints[key] = pos_error_norm
                    if (
                        args_cli.abort_position_error_m > 0.0
                        and pos_error_norm >= args_cli.abort_position_error_m
                    ):
                        stopped_for_divergence = True

                writer.writerow(row)

                if step % 100 == 0:
                    if estimate is None:
                        print(
                            f"[FaultIsolation] step={step} t={t_s:.2f}s waiting for OpenVINS; "
                            f"|imu_a|={np.linalg.norm(imu_acc_b):.3f} "
                            f"|imu_w|={np.linalg.norm(imu_gyro_b):.4f} "
                            f"yaw_err={np.degrees(control_yaw_error):.2f}deg "
                            f"yaw_u={yaw_correction:.4f}"
                        )
                    else:
                        print(
                            f"[FaultIsolation] step={step} t={t_s:.2f}s "
                            f"pos_err={row['pos_err_norm']:.3f}m "
                            f"vel_err={row['vel_err_norm']:.3f}m/s "
                            f"ori_err={np.degrees(row['orientation_err_rad']):.2f}deg "
                            f"yaw_err={np.degrees(control_yaw_error):.2f}deg "
                            f"age={row['odom_age_s']:.4f}s"
                        )

                if stopped_for_divergence:
                    print(
                        "[FaultIsolation] stopping after position error exceeded "
                        f"{args_cli.abort_position_error_m:.1f} m"
                    )
                    break
                if bool(torch.as_tensor(terminated).any()) or bool(torch.as_tensor(truncated).any()):
                    print("[FaultIsolation] environment terminated; restart OpenVINS before another run")
                    break
    finally:
        truth_position_array = np.asarray(truth_positions, dtype=np.float64)
        initial_position = None if not truth_positions else truth_position_array[0]
        max_displacement = None
        if initial_position is not None:
            max_displacement = float(
                np.max(np.linalg.norm(truth_position_array - initial_position, axis=1))
            )
        report = {
            "schema": "isaac_drone_racer.openvins_fault_isolation.v2",
            "profile": args_cli.profile,
            "python_executable": sys.executable,
            "completed_steps": completed_steps,
            "openvins_initialized_step": initialized_step,
            "openvins_initialized_time_s": (
                None if initialized_step is None else initialized_step * float(raw_env.step_dt)
            ),
            "stopped_for_divergence": stopped_for_divergence,
            "control": {
                "yaw_hold_enabled": not args_cli.disable_yaw_hold,
                "target_yaw_rad": target_yaw_rad,
                "yaw_kp": float(args_cli.yaw_kp),
                "yaw_kd": float(args_cli.yaw_kd),
                "yaw_max_correction": float(args_cli.yaw_max_correction),
                "yaw_error_rad": _scalar_summary([abs(v) for v in yaw_errors]),
                "yaw_correction": _scalar_summary([abs(v) for v in yaw_corrections]),
            },
            "truth_motion": {
                "max_displacement_from_start_m": max_displacement,
                "velocity_w_mps": _vector_summary(truth_velocities),
                "rpy_rad": _vector_summary(truth_rpy),
                "angular_velocity_b_radps": _vector_summary(truth_angular_velocities),
            },
            "imu_all": {
                "accel_b_mps2": _vector_summary(imu_accels),
                "gyro_b_radps": _vector_summary(imu_gyros),
                "accel_norm_mps2": _scalar_summary([float(np.linalg.norm(v)) for v in imu_accels]),
                "gyro_norm_radps": _scalar_summary([float(np.linalg.norm(v)) for v in imu_gyros]),
            },
            "imu_preinit": {
                "accel_b_mps2": _vector_summary(preinit_imu_accels),
                "gyro_b_radps": _vector_summary(preinit_imu_gyros),
                "accel_norm_mps2": _scalar_summary(
                    [float(np.linalg.norm(v)) for v in preinit_imu_accels]
                ),
                "gyro_norm_radps": _scalar_summary(
                    [float(np.linalg.norm(v)) for v in preinit_imu_gyros]
                ),
            },
            "vio_errors": {
                "position_norm_m": _scalar_summary(pos_error_norms),
                "velocity_norm_mps": _scalar_summary(vel_error_norms),
                "orientation_rad": _scalar_summary(orientation_errors),
                "orientation_deg": _scalar_summary([float(np.degrees(v)) for v in orientation_errors]),
                "position_error_checkpoints_after_init": error_checkpoints,
            },
            "files": {"trace_csv": trace_path.name},
        }
        summary_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"[FaultIsolation] wrote {trace_path}")
        print(f"[FaultIsolation] wrote {summary_path}")
        env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
