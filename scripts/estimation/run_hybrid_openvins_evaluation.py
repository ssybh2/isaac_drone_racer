"""Evaluate raw OpenVINS versus learned-motion correction and optional Swift gate fusion."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--checkpoint", type=Path, required=True)
parser.add_argument("--steps", type=int, default=4000)
parser.add_argument(
    "--profile", choices=("hover", "translate_x", "lissajous"), default="translate_x"
)
parser.add_argument("--motion_start_s", type=float, default=4.0)
parser.add_argument("--translation_m", type=float, default=1.5)
parser.add_argument("--translation_duration_s", type=float, default=8.0)
parser.add_argument("--lissajous_amplitude_m", type=float, default=0.6)
parser.add_argument("--lissajous_frequency_hz", type=float, default=0.08)
parser.add_argument("--detector_checkpoint", type=Path, default=None)
parser.add_argument("--visibility_checkpoint", type=Path, default=None)
parser.add_argument("--output_dir", type=Path, default=Path("artifacts/hybrid_openvins_evaluation"))
parser.add_argument("--task", default="Isaac-Drone-Racer-Swift-Hybrid-OpenVINS-v0")
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


def _quat_to_yaw(q) -> float:
    w, x, y, z = [float(v) for v in q]
    return float(np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))


def _wrap_angle(angle: float) -> float:
    return float(np.arctan2(np.sin(angle), np.cos(angle)))


def _target_xy(initial_xy: torch.Tensor, t_s: float) -> torch.Tensor:
    target = initial_xy.clone()
    if args_cli.profile == "hover" or t_s < args_cli.motion_start_s:
        return target
    tau = t_s - float(args_cli.motion_start_s)
    if args_cli.profile == "translate_x":
        alpha = min(1.0, max(0.0, tau / max(float(args_cli.translation_duration_s), 1.0e-6)))
        target[0] += float(args_cli.translation_m) * alpha
        return target
    omega = 2.0 * np.pi * float(args_cli.lissajous_frequency_hz)
    amp = float(args_cli.lissajous_amplitude_m)
    target[0] += amp * float(np.sin(omega * tau))
    target[1] += 0.5 * amp * float(np.sin(2.0 * omega * tau))
    return target


def _controller_action(raw_env, initial_xy, target_height_m, target_yaw_rad, t_s):
    robot = raw_env.scene["robot"]
    target_xy = _target_xy(initial_xy, t_s)
    height_error = target_height_m - float(robot.data.root_pos_w[0, 2])
    vertical_velocity = float(robot.data.root_lin_vel_w[0, 2])
    common = 0.18 * height_error - 0.09 * vertical_velocity

    xy_error = target_xy - robot.data.root_pos_w[0, :2]
    desired_accel_xy = xy_error - 1.5 * robot.data.root_lin_vel_w[0, :2]
    desired_roll = float(torch.clamp(-desired_accel_xy[1] / 9.81, -0.12, 0.12))
    desired_pitch = float(torch.clamp(desired_accel_xy[0] / 9.81, -0.12, 0.12))
    q = robot.data.root_quat_w[0]
    rates = robot.data.root_ang_vel_b[0]
    roll = float(torch.atan2(2.0 * (q[0] * q[1] + q[2] * q[3]), 1.0 - 2.0 * (q[1] ** 2 + q[2] ** 2)))
    pitch = float(torch.asin(torch.clamp(2.0 * (q[0] * q[2] - q[3] * q[1]), -1.0, 1.0)))
    yaw = _quat_to_yaw(q)
    roll_u = -0.08 * (roll - desired_roll) - 0.015 * float(rates[0])
    pitch_u = -0.08 * (pitch - desired_pitch) - 0.015 * float(rates[1])
    yaw_u = float(np.clip(-0.06 * _wrap_angle(yaw - target_yaw_rad) - 0.03 * float(rates[2]), -0.08, 0.08))
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


def _error(estimate, truth):
    if estimate is None:
        return None
    return {
        "position": float(np.linalg.norm(estimate.position_w_b - truth.position_w_b)),
        "velocity": float(np.linalg.norm(estimate.linear_velocity_w_b - truth.linear_velocity_w_b)),
    }


def _stats(values):
    if not values:
        return {"samples": 0, "rmse": None, "mean": None, "p95": None, "max": None}
    array = np.asarray(values, dtype=np.float64)
    return {
        "samples": int(array.size),
        "rmse": float(np.sqrt(np.mean(array**2))),
        "mean": float(np.mean(array)),
        "p95": float(np.percentile(array, 95)),
        "max": float(np.max(array)),
    }


def main() -> None:
    checkpoint = args_cli.checkpoint.expanduser().resolve()
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)
    output_dir = args_cli.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    trace_path = output_dir / f"{args_cli.profile}_hybrid_trace.csv"
    summary_path = output_dir / f"{args_cli.profile}_hybrid_summary.json"

    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=1)
    env_cfg.learned_motion_checkpoint = str(checkpoint)
    env_cfg.learned_motion_device = args_cli.device
    env_cfg.swift_detector_checkpoint = (
        None if args_cli.detector_checkpoint is None else str(args_cli.detector_checkpoint.expanduser().resolve())
    )
    env_cfg.swift_visibility_checkpoint = (
        None if args_cli.visibility_checkpoint is None else str(args_cli.visibility_checkpoint.expanduser().resolve())
    )
    env_cfg.swift_use_oracle_gate_index = False
    env_cfg.swift_rejection_dump_dir = str(output_dir / "gate_rejections") if args_cli.detector_checkpoint else None
    env_cfg.terminations.collision = None
    env_cfg.terminations.flyaway = None
    env_cfg.episode_length_s = max(float(env_cfg.episode_length_s), args_cli.steps * 0.01 + 2.0)

    env = gym.make(args_cli.task, cfg=env_cfg)
    raw_env = env.unwrapped
    env.reset()
    robot = raw_env.scene["robot"]
    initial_xy = robot.data.root_pos_w[0, :2].clone()
    target_height_m = float(robot.data.root_pos_w[0, 2])
    target_yaw_rad = _quat_to_yaw(robot.data.root_quat_w[0])

    fields = [
        "step", "t_s",
        "truth_px", "truth_py", "truth_pz",
        "raw_px", "raw_py", "raw_pz", "raw_pos_err",
        "learned_px", "learned_py", "learned_pz", "learned_pos_err",
        "gate_px", "gate_py", "gate_pz", "gate_pos_err",
        "raw_vel_err", "learned_vel_err", "gate_vel_err",
        "learned_update_attempted", "learned_update_accepted", "learned_update_rejected",
        "learned_innovation_d2", "learned_correction_norm_m",
    ]
    raw_pos_errors, learned_pos_errors, gate_pos_errors = [], [], []
    raw_vel_errors, learned_vel_errors, gate_vel_errors = [], [], []
    learned_attempted = learned_accepted = learned_rejected = 0

    def put_estimate(row, prefix, estimate, error):
        if estimate is None:
            row.update({f"{prefix}_px": "", f"{prefix}_py": "", f"{prefix}_pz": "", f"{prefix}_pos_err": "", f"{prefix}_vel_err": ""})
            return
        row.update(
            {
                f"{prefix}_px": float(estimate.position_w_b[0]),
                f"{prefix}_py": float(estimate.position_w_b[1]),
                f"{prefix}_pz": float(estimate.position_w_b[2]),
                f"{prefix}_pos_err": error["position"],
                f"{prefix}_vel_err": error["velocity"],
            }
        )

    completed = 0
    try:
        with trace_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for step in range(args_cli.steps):
                if not simulation_app.is_running():
                    break
                action = _controller_action(
                    raw_env,
                    initial_xy,
                    target_height_m,
                    target_yaw_rad,
                    float(raw_env._timestamp_s()),
                )
                _, _, terminated, truncated, _ = env.step(action.unsqueeze(0))
                completed = step + 1
                truth = raw_env._truth_vio_state()
                raw = raw_env.openvins_raw_vio_estimate
                learned = raw_env.openvins_learned_vio_estimate
                gate_fused = raw_env.swift_fused_estimate
                raw_error = _error(raw, truth)
                learned_error = _error(learned, truth)
                gate_error = _error(gate_fused, truth)

                if raw_error:
                    raw_pos_errors.append(raw_error["position"])
                    raw_vel_errors.append(raw_error["velocity"])
                if learned_error:
                    learned_pos_errors.append(learned_error["position"])
                    learned_vel_errors.append(learned_error["velocity"])
                if gate_error:
                    gate_pos_errors.append(gate_error["position"])
                    gate_vel_errors.append(gate_error["velocity"])

                result = raw_env.learned_motion_last_result
                if result is not None:
                    learned_attempted += int(result.learned_update_attempted)
                    learned_accepted += int(result.learned_update_accepted)
                    learned_rejected += int(result.learned_update_rejected)
                correction_norm = ""
                if raw is not None and learned is not None:
                    correction_norm = float(np.linalg.norm(raw.position_w_b - learned.position_w_b))
                row = {
                    "step": step,
                    "t_s": float(raw_env._timestamp_s()),
                    "truth_px": float(truth.position_w_b[0]),
                    "truth_py": float(truth.position_w_b[1]),
                    "truth_pz": float(truth.position_w_b[2]),
                    "learned_update_attempted": 0 if result is None else int(result.learned_update_attempted),
                    "learned_update_accepted": 0 if result is None else int(result.learned_update_accepted),
                    "learned_update_rejected": 0 if result is None else int(result.learned_update_rejected),
                    "learned_innovation_d2": "" if result is None or result.mahalanobis2 is None else float(result.mahalanobis2),
                    "learned_correction_norm_m": correction_norm,
                }
                put_estimate(row, "raw", raw, raw_error)
                put_estimate(row, "learned", learned, learned_error)
                put_estimate(row, "gate", gate_fused, gate_error)
                writer.writerow(row)

                if bool(torch.as_tensor(terminated).any()) or bool(torch.as_tensor(truncated).any()):
                    print("[HybridEval] environment terminated; restart ov_msckf before the next run")
                    break
    finally:
        report = {
            "schema": "isaac_drone_racer.hybrid_openvins_evaluation.v1",
            "checkpoint": str(checkpoint),
            "profile": args_cli.profile,
            "completed_steps": completed,
            "raw_openvins": {"position_error_m": _stats(raw_pos_errors), "velocity_error_mps": _stats(raw_vel_errors)},
            "learned_corrected": {"position_error_m": _stats(learned_pos_errors), "velocity_error_mps": _stats(learned_vel_errors)},
            "gate_fused": {"position_error_m": _stats(gate_pos_errors), "velocity_error_mps": _stats(gate_vel_errors)},
            "learned_updates": {"attempted": learned_attempted, "accepted": learned_accepted, "rejected": learned_rejected},
            "detector_enabled": args_cli.detector_checkpoint is not None,
            "files": {"trace_csv": trace_path.name},
        }
        summary_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"[HybridEval] wrote {trace_path}")
        print(f"[HybridEval] wrote {summary_path}")
        env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
