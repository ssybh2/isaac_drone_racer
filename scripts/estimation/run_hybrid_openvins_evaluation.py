"""Evaluate raw OpenVINS versus learned-motion correction and optional absolute anchors."""

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
    "--profile", choices=("hover", "translate_x", "lissajous", "circle"), default="translate_x"
)
parser.add_argument("--motion_start_s", type=float, default=4.0)
parser.add_argument("--translation_m", type=float, default=1.5)
parser.add_argument("--translation_duration_s", type=float, default=8.0)
parser.add_argument("--lissajous_amplitude_m", type=float, default=0.6)
parser.add_argument("--lissajous_frequency_hz", type=float, default=0.08)
parser.add_argument("--circle_amplitude_m", type=float, default=0.8)
parser.add_argument("--circle_frequency_hz", type=float, default=0.10)
parser.add_argument("--detector_checkpoint", type=Path, default=None)
parser.add_argument("--visibility_checkpoint", type=Path, default=None)
parser.add_argument(
    "--learned_relative_position_only",
    action="store_true",
    help=(
        "Diagnostic A/B mode: learned relative displacement may update position drift "
        "but must not directly change the velocity-drift mean."
    ),
)
parser.add_argument(
    "--learned_drift_velocity_from_displacement",
    action="store_true",
    help=(
        "With --learned_relative_position_only, derive an explicit velocity-drift "
        "measurement from each accepted jump-isolated VIO minus learned displacement window."
    ),
)
parser.add_argument(
    "--learned_drift_velocity_sigma_floor_mps",
    type=float,
    default=0.5,
    help="Minimum 1-sigma uncertainty for the explicit learned drift-velocity measurement.",
)
parser.add_argument(
    "--learned_position_residual_slew",
    action="store_true",
    help=(
        "With --learned_relative_position_only, defer each accepted learned position "
        "residual and release it as a common-mode drift correction at a bounded rate."
    ),
)
parser.add_argument(
    "--learned_position_residual_max_rate_mps",
    type=float,
    default=4.0,
    help="Maximum norm rate used to release deferred learned position drift residuals.",
)
parser.add_argument(
    "--raw_vio_jump_isolation",
    action="store_true",
    help=(
        "Remove implausible sample-to-sample raw OpenVINS position frame shifts "
        "from the internal learned-correction stream while retaining raw VIO for diagnostics."
    ),
)
parser.add_argument(
    "--raw_vio_jump_threshold_m",
    type=float,
    default=0.5,
    help="Detect a raw VIO frame jump when the velocity-predicted position residual exceeds this value.",
)
parser.add_argument(
    "--oracle_absolute_position",
    action="store_true",
    help=(
        "Inject sparse noisy Isaac truth position as an absolute measurement. "
        "Diagnostic upper-bound experiment only; disabled by default."
    ),
)
parser.add_argument("--oracle_position_rate_hz", type=float, default=2.0)
parser.add_argument("--oracle_position_sigma_m", type=float, default=0.10)
parser.add_argument("--oracle_position_seed", type=int, default=0)
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
from estimation.vio_time_buffer import VioWorldEstimateBuffer


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
    if args_cli.profile == "circle":
        omega = 2.0 * np.pi * float(args_cli.circle_frequency_hz)
        amp = float(args_cli.circle_amplitude_m)
        target[0] += amp * float(np.cos(omega * tau) - 1.0)
        target[1] += amp * float(np.sin(omega * tau))
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


def _put_vector(row: dict, prefix: str, value) -> None:
    if value is None:
        row[f"{prefix}_x"] = ""
        row[f"{prefix}_y"] = ""
        row[f"{prefix}_z"] = ""
        return
    vector = np.asarray(value, dtype=np.float64).reshape(3)
    row[f"{prefix}_x"] = float(vector[0])
    row[f"{prefix}_y"] = float(vector[1])
    row[f"{prefix}_z"] = float(vector[2])


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
    env_cfg.learned_motion_relative_position_only = bool(
        args_cli.learned_relative_position_only
    )
    env_cfg.learned_motion_drift_velocity_from_displacement = bool(
        args_cli.learned_drift_velocity_from_displacement
    )
    env_cfg.learned_motion_drift_velocity_sigma_floor_mps = float(
        args_cli.learned_drift_velocity_sigma_floor_mps
    )
    env_cfg.learned_motion_position_residual_slew = bool(
        args_cli.learned_position_residual_slew
    )
    env_cfg.learned_motion_position_residual_max_rate_mps = float(
        args_cli.learned_position_residual_max_rate_mps
    )
    env_cfg.learned_motion_raw_vio_jump_isolation = bool(args_cli.raw_vio_jump_isolation)
    env_cfg.learned_motion_raw_vio_jump_threshold_m = float(args_cli.raw_vio_jump_threshold_m)
    env_cfg.oracle_absolute_position_enabled = bool(args_cli.oracle_absolute_position)
    env_cfg.oracle_position_rate_hz = float(args_cli.oracle_position_rate_hz)
    env_cfg.oracle_position_sigma_m = float(args_cli.oracle_position_sigma_m)
    env_cfg.oracle_position_seed = int(args_cli.oracle_position_seed)
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
    truth_buffer = VioWorldEstimateBuffer(max_age_s=5.0, max_samples=2048)
    truth_buffer.push(raw_env._truth_vio_state())

    fields = [
        "step", "t_s",
        "truth_px", "truth_py", "truth_pz",
        "raw_px", "raw_py", "raw_pz", "raw_pos_err",
        "learned_px", "learned_py", "learned_pz", "learned_pos_err",
        "gate_px", "gate_py", "gate_pz", "gate_pos_err",
        "raw_vel_err", "learned_vel_err", "gate_vel_err",
        "learned_update_attempted", "learned_update_accepted", "learned_update_rejected",
        "learned_innovation_d2", "learned_correction_norm_m",
        "learned_velocity_update_applied", "learned_velocity_d2",
        "learned_velocity_correction_norm_m",
        "learned_position_injection_norm_m", "learned_position_release_norm_m",
        "learned_position_pending_norm_m",
        "prediction_start_s", "prediction_end_s",
        "gt_dp_x", "gt_dp_y", "gt_dp_z",
        "nn_dp_x", "nn_dp_y", "nn_dp_z",
        "vio_dp_x", "vio_dp_y", "vio_dp_z",
        "nn_dp_error_m", "vio_dp_error_m",
        "learned_vd_meas_x", "learned_vd_meas_y", "learned_vd_meas_z",
        "learned_vd_before_x", "learned_vd_before_y", "learned_vd_before_z",
        "learned_vd_after_x", "learned_vd_after_y", "learned_vd_after_z",
        "raw_jump_detected", "raw_jump_residual_m",
        "raw_jump_d_x", "raw_jump_d_y", "raw_jump_d_z",
        "raw_jump_comp_x", "raw_jump_comp_y", "raw_jump_comp_z",
        "oracle_position_update_applied", "oracle_position_d2",
        "oracle_meas_x", "oracle_meas_y", "oracle_meas_z",
        "oracle_vd_before_x", "oracle_vd_before_y", "oracle_vd_before_z",
        "oracle_vd_after_x", "oracle_vd_after_y", "oracle_vd_after_z",
    ]
    raw_pos_errors, learned_pos_errors, gate_pos_errors = [], [], []
    raw_vel_errors, learned_vel_errors, gate_vel_errors = [], [], []
    learned_attempted = learned_accepted = learned_rejected = 0
    learned_velocity_updates = 0
    nn_window_errors, vio_window_errors = [], []

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
                truth_buffer.push(truth)
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
                    learned_velocity_updates += int(result.learned_velocity_update_applied)
                correction_norm = ""
                velocity_correction_norm = ""
                if raw is not None and learned is not None:
                    correction_norm = float(np.linalg.norm(raw.position_w_b - learned.position_w_b))
                    velocity_correction_norm = float(
                        np.linalg.norm(
                            raw.linear_velocity_w_b - learned.linear_velocity_w_b
                        )
                    )

                gt_dp = nn_dp = vio_dp = None
                nn_dp_error = vio_dp_error = ""
                prediction_start = prediction_end = ""
                if (
                    result is not None
                    and result.prediction is not None
                    and result.prediction_start_timestamp_s is not None
                    and result.prediction_end_timestamp_s is not None
                    and result.raw_window_displacement_w is not None
                ):
                    prediction_start = float(result.prediction_start_timestamp_s)
                    prediction_end = float(result.prediction_end_timestamp_s)
                    try:
                        truth_start = truth_buffer.interpolate(prediction_start)
                        truth_end = truth_buffer.interpolate(prediction_end)
                    except ValueError:
                        pass
                    else:
                        gt_dp = truth_end.position_w_b - truth_start.position_w_b
                        nn_dp = np.asarray(result.prediction.displacement_w, dtype=np.float64)
                        vio_dp = np.asarray(result.raw_window_displacement_w, dtype=np.float64)
                        nn_dp_error = float(np.linalg.norm(nn_dp - gt_dp))
                        vio_dp_error = float(np.linalg.norm(vio_dp - gt_dp))
                        nn_window_errors.append(nn_dp_error)
                        vio_window_errors.append(vio_dp_error)

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
                    "learned_velocity_update_applied": 0 if result is None else int(result.learned_velocity_update_applied),
                    "learned_velocity_d2": "" if result is None or result.learned_velocity_mahalanobis2 is None else float(result.learned_velocity_mahalanobis2),
                    "learned_velocity_correction_norm_m": velocity_correction_norm,
                    "learned_position_injection_norm_m": "" if result is None else float(result.learned_position_injection_norm_m),
                    "learned_position_release_norm_m": "" if result is None else float(result.learned_position_release_norm_m),
                    "learned_position_pending_norm_m": "" if result is None else float(result.learned_position_pending_norm_m),
                    "prediction_start_s": prediction_start,
                    "prediction_end_s": prediction_end,
                    "nn_dp_error_m": nn_dp_error,
                    "vio_dp_error_m": vio_dp_error,
                    "raw_jump_detected": 0 if result is None else int(result.raw_vio_jump_detected),
                    "raw_jump_residual_m": "" if result is None or result.raw_vio_jump_residual_m is None else float(result.raw_vio_jump_residual_m),
                    "oracle_position_update_applied": 0 if result is None else int(result.absolute_position_update_applied),
                    "oracle_position_d2": "" if result is None or result.absolute_position_mahalanobis2 is None else float(result.absolute_position_mahalanobis2),
                }
                _put_vector(row, "gt_dp", gt_dp)
                _put_vector(row, "nn_dp", nn_dp)
                _put_vector(row, "vio_dp", vio_dp)
                _put_vector(
                    row,
                    "learned_vd_meas",
                    None if result is None else result.learned_velocity_drift_measurement_w,
                )
                _put_vector(
                    row,
                    "learned_vd_before",
                    None if result is None else result.learned_velocity_drift_before_w,
                )
                _put_vector(
                    row,
                    "learned_vd_after",
                    None if result is None else result.learned_velocity_drift_after_w,
                )
                _put_vector(
                    row,
                    "raw_jump_d",
                    None if result is None else result.raw_vio_jump_residual_w,
                )
                _put_vector(
                    row,
                    "raw_jump_comp",
                    None if result is None else result.raw_vio_jump_compensation_w,
                )
                _put_vector(row, "oracle_meas", raw_env.oracle_position_last_measurement_w if result is not None and result.absolute_position_update_applied else None)
                _put_vector(
                    row,
                    "oracle_vd_before",
                    None if result is None else result.absolute_velocity_drift_before_w,
                )
                _put_vector(
                    row,
                    "oracle_vd_after",
                    None if result is None else result.absolute_velocity_drift_after_w,
                )
                put_estimate(row, "raw", raw, raw_error)
                put_estimate(row, "learned", learned, learned_error)
                put_estimate(row, "gate", gate_fused, gate_error)
                writer.writerow(row)

                if bool(torch.as_tensor(terminated).any()) or bool(torch.as_tensor(truncated).any()):
                    print("[HybridEval] environment terminated; restart ov_msckf before the next run")
                    break
    finally:
        corrector = raw_env.learned_motion_corrector
        jump_count = 0 if corrector is None else int(corrector.raw_vio_jump_count)
        jump_max = 0.0 if corrector is None else float(corrector.raw_vio_jump_max_residual_m)
        jump_total = 0.0 if corrector is None else float(corrector.raw_vio_jump_total_compensation_m)
        jump_final_norm = 0.0 if corrector is None else float(
            np.linalg.norm(corrector.raw_vio_jump_compensation_w)
        )
        position_pending_final_norm = 0.0 if corrector is None else float(
            np.linalg.norm(corrector.pending_position_drift_w)
        )
        report = {
            "schema": "isaac_drone_racer.hybrid_openvins_evaluation.v1",
            "checkpoint": str(checkpoint),
            "profile": args_cli.profile,
            "completed_steps": completed,
            "raw_openvins": {"position_error_m": _stats(raw_pos_errors), "velocity_error_mps": _stats(raw_vel_errors)},
            "learned_corrected": {"position_error_m": _stats(learned_pos_errors), "velocity_error_mps": _stats(learned_vel_errors)},
            "gate_fused": {"position_error_m": _stats(gate_pos_errors), "velocity_error_mps": _stats(gate_vel_errors)},
            "learned_updates": {"attempted": learned_attempted, "accepted": learned_accepted, "rejected": learned_rejected},
            "learned_window_diagnostics": {
                "nn_displacement_error_m": _stats(nn_window_errors),
                "raw_vio_displacement_error_m": _stats(vio_window_errors),
            },
            "learned_relative_position_only": bool(args_cli.learned_relative_position_only),
            "learned_drift_velocity_from_displacement": {
                "enabled": bool(args_cli.learned_drift_velocity_from_displacement),
                "sigma_floor_mps": float(args_cli.learned_drift_velocity_sigma_floor_mps),
                "updates": int(learned_velocity_updates),
            },
            "learned_position_residual_slew": {
                "enabled": bool(args_cli.learned_position_residual_slew),
                "max_rate_mps": float(args_cli.learned_position_residual_max_rate_mps),
                "final_pending_norm_m": position_pending_final_norm,
            },
            "raw_vio_jump_isolation": {
                "enabled": bool(args_cli.raw_vio_jump_isolation),
                "threshold_m": float(args_cli.raw_vio_jump_threshold_m),
                "detected": jump_count,
                "max_residual_m": jump_max,
                "total_compensation_m": jump_total,
                "final_compensation_norm_m": jump_final_norm,
            },
            "oracle_absolute_position": {
                "enabled": bool(args_cli.oracle_absolute_position),
                "rate_hz": float(args_cli.oracle_position_rate_hz),
                "sigma_m": float(args_cli.oracle_position_sigma_m),
                "seed": int(args_cli.oracle_position_seed),
                "updates": int(raw_env.oracle_position_update_count),
            },
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