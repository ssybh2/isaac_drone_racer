"""Single-process worker for one learned-inertial estimator evaluation mode.

This worker intentionally creates exactly one Isaac Lab environment and then
exits. The parent A/B/C orchestrator launches A, B and C in separate processes
so Isaac Sim/SimulationContext is never torn down and recreated inside one
process.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--mode", choices=("A", "S", "B", "P", "C"), required=True)
parser.add_argument("--steps", type=int, default=6000)
parser.add_argument(
    "--profile",
    choices=("hover", "translate_x", "circle", "lissajous"),
    default="lissajous",
)
parser.add_argument("--amplitude_m", type=float, default=1.5)
parser.add_argument("--frequency_hz", type=float, default=0.10)
parser.add_argument("--translation_m", type=float, default=6.0)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--progress-every", type=int, default=100)
parser.add_argument("--imu-noise-seed", type=int, default=0)
parser.add_argument("--imu-accel-white-noise-sigma-mps2", type=float, default=0.0)
parser.add_argument("--imu-gyro-white-noise-sigma-radps", type=float, default=0.0)
parser.add_argument("--imu-accel-initial-bias-sigma-mps2", type=float, default=0.0)
parser.add_argument("--imu-gyro-initial-bias-sigma-radps", type=float, default=0.0)
parser.add_argument("--imu-accel-bias-rw-sigma-mps2-sqrt-s", type=float, default=0.0)
parser.add_argument("--imu-gyro-bias-rw-sigma-radps-sqrt-s", type=float, default=0.0)
parser.add_argument("--ekf-accel-noise-sigma", type=float, default=None)
parser.add_argument("--ekf-gyro-noise-sigma", type=float, default=None)
parser.add_argument("--ekf-accel-bias-rw-sigma", type=float, default=None)
parser.add_argument("--ekf-gyro-bias-rw-sigma", type=float, default=None)
parser.add_argument(
    "--learned-update-rate-hz",
    type=float,
    default=None,
    help="Override learned measurement fusion/evaluation rate for correlation diagnostics.",
)
parser.add_argument(
    "--learned-covariance-multiplier",
    type=float,
    default=None,
    help="Multiply the final learned measurement covariance for fusion-weight diagnostics.",
)
parser.add_argument(
    "--learned-network-bias-m",
    type=float,
    nargs=3,
    metavar=("BX", "BY", "BZ"),
    default=None,
    help=(
        "Offline-calibrated E[prediction-truth] bias [m] in the learned target "
        "frame. This vector is subtracted from network measurements before fusion."
    ),
)
parser.add_argument(
    "--learned-fusion-rate-hz",
    type=float,
    default=None,
    help="Fuse only a subset of learned predictions while preserving the prediction rate.",
)
parser.add_argument(
    "--learned-kalman-gain-mode",
    choices=(
        "full",
        "freeze_attitude_bias",
        "freeze_position",
        "freeze_position_attitude_bias",
        "freeze_clones",
        "freeze_clones_attitude_bias",
        "freeze_kinematic_state",
    ),
    default=None,
    help=(
        "Legacy current-vs-clone Kalman-gain ablation. V6.2 UZH-style "
        "two-clone learned factors always use the full Kalman gain; this option "
        "is retained only for legacy estimator diagnostics."
    ),
)
parser.add_argument(
    "--task",
    default="Isaac-Drone-Racer-Learned-Inertial-v0",
)
parser.add_argument(
    "--learned-checkpoint",
    type=Path,
    default=Path("artifacts/imo_tcn/model_v1.pt"),
)
parser.add_argument(
    "--gate-checkpoint",
    type=Path,
    default=Path(
        "artifacts/stage2_next_steps_20260911/checkpoints/"
        "torchvision_keypointrcnn_best.pt"
    ),
)
parser.add_argument(
    "--visibility-checkpoint",
    type=Path,
    default=Path(
        "artifacts/stage2_next_steps_20260911/checkpoints/"
        "gate_keypoint_net_best.pt"
    ),
)
parser.add_argument("--disable-visibility", action="store_true")
parser.add_argument(
    "--oracle-learned-residual-fusion",
    action="store_true",
    help=(
        "Diagnostic only: keep running the TCN, but fuse exact GT kinematic "
        "residuals to isolate EKF measurement-model correctness."
    ),
)
parser.add_argument(
    "--truth-orientation-for-tcn-features",
    action="store_true",
    help=(
        "Diagnostic only: rotate TCN features with simulator-truth attitude "
        "instead of the EKF attitude."
    ),
)
parser.add_argument("--replay-npz", type=Path, required=True)
parser.add_argument("--output-dir", type=Path, required=True)
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


MODE_LABELS = {
    "A": "imu_only",
    "S": "imu_tcn_shadow",
    "B": "imu_tcn_20hz",
    "P": "imu_tcn_gate_position_only",
    "C": "imu_tcn_gate_pose",
}


def _np(value) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    return np.asarray(value)


def _quat_to_rpy(q) -> tuple[float, float, float]:
    w, x, y, z = [float(v) for v in q]
    roll = np.arctan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch = np.arcsin(np.clip(2.0 * (w * y - z * x), -1.0, 1.0))
    yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return float(roll), float(pitch), float(yaw)


def _quat_to_rotmat(q) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64).reshape(4)
    q = q / np.linalg.norm(q)
    w, x, y, z = q
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _orientation_error_rad(q_est, q_truth) -> float:
    q_est = np.asarray(q_est, dtype=np.float64).reshape(4)
    q_truth = np.asarray(q_truth, dtype=np.float64).reshape(4)
    q_est /= np.linalg.norm(q_est)
    q_truth /= np.linalg.norm(q_truth)
    alignment = float(np.clip(abs(np.dot(q_est, q_truth)), 0.0, 1.0))
    return float(2.0 * np.arccos(alignment))


def _axis_lag_autocorrelation(values: np.ndarray, lag: int) -> list[float | None]:
    values = np.asarray(values, dtype=np.float64).reshape(-1, 3)
    if lag < 1 or len(values) <= lag:
        return [None, None, None]
    result: list[float | None] = []
    for axis in range(3):
        x = values[:-lag, axis]
        y = values[lag:, axis]
        x = x - np.mean(x)
        y = y - np.mean(y)
        denom = float(np.sqrt(np.sum(x * x) * np.sum(y * y)))
        result.append(
            None
            if denom <= 1.0e-15
            else float(np.sum(x * y) / denom)
        )
    return result


def _axis_coverage(normalized: np.ndarray, threshold: float) -> list[float | None]:
    normalized = np.asarray(normalized, dtype=np.float64).reshape(-1, 3)
    result: list[float | None] = []
    for axis in range(3):
        finite = np.isfinite(normalized[:, axis])
        result.append(
            None
            if not np.any(finite)
            else float(
                np.mean(np.abs(normalized[finite, axis]) <= float(threshold))
            )
        )
    return result


def _summarize_fusion_schedule_errors(
    errors: np.ndarray,
    raw_sigmas: np.ndarray,
    used_sigmas: np.ndarray,
) -> dict[str, object]:
    errors = np.asarray(errors, dtype=np.float64).reshape(-1, 3)
    raw_sigmas = np.asarray(raw_sigmas, dtype=np.float64).reshape(-1, 3)
    used_sigmas = np.asarray(used_sigmas, dtype=np.float64).reshape(-1, 3)
    summary: dict[str, object] = {
        "samples": int(len(errors)),
        "axis_rmse_m": None,
        "axis_bias_m": None,
        "norm_mean_m": None,
        "norm_rmse_m": None,
        "norm_max_m": None,
        "lag1_axis_autocorr": [None, None, None],
        "lag2_axis_autocorr": [None, None, None],
        "cumulative_error_final_m": None,
        "cumulative_error_max_norm_m": None,
        "raw_sigma_mean_m": None,
        "used_sigma_mean_m": None,
        "raw_normalized_axis_rmse": None,
        "used_normalized_axis_rmse": None,
        "raw_axis_coverage_leq_1sigma": None,
        "raw_axis_coverage_leq_2sigma": None,
        "raw_axis_coverage_leq_3sigma": None,
        "used_axis_coverage_leq_1sigma": None,
        "used_axis_coverage_leq_2sigma": None,
        "used_axis_coverage_leq_3sigma": None,
    }
    if not len(errors):
        return summary

    error_norm = np.linalg.norm(errors, axis=1)
    cumulative = np.cumsum(errors, axis=0)
    cumulative_norm = np.linalg.norm(cumulative, axis=1)
    raw_normalized = np.divide(
        errors,
        raw_sigmas,
        out=np.full_like(errors, np.nan),
        where=raw_sigmas > 1.0e-12,
    )
    used_normalized = np.divide(
        errors,
        used_sigmas,
        out=np.full_like(errors, np.nan),
        where=used_sigmas > 1.0e-12,
    )

    summary.update(
        {
            "axis_rmse_m": np.sqrt(np.mean(errors**2, axis=0)).tolist(),
            "axis_bias_m": np.mean(errors, axis=0).tolist(),
            "norm_mean_m": float(np.mean(error_norm)),
            "norm_rmse_m": float(np.sqrt(np.mean(error_norm**2))),
            "norm_max_m": float(np.max(error_norm)),
            "lag1_axis_autocorr": _axis_lag_autocorrelation(errors, 1),
            "lag2_axis_autocorr": _axis_lag_autocorrelation(errors, 2),
            "cumulative_error_final_m": cumulative[-1].tolist(),
            "cumulative_error_max_norm_m": float(np.max(cumulative_norm)),
            "raw_sigma_mean_m": np.mean(raw_sigmas, axis=0).tolist(),
            "used_sigma_mean_m": np.mean(used_sigmas, axis=0).tolist(),
            "raw_normalized_axis_rmse": np.sqrt(
                np.nanmean(raw_normalized**2, axis=0)
            ).tolist(),
            "used_normalized_axis_rmse": np.sqrt(
                np.nanmean(used_normalized**2, axis=0)
            ).tolist(),
            "raw_axis_coverage_leq_1sigma": _axis_coverage(raw_normalized, 1.0),
            "raw_axis_coverage_leq_2sigma": _axis_coverage(raw_normalized, 2.0),
            "raw_axis_coverage_leq_3sigma": _axis_coverage(raw_normalized, 3.0),
            "used_axis_coverage_leq_1sigma": _axis_coverage(used_normalized, 1.0),
            "used_axis_coverage_leq_2sigma": _axis_coverage(used_normalized, 2.0),
            "used_axis_coverage_leq_3sigma": _axis_coverage(used_normalized, 3.0),
        }
    )
    return summary


def _target_xy(initial_xy: torch.Tensor, t_s: float, total_duration_s: float) -> torch.Tensor:
    target = initial_xy.clone()
    amp = float(args_cli.amplitude_m)
    omega = 2.0 * np.pi * float(args_cli.frequency_hz)

    if args_cli.profile == "hover":
        return target
    if args_cli.profile == "translate_x":
        duration = max(total_duration_s * 0.7, 1.0e-6)
        target[0] += float(args_cli.translation_m) * np.clip(t_s / duration, 0.0, 1.0)
        return target
    if args_cli.profile == "circle":
        target[0] += amp * (float(np.cos(omega * t_s)) - 1.0)
        target[1] += amp * float(np.sin(omega * t_s))
        return target

    target[0] += amp * float(np.sin(omega * t_s))
    target[1] += 0.7 * amp * float(np.sin(2.0 * omega * t_s + 0.35))
    return target


def _controller_action(
    raw_env,
    initial_xy: torch.Tensor,
    target_height_m: float,
    target_yaw_rad: float,
    t_s: float,
    total_duration_s: float,
) -> torch.Tensor:
    """GT-feedback trajectory generator used only to create Mode-A replay actions."""
    robot = raw_env.scene["robot"]
    target_xy = _target_xy(initial_xy, t_s, total_duration_s)

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


def _prepare_cfg():
    cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=1)
    cfg.scene.num_envs = 1
    cfg.episode_length_s = max(float(cfg.episode_length_s), args_cli.steps * 0.01 + 10.0)
    cfg.learned_debug_oracle_residual_fusion = bool(
        args_cli.oracle_learned_residual_fusion
    )
    cfg.learned_debug_truth_orientation_for_features = bool(
        args_cli.truth_orientation_for_tcn_features
    )
    cfg.terminations.collision = None
    cfg.terminations.flyaway = None
    cfg.commands.target.randomise_start = None
    cfg.events.push_robot = None
    noise_values = (
        args_cli.imu_accel_white_noise_sigma_mps2,
        args_cli.imu_gyro_white_noise_sigma_radps,
        args_cli.imu_accel_initial_bias_sigma_mps2,
        args_cli.imu_gyro_initial_bias_sigma_radps,
        args_cli.imu_accel_bias_rw_sigma_mps2_sqrt_s,
        args_cli.imu_gyro_bias_rw_sigma_radps_sqrt_s,
    )
    if any(value < 0.0 for value in noise_values):
        raise ValueError("IMU corruption sigmas must be non-negative")
    cfg.imu_noise_seed = int(args_cli.imu_noise_seed)
    cfg.imu_accel_white_noise_sigma_mps2 = float(
        args_cli.imu_accel_white_noise_sigma_mps2
    )
    cfg.imu_gyro_white_noise_sigma_radps = float(
        args_cli.imu_gyro_white_noise_sigma_radps
    )
    cfg.imu_accel_initial_bias_sigma_mps2 = float(
        args_cli.imu_accel_initial_bias_sigma_mps2
    )
    cfg.imu_gyro_initial_bias_sigma_radps = float(
        args_cli.imu_gyro_initial_bias_sigma_radps
    )
    cfg.imu_accel_bias_rw_sigma_mps2_sqrt_s = float(
        args_cli.imu_accel_bias_rw_sigma_mps2_sqrt_s
    )
    cfg.imu_gyro_bias_rw_sigma_radps_sqrt_s = float(
        args_cli.imu_gyro_bias_rw_sigma_radps_sqrt_s
    )
    if args_cli.ekf_accel_noise_sigma is not None:
        cfg.ekf_accel_noise_sigma = float(args_cli.ekf_accel_noise_sigma)
    if args_cli.ekf_gyro_noise_sigma is not None:
        cfg.ekf_gyro_noise_sigma = float(args_cli.ekf_gyro_noise_sigma)
    if args_cli.ekf_accel_bias_rw_sigma is not None:
        cfg.ekf_accel_bias_rw_sigma = float(args_cli.ekf_accel_bias_rw_sigma)
    if args_cli.ekf_gyro_bias_rw_sigma is not None:
        cfg.ekf_gyro_bias_rw_sigma = float(args_cli.ekf_gyro_bias_rw_sigma)

    if args_cli.learned_update_rate_hz is not None:
        if args_cli.learned_update_rate_hz <= 0.0:
            raise ValueError("--learned-update-rate-hz must be positive")
        cfg.learned_update_rate_hz = float(args_cli.learned_update_rate_hz)
    if args_cli.learned_covariance_multiplier is not None:
        if args_cli.learned_covariance_multiplier <= 0.0:
            raise ValueError("--learned-covariance-multiplier must be positive")
        cfg.learned_measurement_covariance_multiplier = float(
            args_cli.learned_covariance_multiplier
        )
    if args_cli.learned_network_bias_m is not None:
        bias = np.asarray(args_cli.learned_network_bias_m, dtype=np.float64)
        if bias.shape != (3,) or not np.all(np.isfinite(bias)):
            raise ValueError("--learned-network-bias-m requires three finite values")
        cfg.learned_network_bias_m = tuple(float(v) for v in bias)
    if args_cli.learned_fusion_rate_hz is not None:
        if args_cli.learned_fusion_rate_hz <= 0.0:
            raise ValueError("--learned-fusion-rate-hz must be positive")
        cfg.learned_fusion_rate_hz = float(args_cli.learned_fusion_rate_hz)
    if args_cli.learned_kalman_gain_mode is not None:
        cfg.learned_kalman_gain_mode = str(args_cli.learned_kalman_gain_mode)

    if args_cli.mode == "A":
        cfg.learned_motion_checkpoint = None
        cfg.swift_detector_checkpoint = None
        cfg.swift_visibility_checkpoint = None
    else:
        cfg.learned_motion_checkpoint = str(args_cli.learned_checkpoint.expanduser().resolve())
        cfg.learned_apply_displacement_updates = args_cli.mode != "S"
        if args_cli.mode in ("P", "C"):
            cfg.swift_detector_checkpoint = str(args_cli.gate_checkpoint.expanduser().resolve())
            cfg.swift_visibility_checkpoint = (
                None
                if args_cli.disable_visibility
                else str(args_cli.visibility_checkpoint.expanduser().resolve())
            )
            cfg.gate_use_orientation_update = args_cli.mode == "C"
        else:
            cfg.swift_detector_checkpoint = None
            cfg.swift_visibility_checkpoint = None
    return cfg


CSV_FIELDS = [
    "mode", "mode_label", "step", "t_s", "profile",
    "action_m1", "action_m2", "action_m3", "action_m4",
    "truth_px", "truth_py", "truth_pz",
    "est_px", "est_py", "est_pz",
    "pos_err_x", "pos_err_y", "pos_err_z", "pos_err_norm_m",
    "truth_vx", "truth_vy", "truth_vz",
    "est_vx", "est_vy", "est_vz",
    "vel_err_x", "vel_err_y", "vel_err_z", "vel_err_norm_mps",
    "truth_qw", "truth_qx", "truth_qy", "truth_qz",
    "est_qw", "est_qx", "est_qy", "est_qz",
    "orientation_error_deg",
    "learned_updates", "learned_update_skips", "clone_count",
    "tcn_target_mode", "learned_measurement_source",
    "tcn_innovation_x", "tcn_innovation_y", "tcn_innovation_z",
    "tcn_innovation_norm_m",
    "tcn_pred_x", "tcn_pred_y", "tcn_pred_z",
    "tcn_gt_dp_x", "tcn_gt_dp_y", "tcn_gt_dp_z",
    "tcn_pred_err_x", "tcn_pred_err_y", "tcn_pred_err_z",
    "tcn_pred_err_norm_m",
    "learned_fused_this_update",
    "tcn_raw_sigma_x", "tcn_raw_sigma_y", "tcn_raw_sigma_z",
    "tcn_used_sigma_x", "tcn_used_sigma_y", "tcn_used_sigma_z",
    "tcn_used_normalized_err_x", "tcn_used_normalized_err_y",
    "tcn_used_normalized_err_z",
    "gate_attempts", "gate_accepted", "gate_rejected",
]


def _validate_inputs() -> None:
    if args_cli.steps < 1:
        raise ValueError("--steps must be positive")
    if args_cli.progress_every < 1:
        raise ValueError("--progress-every must be positive")
    if args_cli.mode in ("S", "B", "P", "C") and not args_cli.learned_checkpoint.expanduser().exists():
        raise FileNotFoundError(f"learned checkpoint not found: {args_cli.learned_checkpoint}")
    if args_cli.mode in ("P", "C") and not args_cli.gate_checkpoint.expanduser().exists():
        raise FileNotFoundError(f"gate checkpoint not found: {args_cli.gate_checkpoint}")
    if (
        args_cli.mode in ("P", "C")
        and not args_cli.disable_visibility
        and not args_cli.visibility_checkpoint.expanduser().exists()
    ):
        raise FileNotFoundError(
            f"visibility checkpoint not found: {args_cli.visibility_checkpoint}"
        )
    if args_cli.mode in ("S", "B", "P", "C") and not args_cli.replay_npz.expanduser().exists():
        raise FileNotFoundError(f"replay file not found: {args_cli.replay_npz}")


def _load_replay():
    if args_cli.mode == "A":
        return None, None
    replay = np.load(args_cli.replay_npz.expanduser().resolve())
    return (
        np.asarray(replay["actions"], dtype=np.float64),
        np.asarray(replay["truth_positions"], dtype=np.float64),
    )


def main() -> None:
    _validate_inputs()
    cfg = _prepare_cfg()
    output_dir = args_cli.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    trace_path = output_dir / f"mode_{args_cli.mode}_trace.csv"
    summary_path = output_dir / f"mode_{args_cli.mode}_summary.json"

    replay_actions, reference_truth_positions = _load_replay()
    env = gym.make(args_cli.task, cfg=cfg)
    raw_env = env.unwrapped

    generated_actions: list[np.ndarray] = []
    truth_positions: list[np.ndarray] = []
    position_errors: list[np.ndarray] = []
    velocity_errors: list[np.ndarray] = []
    orientation_errors_rad: list[float] = []
    clone_counts: list[int] = []
    innovation_vectors: list[np.ndarray] = []
    prediction_errors: list[np.ndarray] = []
    fusion_schedule_prediction_errors: list[np.ndarray] = []
    fusion_schedule_raw_sigmas: list[np.ndarray] = []
    fusion_schedule_used_sigmas: list[np.ndarray] = []
    accel_bias_norms: list[float] = []
    gyro_bias_norms: list[float] = []
    update_nis: list[float] = []
    update_dx_theta_norm: list[float] = []
    update_dx_velocity_norm: list[float] = []
    update_dx_position_norm: list[float] = []
    update_dx_accel_bias_norm: list[float] = []
    update_dx_gyro_bias_norm: list[float] = []
    update_dx_clone_velocity_norm: list[float] = []
    update_dx_clone_position_norm: list[float] = []
    update_k_raw_current_norm: list[float] = []
    update_k_current_norm: list[float] = []
    update_k_theta_norm: list[float] = []
    update_k_velocity_norm: list[float] = []
    update_k_position_norm: list[float] = []
    update_k_accel_bias_norm: list[float] = []
    update_k_gyro_bias_norm: list[float] = []
    update_k_clone_velocity_norm: list[float] = []
    update_k_clone_position_norm: list[float] = []

    try:
        env.reset(seed=int(args_cli.seed))
        robot = raw_env.scene["robot"]
        initial_xy = robot.data.root_pos_w[0, :2].clone()
        target_height_m = float(robot.data.root_pos_w[0, 2])
        _, _, target_yaw_rad = _quat_to_rpy(robot.data.root_quat_w[0])

        start_timestamp_s = float(raw_env._timestamp_s())
        step_dt = float(raw_env.step_dt)
        total_duration_s = float(args_cli.steps) * step_dt
        last_seen_innovation_timestamp = None
        last_seen_update_diag = None
        truth_position_by_time = {
            round(start_timestamp_s, 6): _np(robot.data.root_pos_w[0]).astype(np.float64).copy()
        }
        truth_velocity_by_time = {
            round(start_timestamp_s, 6): _np(robot.data.root_lin_vel_w[0]).astype(np.float64).copy()
        }
        truth_orientation_by_time = {
            round(start_timestamp_s, 6): _np(robot.data.root_quat_w[0]).astype(np.float64).copy()
        }

        with trace_path.open("w", newline="", encoding="utf-8", buffering=1) as handle:
            writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
            writer.writeheader()

            for step in range(int(args_cli.steps)):
                if not simulation_app.is_running():
                    break

                elapsed_before_step = float(raw_env._timestamp_s()) - start_timestamp_s
                if args_cli.mode == "A":
                    action = _controller_action(
                        raw_env,
                        initial_xy,
                        target_height_m,
                        target_yaw_rad,
                        elapsed_before_step,
                        total_duration_s,
                    )
                    generated_actions.append(_np(action).astype(np.float64).copy())
                else:
                    if step >= len(replay_actions):
                        raise RuntimeError("replay action sequence is shorter than requested evaluation")
                    action = torch.as_tensor(
                        replay_actions[step],
                        dtype=torch.float32,
                        device=raw_env.device,
                    )

                env.step(action.unsqueeze(0))

                state = raw_env.learned_inertial_state
                truth_p = _np(robot.data.root_pos_w[0]).astype(np.float64)
                truth_v = _np(robot.data.root_lin_vel_w[0]).astype(np.float64)
                truth_q = _np(robot.data.root_quat_w[0]).astype(np.float64)
                est_p = np.asarray(state.position_w_b, dtype=np.float64)
                est_v = np.asarray(state.linear_velocity_w_b, dtype=np.float64)
                est_q = np.asarray(state.orientation_w_b_wxyz, dtype=np.float64)

                pos_err = est_p - truth_p
                vel_err = est_v - truth_v
                ori_err_rad = _orientation_error_rad(est_q, truth_q)

                truth_positions.append(truth_p.copy())
                position_errors.append(pos_err.copy())
                velocity_errors.append(vel_err.copy())
                orientation_errors_rad.append(ori_err_rad)
                clone_counts.append(int(raw_env._lio.clone_count))
                accel_bias_norms.append(float(np.linalg.norm(state.accel_bias_b)))
                gyro_bias_norms.append(float(np.linalg.norm(state.gyro_bias_b)))

                diag = getattr(raw_env._lio, "last_update_diagnostics", None)
                if diag is not None and diag is not last_seen_update_diag:
                    last_seen_update_diag = diag
                    update_nis.append(float(diag["nis"]))
                    update_dx_theta_norm.append(float(np.linalg.norm(diag["dx_theta"])))
                    update_dx_velocity_norm.append(float(np.linalg.norm(diag["dx_velocity"])))
                    update_dx_position_norm.append(float(np.linalg.norm(diag["dx_position"])))
                    update_dx_accel_bias_norm.append(float(np.linalg.norm(diag["dx_accel_bias"])))
                    update_dx_gyro_bias_norm.append(float(np.linalg.norm(diag["dx_gyro_bias"])))
                    update_dx_clone_velocity_norm.append(
                        float(diag["dx_clone_velocity_norm"])
                    )
                    update_dx_clone_position_norm.append(
                        float(diag["dx_clone_position_norm"])
                    )
                    update_k_raw_current_norm.append(
                        float(diag["kalman_gain_raw_current_norm"])
                    )
                    update_k_current_norm.append(
                        float(diag["kalman_gain_current_norm"])
                    )
                    update_k_theta_norm.append(
                        float(diag["kalman_gain_theta_norm"])
                    )
                    update_k_velocity_norm.append(
                        float(diag["kalman_gain_velocity_norm"])
                    )
                    update_k_position_norm.append(
                        float(diag["kalman_gain_position_norm"])
                    )
                    update_k_accel_bias_norm.append(
                        float(diag["kalman_gain_accel_bias_norm"])
                    )
                    update_k_gyro_bias_norm.append(
                        float(diag["kalman_gain_gyro_bias_norm"])
                    )
                    update_k_clone_velocity_norm.append(
                        float(diag["kalman_gain_clone_velocity_norm"])
                    )
                    update_k_clone_position_norm.append(
                        float(diag["kalman_gain_clone_position_norm"])
                    )

                timestamp_key = round(float(raw_env._timestamp_s()), 6)
                truth_position_by_time[timestamp_key] = truth_p.copy()
                truth_velocity_by_time[timestamp_key] = truth_v.copy()
                truth_orientation_by_time[timestamp_key] = truth_q.copy()

                innovation = None
                tcn_prediction = None
                tcn_gt_dp = None
                tcn_pred_error = None
                learned_fused_this_update = False
                tcn_raw_sigma = None
                tcn_used_sigma = None
                tcn_used_normalized_error = None
                innovation_timestamp = raw_env._last_learned_update_timestamp_s
                if (
                    innovation_timestamp is not None
                    and innovation_timestamp != last_seen_innovation_timestamp
                    and raw_env._last_learned_innovation_w is not None
                ):
                    innovation = np.asarray(
                        raw_env._last_learned_innovation_w,
                        dtype=np.float64,
                    ).copy()
                    innovation_vectors.append(innovation)
                    last_seen_innovation_timestamp = innovation_timestamp

                    if raw_env._last_learned_prediction_w is not None:
                        tcn_prediction = np.asarray(
                            raw_env._last_learned_prediction_w,
                            dtype=np.float64,
                        ).copy()
                        start_key = round(float(raw_env._last_learned_window_start_s), 6)
                        end_key = round(float(raw_env._last_learned_window_end_s), 6)
                        if start_key in truth_position_by_time and end_key in truth_position_by_time:
                            tcn_gt_dp = (
                                truth_position_by_time[end_key]
                                - truth_position_by_time[start_key]
                            )
                            target_mode = str(
                                getattr(
                                    raw_env._motion_predictor,
                                    "target_mode",
                                    "displacement",
                                )
                            )
                            if target_mode in (
                                "kinematic_residual",
                                "kinematic_residual_body_end",
                                "kinematic_residual_body_end_gyro_aligned",
                                "kinematic_residual_body_end_gravity_compensated",
                            ):
                                if start_key not in truth_velocity_by_time:
                                    raise RuntimeError(
                                        "missing truth start velocity for residual diagnostic"
                                    )
                                window_dt = float(
                                    raw_env._last_learned_window_end_s
                                    - raw_env._last_learned_window_start_s
                                )
                                tcn_gt_dp = (
                                    tcn_gt_dp
                                    - truth_velocity_by_time[start_key] * window_dt
                                )
                            if target_mode == "kinematic_residual_body_end_gravity_compensated":
                                gravity_w = np.array([0.0, 0.0, -9.81], dtype=np.float64)
                                tcn_gt_dp = (
                                    tcn_gt_dp
                                    - 0.5 * gravity_w * window_dt * window_dt
                                )
                            if target_mode in (
                                "kinematic_residual_body_end",
                                "kinematic_residual_body_end_gyro_aligned",
                                "kinematic_residual_body_end_gravity_compensated",
                            ):
                                if end_key not in truth_orientation_by_time:
                                    raise RuntimeError(
                                        "missing truth endpoint attitude for body residual diagnostic"
                                    )
                                R_end_gt = _quat_to_rotmat(
                                    truth_orientation_by_time[end_key]
                                )
                                tcn_gt_dp = R_end_gt.T @ tcn_gt_dp
                            tcn_pred_error = tcn_prediction - tcn_gt_dp
                            prediction_errors.append(tcn_pred_error.copy())

                            learned_fused_this_update = bool(
                                getattr(raw_env, "_last_learned_fused", False)
                            )
                            if learned_fused_this_update:
                                raw_cov = np.asarray(
                                    raw_env._last_learned_covariance_raw_w,
                                    dtype=np.float64,
                                ).reshape(3, 3)
                                used_cov = np.asarray(
                                    raw_env._last_learned_covariance_used_w,
                                    dtype=np.float64,
                                ).reshape(3, 3)
                                tcn_raw_sigma = np.sqrt(
                                    np.maximum(np.diag(raw_cov), 0.0)
                                )
                                tcn_used_sigma = np.sqrt(
                                    np.maximum(np.diag(used_cov), 0.0)
                                )
                                tcn_used_normalized_error = np.divide(
                                    tcn_pred_error,
                                    tcn_used_sigma,
                                    out=np.full(3, np.nan, dtype=np.float64),
                                    where=tcn_used_sigma > 1.0e-12,
                                )
                                fusion_schedule_prediction_errors.append(
                                    tcn_pred_error.copy()
                                )
                                fusion_schedule_raw_sigmas.append(
                                    tcn_raw_sigma.copy()
                                )
                                fusion_schedule_used_sigmas.append(
                                    tcn_used_sigma.copy()
                                )

                action_np = _np(action).astype(np.float64)
                t_s = float(raw_env._timestamp_s()) - start_timestamp_s
                writer.writerow(
                    {
                        "mode": args_cli.mode,
                        "mode_label": MODE_LABELS[args_cli.mode],
                        "step": step,
                        "t_s": t_s,
                        "profile": args_cli.profile,
                        "action_m1": action_np[0],
                        "action_m2": action_np[1],
                        "action_m3": action_np[2],
                        "action_m4": action_np[3],
                        "truth_px": truth_p[0],
                        "truth_py": truth_p[1],
                        "truth_pz": truth_p[2],
                        "est_px": est_p[0],
                        "est_py": est_p[1],
                        "est_pz": est_p[2],
                        "pos_err_x": pos_err[0],
                        "pos_err_y": pos_err[1],
                        "pos_err_z": pos_err[2],
                        "pos_err_norm_m": float(np.linalg.norm(pos_err)),
                        "truth_vx": truth_v[0],
                        "truth_vy": truth_v[1],
                        "truth_vz": truth_v[2],
                        "est_vx": est_v[0],
                        "est_vy": est_v[1],
                        "est_vz": est_v[2],
                        "vel_err_x": vel_err[0],
                        "vel_err_y": vel_err[1],
                        "vel_err_z": vel_err[2],
                        "vel_err_norm_mps": float(np.linalg.norm(vel_err)),
                        "truth_qw": truth_q[0],
                        "truth_qx": truth_q[1],
                        "truth_qy": truth_q[2],
                        "truth_qz": truth_q[3],
                        "est_qw": est_q[0],
                        "est_qx": est_q[1],
                        "est_qy": est_q[2],
                        "est_qz": est_q[3],
                        "orientation_error_deg": float(np.degrees(ori_err_rad)),
                        "learned_updates": int(raw_env._learned_update_count),
                        "learned_update_skips": int(raw_env._learned_update_skip_count),
                        "clone_count": int(raw_env._lio.clone_count),
                        "tcn_target_mode": (
                            None
                            if raw_env._motion_predictor is None
                            else str(
                                getattr(
                                    raw_env._motion_predictor,
                                    "target_mode",
                                    "displacement",
                                )
                            )
                        ),
                        "learned_measurement_source": (
                            raw_env._last_learned_measurement_source
                        ),
                        "tcn_innovation_x": None if innovation is None else innovation[0],
                        "tcn_innovation_y": None if innovation is None else innovation[1],
                        "tcn_innovation_z": None if innovation is None else innovation[2],
                        "tcn_innovation_norm_m": (
                            None if innovation is None else float(np.linalg.norm(innovation))
                        ),
                        "tcn_pred_x": None if tcn_prediction is None else tcn_prediction[0],
                        "tcn_pred_y": None if tcn_prediction is None else tcn_prediction[1],
                        "tcn_pred_z": None if tcn_prediction is None else tcn_prediction[2],
                        "tcn_gt_dp_x": None if tcn_gt_dp is None else tcn_gt_dp[0],
                        "tcn_gt_dp_y": None if tcn_gt_dp is None else tcn_gt_dp[1],
                        "tcn_gt_dp_z": None if tcn_gt_dp is None else tcn_gt_dp[2],
                        "tcn_pred_err_x": None if tcn_pred_error is None else tcn_pred_error[0],
                        "tcn_pred_err_y": None if tcn_pred_error is None else tcn_pred_error[1],
                        "tcn_pred_err_z": None if tcn_pred_error is None else tcn_pred_error[2],
                        "tcn_pred_err_norm_m": (
                            None if tcn_pred_error is None else float(np.linalg.norm(tcn_pred_error))
                        ),
                        "learned_fused_this_update": int(learned_fused_this_update),
                        "tcn_raw_sigma_x": None if tcn_raw_sigma is None else tcn_raw_sigma[0],
                        "tcn_raw_sigma_y": None if tcn_raw_sigma is None else tcn_raw_sigma[1],
                        "tcn_raw_sigma_z": None if tcn_raw_sigma is None else tcn_raw_sigma[2],
                        "tcn_used_sigma_x": None if tcn_used_sigma is None else tcn_used_sigma[0],
                        "tcn_used_sigma_y": None if tcn_used_sigma is None else tcn_used_sigma[1],
                        "tcn_used_sigma_z": None if tcn_used_sigma is None else tcn_used_sigma[2],
                        "tcn_used_normalized_err_x": (
                            None
                            if tcn_used_normalized_error is None
                            else tcn_used_normalized_error[0]
                        ),
                        "tcn_used_normalized_err_y": (
                            None
                            if tcn_used_normalized_error is None
                            else tcn_used_normalized_error[1]
                        ),
                        "tcn_used_normalized_err_z": (
                            None
                            if tcn_used_normalized_error is None
                            else tcn_used_normalized_error[2]
                        ),
                        "gate_attempts": int(raw_env._gate_attempt_count),
                        "gate_accepted": int(raw_env._gate_update_count),
                        "gate_rejected": int(raw_env._gate_reject_count),
                    }
                )

                if (
                    (step + 1) % int(args_cli.progress_every) == 0
                    or step + 1 == int(args_cli.steps)
                ):
                    print(
                        f"[estimator-ab:{args_cli.mode}] step={step + 1}/{args_cli.steps} "
                        f"learned={raw_env._learned_update_count} "
                        f"fused={raw_env._learned_fusion_count} "
                        f"clones={raw_env._lio.clone_count} "
                        f"skips={raw_env._learned_update_skip_count} "
                        f"gate={raw_env._gate_update_count}/{raw_env._gate_attempt_count} "
                        f"rejected={raw_env._gate_reject_count}",
                        flush=True,
                    )

        if not position_errors:
            raise RuntimeError(f"mode {args_cli.mode} produced no evaluation samples")

        pos = np.asarray(position_errors, dtype=np.float64)
        vel = np.asarray(velocity_errors, dtype=np.float64)
        ori = np.asarray(orientation_errors_rad, dtype=np.float64)
        clones = np.asarray(clone_counts, dtype=np.float64)
        truth = np.asarray(truth_positions, dtype=np.float64)
        innovations = (
            np.asarray(innovation_vectors, dtype=np.float64)
            if innovation_vectors
            else np.empty((0, 3), dtype=np.float64)
        )
        pred_errors = (
            np.asarray(prediction_errors, dtype=np.float64)
            if prediction_errors
            else np.empty((0, 3), dtype=np.float64)
        )
        fusion_pred_errors = (
            np.asarray(fusion_schedule_prediction_errors, dtype=np.float64)
            if fusion_schedule_prediction_errors
            else np.empty((0, 3), dtype=np.float64)
        )
        fusion_raw_sigmas = (
            np.asarray(fusion_schedule_raw_sigmas, dtype=np.float64)
            if fusion_schedule_raw_sigmas
            else np.empty((0, 3), dtype=np.float64)
        )
        fusion_used_sigmas = (
            np.asarray(fusion_schedule_used_sigmas, dtype=np.float64)
            if fusion_schedule_used_sigmas
            else np.empty((0, 3), dtype=np.float64)
        )

        duration_s = len(pos) * step_dt
        learned_updates = int(raw_env._learned_update_count)
        learned_fusions = int(raw_env._learned_fusion_count)
        warm_duration_s = max(duration_s - float(cfg.learned_window_time_s), 1.0e-12)

        replay_position_max_diff_m = None
        replay_position_rmse_m = None
        if reference_truth_positions is not None:
            n = min(len(reference_truth_positions), len(truth))
            delta = truth[:n] - reference_truth_positions[:n]
            replay_norm = np.linalg.norm(delta, axis=1)
            replay_position_max_diff_m = float(np.max(replay_norm))
            replay_position_rmse_m = float(np.sqrt(np.mean(replay_norm**2)))

        innovation_summary = {
            "samples": int(len(innovations)),
            "axis_rmse_m": None,
            "axis_bias_m": None,
            "norm_mean_m": None,
            "norm_rmse_m": None,
            "norm_max_m": None,
        }
        if len(innovations):
            innovation_norm = np.linalg.norm(innovations, axis=1)
            innovation_summary.update(
                {
                    "axis_rmse_m": np.sqrt(np.mean(innovations**2, axis=0)).tolist(),
                    "axis_bias_m": np.mean(innovations, axis=0).tolist(),
                    "norm_mean_m": float(np.mean(innovation_norm)),
                    "norm_rmse_m": float(np.sqrt(np.mean(innovation_norm**2))),
                    "norm_max_m": float(np.max(innovation_norm)),
                }
            )

        prediction_summary = {
            "samples": int(len(pred_errors)),
            "axis_rmse_m": None,
            "axis_bias_m": None,
            "norm_rmse_m": None,
            "norm_mean_m": None,
            "norm_max_m": None,
        }
        if len(pred_errors):
            pred_norm = np.linalg.norm(pred_errors, axis=1)
            prediction_summary.update(
                {
                    "axis_rmse_m": np.sqrt(np.mean(pred_errors**2, axis=0)).tolist(),
                    "axis_bias_m": np.mean(pred_errors, axis=0).tolist(),
                    "norm_rmse_m": float(np.sqrt(np.mean(pred_norm**2))),
                    "norm_mean_m": float(np.mean(pred_norm)),
                    "norm_max_m": float(np.max(pred_norm)),
                }
            )

        fusion_schedule_prediction_summary = _summarize_fusion_schedule_errors(
            fusion_pred_errors,
            fusion_raw_sigmas,
            fusion_used_sigmas,
        )

        gate_attempts = int(raw_env._gate_attempt_count)
        gate_accepted = int(raw_env._gate_update_count)
        gate_rejected = int(raw_env._gate_reject_count)

        tail_count = max(1, int(round(0.2 * len(pos))))
        pos_norm = np.linalg.norm(pos, axis=1)
        vel_norm = np.linalg.norm(vel, axis=1)
        summary = {
            "mode": args_cli.mode,
            "label": MODE_LABELS[args_cli.mode],
            "tcn_target_mode": (
                None
                if raw_env._motion_predictor is None
                else str(
                    getattr(
                        raw_env._motion_predictor,
                        "target_mode",
                        "displacement",
                    )
                )
            ),
            "tcn_feature_frame": (
                None
                if raw_env._motion_predictor is None
                else str(getattr(raw_env._motion_predictor, "feature_frame", "world"))
            ),
            "oracle_learned_residual_fusion": bool(
                cfg.learned_debug_oracle_residual_fusion
            ),
            "last_learned_measurement_source": (
                raw_env._last_learned_measurement_source
            ),
            "truth_orientation_for_tcn_features": bool(
                cfg.learned_debug_truth_orientation_for_features
            ),
            "samples": int(len(pos)),
            "duration_s": float(duration_s),
            "imu_corruption": {
                "seed": int(cfg.imu_noise_seed),
                "accel_white_noise_sigma_mps2": float(
                    cfg.imu_accel_white_noise_sigma_mps2
                ),
                "gyro_white_noise_sigma_radps": float(
                    cfg.imu_gyro_white_noise_sigma_radps
                ),
                "accel_initial_bias_sigma_mps2": float(
                    cfg.imu_accel_initial_bias_sigma_mps2
                ),
                "gyro_initial_bias_sigma_radps": float(
                    cfg.imu_gyro_initial_bias_sigma_radps
                ),
                "accel_bias_rw_sigma_mps2_sqrt_s": float(
                    cfg.imu_accel_bias_rw_sigma_mps2_sqrt_s
                ),
                "gyro_bias_rw_sigma_radps_sqrt_s": float(
                    cfg.imu_gyro_bias_rw_sigma_radps_sqrt_s
                ),
            },
            "ekf_process_noise": {
                "accel_noise_sigma": float(cfg.ekf_accel_noise_sigma),
                "gyro_noise_sigma": float(cfg.ekf_gyro_noise_sigma),
                "accel_bias_rw_sigma": float(cfg.ekf_accel_bias_rw_sigma),
                "gyro_bias_rw_sigma": float(cfg.ekf_gyro_bias_rw_sigma),
            },
            "configured_learned_update_rate_hz": float(cfg.learned_update_rate_hz),
            "configured_learned_fusion_rate_hz": (
                None
                if cfg.learned_fusion_rate_hz is None
                else float(cfg.learned_fusion_rate_hz)
            ),
            "learned_measurement_covariance_multiplier": float(
                cfg.learned_measurement_covariance_multiplier
            ),
            "configured_learned_network_bias_m": [
                float(v) for v in cfg.learned_network_bias_m
            ],
            "configured_learned_kalman_gain_mode": str(
                cfg.learned_kalman_gain_mode
            ),
            "learned_filter_structure": str(
                getattr(cfg, "learned_filter_structure", "legacy_current_clone")
            ),
            "position_rmse_m": float(np.sqrt(np.mean(np.sum(pos**2, axis=1)))),
            "position_axis_rmse_m": np.sqrt(np.mean(pos**2, axis=0)).tolist(),
            "position_max_error_m": float(np.max(pos_norm)),
            "position_final_error_m": float(pos_norm[-1]),
            "position_tail20_rmse_m": float(
                np.sqrt(np.mean(pos_norm[-tail_count:] ** 2))
            ),
            "velocity_rmse_mps": float(np.sqrt(np.mean(np.sum(vel**2, axis=1)))),
            "velocity_axis_rmse_mps": np.sqrt(np.mean(vel**2, axis=0)).tolist(),
            "velocity_max_error_mps": float(np.max(vel_norm)),
            "velocity_final_error_mps": float(vel_norm[-1]),
            "velocity_tail20_rmse_mps": float(
                np.sqrt(np.mean(vel_norm[-tail_count:] ** 2))
            ),
            "orientation_rmse_deg": float(np.degrees(np.sqrt(np.mean(ori**2)))),
            "orientation_max_error_deg": float(np.degrees(np.max(ori))),
            "orientation_final_error_deg": float(np.degrees(ori[-1])),
            "orientation_tail20_rmse_deg": float(
                np.degrees(np.sqrt(np.mean(ori[-tail_count:] ** 2)))
            ),
            "learned_updates": learned_updates,
            "learned_fusions": learned_fusions,
            "learned_update_hz_total": float(learned_updates / max(duration_s, 1.0e-12)),
            "learned_update_hz_after_warmup": float(learned_updates / warm_duration_s),
            "learned_fusion_hz_total": float(learned_fusions / max(duration_s, 1.0e-12)),
            "learned_fusion_hz_after_warmup": float(learned_fusions / warm_duration_s),
            "learned_update_skips": int(raw_env._learned_update_skip_count),
            "ekf_bias_diagnostics": {
                "accel_bias_norm_final": float(accel_bias_norms[-1]),
                "accel_bias_norm_max": float(np.max(accel_bias_norms)),
                "gyro_bias_norm_final": float(gyro_bias_norms[-1]),
                "gyro_bias_norm_max": float(np.max(gyro_bias_norms)),
            },
            "learned_update_diagnostics": {
                "samples": int(len(update_nis)),
                "nis_mean": None if not update_nis else float(np.nanmean(update_nis)),
                "nis_max": None if not update_nis else float(np.nanmax(update_nis)),
                "dx_theta_norm_mean_rad": None if not update_dx_theta_norm else float(np.mean(update_dx_theta_norm)),
                "dx_theta_norm_max_rad": None if not update_dx_theta_norm else float(np.max(update_dx_theta_norm)),
                "dx_velocity_norm_mean_mps": None if not update_dx_velocity_norm else float(np.mean(update_dx_velocity_norm)),
                "dx_velocity_norm_max_mps": None if not update_dx_velocity_norm else float(np.max(update_dx_velocity_norm)),
                "dx_position_norm_mean_m": None if not update_dx_position_norm else float(np.mean(update_dx_position_norm)),
                "dx_position_norm_max_m": None if not update_dx_position_norm else float(np.max(update_dx_position_norm)),
                "dx_accel_bias_norm_mean_mps2": None if not update_dx_accel_bias_norm else float(np.mean(update_dx_accel_bias_norm)),
                "dx_accel_bias_norm_max_mps2": None if not update_dx_accel_bias_norm else float(np.max(update_dx_accel_bias_norm)),
                "dx_gyro_bias_norm_mean_radps": None if not update_dx_gyro_bias_norm else float(np.mean(update_dx_gyro_bias_norm)),
                "dx_gyro_bias_norm_max_radps": None if not update_dx_gyro_bias_norm else float(np.max(update_dx_gyro_bias_norm)),
                "dx_clone_velocity_norm_mean_mps": None if not update_dx_clone_velocity_norm else float(np.mean(update_dx_clone_velocity_norm)),
                "dx_clone_velocity_norm_max_mps": None if not update_dx_clone_velocity_norm else float(np.max(update_dx_clone_velocity_norm)),
                "dx_clone_position_norm_mean_m": None if not update_dx_clone_position_norm else float(np.mean(update_dx_clone_position_norm)),
                "dx_clone_position_norm_max_m": None if not update_dx_clone_position_norm else float(np.max(update_dx_clone_position_norm)),
                "kalman_gain_raw_current_norm_mean": None if not update_k_raw_current_norm else float(np.mean(update_k_raw_current_norm)),
                "kalman_gain_raw_current_norm_max": None if not update_k_raw_current_norm else float(np.max(update_k_raw_current_norm)),
                "kalman_gain_current_norm_mean": None if not update_k_current_norm else float(np.mean(update_k_current_norm)),
                "kalman_gain_current_norm_max": None if not update_k_current_norm else float(np.max(update_k_current_norm)),
                "kalman_gain_theta_norm_mean": None if not update_k_theta_norm else float(np.mean(update_k_theta_norm)),
                "kalman_gain_theta_norm_max": None if not update_k_theta_norm else float(np.max(update_k_theta_norm)),
                "kalman_gain_velocity_norm_mean": None if not update_k_velocity_norm else float(np.mean(update_k_velocity_norm)),
                "kalman_gain_velocity_norm_max": None if not update_k_velocity_norm else float(np.max(update_k_velocity_norm)),
                "kalman_gain_position_norm_mean": None if not update_k_position_norm else float(np.mean(update_k_position_norm)),
                "kalman_gain_position_norm_max": None if not update_k_position_norm else float(np.max(update_k_position_norm)),
                "kalman_gain_accel_bias_norm_mean": None if not update_k_accel_bias_norm else float(np.mean(update_k_accel_bias_norm)),
                "kalman_gain_accel_bias_norm_max": None if not update_k_accel_bias_norm else float(np.max(update_k_accel_bias_norm)),
                "kalman_gain_gyro_bias_norm_mean": None if not update_k_gyro_bias_norm else float(np.mean(update_k_gyro_bias_norm)),
                "kalman_gain_gyro_bias_norm_max": None if not update_k_gyro_bias_norm else float(np.max(update_k_gyro_bias_norm)),
                "kalman_gain_clone_velocity_norm_mean": None if not update_k_clone_velocity_norm else float(np.mean(update_k_clone_velocity_norm)),
                "kalman_gain_clone_velocity_norm_max": None if not update_k_clone_velocity_norm else float(np.max(update_k_clone_velocity_norm)),
                "kalman_gain_clone_position_norm_mean": None if not update_k_clone_position_norm else float(np.mean(update_k_clone_position_norm)),
                "kalman_gain_clone_position_norm_max": None if not update_k_clone_position_norm else float(np.max(update_k_clone_position_norm)),
            },
            "clone_count_mean": float(np.mean(clones)),
            "clone_count_max": int(np.max(clones)),
            "clone_count_final": int(raw_env._lio.clone_count),
            "tcn_innovation": innovation_summary,
            "tcn_prediction_error_vs_gt": prediction_summary,
            "tcn_fusion_schedule_prediction_error_vs_gt": (
                fusion_schedule_prediction_summary
            ),
            "tcn_fusion_schedule_sample_count_matches_learned_fusions": (
                int(fusion_schedule_prediction_summary["samples"])
                == int(learned_fusions)
            ),
            "gate_attempts": gate_attempts,
            "gate_accepted": gate_accepted,
            "gate_rejected": gate_rejected,
            "gate_acceptance_rate": (
                None if gate_attempts == 0 else float(gate_accepted / gate_attempts)
            ),
            "replay_truth_position_rmse_vs_A_m": replay_position_rmse_m,
            "replay_truth_position_max_diff_vs_A_m": replay_position_max_diff_m,
        }

        summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

        if args_cli.mode == "A":
            replay_path = args_cli.replay_npz.expanduser().resolve()
            replay_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                replay_path,
                actions=np.asarray(generated_actions, dtype=np.float64),
                truth_positions=truth,
            )

        print(
            f"[estimator-ab:{args_cli.mode}] complete "
            f"pos_rmse={summary['position_rmse_m']:.4f}m "
            f"vel_rmse={summary['velocity_rmse_mps']:.4f}m/s "
            f"ori_rmse={summary['orientation_rmse_deg']:.3f}deg",
            flush=True,
        )
        print(f"[estimator-ab:{args_cli.mode}] trace={trace_path}", flush=True)
        print(f"[estimator-ab:{args_cli.mode}] summary={summary_path}", flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
