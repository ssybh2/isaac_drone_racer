"""Unified A/B/C evaluation for the learned-inertial racing estimator.

One invocation evaluates three estimator configurations with an identical action
sequence:

A: IMU propagation only
B: IMU + 20 Hz overlapping TCN displacement updates
C: IMU + 20 Hz TCN + Stage2 mapped-gate PnP absolute updates

Mode A uses a deterministic GT-feedback controller only to generate the replay
action sequence. Modes B and C replay those exact actions. Simulator ground
truth is used only for evaluation metrics and for the Mode-A trajectory
generator; it is never injected into any estimator state after the fixed known
start pose.
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
parser.add_argument("--amplitude_m", type=float, default=1.5)
parser.add_argument("--frequency_hz", type=float, default=0.10)
parser.add_argument("--translation_m", type=float, default=6.0)
parser.add_argument("--seed", type=int, default=0)
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
parser.add_argument(
    "--disable-visibility",
    action="store_true",
    help="Run Mode C without the optional visibility-guard checkpoint.",
)
parser.add_argument(
    "--output-dir",
    type=Path,
    default=Path("artifacts/learned_inertial_ab"),
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


MODE_LABELS = {
    "A": "imu_only",
    "B": "imu_tcn_20hz",
    "C": "imu_tcn_gate",
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


def _orientation_error_rad(q_est, q_truth) -> float:
    q_est = np.asarray(q_est, dtype=np.float64).reshape(4)
    q_truth = np.asarray(q_truth, dtype=np.float64).reshape(4)
    q_est /= np.linalg.norm(q_est)
    q_truth /= np.linalg.norm(q_truth)
    alignment = float(np.clip(abs(np.dot(q_est, q_truth)), 0.0, 1.0))
    return float(2.0 * np.arccos(alignment))


def _target_xy(
    initial_xy: torch.Tensor,
    t_s: float,
    total_duration_s: float,
) -> torch.Tensor:
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
    yaw_error = float(
        np.arctan2(
            np.sin(yaw - target_yaw_rad),
            np.cos(yaw - target_yaw_rad),
        )
    )
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


def _prepare_cfg(mode: str):
    cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=1)
    cfg.scene.num_envs = 1
    cfg.episode_length_s = max(float(cfg.episode_length_s), args_cli.steps * 0.01 + 10.0)
    cfg.terminations.collision = None
    cfg.terminations.flyaway = None
    cfg.commands.target.randomise_start = None
    cfg.events.push_robot = None

    if mode == "A":
        cfg.learned_motion_checkpoint = None
        cfg.swift_detector_checkpoint = None
        cfg.swift_visibility_checkpoint = None
    elif mode == "B":
        cfg.learned_motion_checkpoint = str(args_cli.learned_checkpoint.resolve())
        cfg.swift_detector_checkpoint = None
        cfg.swift_visibility_checkpoint = None
    elif mode == "C":
        cfg.learned_motion_checkpoint = str(args_cli.learned_checkpoint.resolve())
        cfg.swift_detector_checkpoint = str(args_cli.gate_checkpoint.resolve())
        cfg.swift_visibility_checkpoint = (
            None
            if args_cli.disable_visibility
            else str(args_cli.visibility_checkpoint.resolve())
        )
    else:
        raise ValueError(f"unsupported mode: {mode}")
    return cfg


CSV_FIELDS = [
    "mode",
    "mode_label",
    "step",
    "t_s",
    "profile",
    "action_m1",
    "action_m2",
    "action_m3",
    "action_m4",
    "truth_px",
    "truth_py",
    "truth_pz",
    "est_px",
    "est_py",
    "est_pz",
    "pos_err_x",
    "pos_err_y",
    "pos_err_z",
    "pos_err_norm_m",
    "truth_vx",
    "truth_vy",
    "truth_vz",
    "est_vx",
    "est_vy",
    "est_vz",
    "vel_err_x",
    "vel_err_y",
    "vel_err_z",
    "vel_err_norm_mps",
    "truth_qw",
    "truth_qx",
    "truth_qy",
    "truth_qz",
    "est_qw",
    "est_qx",
    "est_qy",
    "est_qz",
    "orientation_error_deg",
    "learned_updates",
    "learned_update_skips",
    "clone_count",
    "tcn_innovation_x",
    "tcn_innovation_y",
    "tcn_innovation_z",
    "tcn_innovation_norm_m",
    "gate_attempts",
    "gate_accepted",
    "gate_rejected",
]


def _run_mode(
    mode: str,
    writer: csv.DictWriter,
    replay_actions: list[np.ndarray] | None,
    reference_truth_positions: np.ndarray | None,
):
    cfg = _prepare_cfg(mode)
    env = gym.make(args_cli.task, cfg=cfg)
    raw_env = env.unwrapped

    generated_actions: list[np.ndarray] = []
    truth_positions: list[np.ndarray] = []
    position_errors: list[np.ndarray] = []
    velocity_errors: list[np.ndarray] = []
    orientation_errors_rad: list[float] = []
    clone_counts: list[int] = []
    innovation_vectors: list[np.ndarray] = []

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

        for step in range(int(args_cli.steps)):
            if not simulation_app.is_running():
                break

            elapsed_before_step = float(raw_env._timestamp_s()) - start_timestamp_s
            if mode == "A":
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
                if replay_actions is None or step >= len(replay_actions):
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

            innovation = None
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

            action_np = _np(action).astype(np.float64)
            t_s = float(raw_env._timestamp_s()) - start_timestamp_s
            row = {
                "mode": mode,
                "mode_label": MODE_LABELS[mode],
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
                "tcn_innovation_x": None if innovation is None else innovation[0],
                "tcn_innovation_y": None if innovation is None else innovation[1],
                "tcn_innovation_z": None if innovation is None else innovation[2],
                "tcn_innovation_norm_m": (
                    None if innovation is None else float(np.linalg.norm(innovation))
                ),
                "gate_attempts": int(raw_env._gate_attempt_count),
                "gate_accepted": int(raw_env._gate_update_count),
                "gate_rejected": int(raw_env._gate_reject_count),
            }
            writer.writerow(row)

        if not position_errors:
            raise RuntimeError(f"mode {mode} produced no evaluation samples")

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

        duration_s = len(pos) * step_dt
        learned_updates = int(raw_env._learned_update_count)
        warm_duration_s = max(
            duration_s - float(cfg.learned_window_time_s),
            1.0e-12,
        )

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
            "norm_mean_m": None,
            "norm_rmse_m": None,
            "norm_max_m": None,
        }
        if len(innovations):
            innovation_norm = np.linalg.norm(innovations, axis=1)
            innovation_summary.update(
                {
                    "axis_rmse_m": np.sqrt(np.mean(innovations**2, axis=0)).tolist(),
                    "norm_mean_m": float(np.mean(innovation_norm)),
                    "norm_rmse_m": float(np.sqrt(np.mean(innovation_norm**2))),
                    "norm_max_m": float(np.max(innovation_norm)),
                }
            )

        gate_attempts = int(raw_env._gate_attempt_count)
        gate_accepted = int(raw_env._gate_update_count)
        gate_rejected = int(raw_env._gate_reject_count)

        summary = {
            "mode": mode,
            "label": MODE_LABELS[mode],
            "samples": int(len(pos)),
            "duration_s": float(duration_s),
            "position_rmse_m": float(np.sqrt(np.mean(np.sum(pos**2, axis=1)))),
            "position_axis_rmse_m": np.sqrt(np.mean(pos**2, axis=0)).tolist(),
            "position_max_error_m": float(np.max(np.linalg.norm(pos, axis=1))),
            "velocity_rmse_mps": float(np.sqrt(np.mean(np.sum(vel**2, axis=1)))),
            "velocity_axis_rmse_mps": np.sqrt(np.mean(vel**2, axis=0)).tolist(),
            "velocity_max_error_mps": float(np.max(np.linalg.norm(vel, axis=1))),
            "orientation_rmse_deg": float(np.degrees(np.sqrt(np.mean(ori**2)))),
            "orientation_max_error_deg": float(np.degrees(np.max(ori))),
            "learned_updates": learned_updates,
            "learned_update_hz_total": float(learned_updates / max(duration_s, 1.0e-12)),
            "learned_update_hz_after_warmup": float(learned_updates / warm_duration_s),
            "learned_update_skips": int(raw_env._learned_update_skip_count),
            "clone_count_mean": float(np.mean(clones)),
            "clone_count_max": int(np.max(clones)),
            "clone_count_final": int(raw_env._lio.clone_count),
            "tcn_innovation": innovation_summary,
            "gate_attempts": gate_attempts,
            "gate_accepted": gate_accepted,
            "gate_rejected": gate_rejected,
            "gate_acceptance_rate": (
                None if gate_attempts == 0 else float(gate_accepted / gate_attempts)
            ),
            "replay_truth_position_rmse_vs_A_m": replay_position_rmse_m,
            "replay_truth_position_max_diff_vs_A_m": replay_position_max_diff_m,
        }

        return (
            summary,
            generated_actions if mode == "A" else replay_actions,
            truth,
        )
    finally:
        env.close()


def _validate_inputs() -> None:
    if args_cli.steps < 1:
        raise ValueError("--steps must be positive")
    if not args_cli.learned_checkpoint.expanduser().exists():
        raise FileNotFoundError(
            f"learned checkpoint not found: {args_cli.learned_checkpoint}"
        )
    if not args_cli.gate_checkpoint.expanduser().exists():
        raise FileNotFoundError(
            f"gate checkpoint not found: {args_cli.gate_checkpoint}"
        )
    if (
        not args_cli.disable_visibility
        and not args_cli.visibility_checkpoint.expanduser().exists()
    ):
        raise FileNotFoundError(
            f"visibility checkpoint not found: {args_cli.visibility_checkpoint}"
        )


def main() -> None:
    _validate_inputs()
    output_dir = args_cli.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "estimator_ab_trace.csv"
    json_path = output_dir / "estimator_ab_summary.json"

    summaries = {}
    replay_actions = None
    reference_truth_positions = None

    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()

        for mode in ("A", "B", "C"):
            print(f"[estimator-ab] starting mode {mode}: {MODE_LABELS[mode]}")
            summary, replay_actions, truth_positions = _run_mode(
                mode,
                writer,
                replay_actions,
                reference_truth_positions,
            )
            summaries[mode] = summary
            if mode == "A":
                reference_truth_positions = truth_positions
            print(
                f"[estimator-ab] mode {mode} "
                f"pos_rmse={summary['position_rmse_m']:.4f}m "
                f"vel_rmse={summary['velocity_rmse_mps']:.4f}m/s "
                f"ori_rmse={summary['orientation_rmse_deg']:.3f}deg"
            )

    report = {
        "schema": "isaac_drone_racer.learned_inertial_ab.v1",
        "task": args_cli.task,
        "profile": args_cli.profile,
        "steps": int(args_cli.steps),
        "seed": int(args_cli.seed),
        "same_trajectory_contract": (
            "Mode A generates the action sequence once; Modes B and C replay "
            "the exact same actions. GT is used only for metrics and the Mode-A "
            "evaluation controller, never as an estimator update."
        ),
        "checkpoints": {
            "learned_motion": str(args_cli.learned_checkpoint.expanduser().resolve()),
            "gate_detector": str(args_cli.gate_checkpoint.expanduser().resolve()),
            "visibility": (
                None
                if args_cli.disable_visibility
                else str(args_cli.visibility_checkpoint.expanduser().resolve())
            ),
        },
        "modes": summaries,
    }
    json_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print(f"[estimator-ab] wrote trace: {csv_path}")
    print(f"[estimator-ab] wrote summary: {json_path}")
    print("\n=== A/B/C SUMMARY ===")
    for mode in ("A", "B", "C"):
        s = summaries[mode]
        print(
            f"{mode} {s['label']:16s} "
            f"pos={s['position_rmse_m']:.4f}m "
            f"vel={s['velocity_rmse_mps']:.4f}m/s "
            f"ori={s['orientation_rmse_deg']:.3f}deg "
            f"max_pos={s['position_max_error_m']:.4f}m "
            f"learned_hz={s['learned_update_hz_after_warmup']:.2f} "
            f"clones_max={s['clone_count_max']} "
            f"skips={s['learned_update_skips']} "
            f"gate={s['gate_accepted']}/{s['gate_attempts']} "
            f"rejected={s['gate_rejected']}"
        )


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
