"""Run the single-vehicle Isaac -> ROS2 -> OpenVINS diagnostic path without a policy checkpoint."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Validate Isaac RGB/IMU transport and OpenVINS alignment.")
parser.add_argument("--steps", type=int, default=5000, help="Maximum control steps to run.")
parser.add_argument(
    "--hover_action",
    type=float,
    default=0.0,
    help="Nominal equal normalized rotor command used by the stationary controller.",
)
parser.add_argument(
    "--output",
    type=Path,
    default=Path("outputs/swift_openvins_diagnostic.json"),
    help="JSON report containing transport, estimator, gate and fusion diagnostics.",
)
parser.add_argument(
    "--detector_checkpoint",
    type=str,
    default=None,
    help="Optional Torchvision gate keypoint checkpoint; enables detector->IPPE->VIO fusion.",
)
parser.add_argument(
    "--detector_device", type=str, default="cuda", help="Torch device for the optional detector."
)
parser.add_argument(
    "--visibility_checkpoint",
    type=str,
    default=None,
    help="Optional compact checkpoint used as a calibrated learned visibility guard.",
)
parser.add_argument(
    "--visibility_threshold",
    type=float,
    default=0.75,
    help="Per-corner visibility threshold for --visibility_checkpoint.",
)
parser.add_argument(
    "--corner_sigma_px",
    type=float,
    default=0.8472250465393066,
    help="Calibrated ordinary per-component corner noise used by IPPE perturbations.",
)
parser.add_argument(
    "--oracle_gate_index",
    action="store_true",
    help=(
        "Diagnostic ablation only: use Isaac task next_gate_idx instead of Swift-style "
        "known-map/VIO gate association."
    ),
)
parser.add_argument(
    "--rejection_dump_dir",
    type=str,
    default=None,
    help="Optional directory for rejected detector/IPPE/Kalman RGB frames and JSON reasons.",
)
parser.add_argument(
    "--rejection_dump_limit",
    type=int,
    default=200,
    help="Maximum number of rejected frames to persist during one diagnostic run.",
)
parser.add_argument(
    "--task",
    type=str,
    default="Isaac-Drone-Racer-Swift-OpenVINS-v0",
    help="Registered diagnostic task.",
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


def _error_summary(values: list[float]) -> dict:
    if not values:
        return {"samples": 0, "mean_m": None, "rmse_m": None, "p95_m": None}
    array = np.asarray(values, dtype=np.float64)
    return {
        "samples": int(array.size),
        "mean_m": float(np.mean(array)),
        "rmse_m": float(np.sqrt(np.mean(np.square(array)))),
        "p95_m": float(np.percentile(array, 95)),
    }


def _second_difference_summary(positions: list[np.ndarray]) -> dict:
    if len(positions) < 3:
        return {"samples": 0, "median_m": None, "p95_m": None}
    values = np.linalg.norm(np.diff(np.asarray(positions), n=2, axis=0), axis=1)
    return {
        "samples": int(values.size),
        "median_m": float(np.median(values)),
        "p95_m": float(np.percentile(values, 95)),
    }


def main() -> None:
    if args_cli.rejection_dump_limit < 0:
        raise ValueError("--rejection_dump_limit must be non-negative")

    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=1)
    if args_cli.detector_checkpoint is not None:
        env_cfg.swift_detector_checkpoint = args_cli.detector_checkpoint
        env_cfg.swift_detector_device = args_cli.detector_device
        env_cfg.swift_visibility_checkpoint = args_cli.visibility_checkpoint
        env_cfg.swift_visibility_threshold = float(args_cli.visibility_threshold)
        env_cfg.swift_corner_sigma_px = float(args_cli.corner_sigma_px)
    elif args_cli.visibility_checkpoint is not None:
        raise ValueError("--visibility_checkpoint requires --detector_checkpoint")
    env_cfg.swift_use_oracle_gate_index = bool(args_cli.oracle_gate_index)
    env_cfg.swift_rejection_dump_dir = args_cli.rejection_dump_dir
    env_cfg.swift_rejection_dump_limit = int(args_cli.rejection_dump_limit)
    # A diagnostic segment must not silently reset the external estimator.
    # Preserve physical contacts in the simulation, but disable racing-task
    # episode termination and make the timeout longer than this requested run.
    env_cfg.terminations.collision = None
    env_cfg.terminations.flyaway = None
    env_cfg.episode_length_s = max(
        float(env_cfg.episode_length_s), int(args_cli.steps) * 0.01 + 1.0
    )

    env = gym.make(args_cli.task, cfg=env_cfg)
    raw_env = env.unwrapped
    env.reset()

    action_dim = int(raw_env.action_manager.total_action_dim)
    # Equal rotor commands keep the vehicle level and make this transport
    # diagnostic independent of an RL checkpoint.
    actions = torch.full(
        (1, action_dim),
        float(args_cli.hover_action),
        dtype=torch.float32,
        device=raw_env.device,
    )
    target_height_m = float(raw_env.scene["robot"].data.root_pos_w[0, 2])
    target_xy_w = raw_env.scene["robot"].data.root_pos_w[0, :2].clone()
    vio_position_errors: list[float] = []
    fused_position_errors: list[float] = []
    gate_distances: list[float] = []
    gate_covariance_traces: list[float] = []
    truth_positions: list[np.ndarray] = []
    truth_roll_pitch_rad: list[np.ndarray] = []
    raw_gate_positions: list[np.ndarray] = []
    fused_gate_frame_positions: list[np.ndarray] = []
    orientation_differences_rad: list[float] = []
    fusion_rejections: Counter[str] = Counter()
    measurement_attempts = 0
    measurement_accepts = 0
    initialized_step = None
    completed_steps = 0

    try:
        for step in range(int(args_cli.steps)):
            if not simulation_app.is_running():
                break
            robot = raw_env.scene["robot"]
            height_error = target_height_m - float(robot.data.root_pos_w[0, 2])
            vertical_velocity = float(robot.data.root_lin_vel_w[0, 2])
            common = float(args_cli.hover_action) + 0.18 * height_error - 0.09 * vertical_velocity
            xy_error = target_xy_w - robot.data.root_pos_w[0, :2]
            desired_acceleration_xy = xy_error - 1.5 * robot.data.root_lin_vel_w[0, :2]
            desired_roll = float(torch.clamp(-desired_acceleration_xy[1] / 9.81, -0.1, 0.1))
            desired_pitch = float(torch.clamp(desired_acceleration_xy[0] / 9.81, -0.1, 0.1))
            quaternion = robot.data.root_quat_w[0]
            angular_velocity = robot.data.root_ang_vel_b[0]
            # Small-angle roll/pitch stabilization expressed directly as
            # opposite motor-pair commands. This is only a stationary source
            # generator, not a racing controller.
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
            roll_correction = -0.08 * (roll - desired_roll) - 0.015 * float(
                angular_velocity[0]
            )
            pitch_correction = -0.08 * (pitch - desired_pitch) - 0.015 * float(
                angular_velocity[1]
            )
            actions[0] = torch.tensor(
                [
                    common + roll_correction - pitch_correction,
                    common - roll_correction - pitch_correction,
                    common - roll_correction + pitch_correction,
                    common + roll_correction + pitch_correction,
                ],
                dtype=actions.dtype,
                device=actions.device,
            ).clamp(-1.0, 1.0)
            _, _, terminated, truncated, _ = env.step(actions)
            completed_steps = step + 1
            estimate = raw_env.openvins_vio_estimate
            truth = raw_env._truth_vio_state()
            truth_positions.append(truth.position_w_b.copy())
            q_w, q_x, q_y, q_z = truth.orientation_w_b_wxyz
            truth_roll_pitch_rad.append(
                np.array(
                    [
                        np.arctan2(
                            2.0 * (q_w * q_x + q_y * q_z),
                            1.0 - 2.0 * (q_x * q_x + q_y * q_y),
                        ),
                        np.arcsin(np.clip(2.0 * (q_w * q_y - q_z * q_x), -1.0, 1.0)),
                    ],
                    dtype=np.float64,
                )
            )
            if estimate is not None:
                if initialized_step is None:
                    initialized_step = step
                vio_position_errors.append(
                    float(np.linalg.norm(estimate.position_w_b - truth.position_w_b))
                )
            fusion = raw_env.swift_last_fusion_result
            if fusion is not None and estimate is not None:
                fused = fusion.fused_state
                fused_position_errors.append(
                    float(np.linalg.norm(fused.position_w_b - truth.position_w_b))
                )
                quaternion_dot = float(
                    np.clip(
                        abs(np.dot(fused.orientation_w_b_wxyz, estimate.orientation_w_b_wxyz)),
                        0.0,
                        1.0,
                    )
                )
                orientation_differences_rad.append(2.0 * float(np.arccos(quaternion_dot)))
                if fusion.rejection_reason != "no gate observation":
                    measurement_attempts += 1
                    if fusion.measurement_accepted:
                        measurement_accepts += 1
                    else:
                        fusion_rejections[fusion.rejection_reason] += 1
                measurement = fusion.gate_measurement
                if measurement is not None:
                    gate_distances.append(float(np.linalg.norm(measurement.T_cg.t)))
                    gate_covariance_traces.append(
                        float(np.trace(measurement.position_covariance_w))
                    )
                    raw_gate_positions.append(measurement.position_w_b.copy())
                    fused_gate_frame_positions.append(fused.position_w_b.copy())
            if step % 100 == 0:
                if estimate is None:
                    print(f"[OpenVINS] step={step}: waiting for initialized odometry")
                else:
                    age = max(0.0, raw_env._timestamp_s() - estimate.timestamp_s)
                    drained = raw_env._openvins_bridge.last_drain_count
                    print(
                        f"[OpenVINS] step={step}: aligned={raw_env.openvins_alignment is not None} "
                        f"age={age:.4f}s drained={drained} pos={estimate.position_w_b.tolist()}"
                    )
                print(
                    f"[Truth] step={step}: pos={truth.position_w_b.tolist()} "
                    f"roll_pitch={truth_roll_pitch_rad[-1].tolist()}"
                )
                if fusion is not None and not fusion.measurement_accepted:
                    print(
                        f"[SwiftFusion] rejected: {fusion.rejection_reason} "
                        f"d2={fusion.innovation_mahalanobis2}"
                    )
            if bool(torch.as_tensor(terminated).any()) or bool(torch.as_tensor(truncated).any()):
                term_values = dict(raw_env.termination_manager.get_active_iterable_terms(0))
                print(
                    "[OpenVINS] Isaac episode reset/termination occurred. "
                    "Restart ov_msckf before trusting a new aligned segment. "
                    f"terms={term_values}"
                )
                break
    finally:
        vio_summary = _error_summary(vio_position_errors)
        fused_summary = _error_summary(fused_position_errors)
        covariance_distance = {
            "samples": len(gate_distances),
            "near_distance_median_m": None,
            "near_covariance_trace_m2_median": None,
            "far_distance_median_m": None,
            "far_covariance_trace_m2_median": None,
            "far_trace_larger_than_near": None,
        }
        if len(gate_distances) >= 4:
            order = np.argsort(gate_distances)
            quartile = max(1, len(order) // 4)
            near = order[:quartile]
            far = order[-quartile:]
            near_trace = float(np.median(np.asarray(gate_covariance_traces)[near]))
            far_trace = float(np.median(np.asarray(gate_covariance_traces)[far]))
            covariance_distance.update(
                {
                    "near_distance_median_m": float(np.median(np.asarray(gate_distances)[near])),
                    "near_covariance_trace_m2_median": near_trace,
                    "far_distance_median_m": float(np.median(np.asarray(gate_distances)[far])),
                    "far_covariance_trace_m2_median": far_trace,
                    "far_trace_larger_than_near": bool(far_trace > near_trace),
                }
            )
        report = {
            "schema": "isaac_drone_racer.swift_openvins_diagnostic.v1",
            "completed_steps": completed_steps,
            "openvins_initialized_step": initialized_step,
            "camera_contract_validated": bool(raw_env._camera_contract_validated),
            "truth_motion": {
                "samples": len(truth_positions),
                "initial_position_w_b": (
                    None if not truth_positions else truth_positions[0].tolist()
                ),
                "final_position_w_b": (
                    None if not truth_positions else truth_positions[-1].tolist()
                ),
                "max_displacement_from_start_m": (
                    None
                    if not truth_positions
                    else float(
                        np.max(
                            np.linalg.norm(
                                np.asarray(truth_positions) - truth_positions[0], axis=1
                            )
                        )
                    )
                ),
                "max_abs_roll_pitch_rad": (
                    None
                    if not truth_roll_pitch_rad
                    else np.max(np.abs(np.asarray(truth_roll_pitch_rad)), axis=0).tolist()
                ),
            },
            "vio_position_error": vio_summary,
            "fused_position_error": fused_summary,
            "fused_rmse_improves_over_vio": (
                None
                if vio_summary["rmse_m"] is None or fused_summary["rmse_m"] is None
                else bool(fused_summary["rmse_m"] < vio_summary["rmse_m"])
            ),
            "orientation_owned_by_vio": {
                "samples": len(orientation_differences_rad),
                "max_difference_rad": (
                    None
                    if not orientation_differences_rad
                    else float(np.max(orientation_differences_rad))
                ),
            },
            "gate_measurements": {
                "attempted_frames": measurement_attempts,
                "accepted_frames": measurement_accepts,
                "rejected_frames": measurement_attempts - measurement_accepts,
                "rejection_reasons": dict(fusion_rejections),
                "covariance_by_distance": covariance_distance,
                "raw_gate_second_difference": _second_difference_summary(raw_gate_positions),
                "fused_second_difference": _second_difference_summary(
                    fused_gate_frame_positions
                ),
            },
            "rejection_dump_count": int(raw_env._rejection_dump_count),
        }
        args_cli.output.parent.mkdir(parents=True, exist_ok=True)
        args_cli.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"[Diagnostic] wrote {args_cli.output}")
        if raw_env._rejection_dump_count:
            print(f"[SwiftFusion] wrote {raw_env._rejection_dump_count} rejected-frame records")
        env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
