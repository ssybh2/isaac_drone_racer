"""Scripted gate-flight validation for the Swift CTBR control stack.

This is deliberately NOT an RL policy. The outer loop uses only the
learned-inertial estimate and the known mapped next gate to generate desired
collective thrust and body rates. The new SwiftCTBRAction then closes the
low-level body-rate loop and mixes to motors.

Simulator root state is read only for diagnostics.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--task",
    default="Isaac-Drone-Racer-Learned-Inertial-Swift-CTBR-v0",
)
parser.add_argument("--duration-s", type=float, default=20.0)
parser.add_argument("--max-speed-mps", type=float, default=2.0)
parser.add_argument("--lookahead-m", type=float, default=1.0)
parser.add_argument("--position-gain", type=float, default=0.8)
parser.add_argument("--velocity-gain", type=float, default=2.0)
parser.add_argument("--attitude-rate-gain", type=float, default=4.0)
parser.add_argument("--max-accel-mps2", type=float, default=4.0)
parser.add_argument("--progress-every", type=int, default=200)
parser.add_argument("--output-json", type=Path, default=None)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

import tasks  # noqa: F401,E402


MASS_KG = 0.6076
GRAVITY_MPS2 = 9.81


def _rotation_from_quat_wxyz(q) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64).reshape(4)
    q = q / max(np.linalg.norm(q), 1.0e-12)
    w, x, y, z = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _vee(skew: np.ndarray) -> np.ndarray:
    return np.array([skew[2, 1], skew[0, 2], skew[1, 0]], dtype=np.float64)


def _physical_ctbr_to_normalized(raw_env, collective_accel, body_rate) -> torch.Tensor:
    action_term = raw_env.action_manager.get_term("control_action")
    max_collective = float(action_term._max_collective_accel)
    rate_max = np.asarray(action_term.cfg.body_rate_max_radps, dtype=np.float64)

    collective_accel = float(np.clip(collective_accel, 0.0, max_collective))
    if collective_accel >= GRAVITY_MPS2:
        a0 = (collective_accel - GRAVITY_MPS2) / max(
            max_collective - GRAVITY_MPS2, 1.0e-9
        )
    else:
        a0 = (collective_accel - GRAVITY_MPS2) / GRAVITY_MPS2

    rate_action = np.asarray(body_rate, dtype=np.float64) / rate_max
    action = np.concatenate(([a0], rate_action))
    action = np.clip(action, -1.0, 1.0)
    return torch.as_tensor(
        action,
        dtype=torch.float32,
        device=raw_env.device,
    ).view(1, 4)


def _controller_action(raw_env) -> tuple[torch.Tensor, dict]:
    state = raw_env.learned_inertial_state
    if state is None:
        raise RuntimeError("learned inertial state is not initialized")

    p_w = np.asarray(state.position_w_b, dtype=np.float64).reshape(3)
    v_w = np.asarray(state.linear_velocity_w_b, dtype=np.float64).reshape(3)
    R_wb = _rotation_from_quat_wxyz(state.orientation_w_b_wxyz)

    command = raw_env.command_manager.get_term("target")
    gate_pose = command.command[0].detach().cpu().numpy().astype(np.float64)
    gate_center_w = gate_pose[:3]
    R_wg = _rotation_from_quat_wxyz(gate_pose[3:7])
    gate_normal_w = R_wg[:, 0]

    goal_w = gate_center_w + float(args_cli.lookahead_m) * gate_normal_w
    position_error = goal_w - p_w

    v_des = float(args_cli.position_gain) * position_error
    speed = float(np.linalg.norm(v_des))
    if speed > float(args_cli.max_speed_mps):
        v_des *= float(args_cli.max_speed_mps) / max(speed, 1.0e-12)

    desired_accel_w = float(args_cli.velocity_gain) * (v_des - v_w)
    accel_norm = float(np.linalg.norm(desired_accel_w))
    if accel_norm > float(args_cli.max_accel_mps2):
        desired_accel_w *= float(args_cli.max_accel_mps2) / max(accel_norm, 1.0e-12)

    desired_force_w = MASS_KG * (
        desired_accel_w + np.array([0.0, 0.0, GRAVITY_MPS2])
    )
    force_norm = max(float(np.linalg.norm(desired_force_w)), 1.0e-12)
    z_des = desired_force_w / force_norm

    heading = goal_w - p_w
    heading[2] = 0.0
    if np.linalg.norm(heading) < 1.0e-6:
        heading = gate_normal_w.copy()
        heading[2] = 0.0
    if np.linalg.norm(heading) < 1.0e-6:
        heading = np.array([1.0, 0.0, 0.0])
    heading /= max(np.linalg.norm(heading), 1.0e-12)

    y_des = np.cross(z_des, heading)
    if np.linalg.norm(y_des) < 1.0e-6:
        heading = np.array([0.0, 1.0, 0.0])
        y_des = np.cross(z_des, heading)
    y_des /= max(np.linalg.norm(y_des), 1.0e-12)
    x_des = np.cross(y_des, z_des)
    x_des /= max(np.linalg.norm(x_des), 1.0e-12)
    R_des = np.column_stack((x_des, y_des, z_des))

    attitude_error = 0.5 * _vee(R_des.T @ R_wb - R_wb.T @ R_des)
    desired_body_rate = -float(args_cli.attitude_rate_gain) * attitude_error

    collective_accel = float(np.dot(desired_force_w, R_wb[:, 2]) / MASS_KG)
    action = _physical_ctbr_to_normalized(
        raw_env,
        collective_accel,
        desired_body_rate,
    )

    return action, {
        "target_gate_index": int(command.next_gate_idx[0].item()),
        "estimated_position_w": p_w.tolist(),
        "estimated_velocity_w": v_w.tolist(),
        "goal_w": goal_w.tolist(),
        "position_error_norm_m": float(np.linalg.norm(position_error)),
        "collective_accel_mps2": collective_accel,
        "desired_body_rate_radps": desired_body_rate.tolist(),
        "normalized_action": action[0].detach().cpu().numpy().tolist(),
    }


def main() -> None:
    cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=1)
    cfg.scene.num_envs = 1
    cfg.episode_length_s = max(float(args_cli.duration_s), 1.0)
    cfg.commands.target.debug_vis = False

    env = gym.make(args_cli.task, cfg=cfg)
    raw_env = env.unwrapped
    robot = raw_env.scene["robot"]

    try:
        env.reset()
        command = raw_env.command_manager.get_term("target")
        mission_passes = 0
        truth_passes = 0
        max_position_error = 0.0
        max_velocity_error = 0.0
        max_abs_action = 0.0
        last_control = None
        termination = "none"

        total_steps = int(np.ceil(float(args_cli.duration_s) / raw_env.step_dt))
        for step in range(total_steps):
            state = raw_env.learned_inertial_state
            gt_p = robot.data.root_pos_w[0].detach().cpu().numpy().astype(np.float64)
            gt_v = robot.data.root_lin_vel_w[0].detach().cpu().numpy().astype(np.float64)
            est_p = np.asarray(state.position_w_b, dtype=np.float64)
            est_v = np.asarray(state.linear_velocity_w_b, dtype=np.float64)
            max_position_error = max(
                max_position_error,
                float(np.linalg.norm(est_p - gt_p)),
            )
            max_velocity_error = max(
                max_velocity_error,
                float(np.linalg.norm(est_v - gt_v)),
            )

            with torch.inference_mode():
                action, last_control = _controller_action(raw_env)
                max_abs_action = max(
                    max_abs_action,
                    float(action.abs().max().item()),
                )
                _, _, terminated, truncated, _ = env.step(action)

            done = bool(terminated.reshape(-1)[0].item()) or bool(
                truncated.reshape(-1)[0].item()
            )
            if not done:
                mission_passes += int(command.mission_gate_passed[0].item())
                truth_passes += int(command.gate_passed[0].item())

            if step == 0 or (step + 1) % int(args_cli.progress_every) == 0:
                print(
                    "[ctbr-gate-smoke] "
                    f"step={step + 1}/{total_steps} "
                    f"gate={last_control['target_gate_index']} "
                    f"mission={mission_passes} truth={truth_passes} "
                    f"p_err={max_position_error:.3f}m "
                    f"v_err={max_velocity_error:.3f}m/s "
                    f"|a|max={max_abs_action:.3f}",
                    flush=True,
                )

            if done:
                diag = raw_env.last_episode_diagnostic
                if diag is not None:
                    if bool(diag.get("collision", False)):
                        termination = "collision"
                    elif bool(diag.get("flyaway", False)):
                        termination = "flyaway"
                    elif bool(diag.get("time_out", False)):
                        termination = "timeout"
                    truth_passes = max(
                        truth_passes,
                        int(diag.get("truth_gates_passed", truth_passes)),
                    )
                    mission_passes = max(
                        mission_passes,
                        int(diag.get("mission_gates_passed", mission_passes)),
                    )
                else:
                    termination = "terminated"
                break

        summary = {
            "task": args_cli.task,
            "duration_requested_s": float(args_cli.duration_s),
            "mission_gate_passes": int(mission_passes),
            "truth_gate_passes": int(truth_passes),
            "termination": termination,
            "maximum_position_error_m": float(max_position_error),
            "maximum_velocity_error_mps": float(max_velocity_error),
            "maximum_abs_normalized_action": float(max_abs_action),
            "last_control": last_control,
            "controller_uses_simulator_root_state": False,
            "mission_progression_uses_simulator_root_state": False,
        }

        print("=" * 96)
        print("SWIFT CTBR ESTIMATED-STATE GATE-FLIGHT SMOKE")
        print("=" * 96)
        print(json.dumps(summary, indent=2))

        if args_cli.output_json is not None:
            output = args_cli.output_json.expanduser().resolve()
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps(summary, indent=2) + "\n")
            print(f"[ctbr-gate-smoke] wrote: {output}")
    finally:
        env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
