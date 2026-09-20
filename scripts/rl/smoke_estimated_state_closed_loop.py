"""Closed-loop integration smoke using only learned-inertial state for control.

The controller path deliberately consumes only:
  * LearnedInertialOdometry state
  * onboard/synthetic gyro measurement
  * the actor-visible mapped gate command

Simulator root state is read only after each step for diagnostic error reporting.
It is never used to compute an action or to advance the mission gate index.
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
    default="Isaac-Drone-Racer-Learned-Inertial-RL-v0",
)
parser.add_argument("--duration-s", type=float, default=40.0)
parser.add_argument("--max-speed-mps", type=float, default=2.0)
parser.add_argument("--lookahead-m", type=float, default=1.0)
parser.add_argument("--position-gain", type=float, default=0.8)
parser.add_argument("--velocity-gain", type=float, default=2.0)
parser.add_argument("--attitude-gain", type=float, default=0.08)
parser.add_argument("--rate-gain", type=float, default=0.018)
parser.add_argument("--progress-every", type=int, default=200)
parser.add_argument("--output-json", type=Path, default=None)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# This task always needs its onboard camera, including in headless mode.
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import tasks  # noqa: F401,E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402


MASS_KG = 0.6076
GRAVITY_MPS2 = 9.81
ARM_LENGTH_M = 0.035
THRUST_COEFF = 2.25e-7
DRAG_COEFF = 1.5e-9
OMEGA_MAX_RADPS = 5145.0


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


def _vee(skew_matrix: np.ndarray) -> np.ndarray:
    return np.array(
        [
            skew_matrix[2, 1],
            skew_matrix[0, 2],
            skew_matrix[1, 0],
        ],
        dtype=np.float64,
    )


def _allocation_matrix() -> np.ndarray:
    arm = ARM_LENGTH_M / np.sqrt(2.0)
    yaw = DRAG_COEFF / THRUST_COEFF
    return np.array(
        [
            [1.0, 1.0, 1.0, 1.0],
            [arm, -arm, -arm, arm],
            [-arm, -arm, arm, arm],
            [yaw, -yaw, yaw, -yaw],
        ],
        dtype=np.float64,
    )


ALLOCATION_INV = np.linalg.inv(_allocation_matrix())
MAX_ROTOR_THRUST_N = THRUST_COEFF * OMEGA_MAX_RADPS**2


def _controller_action(raw_env) -> tuple[torch.Tensor, dict]:
    """Compute one action without reading simulator root pose/state."""
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

    # Aim slightly through the gate instead of braking at its plane.
    goal_w = gate_center_w + float(args_cli.lookahead_m) * gate_normal_w
    position_error = goal_w - p_w

    # Bounded velocity command followed by velocity feedback.
    v_des = float(args_cli.position_gain) * position_error
    v_norm = float(np.linalg.norm(v_des))
    max_speed = float(args_cli.max_speed_mps)
    if v_norm > max_speed:
        v_des *= max_speed / max(v_norm, 1.0e-12)

    desired_accel_w = float(args_cli.velocity_gain) * (v_des - v_w)
    accel_norm = float(np.linalg.norm(desired_accel_w))
    if accel_norm > 4.0:
        desired_accel_w *= 4.0 / accel_norm

    desired_force_w = MASS_KG * (
        desired_accel_w + np.array([0.0, 0.0, GRAVITY_MPS2])
    )
    force_norm = float(np.linalg.norm(desired_force_w))
    if force_norm < 1.0e-9:
        desired_force_w = np.array([0.0, 0.0, MASS_KG * GRAVITY_MPS2])
        force_norm = float(np.linalg.norm(desired_force_w))

    z_des = desired_force_w / force_norm
    heading = gate_normal_w.copy()
    heading[2] = 0.0
    if np.linalg.norm(heading) < 1.0e-6:
        heading = np.array([1.0, 0.0, 0.0])
    heading /= np.linalg.norm(heading)

    y_des = np.cross(z_des, heading)
    if np.linalg.norm(y_des) < 1.0e-6:
        heading = np.array([0.0, 1.0, 0.0])
        y_des = np.cross(z_des, heading)
    y_des /= max(np.linalg.norm(y_des), 1.0e-12)
    x_des = np.cross(y_des, z_des)
    x_des /= max(np.linalg.norm(x_des), 1.0e-12)
    R_des = np.column_stack((x_des, y_des, z_des))

    attitude_error = 0.5 * _vee(R_des.T @ R_wb - R_wb.T @ R_des)

    gyro_meas = getattr(raw_env, "_last_imu_gyro_b_meas", None)
    if gyro_meas is None:
        omega_b = (
            raw_env.scene["imu"].data.ang_vel_b[0]
            .detach()
            .cpu()
            .numpy()
            .astype(np.float64)
        )
    else:
        omega_b = np.asarray(gyro_meas, dtype=np.float64).reshape(3)

    thrust_n = float(np.dot(desired_force_w, R_wb[:, 2]))
    thrust_n = float(np.clip(thrust_n, 0.0, 4.0 * MAX_ROTOR_THRUST_N))

    moment_nm = (
        -float(args_cli.attitude_gain) * attitude_error
        -float(args_cli.rate_gain) * omega_b
    )

    wrench = np.concatenate(([thrust_n], moment_nm))
    rotor_thrust_n = ALLOCATION_INV @ wrench
    rotor_thrust_n = np.clip(
        rotor_thrust_n,
        0.0,
        MAX_ROTOR_THRUST_N,
    )
    rotor_omega = np.sqrt(rotor_thrust_n / THRUST_COEFF)
    action_np = 2.0 * rotor_omega / OMEGA_MAX_RADPS - 1.0
    action_np = np.clip(action_np, -1.0, 1.0)

    action = torch.as_tensor(
        action_np,
        dtype=torch.float32,
        device=raw_env.device,
    ).view(1, 4)

    return action, {
        "estimated_position_w": p_w.tolist(),
        "estimated_velocity_w": v_w.tolist(),
        "goal_w": goal_w.tolist(),
        "position_error_norm_m": float(np.linalg.norm(position_error)),
        "thrust_n": thrust_n,
        "moment_nm": moment_nm.tolist(),
        "action": action_np.tolist(),
        "target_gate_index": int(command.next_gate_idx[0].item()),
    }


def _diagnostic_truth_error(raw_env) -> dict:
    """Ground truth is isolated here and never enters _controller_action."""
    robot = raw_env.scene["robot"]
    gt_p = robot.data.root_pos_w[0].detach().cpu().numpy().astype(np.float64)
    gt_v = robot.data.root_lin_vel_w[0].detach().cpu().numpy().astype(np.float64)
    state = raw_env.learned_inertial_state
    est_p = np.asarray(state.position_w_b, dtype=np.float64)
    est_v = np.asarray(state.linear_velocity_w_b, dtype=np.float64)
    return {
        "position_error_m": float(np.linalg.norm(est_p - gt_p)),
        "velocity_error_mps": float(np.linalg.norm(est_v - gt_v)),
    }


def main() -> None:
    cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=1,
    )
    cfg.scene.num_envs = 1
    cfg.episode_length_s = max(float(args_cli.duration_s), 1.0)
    cfg.commands.target.debug_vis = False

    env = gym.make(args_cli.task, cfg=cfg)
    raw_env = env.unwrapped

    try:
        observation, _ = env.reset()
        del observation

        command = raw_env.command_manager.get_term("target")
        mission_passes = 0
        truth_passes = 0
        maximum_position_error = 0.0
        maximum_velocity_error = 0.0
        last_control = None
        terminated_early = False
        termination_step = None

        total_steps = int(np.ceil(float(args_cli.duration_s) / raw_env.step_dt))

        for step in range(total_steps):
            with torch.inference_mode():
                action, last_control = _controller_action(raw_env)
                _, _, terminated, truncated, _ = env.step(action)

            mission_passes += int(command.mission_gate_passed[0].item())
            truth_passes += int(command.gate_passed[0].item())

            truth_diag = _diagnostic_truth_error(raw_env)
            maximum_position_error = max(
                maximum_position_error,
                truth_diag["position_error_m"],
            )
            maximum_velocity_error = max(
                maximum_velocity_error,
                truth_diag["velocity_error_mps"],
            )

            if step == 0 or (step + 1) % int(args_cli.progress_every) == 0:
                print(
                    "[estimated-state-smoke] "
                    f"step={step + 1}/{total_steps} "
                    f"gate={int(command.next_gate_idx[0].item())} "
                    f"mission_passes={mission_passes} "
                    f"truth_passes={truth_passes} "
                    f"p_err={truth_diag['position_error_m']:.3f}m "
                    f"v_err={truth_diag['velocity_error_mps']:.3f}m/s",
                    flush=True,
                )

            done = bool(terminated.reshape(-1)[0].item()) or bool(
                truncated.reshape(-1)[0].item()
            )
            if done:
                terminated_early = True
                termination_step = step + 1
                break

        summary = {
            "task": args_cli.task,
            "duration_requested_s": float(args_cli.duration_s),
            "steps_executed": (
                termination_step if termination_step is not None else total_steps
            ),
            "step_dt_s": float(raw_env.step_dt),
            "mission_gate_passes": int(mission_passes),
            "truth_gate_passes": int(truth_passes),
            "final_target_gate_index": int(command.next_gate_idx[0].item()),
            "maximum_position_error_m": float(maximum_position_error),
            "maximum_velocity_error_mps": float(maximum_velocity_error),
            "terminated_early": bool(terminated_early),
            "last_control": last_control,
            "controller_uses_simulator_root_state": False,
            "mission_progression_uses_simulator_root_state": False,
        }

        print("=" * 90)
        print("ESTIMATED-STATE CLOSED-LOOP SMOKE SUMMARY")
        print("=" * 90)
        print(json.dumps(summary, indent=2))

        if args_cli.output_json is not None:
            output = args_cli.output_json.expanduser().resolve()
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps(summary, indent=2) + "\n")

        if mission_passes < 1:
            raise RuntimeError(
                "closed-loop smoke did not pass any gate from estimated state"
            )
    finally:
        env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
