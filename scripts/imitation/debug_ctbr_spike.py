"""Temporary, read-only CTBR/PhysX first-spike probe for Circular-12 BC."""

from __future__ import annotations

import argparse
import json
import sys
from collections import deque
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--checkpoint", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--episodes", type=int, default=5)
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--post-steps", type=int, default=10)
parser.add_argument("--spike-threshold-radps", type=float, default=5.0)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
simulation_app = AppLauncher(args).app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
import isaaclab.utils.math as math_utils  # noqa: E402
from isaaclab_rl.skrl import SkrlVecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

import tasks  # noqa: F401,E402
from imitation.bc_policy import Circular12BCPolicy  # noqa: E402
from imitation.circular12_expert import (  # noqa: E402
    circular12_expert_action,
    config_from_ctbr_action_cfg,
)


def vec(value):
    return value.detach().cpu().reshape(-1).tolist()


def sensor_forces(sensor, dt):
    data = sensor.data
    result = {}
    for name in ("net_forces_w", "net_forces_w_history"):
        value = getattr(data, name, None)
        if value is not None:
            result[name + "_max_n"] = float(torch.linalg.vector_norm(value.reshape(-1, 3), dim=-1).max().item())
    try:
        direct = sensor.contact_physx_view.get_net_contact_forces(dt=dt)
        result["physx_view_max_n"] = float(torch.linalg.vector_norm(direct.reshape(-1, 3), dim=-1).max().item())
    except Exception as error:
        result["physx_view_error"] = str(error)
    return result


def main():
    task = "Isaac-Drone-Racer-Swift-CTBR-GT-Circular12-ExpertDemo-v0"
    cfg = parse_env_cfg(task, device=args.device, num_envs=1)
    cfg.scene.num_envs = 1
    cfg.seed = args.seed
    env = gym.make(task, cfg=cfg)
    raw = env.unwrapped
    wrapped = SkrlVecEnvWrapper(env, ml_framework="torch")
    policy, metadata = Circular12BCPolicy.load(args.checkpoint, map_location=raw.device)
    policy = policy.to(raw.device).eval()
    robot = raw.scene["robot"]
    sensor = raw.scene.sensors["collision_sensor"]
    command = raw.command_manager.get_term("target")
    term = raw.action_manager.get_term("control_action")
    controller = term._rate_controller
    expert_cfg = config_from_ctbr_action_cfg(raw.cfg.actions.control_action, target_speed_mps=14.0, radius_m=12.0, height_m=2.07)
    inertia = robot.root_physx_view.get_inertias().detach().cpu().tolist()

    current_substeps = []
    controller_diag = {}
    original_update = raw.scene.update
    original_process = term.process_actions

    def traced_update(dt):
        original_update(dt)
        current_substeps.append({
            "omega_b": vec(robot.data.root_ang_vel_b[0]),
            "omega_w": vec(robot.data.root_ang_vel_w[0]),
            "position_w": vec(robot.data.root_pos_w[0]),
            "contact": sensor_forces(sensor, raw.physics_dt),
        })

    def traced_process(actions):
        measured = robot.data.root_ang_vel_b.clone()
        desired = actions[:, 1:4].clamp(-1, 1) * term._body_rate_max
        previous = controller.previous_rate.clone()
        initialized = bool(controller.initialized[0, 0].item())
        original_process(actions)
        derivative = (measured - previous) / controller.dt if initialized else torch.zeros_like(measured)
        p = controller.kp * (desired - measured)
        i = controller.ki * controller.integral
        d = -controller.kd * derivative
        controller_diag.clear()
        controller_diag.update({
            "desired_rate_b": vec(desired[0]),
            "measured_rate_b": vec(measured[0]),
            "rate_error_b": vec((desired - measured)[0]),
            "pid_p_nm": vec(p[0]),
            "pid_i_nm": vec(i[0]),
            "pid_d_nm": vec(d[0]),
            "moment_cmd_nm": vec(torch.maximum(torch.minimum(p + i + d, controller.moment_limit), -controller.moment_limit)[0]),
            "desired_wrench": [float(term._ctbr_command[0, 0].item() * term.cfg.vehicle_mass_kg)] + vec(torch.maximum(torch.minimum(p + i + d, controller.moment_limit), -controller.moment_limit)[0]),
            "rotor_thrusts_n": vec(term._rotor_thrusts[0]),
            "omega_ref_radps": vec(term._omega_ref[0]),
            "omega_real_radps": vec(term._omega_real[0]),
            "actual_wrench": vec(term._processed_actions[0]),
        })

    raw.scene.update = traced_update
    term.process_actions = traced_process
    obs, _ = wrapped.reset()
    recent = deque(maxlen=12)
    first_spike = None
    post_remaining = 0
    episode = 0
    step = 0
    try:
        while episode < args.episodes:
            with torch.inference_mode():
                action = policy(obs)
            expert = circular12_expert_action(robot.data.root_pos_w, robot.data.root_lin_vel_w, math_utils.matrix_from_quat(robot.data.root_quat_w), expert_cfg)
            pre_omega = robot.data.root_ang_vel_b[0].detach().clone()
            pre = {
                "episode": episode + 1,
                "step": step,
                "time_s": step * raw.step_dt,
                "gate_index_pre": int(command.next_gate_idx[0].item()),
                "gate_center_pre": vec(command.command[0, :3]),
                "omega_b_pre": vec(pre_omega),
                "omega_w_pre": vec(robot.data.root_ang_vel_w[0]),
                "position_w_pre": vec(robot.data.root_pos_w[0]),
                "velocity_w_pre": vec(robot.data.root_lin_vel_w[0]),
                "quat_wb_pre": vec(robot.data.root_quat_w[0]),
                "contact_pre": sensor_forces(sensor, raw.physics_dt),
                "bc_action": vec(action[0]),
                "expert_action": vec(expert.action[0]),
                "bc_expert_mae": float((action - expert.action).abs().mean().item()),
            }
            current_substeps.clear()
            obs, _, terminated, truncated, _ = wrapped.step(action)
            pre.update(controller_diag)
            pre["external_force_b"] = vec(term._thrust[0, 0])
            pre["external_torque_b"] = vec(term._moment[0, 0])
            pre["substeps"] = list(current_substeps)
            pre["gate_index_post"] = int(command.next_gate_idx[0].item())
            pre["gate_passed"] = bool(command.gate_passed[0].item())
            pre["gate_missed"] = bool(command.gate_missed[0].item())
            pre["contact_post"] = sensor_forces(sensor, raw.physics_dt)
            pre["terminated"] = bool(terminated[0].item())
            pre["truncated"] = bool(truncated[0].item())
            final_omega = torch.tensor(current_substeps[-1]["omega_b"], device=raw.device)
            pre["delta_omega_b"] = vec(final_omega - pre_omega)
            pre["delta_omega_norm"] = float(torch.linalg.vector_norm(final_omega - pre_omega).item())
            recent.append(pre)
            if first_spike is None and pre["delta_omega_norm"] > args.spike_threshold_radps:
                first_spike = {
                    "threshold_radps": args.spike_threshold_radps,
                    "physics_dt_s": raw.physics_dt,
                    "control_dt_s": raw.step_dt,
                    "inertia_raw": inertia,
                    "checkpoint_metadata": {"target_speed_mps": metadata.get("target_speed_mps")},
                    "trace": list(recent),
                }
                post_remaining = args.post_steps
                print(f"FIRST SPIKE ep={episode+1} step={step} delta={pre['delta_omega_norm']:.3f}", flush=True)
            elif first_spike is not None:
                first_spike["trace"].append(pre)
                post_remaining -= 1
                if post_remaining <= 0:
                    break
            step += 1
            if pre["terminated"] or pre["truncated"]:
                episode += 1
                step = 0
                recent.clear()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps({"first_spike": first_spike, "episodes_seen": episode + 1}, indent=2) + "\n")
        print(f"wrote {args.output}; found={first_spike is not None}", flush=True)
    finally:
        wrapped.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
