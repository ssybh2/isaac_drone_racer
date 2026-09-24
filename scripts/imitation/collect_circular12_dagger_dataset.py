"""Collect DAgger labels on states visited by the current BC student."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--task",
    default="Isaac-Drone-Racer-Swift-CTBR-GT-Circular12-ExpertDemo-v0",
)
parser.add_argument("--student", type=Path, required=True)
parser.add_argument("--episodes", type=int, default=10)
parser.add_argument("--target-speed-mps", type=float, default=14.0)
parser.add_argument(
    "--mixing-mode",
    choices=["bernoulli", "safety"],
    default="safety",
    help=(
        "DAgger execution policy. 'bernoulli' reproduces per-step random "
        "expert mixing; 'safety' lets the student fly continuously until a "
        "state-quality threshold is crossed, then keeps the expert in control "
        "until the trajectory has recovered."
    ),
)
parser.add_argument(
    "--expert-prob",
    type=float,
    default=0.25,
    help="Bernoulli-mode probability of executing the expert action.",
)
parser.add_argument("--safety-radial-error-m", type=float, default=0.35)
parser.add_argument("--safety-height-error-m", type=float, default=0.20)
parser.add_argument("--safety-speed-error-mps", type=float, default=1.50)
parser.add_argument("--safety-attitude-error-deg", type=float, default=15.0)
parser.add_argument("--safety-body-rate-radps", type=float, default=2.50)
parser.add_argument("--safety-gate-plane-distance-m", type=float, default=1.5)
parser.add_argument("--safety-gate-center-limit-m", type=float, default=0.40)
parser.add_argument(
    "--recovery-hold-steps",
    type=int,
    default=25,
    help="Consecutive safe expert-controlled steps required before returning control to the student.",
)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--seed", type=int, default=1)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if not 0.0 <= args.expert_prob <= 1.0:
    parser.error("--expert-prob must be in [0, 1]")
if args.recovery_hold_steps < 1:
    parser.error("--recovery-hold-steps must be positive")
for name in (
    "safety_radial_error_m",
    "safety_height_error_m",
    "safety_speed_error_mps",
    "safety_attitude_error_deg",
    "safety_body_rate_radps",
    "safety_gate_plane_distance_m",
    "safety_gate_center_limit_m",
):
    if float(getattr(args, name)) <= 0.0:
        parser.error(f"--{name.replace('_', '-')} must be positive")

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
from imitation.dataset import save_dataset  # noqa: E402


def _gate_clearance_metrics(
    position_w: torch.Tensor,
    gate_pose_w: torch.Tensor,
    *,
    plane_distance_m: float,
    center_limit_m: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return absolute gate-frame offsets and near-plane clearance risk."""
    gate_pos_w = gate_pose_w[:, :3]
    gate_quat_w = gate_pose_w[:, 3:7]
    position_gate = math_utils.quat_apply(
        math_utils.quat_inv(gate_quat_w), position_w - gate_pos_w
    )
    plane_distance = position_gate[:, 0].abs()
    lateral_error = position_gate[:, 1].abs()
    vertical_error = position_gate[:, 2].abs()
    clearance_unsafe = (
        (plane_distance <= plane_distance_m)
        & ((lateral_error > center_limit_m) | (vertical_error > center_limit_m))
    )
    return plane_distance, lateral_error, vertical_error, clearance_unsafe


def main() -> None:
    env_cfg = parse_env_cfg(
        args.task, device=args.device, num_envs=1
    )
    env_cfg.scene.num_envs = 1
    env_cfg.seed = int(args.seed)
    if getattr(env_cfg.events, "reset_base", None) is not None:
        env_cfg.events.reset_base.params["target_speed_mps"] = float(
            args.target_speed_mps
        )
    env = gym.make(args.task, cfg=env_cfg)
    raw = env.unwrapped
    wrapped = SkrlVecEnvWrapper(env, ml_framework="torch")

    student, student_metadata = Circular12BCPolicy.load(
        args.student, map_location=raw.device
    )
    trained_speed = student_metadata.get("target_speed_mps")
    if trained_speed is not None and abs(
        float(trained_speed) - float(args.target_speed_mps)
    ) > 1.0e-6:
        raise ValueError(
            "DAgger target speed must match the student BC checkpoint. "
            f"checkpoint={trained_speed} requested={args.target_speed_mps}. "
            "Finish this DAgger round before advancing the speed curriculum."
        )
    student = student.to(raw.device).eval()
    command = raw.command_manager.get_term("target")
    expert_cfg = config_from_ctbr_action_cfg(
        raw.cfg.actions.control_action,
        target_speed_mps=float(args.target_speed_mps),
        height_m=2.07,
        radius_m=12.0,
    )
    generator = torch.Generator(device=raw.device)
    generator.manual_seed(int(args.seed))

    obs, _ = wrapped.reset()
    observations: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    student_actions: list[np.ndarray] = []
    executed_actions: list[np.ndarray] = []
    expert_executed: list[bool] = []
    radial_error_m: list[float] = []
    height_error_m: list[float] = []
    speed_error_mps: list[float] = []
    attitude_error_deg: list[float] = []
    body_rate_radps: list[float] = []
    gate_plane_distance_m: list[float] = []
    gate_lateral_error_m: list[float] = []
    gate_vertical_error_m: list[float] = []
    gate_clearance_unsafe: list[bool] = []
    episode_ids: list[int] = []
    episode_steps: list[int] = []

    episode = 0
    step = 0
    expert_recovery_active = False
    recovered_safe_steps = 0
    try:
        while episode < int(args.episodes):
            robot = raw.scene["robot"]
            p_w = robot.data.root_pos_w
            v_w = robot.data.root_lin_vel_w
            R_wb = math_utils.matrix_from_quat(robot.data.root_quat_w)
            expert = circular12_expert_action(
                p_w, v_w, R_wb, expert_cfg
            )
            with torch.inference_mode():
                student_action = student(obs)
            radius_now = torch.linalg.vector_norm(
                p_w[:, :2]
                - torch.tensor(
                    [[0.0, 12.0]], dtype=p_w.dtype, device=p_w.device
                ),
                dim=-1,
            )
            radial_err = torch.abs(radius_now - 12.0)
            height_err = torch.abs(p_w[:, 2] - 2.07)
            speed_err = torch.abs(
                torch.linalg.vector_norm(v_w[:, :2], dim=-1)
                - float(args.target_speed_mps)
            )
            attitude_err = torch.rad2deg(
                torch.linalg.vector_norm(
                    expert.attitude_error_rotvec_b, dim=-1
                )
            )
            body_rate = torch.linalg.vector_norm(
                robot.data.root_ang_vel_b, dim=-1
            )
            (
                gate_plane_distance,
                gate_lateral_error,
                gate_vertical_error,
                clearance_unsafe,
            ) = _gate_clearance_metrics(
                p_w,
                command.command,
                plane_distance_m=float(args.safety_gate_plane_distance_m),
                center_limit_m=float(args.safety_gate_center_limit_m),
            )

            unsafe = bool(
                (
                    (radial_err[0] > float(args.safety_radial_error_m))
                    | (height_err[0] > float(args.safety_height_error_m))
                    | (speed_err[0] > float(args.safety_speed_error_mps))
                    | (
                        attitude_err[0]
                        > float(args.safety_attitude_error_deg)
                    )
                    | (
                        body_rate[0]
                        > float(args.safety_body_rate_radps)
                    )
                    | clearance_unsafe[0]
                ).item()
            )

            if args.mixing_mode == "bernoulli":
                use_expert = bool(
                    (
                        torch.rand(
                            (), generator=generator, device=raw.device
                        ) < float(args.expert_prob)
                    ).item()
                )
            else:
                if unsafe:
                    expert_recovery_active = True
                    recovered_safe_steps = 0
                elif expert_recovery_active:
                    recovered_safe_steps += 1
                    if recovered_safe_steps >= int(args.recovery_hold_steps):
                        expert_recovery_active = False
                        recovered_safe_steps = 0
                use_expert = expert_recovery_active

            executed = expert.action if use_expert else student_action

            observations.append(
                obs.detach().cpu().numpy().reshape(-1).astype(np.float32)
            )
            labels.append(
                expert.action.detach().cpu().numpy().reshape(-1).astype(np.float32)
            )
            student_actions.append(
                student_action.detach().cpu().numpy().reshape(-1).astype(np.float32)
            )
            executed_actions.append(
                executed.detach().cpu().numpy().reshape(-1).astype(np.float32)
            )
            expert_executed.append(use_expert)
            radial_error_m.append(float(radial_err[0].item()))
            height_error_m.append(float(height_err[0].item()))
            speed_error_mps.append(float(speed_err[0].item()))
            attitude_error_deg.append(float(attitude_err[0].item()))
            body_rate_radps.append(float(body_rate[0].item()))
            gate_plane_distance_m.append(float(gate_plane_distance[0].item()))
            gate_lateral_error_m.append(float(gate_lateral_error[0].item()))
            gate_vertical_error_m.append(float(gate_vertical_error[0].item()))
            gate_clearance_unsafe.append(bool(clearance_unsafe[0].item()))
            episode_ids.append(episode)
            episode_steps.append(step)

            obs, _, terminated, truncated, _ = wrapped.step(executed)
            step += 1
            done = bool(terminated.reshape(-1)[0].item()) or bool(
                truncated.reshape(-1)[0].item()
            )
            if not done:
                continue
            episode += 1
            step = 0
            expert_recovery_active = False
            recovered_safe_steps = 0

            save_dataset(
                args.output,
                {
                    "observation": np.stack(observations),
                    "expert_action": np.stack(labels),
                    "student_action": np.stack(student_actions),
                    "executed_action": np.stack(executed_actions),
                    "expert_executed": np.asarray(
                        expert_executed, dtype=np.bool_
                    ),
                    "radial_error_m": np.asarray(
                        radial_error_m, dtype=np.float32
                    ),
                    "height_error_m": np.asarray(
                        height_error_m, dtype=np.float32
                    ),
                    "speed_error_mps": np.asarray(
                        speed_error_mps, dtype=np.float32
                    ),
                    "attitude_error_deg": np.asarray(
                        attitude_error_deg, dtype=np.float32
                    ),
                    "body_rate_radps": np.asarray(
                        body_rate_radps, dtype=np.float32
                    ),
                    "gate_plane_distance_m": np.asarray(
                        gate_plane_distance_m, dtype=np.float32
                    ),
                    "gate_lateral_error_m": np.asarray(
                        gate_lateral_error_m, dtype=np.float32
                    ),
                    "gate_vertical_error_m": np.asarray(
                        gate_vertical_error_m, dtype=np.float32
                    ),
                    "gate_clearance_unsafe": np.asarray(
                        gate_clearance_unsafe, dtype=np.bool_
                    ),
                    "episode_id": np.asarray(
                        episode_ids, dtype=np.int32
                    ),
                    "episode_step": np.asarray(
                        episode_steps, dtype=np.int32
                    ),
                    "target_speed_mps": np.asarray(
                        float(args.target_speed_mps), dtype=np.float32
                    ),
                    "mixing_mode": np.asarray(str(args.mixing_mode)),
                    "expert_probability": np.asarray(
                        float(args.expert_prob), dtype=np.float32
                    ),
                    "safety_radial_error_m": np.asarray(
                        float(args.safety_radial_error_m), dtype=np.float32
                    ),
                    "safety_height_error_m": np.asarray(
                        float(args.safety_height_error_m), dtype=np.float32
                    ),
                    "safety_speed_error_mps": np.asarray(
                        float(args.safety_speed_error_mps), dtype=np.float32
                    ),
                    "safety_attitude_error_deg": np.asarray(
                        float(args.safety_attitude_error_deg), dtype=np.float32
                    ),
                    "safety_body_rate_radps": np.asarray(
                        float(args.safety_body_rate_radps), dtype=np.float32
                    ),
                    "safety_gate_plane_distance_m": np.asarray(
                        float(args.safety_gate_plane_distance_m), dtype=np.float32
                    ),
                    "safety_gate_center_limit_m": np.asarray(
                        float(args.safety_gate_center_limit_m), dtype=np.float32
                    ),
                    "recovery_hold_steps": np.asarray(
                        int(args.recovery_hold_steps), dtype=np.int32
                    ),
                    "schema_version": np.asarray(
                        "circular12_dagger_dataset_v2"
                    ),
                },
            )
            print(
                f"[dagger] episode {episode}/{args.episodes} "
                f"samples={len(observations)} "
                f"expert_fraction={float(np.mean(expert_executed)):.3f} "
                f"mode={args.mixing_mode} output={args.output}",
                flush=True,
            )
    finally:
        wrapped.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
