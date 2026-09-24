"""Collect physics-valid Circular-12 demonstrations from the GT CTBR expert."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--task",
    default="Isaac-Drone-Racer-Swift-CTBR-GT-Circular12-ImitationFineTune-v0",
)
parser.add_argument("--episodes", type=int, default=20)
parser.add_argument("--target-speed-mps", type=float, default=14.0)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--seed", type=int, default=1)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.episodes <= 0:
    parser.error("--episodes must be positive")
if args.target_speed_mps <= 0.0:
    parser.error("--target-speed-mps must be positive")

simulation_app = AppLauncher(args).app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
import isaaclab.utils.math as math_utils  # noqa: E402
from isaaclab_rl.skrl import SkrlVecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

import tasks  # noqa: F401,E402
from imitation.circular12_expert import (  # noqa: E402
    circular12_expert_action,
    config_from_ctbr_action_cfg,
)
from imitation.dataset import save_dataset  # noqa: E402


def main() -> None:
    env_cfg = parse_env_cfg(
        args.task, device=args.device, num_envs=1
    )
    env_cfg.scene.num_envs = 1
    env_cfg.seed = int(args.seed)
    env = gym.make(args.task, cfg=env_cfg)
    raw = env.unwrapped
    wrapped = SkrlVecEnvWrapper(env, ml_framework="torch")

    action_cfg = raw.cfg.actions.control_action
    expert_cfg = config_from_ctbr_action_cfg(
        action_cfg,
        target_speed_mps=float(args.target_speed_mps),
        height_m=2.07,
        radius_m=12.0,
    )

    obs, _ = wrapped.reset()
    observations: list[np.ndarray] = []
    expert_actions: list[np.ndarray] = []
    episode_ids: list[int] = []
    episode_steps: list[int] = []
    position_error: list[float] = []
    velocity_error: list[float] = []
    attitude_error: list[float] = []
    body_rate_norm: list[float] = []

    episode = 0
    step = 0
    try:
        while episode < int(args.episodes):
            robot = raw.scene["robot"]
            p_w = robot.data.root_pos_w
            v_w = robot.data.root_lin_vel_w
            R_wb = math_utils.matrix_from_quat(robot.data.root_quat_w)
            expert = circular12_expert_action(
                p_w, v_w, R_wb, expert_cfg
            )

            observations.append(
                obs.detach().cpu().numpy().reshape(-1).astype(np.float32)
            )
            expert_actions.append(
                expert.action.detach().cpu().numpy().reshape(-1).astype(np.float32)
            )
            episode_ids.append(episode)
            episode_steps.append(step)
            position_error.append(
                float(
                    torch.linalg.vector_norm(
                        expert.reference_position_w - p_w, dim=-1
                    )[0].item()
                )
            )
            velocity_error.append(
                float(
                    torch.linalg.vector_norm(
                        expert.reference_velocity_w - v_w, dim=-1
                    )[0].item()
                )
            )
            attitude_error.append(
                float(
                    torch.linalg.vector_norm(
                        expert.attitude_error_rotvec_b, dim=-1
                    )[0].item()
                )
            )
            body_rate_norm.append(
                float(
                    torch.linalg.vector_norm(
                        robot.data.root_ang_vel_b, dim=-1
                    )[0].item()
                )
            )

            obs, _, terminated, truncated, _ = wrapped.step(
                expert.action
            )
            step += 1
            done = bool(terminated.reshape(-1)[0].item()) or bool(
                truncated.reshape(-1)[0].item()
            )
            if not done:
                continue

            episode += 1
            step = 0
            arrays = {
                "observation": np.stack(observations),
                "expert_action": np.stack(expert_actions),
                "episode_id": np.asarray(episode_ids, dtype=np.int32),
                "episode_step": np.asarray(episode_steps, dtype=np.int32),
                "reference_position_error_m": np.asarray(
                    position_error, dtype=np.float32
                ),
                "reference_velocity_error_mps": np.asarray(
                    velocity_error, dtype=np.float32
                ),
                "reference_attitude_error_rad": np.asarray(
                    attitude_error, dtype=np.float32
                ),
                "body_rate_norm_radps": np.asarray(
                    body_rate_norm, dtype=np.float32
                ),
                "target_speed_mps": np.asarray(
                    float(args.target_speed_mps), dtype=np.float32
                ),
                "schema_version": np.asarray(
                    "circular12_expert_dataset_v1"
                ),
            }
            save_dataset(args.output, arrays)
            print(
                f"[expert-data] saved episode {episode}/{args.episodes}: "
                f"{args.output} samples={len(observations)}",
                flush=True,
            )

        summary = {
            "episodes": int(args.episodes),
            "samples": len(observations),
            "target_speed_mps": float(args.target_speed_mps),
            "position_error_rmse_m": float(
                np.sqrt(np.mean(np.square(position_error)))
            ),
            "velocity_error_rmse_mps": float(
                np.sqrt(np.mean(np.square(velocity_error)))
            ),
            "attitude_error_rmse_deg": float(
                np.degrees(np.sqrt(np.mean(np.square(attitude_error))))
            ),
            "mean_body_rate_radps": float(np.mean(body_rate_norm)),
        }
        summary_path = args.output.with_suffix(
            args.output.suffix + ".json"
        )
        summary_path.write_text(json.dumps(summary, indent=2) + "\n")
        print(json.dumps(summary, indent=2), flush=True)
    finally:
        wrapped.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
