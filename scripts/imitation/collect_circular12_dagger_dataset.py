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
    default="Isaac-Drone-Racer-Swift-CTBR-GT-Circular12-ImitationFineTune-v0",
)
parser.add_argument("--student", type=Path, required=True)
parser.add_argument("--episodes", type=int, default=10)
parser.add_argument("--target-speed-mps", type=float, default=14.0)
parser.add_argument(
    "--expert-prob",
    type=float,
    default=0.25,
    help="Probability of executing the expert action; labels are always expert.",
)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--seed", type=int, default=1)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if not 0.0 <= args.expert_prob <= 1.0:
    parser.error("--expert-prob must be in [0, 1]")

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


def main() -> None:
    env_cfg = parse_env_cfg(
        args.task, device=args.device, num_envs=1
    )
    env_cfg.scene.num_envs = 1
    env_cfg.seed = int(args.seed)
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
    episode_ids: list[int] = []
    episode_steps: list[int] = []

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
            with torch.inference_mode():
                student_action = student(obs)
            use_expert = bool(
                (
                    torch.rand(
                        (), generator=generator, device=raw.device
                    ) < float(args.expert_prob)
                ).item()
            )
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
                    "episode_id": np.asarray(
                        episode_ids, dtype=np.int32
                    ),
                    "episode_step": np.asarray(
                        episode_steps, dtype=np.int32
                    ),
                    "target_speed_mps": np.asarray(
                        float(args.target_speed_mps), dtype=np.float32
                    ),
                    "expert_probability": np.asarray(
                        float(args.expert_prob), dtype=np.float32
                    ),
                    "schema_version": np.asarray(
                        "circular12_dagger_dataset_v1"
                    ),
                },
            )
            print(
                f"[dagger] episode {episode}/{args.episodes} "
                f"samples={len(observations)} output={args.output}",
                flush=True,
            )
    finally:
        wrapped.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
