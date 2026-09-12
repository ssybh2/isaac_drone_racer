"""Run the single-vehicle Isaac -> ROS2 -> OpenVINS diagnostic path without a policy checkpoint."""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Validate Isaac RGB/IMU transport and OpenVINS alignment.")
parser.add_argument("--steps", type=int, default=5000, help="Maximum control steps to run.")
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
import torch

from isaaclab_tasks.utils import parse_env_cfg

import tasks  # noqa: F401


def main() -> None:
    if args_cli.rejection_dump_limit < 0:
        raise ValueError("--rejection_dump_limit must be non-negative")

    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=1)
    if args_cli.detector_checkpoint is not None:
        env_cfg.swift_detector_checkpoint = args_cli.detector_checkpoint
        env_cfg.swift_detector_device = args_cli.detector_device
    env_cfg.swift_use_oracle_gate_index = bool(args_cli.oracle_gate_index)
    env_cfg.swift_rejection_dump_dir = args_cli.rejection_dump_dir
    env_cfg.swift_rejection_dump_limit = int(args_cli.rejection_dump_limit)

    env = gym.make(args_cli.task, cfg=env_cfg)
    raw_env = env.unwrapped
    env.reset()

    action_dim = int(raw_env.action_manager.total_action_dim)
    # In the repository motor mapping, zero normalized command is approximately
    # the configured hover rotor speed. This keeps the transport diagnostic
    # independent of an RL checkpoint.
    actions = torch.zeros((1, action_dim), dtype=torch.float32, device=raw_env.device)

    try:
        for step in range(int(args_cli.steps)):
            if not simulation_app.is_running():
                break
            _, _, terminated, truncated, _ = env.step(actions)
            if step % 100 == 0:
                estimate = raw_env.openvins_vio_estimate
                if estimate is None:
                    print(f"[OpenVINS] step={step}: waiting for initialized odometry")
                else:
                    age = max(0.0, raw_env._timestamp_s() - estimate.timestamp_s)
                    drained = raw_env._openvins_bridge.last_drain_count
                    print(
                        f"[OpenVINS] step={step}: aligned={raw_env.openvins_alignment is not None} "
                        f"age={age:.4f}s drained={drained} pos={estimate.position_w_b.tolist()}"
                    )
                fusion = raw_env.swift_last_fusion_result
                if fusion is not None and not fusion.measurement_accepted:
                    print(
                        f"[SwiftFusion] rejected: {fusion.rejection_reason} "
                        f"d2={fusion.innovation_mahalanobis2}"
                    )
            if bool(torch.as_tensor(terminated).any()) or bool(torch.as_tensor(truncated).any()):
                print(
                    "[OpenVINS] Isaac episode reset/termination occurred. "
                    "Restart ov_msckf before trusting a new aligned segment."
                )
                break
    finally:
        if raw_env._rejection_dump_count:
            print(f"[SwiftFusion] wrote {raw_env._rejection_dump_count} rejected-frame records")
        env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
