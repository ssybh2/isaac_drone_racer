"""Collect randomized Isaac RGB frames and Stage2A oracle corner labels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--num_envs", type=int, default=16)
parser.add_argument("--batches", type=int, default=50)
parser.add_argument("--width", type=int, default=256)
parser.add_argument("--height", type=int, default=256)
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--min_distance", type=float, default=2.5)
parser.add_argument("--max_distance", type=float, default=8.0)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if not args.enable_cameras:
    parser.error("Stage2 dataset collection requires --enable_cameras")
app = AppLauncher(args).app

sys.path.insert(0, str(Path(__file__).parents[2]))

import gymnasium as gym
import torch
import isaaclab.utils.math as math_utils

import tasks  # noqa: F401,E402
from perception.dataset import Stage2DatasetWriter  # noqa: E402
from perception.dataset_collector import IsaacStage2DatasetCollector  # noqa: E402
from perception.gate_usd_config import DEFAULT_GATE_USD, load_gate_keypoint_calibration  # noqa: E402
from tasks.drone_racer.drone_racer_stage2_env_cfg import DroneRacerStage2DataEnvCfg  # noqa: E402


def main() -> None:
    if args.num_envs <= 0 or args.batches <= 0:
        raise ValueError("num_envs and batches must be positive")
    if not 0.5 < args.min_distance < args.max_distance:
        raise ValueError("Require 0.5 < min_distance < max_distance")

    torch.manual_seed(args.seed)
    cfg = DroneRacerStage2DataEnvCfg()
    cfg.scene.num_envs = args.num_envs
    cfg.scene.tiled_camera.width = args.width
    cfg.scene.tiled_camera.height = args.height
    cfg.scene.collision_sensor.debug_vis = False
    cfg.commands.target.debug_vis = False
    cfg.events.push_robot = None
    cfg.seed = args.seed

    env = gym.make("Isaac-Drone-Racer-Stage2-Data-v0", cfg=cfg)
    unwrapped = env.unwrapped
    writer = Stage2DatasetWriter(args.output)
    geometry = load_gate_keypoint_calibration(DEFAULT_GATE_USD.keypoint_calibration_path)
    collector = IsaacStage2DatasetCollector(unwrapped, geometry, writer)
    generator = torch.Generator(device=unwrapped.device).manual_seed(args.seed)
    robot = unwrapped.scene["robot"]
    track = unwrapped.scene["track"]
    command = unwrapped.command_manager.get_term("target")
    env_ids = torch.arange(args.num_envs, device=unwrapped.device)
    print(
        f"DATA_ENV_READY envs={args.num_envs} resolution={args.width}x{args.height} "
        f"target_samples={args.batches * args.num_envs}",
        flush=True,
    )

    try:
        for batch in range(args.batches):
            command.next_gate_idx[:] = 0
            gate_pos = track.data.object_pos_w[:, 0]
            gate_quat = track.data.object_quat_w[:, 0]

            distance = torch.empty(args.num_envs, device=unwrapped.device).uniform_(
                args.min_distance, args.max_distance, generator=generator
            )
            lateral = torch.empty(args.num_envs, device=unwrapped.device).uniform_(-1.25, 1.25, generator=generator)
            vertical = torch.empty(args.num_envs, device=unwrapped.device).uniform_(-0.8, 0.8, generator=generator)
            local_position = torch.stack(
                (-distance, lateral, torch.full_like(distance, float(geometry.center_g[2])) + vertical), dim=-1
            )
            world_position = gate_pos + math_utils.quat_apply(gate_quat, local_position)

            opening_center_local = torch.as_tensor(
                geometry.center_g, dtype=torch.float32, device=unwrapped.device
            ).expand(args.num_envs, -1)
            opening_center_world = gate_pos + math_utils.quat_apply(gate_quat, opening_center_local)
            direction = opening_center_world - world_position
            horizontal = torch.linalg.vector_norm(direction[:, :2], dim=-1)
            yaw = torch.atan2(direction[:, 1], direction[:, 0])
            pitch = -torch.atan2(direction[:, 2], horizontal)
            # Mostly complete gates plus enough edge cases for visibility training.
            yaw += torch.empty(args.num_envs, device=unwrapped.device).uniform_(-0.22, 0.22, generator=generator)
            pitch += torch.empty(args.num_envs, device=unwrapped.device).uniform_(-0.15, 0.15, generator=generator)
            roll = torch.empty(args.num_envs, device=unwrapped.device).uniform_(-0.18, 0.18, generator=generator)
            orientation = math_utils.quat_from_euler_xyz(roll, pitch, yaw)

            robot.write_root_pose_to_sim(torch.cat((world_position, orientation), dim=-1), env_ids=env_ids)
            robot.write_root_velocity_to_sim(torch.zeros((args.num_envs, 6), device=unwrapped.device), env_ids=env_ids)
            if batch == 0:
                print("RENDERING_FIRST_BATCH", flush=True)
            # Dataset collection does not need the RL manager step. Advance one
            # physics/render frame directly so the tiled camera sees the pose
            # written above without rewards, commands, or reset side effects.
            unwrapped.scene.write_data_to_sim()
            unwrapped.sim.step(render=True)
            unwrapped.scene.update(dt=unwrapped.physics_dt)
            if batch == 0:
                print("CAPTURING_FIRST_BATCH", flush=True)
            collector.capture_batch(f"batch{batch:05d}", range(args.num_envs))
            if batch == 0 or (batch + 1) % 10 == 0:
                print(f"COLLECTED {(batch + 1) * args.num_envs}/{args.batches * args.num_envs}", flush=True)

        manifest = {
            "schema": "isaac_drone_racer.stage2_dataset_manifest.v1",
            "samples": args.batches * args.num_envs,
            "seed": args.seed,
            "image_width": args.width,
            "image_height": args.height,
            "num_envs": args.num_envs,
            "batches": args.batches,
            "distance_m": [args.min_distance, args.max_distance],
            "gate_calibration": DEFAULT_GATE_USD.keypoint_calibration_path,
            "camera_model": "pinhole",
        }
        (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        print(f"COLLECTION_COMPLETE output={args.output} samples={manifest['samples']}")
    finally:
        env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        app.close()
