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
parser.add_argument("--lateral_range", type=float, default=1.25)
parser.add_argument("--vertical_range", type=float, default=0.8)
parser.add_argument("--yaw_range", type=float, default=0.22)
parser.add_argument("--pitch_range", type=float, default=0.15)
parser.add_argument("--roll_range", type=float, default=0.18)
parser.add_argument("--random_gates", action="store_true")
parser.add_argument("--randomize_images", action="store_true")
parser.add_argument("--randomize_lighting", action="store_true")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if not args.enable_cameras:
    parser.error("Stage2 dataset collection requires --enable_cameras")
app = AppLauncher(args).app

sys.path.insert(0, str(Path(__file__).parents[2]))

import gymnasium as gym
import cv2
import numpy as np
import omni.usd
import torch
import isaaclab.utils.math as math_utils
from pxr import Gf

import tasks  # noqa: F401,E402
from perception.dataset import Stage2DatasetWriter  # noqa: E402
from perception.dataset_collector import IsaacStage2DatasetCollector  # noqa: E402
from perception.stage2_calibration import (  # noqa: E402
    CAMERA_MODEL,
    GATE_KEYPOINT_CALIBRATION_PATH,
    load_stage2_gate_geometry,
)
from tasks.drone_racer.drone_racer_stage2_env_cfg import DroneRacerStage2DataEnvCfg  # noqa: E402


def image_randomizer(seed: int):
    """Create a deterministic photometric transform that preserves keypoint geometry."""
    rng = np.random.default_rng(seed)

    def transform(rgb: np.ndarray) -> tuple[np.ndarray, dict]:
        gain = float(rng.uniform(0.65, 1.35))
        contrast = float(rng.uniform(0.8, 1.2))
        noise_std = float(rng.uniform(0.0, 6.0))
        blur_kind = "none"
        image = np.asarray(rgb, dtype=np.float32)
        channel_mean = image.mean(axis=(0, 1), keepdims=True)
        image = (image - channel_mean) * contrast + channel_mean
        image *= gain
        draw = float(rng.random())
        if draw < 0.15:
            sigma = float(rng.uniform(0.4, 1.2))
            image = cv2.GaussianBlur(image, (5, 5), sigmaX=sigma)
            blur_kind = f"gaussian_sigma_{sigma:.4f}"
        elif draw < 0.25:
            kernel_size = int(rng.choice((3, 5, 7)))
            kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
            if rng.random() < 0.5:
                kernel[kernel_size // 2, :] = 1.0 / kernel_size
                direction = "horizontal"
            else:
                kernel[:, kernel_size // 2] = 1.0 / kernel_size
                direction = "vertical"
            image = cv2.filter2D(image, -1, kernel)
            blur_kind = f"motion_{direction}_{kernel_size}px"
        if noise_std > 0.0:
            image += rng.normal(0.0, noise_std, size=image.shape).astype(np.float32)
        return np.clip(image, 0.0, 255.0).astype(np.uint8), {
            "brightness_gain": gain,
            "contrast": contrast,
            "noise_std_8bit": noise_std,
            "blur": blur_kind,
        }

    return transform


def set_random_light(generator: torch.Generator) -> dict:
    """Randomize the shared dome light once per rendered batch."""
    intensity = float(
        torch.empty(1, device=generator.device).uniform_(1800.0, 4200.0, generator=generator).item()
    )
    color = (
        torch.empty(3, device=generator.device)
        .uniform_(0.65, 1.0, generator=generator)
        .cpu()
        .tolist()
    )
    prim = omni.usd.get_context().get_stage().GetPrimAtPath("/World/Light")
    if not prim.IsValid():
        raise RuntimeError("Expected Stage2 dome light at /World/Light")
    prim.GetAttribute("inputs:intensity").Set(intensity)
    prim.GetAttribute("inputs:color").Set(Gf.Vec3f(*color))
    return {"dome_intensity": intensity, "dome_color_rgb": color}


def main() -> None:
    if args.num_envs <= 0 or args.batches <= 0:
        raise ValueError("num_envs and batches must be positive")
    if not 0.5 < args.min_distance < args.max_distance:
        raise ValueError("Require 0.5 < min_distance < max_distance")
    if min(args.lateral_range, args.vertical_range, args.yaw_range, args.pitch_range, args.roll_range) < 0.0:
        raise ValueError("Pose randomization ranges must be non-negative")

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
    geometry = load_stage2_gate_geometry()
    collector = IsaacStage2DatasetCollector(
        unwrapped,
        geometry,
        writer,
        image_transform=image_randomizer(args.seed) if args.randomize_images else None,
    )
    generator = torch.Generator(device=unwrapped.device).manual_seed(args.seed)
    robot = unwrapped.scene["robot"]
    track = unwrapped.scene["track"]
    command = unwrapped.command_manager.get_term("target")
    env_ids = torch.arange(args.num_envs, device=unwrapped.device)
    complete_samples = 0
    partial_samples = 0
    invisible_samples = 0
    represented_gate_indices: set[int] = set()
    print(
        f"DATA_ENV_READY envs={args.num_envs} resolution={args.width}x{args.height} "
        f"target_samples={args.batches * args.num_envs}",
        flush=True,
    )

    try:
        for batch in range(args.batches):
            if args.random_gates:
                gate_indices = torch.randint(
                    track.data.object_pos_w.shape[1],
                    (args.num_envs,),
                    device=unwrapped.device,
                    generator=generator,
                )
            else:
                gate_indices = torch.zeros(args.num_envs, dtype=torch.long, device=unwrapped.device)
            command.next_gate_idx.copy_(gate_indices)
            gate_pos = track.data.object_pos_w[env_ids, gate_indices]
            gate_quat = track.data.object_quat_w[env_ids, gate_indices]

            distance = torch.empty(args.num_envs, device=unwrapped.device).uniform_(
                args.min_distance, args.max_distance, generator=generator
            )
            lateral = torch.empty(args.num_envs, device=unwrapped.device).uniform_(
                -args.lateral_range, args.lateral_range, generator=generator
            )
            vertical = torch.empty(args.num_envs, device=unwrapped.device).uniform_(
                -args.vertical_range, args.vertical_range, generator=generator
            )
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
            yaw_perturbation = torch.empty(args.num_envs, device=unwrapped.device).uniform_(
                -args.yaw_range, args.yaw_range, generator=generator
            )
            pitch_perturbation = torch.empty(args.num_envs, device=unwrapped.device).uniform_(
                -args.pitch_range, args.pitch_range, generator=generator
            )
            yaw += yaw_perturbation
            pitch += pitch_perturbation
            roll = torch.empty(args.num_envs, device=unwrapped.device).uniform_(
                -args.roll_range, args.roll_range, generator=generator
            )
            orientation = math_utils.quat_from_euler_xyz(roll, pitch, yaw)

            light_metadata = set_random_light(generator) if args.randomize_lighting else None
            extras = []
            for env_id in range(args.num_envs):
                sample_metadata = {
                    "approach_distance_m": float(distance[env_id]),
                    "lateral_offset_m": float(lateral[env_id]),
                    "vertical_offset_m": float(vertical[env_id]),
                    "yaw_perturbation_rad": float(yaw_perturbation[env_id]),
                    "pitch_perturbation_rad": float(pitch_perturbation[env_id]),
                    "roll_rad": float(roll[env_id]),
                }
                if light_metadata is not None:
                    sample_metadata["lighting_randomization"] = light_metadata
                extras.append(sample_metadata)

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
            written_samples = collector.capture_batch(
                f"batch{batch:05d}", range(args.num_envs), extras=extras
            )
            for _, label_path in written_samples:
                label = json.loads(label_path.read_text(encoding="utf-8"))
                visible_count = sum(bool(value) for value in label["visible"])
                complete_samples += visible_count == 4
                partial_samples += 0 < visible_count < 4
                invisible_samples += visible_count == 0
                represented_gate_indices.add(int(label["gate_index"]))
            if batch == 0 or (batch + 1) % 10 == 0:
                print(f"COLLECTED {(batch + 1) * args.num_envs}/{args.batches * args.num_envs}", flush=True)

        manifest = {
            "schema": "isaac_drone_racer.stage2_dataset_manifest.v2",
            "samples": args.batches * args.num_envs,
            "seed": args.seed,
            "image_width": args.width,
            "image_height": args.height,
            "num_envs": args.num_envs,
            "batches": args.batches,
            "distance_m": [args.min_distance, args.max_distance],
            "lateral_offset_m": [-args.lateral_range, args.lateral_range],
            "vertical_offset_m": [-args.vertical_range, args.vertical_range],
            "yaw_perturbation_rad": [-args.yaw_range, args.yaw_range],
            "pitch_perturbation_rad": [-args.pitch_range, args.pitch_range],
            "roll_rad": [-args.roll_range, args.roll_range],
            "gate_sampling": "uniform_random" if args.random_gates else "fixed_gate_0",
            "gate_indices": sorted(represented_gate_indices),
            "complete_samples": complete_samples,
            "partial_samples": partial_samples,
            "fully_out_of_frame_samples": invisible_samples,
            "visibility_semantics": "geometric_in_front_and_in_frame",
            "domain_randomization": {
                "dome_light_intensity": [1800.0, 4200.0] if args.randomize_lighting else None,
                "dome_light_color_rgb": [0.65, 1.0] if args.randomize_lighting else None,
                "brightness_gain": [0.65, 1.35] if args.randomize_images else None,
                "contrast": [0.8, 1.2] if args.randomize_images else None,
                "image_noise_std_8bit": [0.0, 6.0] if args.randomize_images else None,
                "gaussian_blur_probability": 0.15 if args.randomize_images else 0.0,
                "motion_blur_probability": 0.10 if args.randomize_images else 0.0,
                "gate_material": None,
                "camera_calibration": None,
            },
            "gate_calibration": str(GATE_KEYPOINT_CALIBRATION_PATH.relative_to(Path(__file__).parents[2])),
            "camera_model": CAMERA_MODEL,
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
