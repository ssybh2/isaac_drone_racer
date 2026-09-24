"""Record FPV for a deterministic BC episode replay."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--task",
    default="Isaac-Drone-Racer-Swift-CTBR-GT-Circular12-ExpertDemo-v0",
)
parser.add_argument("--checkpoint", type=Path, required=True)
parser.add_argument("--episode-index", type=int, default=19)
parser.add_argument("--target-speed-mps", type=float, default=14.0)
parser.add_argument("--camera-pitch-up-deg", type=float, default=20.0)
parser.add_argument("--fps", type=int, default=25)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--seed", type=int, default=1)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.episode_index < 1:
    parser.error("--episode-index must be >= 1")
if args.fps <= 0:
    parser.error("--fps must be positive")
args.enable_cameras = True
simulation_app = AppLauncher(args).app

import cv2  # noqa: E402
import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from isaaclab_rl.skrl import SkrlVecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

import tasks  # noqa: F401,E402
from imitation.bc_policy import Circular12BCPolicy  # noqa: E402
from perception.isaac_adapter import IsaacStage2TruthAdapter  # noqa: E402
from tasks.drone_racer.drone_racer_stage2_env_cfg import (  # noqa: E402
    stage2_reference_camera_cfg,
)


def _done(terminated: torch.Tensor, truncated: torch.Tensor) -> bool:
    return bool(terminated.reshape(-1)[0].item()) or bool(
        truncated.reshape(-1)[0].item()
    )


def main() -> None:
    env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=1)
    env_cfg.scene.num_envs = 1
    env_cfg.seed = int(args.seed)
    if getattr(env_cfg.events, "reset_base", None) is not None:
        env_cfg.events.reset_base.params["target_speed_mps"] = float(
            args.target_speed_mps
        )

    camera_cfg = stage2_reference_camera_cfg(
        pitch_up_deg=float(args.camera_pitch_up_deg)
    )
    camera_cfg.width = 256
    camera_cfg.height = 256
    env_cfg.scene.tiled_camera = camera_cfg
    env_cfg.scene.collision_sensor.debug_vis = False
    env_cfg.commands.target.debug_vis = False

    env = gym.make(args.task, cfg=env_cfg)
    raw = env.unwrapped
    wrapped = SkrlVecEnvWrapper(env, ml_framework="torch")
    adapter = IsaacStage2TruthAdapter(raw)

    policy, metadata = Circular12BCPolicy.load(
        args.checkpoint, map_location=raw.device
    )
    trained_speed = metadata.get("target_speed_mps")
    if trained_speed is not None and abs(
        float(trained_speed) - float(args.target_speed_mps)
    ) > 1.0e-6:
        raise ValueError(
            f"checkpoint speed={trained_speed} != replay speed={args.target_speed_mps}"
        )
    policy = policy.to(raw.device).eval()

    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output),
        cv2.VideoWriter_fourcc(*"mp4v"),
        int(args.fps),
        (256, 256),
    )
    if not writer.isOpened():
        raise RuntimeError(f"cannot open video writer: {output}")

    step_dt = float(raw.step_dt)
    capture_every = max(1, int(round(1.0 / (step_dt * float(args.fps)))))
    effective_fps = 1.0 / (step_dt * capture_every)

    obs, _ = wrapped.reset()
    episode = 1
    episode_step = 0
    frames = 0
    recorded_steps = 0
    termination = "unknown"

    try:
        while episode <= int(args.episode_index):
            with torch.inference_mode():
                action = policy(obs)
            obs, _, terminated, truncated, _ = wrapped.step(action)
            episode_step += 1
            done = _done(terminated, truncated)

            if episode == int(args.episode_index):
                if episode_step % capture_every == 0 or done:
                    rgb = adapter.rgb(0)
                    writer.write(
                        cv2.cvtColor(
                            np.ascontiguousarray(rgb),
                            cv2.COLOR_RGB2BGR,
                        )
                    )
                    frames += 1
                recorded_steps += 1

            if not done:
                continue

            if episode == int(args.episode_index):
                manager = raw.termination_manager
                fired = {
                    name
                    for name in manager.active_terms
                    if bool(manager.get_term(name).reshape(-1)[0].item())
                }
                if "collision" in fired:
                    termination = "collision"
                elif "flyaway" in fired:
                    termination = "flyaway"
                elif "time_out" in fired:
                    termination = "timeout"
                else:
                    termination = "other"
                break

            print(
                f"[fpv-replay] skipped episode {episode}/"
                f"{args.episode_index - 1}",
                flush=True,
            )
            episode += 1
            episode_step = 0

        sidecar = {
            "checkpoint": str(args.checkpoint.expanduser().resolve()),
            "task": args.task,
            "seed": int(args.seed),
            "episode_index": int(args.episode_index),
            "target_speed_mps": float(args.target_speed_mps),
            "camera_pitch_up_deg": float(args.camera_pitch_up_deg),
            "requested_fps": int(args.fps),
            "effective_fps": float(effective_fps),
            "capture_every_control_steps": int(capture_every),
            "recorded_control_steps": int(recorded_steps),
            "frames": int(frames),
            "termination": termination,
            "video": str(output),
        }
        output.with_suffix(".json").write_text(
            json.dumps(sidecar, indent=2) + "\n"
        )
        print(json.dumps(sidecar, indent=2), flush=True)
        print(f"[fpv-replay] saved: {output}", flush=True)
    finally:
        writer.release()
        wrapped.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
