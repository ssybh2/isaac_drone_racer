"""Record FPV for multiple BC episodes and keep the representative one.

All flight-quality metrics and videos are produced in the same camera-enabled
run. This avoids assuming that a numbered episode from a camera-free audit is
reproducible after camera sensors are enabled.
"""

from __future__ import annotations

import argparse
import json
import shutil
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
parser.add_argument("--episodes", type=int, default=20)
parser.add_argument("--target-speed-mps", type=float, default=14.0)
parser.add_argument("--camera-pitch-up-deg", type=float, default=20.0)
parser.add_argument("--fps", type=int, default=25)
parser.add_argument("--output-dir", type=Path, required=True)
parser.add_argument("--seed", type=int, default=1)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.episodes <= 0:
    parser.error("--episodes must be positive")
if args.fps <= 0:
    parser.error("--fps must be positive")
args.enable_cameras = True
simulation_app = AppLauncher(args).app

import cv2  # noqa: E402
import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
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
from perception.isaac_adapter import IsaacStage2TruthAdapter  # noqa: E402
from tasks.drone_racer.drone_racer_stage2_env_cfg import (  # noqa: E402
    stage2_reference_camera_cfg,
)


def _done(terminated: torch.Tensor, truncated: torch.Tensor) -> bool:
    return bool(terminated.reshape(-1)[0].item()) or bool(
        truncated.reshape(-1)[0].item()
    )


def _termination_cause(raw) -> str:
    manager = raw.termination_manager
    fired = {
        name
        for name in manager.active_terms
        if bool(manager.get_term(name).reshape(-1)[0].item())
    }
    if "collision" in fired:
        return "collision"
    if "flyaway" in fired:
        return "flyaway"
    if "time_out" in fired:
        return "timeout"
    return "other"


def _representative_index(rows: list[dict]) -> tuple[int, list[float]]:
    # Standardized Euclidean distance to the run mean. These dimensions span
    # mission progress and the major instability signatures visible in FPV.
    keys = (
        "gates",
        "inversion_events",
        "gross_attitude_excursion_events",
        "body_rate_p95_radps",
        "radius_rmse_m",
        "height_rmse_m",
        "speed_mean_mps",
    )
    values = np.asarray(
        [[float(row[key]) for key in keys] for row in rows],
        dtype=np.float64,
    )
    mean = values.mean(axis=0)
    std = values.std(axis=0)
    std = np.where(std < 1.0e-9, 1.0, std)
    z = (values - mean) / std
    score = np.sum(z * z, axis=1)
    return int(np.argmin(score)), score.tolist()


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

    expert_cfg = config_from_ctbr_action_cfg(
        raw.cfg.actions.control_action,
        target_speed_mps=float(args.target_speed_mps),
        radius_m=12.0,
        height_m=2.07,
    )

    out_dir = args.output_dir.expanduser().resolve()
    clips_dir = out_dir / "episodes"
    clips_dir.mkdir(parents=True, exist_ok=True)

    step_dt = float(raw.step_dt)
    capture_every = max(1, int(round(1.0 / (step_dt * float(args.fps)))))
    effective_fps = 1.0 / (step_dt * capture_every)

    obs, _ = wrapped.reset()
    command = raw.command_manager.get_term("target")
    rows: list[dict] = []
    clip_paths: list[Path] = []

    try:
        for episode in range(1, int(args.episodes) + 1):
            clip_path = clips_dir / f"ep_{episode:02d}.mp4"
            clip_paths.append(clip_path)
            writer = cv2.VideoWriter(
                str(clip_path),
                cv2.VideoWriter_fourcc(*"mp4v"),
                int(args.fps),
                (256, 256),
            )
            if not writer.isOpened():
                raise RuntimeError(f"cannot open video writer: {clip_path}")

            ep_step = 0
            ep_gates = 0
            prev_gate = int(command.next_gate_idx[0].item())
            speed_samples: list[float] = []
            rate_samples: list[float] = []
            radius_error: list[float] = []
            height_error: list[float] = []
            attitude_error: list[float] = []
            inverted_samples = 0
            inversion_events = 0
            gross_events = 0
            inverted_active = False
            gross_active = False
            frames = 0

            try:
                while True:
                    robot = raw.scene["robot"]
                    p_w = robot.data.root_pos_w
                    v_w = robot.data.root_lin_vel_w
                    R_wb = math_utils.matrix_from_quat(
                        robot.data.root_quat_w
                    )
                    expert = circular12_expert_action(
                        p_w, v_w, R_wb, expert_cfg
                    )

                    speed_samples.append(
                        float(torch.linalg.vector_norm(v_w[0, :2]).item())
                    )
                    rate_samples.append(
                        float(
                            torch.linalg.vector_norm(
                                robot.data.root_ang_vel_b[0]
                            ).item()
                        )
                    )
                    radius = float(
                        torch.linalg.vector_norm(
                            p_w[0, :2]
                            - torch.tensor(
                                [0.0, 12.0],
                                dtype=p_w.dtype,
                                device=p_w.device,
                            )
                        ).item()
                    )
                    radius_error.append(radius - 12.0)
                    height_error.append(float(p_w[0, 2].item()) - 2.07)
                    att_err = float(
                        torch.linalg.vector_norm(
                            expert.attitude_error_rotvec_b[0]
                        ).item()
                    )
                    attitude_error.append(att_err)

                    inverted_now = bool((R_wb[0, 2, 2] < 0.0).item())
                    gross_now = att_err > float(np.deg2rad(120.0))
                    if inverted_now and not inverted_active:
                        inversion_events += 1
                    if gross_now and not gross_active:
                        gross_events += 1
                    inverted_active = inverted_now
                    gross_active = gross_now
                    inverted_samples += int(inverted_now)

                    with torch.inference_mode():
                        action = policy(obs)
                    obs, _, terminated, truncated, _ = wrapped.step(action)
                    ep_step += 1
                    done = _done(terminated, truncated)

                    current_gate = int(command.next_gate_idx[0].item())
                    if not done and current_gate != prev_gate:
                        ep_gates += int(
                            (current_gate - prev_gate)
                            % int(command.num_gates)
                        )
                    prev_gate = current_gate

                    if (
                        ep_step == 1
                        or ep_step % capture_every == 0
                        or done
                    ):
                        rgb = adapter.rgb(0)
                        writer.write(
                            cv2.cvtColor(
                                np.ascontiguousarray(rgb),
                                cv2.COLOR_RGB2BGR,
                            )
                        )
                        frames += 1

                    if done:
                        break
            finally:
                writer.release()

            row = {
                "episode": episode,
                "steps": ep_step,
                "frames": frames,
                "duration_s": frames / float(args.fps),
                "gates": ep_gates,
                "termination": _termination_cause(raw),
                "speed_mean_mps": float(np.mean(speed_samples)),
                "body_rate_p95_radps": float(
                    np.percentile(rate_samples, 95)
                ),
                "radius_rmse_m": float(
                    np.sqrt(np.mean(np.square(radius_error)))
                ),
                "height_rmse_m": float(
                    np.sqrt(np.mean(np.square(height_error)))
                ),
                "attitude_error_p95_deg": float(
                    np.degrees(np.percentile(attitude_error, 95))
                ),
                "inverted_fraction": float(
                    inverted_samples / max(ep_step, 1)
                ),
                "inversion_events": int(inversion_events),
                "gross_attitude_excursion_events": int(gross_events),
                "video": str(clip_path),
            }
            rows.append(row)
            print(
                "[fpv-audit] "
                f"ep={episode:02d}/{args.episodes} "
                f"gates={ep_gates} cause={row['termination']} "
                f"inv={inversion_events} gross={gross_events} "
                f"rate95={row['body_rate_p95_radps']:.3f} "
                f"video={clip_path.name}",
                flush=True,
            )

        selected, scores = _representative_index(rows)
        selected_row = dict(rows[selected])
        representative_path = out_dir / "fpv_representative.mp4"
        shutil.copy2(clip_paths[selected], representative_path)

        summary = {
            "checkpoint": str(args.checkpoint.expanduser().resolve()),
            "task": args.task,
            "episodes": int(args.episodes),
            "seed": int(args.seed),
            "target_speed_mps": float(args.target_speed_mps),
            "camera_pitch_up_deg": float(args.camera_pitch_up_deg),
            "requested_fps": int(args.fps),
            "effective_fps": float(effective_fps),
            "capture_every_control_steps": int(capture_every),
            "representative_episode": int(selected_row["episode"]),
            "representative_score": float(scores[selected]),
            "representative_metrics": selected_row,
            "representative_video": str(representative_path),
            "selection_method": (
                "minimum standardized Euclidean distance to the run mean over "
                "gates, inversion events, gross attitude excursions, body-rate "
                "p95, radius RMSE, height RMSE and mean speed"
            ),
            "episodes_metrics": rows,
        }
        (out_dir / "fpv_summary.json").write_text(
            json.dumps(summary, indent=2) + "\n"
        )
        print(
            f"[fpv-audit] representative episode="
            f"{selected_row['episode']} -> {representative_path}",
            flush=True,
        )
        print(json.dumps(summary["representative_metrics"], indent=2))
    finally:
        wrapped.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
