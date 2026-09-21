"""Collect high-speed racing vision data from the frozen GT CTBR policy.

The collector deliberately keeps control independent from perception:
- the frozen GT policy receives the same 31-D simulator-truth observation used
  by the successful racing benchmark;
- a production-calibrated Stage2 camera is enabled only for data capture;
- exact gate-corner labels are generated from simulator geometry;
- failed/short episodes are discarded so the retained dataset represents the
  true high-speed racing distribution.

The resulting directory contains train/val/test Stage2-compatible datasets.
Splits are assigned by complete accepted episode, never by individual frame.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from collections import Counter
from pathlib import Path

import numpy as np

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--task",
    default="Isaac-Drone-Racer-Swift-CTBR-GT-Racing-v0",
)
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--output-dir", type=Path, required=True)
parser.add_argument("--target-successful-episodes", type=int, default=8)
parser.add_argument("--val-episodes", type=int, default=1)
parser.add_argument("--test-episodes", type=int, default=1)
parser.add_argument("--min-gates", type=int, default=30)
parser.add_argument("--max-attempts", type=int, default=40)
parser.add_argument(
    "--capture-every-steps",
    type=int,
    default=4,
    help="Capture every N 100-Hz policy steps. Default 4 gives 25 Hz.",
)
parser.add_argument("--width", type=int, default=256)
parser.add_argument("--height", type=int, default=256)
parser.add_argument("--seed", type=int, default=1)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Camera rendering is required even in headless mode.
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from isaaclab_rl.skrl import SkrlVecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils import load_cfg_from_registry, parse_env_cfg  # noqa: E402
from skrl.utils.runner.torch import Runner  # noqa: E402

import tasks  # noqa: F401,E402
from perception.dataset import Stage2DatasetWriter  # noqa: E402
from perception.dataset_collector import IsaacStage2DatasetCollector  # noqa: E402
from perception.stage2_calibration import load_stage2_gate_geometry  # noqa: E402
from tasks.drone_racer.drone_racer_stage2_env_cfg import (  # noqa: E402
    stage2_reference_camera_cfg,
)


def _termination_cause(raw_env) -> str:
    manager = raw_env.termination_manager
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


def _prepare_output(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val", "test"):
        (root / split / "images").mkdir(parents=True, exist_ok=True)
        (root / split / "labels").mkdir(parents=True, exist_ok=True)
    staging = root / "_staging"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True, exist_ok=True)


def _split_for_success(
    success_index: int,
    *,
    target_episodes: int,
    val_episodes: int,
    test_episodes: int,
) -> str:
    train_episodes = target_episodes - val_episodes - test_episodes
    if success_index < train_episodes:
        return "train"
    if success_index < train_episodes + val_episodes:
        return "val"
    return "test"


def _promote_episode(
    staging_root: Path,
    split_root: Path,
    *,
    accepted_episode_index: int,
    attempt_index: int,
    gates_passed: int,
    termination: str,
) -> tuple[int, Counter, list[float], list[float]]:
    label_paths = sorted((staging_root / "labels").glob("*.json"))
    visibility_hist = Counter()
    speeds: list[float] = []
    body_rates: list[float] = []

    for frame_index, label_path in enumerate(label_paths):
        payload = json.loads(label_path.read_text(encoding="utf-8"))
        extra = dict(payload.get("extra") or {})
        visible_count = int(sum(bool(v) for v in payload["visible"]))
        visibility_hist[visible_count] += 1
        speeds.append(float(extra.get("speed_mps", 0.0)))
        body_rates.append(float(extra.get("body_rate_norm_radps", 0.0)))

        new_id = (
            f"ep{accepted_episode_index:03d}_"
            f"attempt{attempt_index:03d}_frame{frame_index:05d}"
        )
        old_image = staging_root / "images" / f"{label_path.stem}.png"
        new_image = split_root / "images" / f"{new_id}.png"
        new_label = split_root / "labels" / f"{new_id}.json"

        extra.update(
            {
                "accepted_episode_index": int(accepted_episode_index),
                "attempt_index": int(attempt_index),
                "episode_gates_passed": int(gates_passed),
                "episode_termination": termination,
                "dataset_split": split_root.name,
            }
        )
        payload["sample_id"] = new_id
        payload["extra"] = extra
        shutil.move(str(old_image), str(new_image))
        new_label.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    shutil.rmtree(staging_root)
    return len(label_paths), visibility_hist, speeds, body_rates


def _write_split_manifest(
    root: Path,
    split: str,
    episodes: list[dict],
    *,
    width: int,
    height: int,
    capture_every_steps: int,
    step_dt: float,
) -> None:
    labels = list((root / split / "labels").glob("*.json"))
    visibility_hist = Counter()
    speeds: list[float] = []
    body_rates: list[float] = []
    for label_path in labels:
        payload = json.loads(label_path.read_text(encoding="utf-8"))
        visibility_hist[int(sum(bool(v) for v in payload["visible"]))] += 1
        extra = payload.get("extra") or {}
        speeds.append(float(extra.get("speed_mps", 0.0)))
        body_rates.append(float(extra.get("body_rate_norm_radps", 0.0)))

    manifest = {
        "schema": "isaac_drone_racer.racing_vision_split.v1",
        "split": split,
        "samples": len(labels),
        "episodes": episodes,
        "image_width": int(width),
        "image_height": int(height),
        "capture_every_steps": int(capture_every_steps),
        "capture_rate_hz": float(1.0 / (step_dt * capture_every_steps)),
        "visible_corner_histogram": {
            str(k): int(v) for k, v in sorted(visibility_hist.items())
        },
        "speed_mps": {
            "mean": float(np.mean(speeds)) if speeds else 0.0,
            "max": float(np.max(speeds)) if speeds else 0.0,
        },
        "body_rate_norm_radps": {
            "mean": float(np.mean(body_rates)) if body_rates else 0.0,
            "max": float(np.max(body_rates)) if body_rates else 0.0,
        },
    }
    (root / split / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    if args_cli.target_successful_episodes <= 0:
        raise ValueError("--target-successful-episodes must be positive")
    if args_cli.val_episodes < 0 or args_cli.test_episodes < 0:
        raise ValueError("validation/test episode counts must be non-negative")
    if args_cli.val_episodes + args_cli.test_episodes >= args_cli.target_successful_episodes:
        raise ValueError("At least one successful episode must remain for training")
    if args_cli.min_gates <= 0:
        raise ValueError("--min-gates must be positive")
    if args_cli.max_attempts < args_cli.target_successful_episodes:
        raise ValueError("--max-attempts must be >= --target-successful-episodes")
    if args_cli.capture_every_steps <= 0:
        raise ValueError("--capture-every-steps must be positive")

    output_root = args_cli.output_dir.expanduser().resolve()
    _prepare_output(output_root)

    cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=1,
    )
    cfg.scene.num_envs = 1
    cfg.seed = int(args_cli.seed)

    # Re-enable only the production Stage2 camera. No estimator, detector or
    # visual feedback is introduced into the GT racing control loop.
    cfg.scene.tiled_camera = stage2_reference_camera_cfg()
    cfg.scene.tiled_camera.width = int(args_cli.width)
    cfg.scene.tiled_camera.height = int(args_cli.height)
    cfg.scene.collision_sensor.debug_vis = False
    # PhysX GPU contact forces can stay latched after a real collision when
    # ContactSensor uses the default history_length=0 and data is only read at
    # policy rate (this task has decimation=4). A non-zero history forces the
    # contact sensor to refresh every physics step, preventing one collision
    # from poisoning all later dataset episodes with a stale force sample.
    cfg.scene.collision_sensor.history_length = 1
    cfg.commands.target.debug_vis = False

    env = gym.make(args_cli.task, cfg=cfg)
    raw_env = env.unwrapped
    wrapped = SkrlVecEnvWrapper(env, ml_framework="torch")

    agent_cfg = load_cfg_from_registry(args_cli.task, "skrl_cfg_entry_point")
    agent_cfg["seed"] = int(args_cli.seed)
    agent_cfg["trainer"]["close_environment_at_exit"] = False
    agent_cfg["agent"]["experiment"]["write_interval"] = 0
    agent_cfg["agent"]["experiment"]["checkpoint_interval"] = 0

    runner = Runner(wrapped, agent_cfg)
    checkpoint = str(Path(args_cli.checkpoint).expanduser().resolve())
    print(f"[racing-vision] loading checkpoint: {checkpoint}", flush=True)
    runner.agent.load(checkpoint)
    runner.agent.set_running_mode("eval")
    runner.agent.set_mode("eval")

    geometry = load_stage2_gate_geometry()
    command = raw_env.command_manager.get_term("target")
    robot = raw_env.scene["robot"]
    action_term = raw_env.action_manager.get_term("control_action")
    num_gates = int(command.num_gates)

    split_episodes: dict[str, list[dict]] = {
        "train": [],
        "val": [],
        "test": [],
    }
    episode_rows: list[dict] = []
    accepted_count = 0
    attempt_count = 0

    obs, _ = wrapped.reset()
    prev_gate_idx = int(command.next_gate_idx[0].item())

    try:
        while (
            accepted_count < int(args_cli.target_successful_episodes)
            and attempt_count < int(args_cli.max_attempts)
        ):
            attempt_count += 1
            ep_step = 0
            ep_gates = 0
            ep_return = 0.0
            staging_root = output_root / "_staging" / f"attempt_{attempt_count:03d}"
            writer = Stage2DatasetWriter(staging_root)
            collector = IsaacStage2DatasetCollector(
                raw_env,
                geometry,
                writer,
            )

            while True:
                with torch.inference_mode():
                    outputs = runner.agent.act(obs, timestep=0, timesteps=0)
                    actions = outputs[-1].get("mean_actions", outputs[0])

                obs, reward, terminated, truncated, _ = wrapped.step(actions)
                ep_step += 1
                ep_return += float(reward.reshape(-1)[0].item())

                done = bool(terminated.reshape(-1)[0].item()) or bool(
                    truncated.reshape(-1)[0].item()
                )

                current_gate_idx = int(command.next_gate_idx[0].item())
                if not done and current_gate_idx != prev_gate_idx:
                    ep_gates += int((current_gate_idx - prev_gate_idx) % num_gates)
                prev_gate_idx = current_gate_idx

                if not done and ep_step % int(args_cli.capture_every_steps) == 0:
                    velocity_w = robot.data.root_lin_vel_w[0]
                    body_rate_b = robot.data.root_ang_vel_b[0]
                    root_quat = robot.data.root_quat_w[0]
                    ctbr = action_term.ctbr_command[0]
                    extra = {
                        "attempt_index": int(attempt_count),
                        "episode_step": int(ep_step),
                        "gates_passed_so_far": int(ep_gates),
                        "active_gate_index": int(current_gate_idx),
                        "speed_mps": float(torch.linalg.vector_norm(velocity_w).item()),
                        "body_rate_norm_radps": float(
                            torch.linalg.vector_norm(body_rate_b).item()
                        ),
                        "truth_position_w_m": robot.data.root_pos_w[0].detach().cpu().tolist(),
                        "truth_velocity_w_mps": velocity_w.detach().cpu().tolist(),
                        "truth_orientation_wxyz": root_quat.detach().cpu().tolist(),
                        "truth_body_rate_radps": body_rate_b.detach().cpu().tolist(),
                        "normalized_policy_action": actions[0].detach().cpu().tolist(),
                        "ctbr_command": ctbr.detach().cpu().tolist(),
                    }
                    collector.capture(
                        f"attempt{attempt_count:03d}_step{ep_step:06d}",
                        0,
                        extra=extra,
                    )

                if not done:
                    continue

                cause = _termination_cause(raw_env)
                keep = ep_gates >= int(args_cli.min_gates)
                row = {
                    "attempt": int(attempt_count),
                    "accepted": bool(keep),
                    "gates_passed": int(ep_gates),
                    "steps": int(ep_step),
                    "duration_s": float(ep_step * raw_env.step_dt),
                    "return": float(ep_return),
                    "termination": cause,
                }

                if keep:
                    split = _split_for_success(
                        accepted_count,
                        target_episodes=int(args_cli.target_successful_episodes),
                        val_episodes=int(args_cli.val_episodes),
                        test_episodes=int(args_cli.test_episodes),
                    )
                    sample_count, visible_hist, speeds, rates = _promote_episode(
                        staging_root,
                        output_root / split,
                        accepted_episode_index=accepted_count,
                        attempt_index=attempt_count,
                        gates_passed=ep_gates,
                        termination=cause,
                    )
                    row.update(
                        {
                            "split": split,
                            "samples": int(sample_count),
                            "visible_corner_histogram": dict(visible_hist),
                            "speed_max_mps": float(max(speeds, default=0.0)),
                            "body_rate_max_radps": float(max(rates, default=0.0)),
                        }
                    )
                    split_episodes[split].append(dict(row))
                    accepted_count += 1
                    print(
                        "[racing-vision] KEEP "
                        f"attempt={attempt_count:02d} "
                        f"success={accepted_count}/{args_cli.target_successful_episodes} "
                        f"split={split} gates={ep_gates} "
                        f"samples={sample_count} cause={cause}",
                        flush=True,
                    )
                else:
                    shutil.rmtree(staging_root)
                    print(
                        "[racing-vision] DROP "
                        f"attempt={attempt_count:02d} gates={ep_gates} "
                        f"steps={ep_step} cause={cause}",
                        flush=True,
                    )

                episode_rows.append(row)

                # Do not rely on ManagerBasedRLEnv.step()'s in-step auto-reset
                # as the sole initialization path for the next dataset episode.
                # A real contact termination can leave contact-sensor / derived
                # scene state stale for the first post-reset step, which then
                # appears as an endless sequence of one-step collisions.
                #
                # The explicit reset path performs a full scene/manager reset
                # before the next policy action and is also consistent after
                # timeouts, so every accepted/rejected episode starts from the
                # same clean contract.
                obs, _ = wrapped.reset()
                prev_gate_idx = int(command.next_gate_idx[0].item())
                break

        if accepted_count < int(args_cli.target_successful_episodes):
            raise RuntimeError(
                "Unable to collect the requested number of successful racing "
                f"episodes: accepted={accepted_count}, attempts={attempt_count}"
            )

        for split in ("train", "val", "test"):
            _write_split_manifest(
                output_root,
                split,
                split_episodes[split],
                width=int(args_cli.width),
                height=int(args_cli.height),
                capture_every_steps=int(args_cli.capture_every_steps),
                step_dt=float(raw_env.step_dt),
            )

        with (output_root / "episodes.csv").open("w", newline="", encoding="utf-8") as handle:
            scalar_fields = [
                "attempt",
                "accepted",
                "gates_passed",
                "steps",
                "duration_s",
                "return",
                "termination",
                "split",
                "samples",
                "speed_max_mps",
                "body_rate_max_radps",
            ]
            writer = csv.DictWriter(handle, fieldnames=scalar_fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(episode_rows)

        summary = {
            "schema": "isaac_drone_racer.racing_vision_collection.v1",
            "checkpoint": checkpoint,
            "task": args_cli.task,
            "seed": int(args_cli.seed),
            "target_successful_episodes": int(args_cli.target_successful_episodes),
            "accepted_episodes": int(accepted_count),
            "attempted_episodes": int(attempt_count),
            "minimum_gates_to_keep": int(args_cli.min_gates),
            "num_track_gates": num_gates,
            "image_size": [int(args_cli.width), int(args_cli.height)],
            "capture_every_steps": int(args_cli.capture_every_steps),
            "capture_rate_hz": float(
                1.0 / (float(raw_env.step_dt) * int(args_cli.capture_every_steps))
            ),
            "split_episode_counts": {
                key: len(value) for key, value in split_episodes.items()
            },
            "gate_geometry_source": geometry.source,
        }
        (output_root / "manifest.json").write_text(
            json.dumps(summary, indent=2) + "\n",
            encoding="utf-8",
        )
        print("=" * 96)
        print("RACING VISION DATASET COMPLETE")
        print("=" * 96)
        print(json.dumps(summary, indent=2))
        print(f"[racing-vision] dataset: {output_root}")
    finally:
        wrapped.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
