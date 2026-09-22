"""Collect synchronized high-speed racing data for estimator retraining.

A frozen GT Circular-12 CTBR policy controls the vehicle.  The collector never
feeds camera, IMU or a learned estimator back into the actor.  It records two
time-aligned products from each accepted high-speed episode:

* 100-Hz IMO-compatible traces: GT p/v/q + body gyro + body acceleration +
  mass-normalized applied collective thrust.  These CSVs can be consumed
  directly by train_learned_motion.py with
  target_mode=delta_velocity_body_end_gyro_aligned.
* 25-Hz production-camera frames with exact labels for EVERY mapped gate, not
  only the active mission gate.  This avoids treating visible non-active gates
  as unlabeled background and supports later multi-instance detector training.

Train/val/test membership is assigned by complete accepted episode.
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
    default="Isaac-Drone-Racer-Swift-CTBR-GT-Circular12-v0",
)
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--output-dir", type=Path, required=True)
parser.add_argument("--target-successful-episodes", type=int, default=30)
parser.add_argument("--val-episodes", type=int, default=5)
parser.add_argument("--test-episodes", type=int, default=5)
parser.add_argument("--min-gates", type=int, default=45)
parser.add_argument("--max-attempts", type=int, default=80)
parser.add_argument(
    "--capture-every-steps",
    type=int,
    default=4,
    help="Capture RGB every N 100-Hz control steps. Default 4 -> 25 Hz.",
)
parser.add_argument("--width", type=int, default=256)
parser.add_argument("--height", type=int, default=256)
parser.add_argument(
    "--camera-pitch-up-deg",
    type=float,
    default=40.0,
    help=(
        "Camera optical pitch-up angle relative to body horizontal. "
        "Positive tilts optical +Z from body +X toward body +Z. "
        "The new Circular-12 vision campaign uses 40 deg."
    ),
)
parser.add_argument("--seed", type=int, default=1)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from isaaclab.sensors import ImuCfg  # noqa: E402
from isaaclab_rl.skrl import SkrlVecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils import load_cfg_from_registry, parse_env_cfg  # noqa: E402
from skrl.utils.runner.torch import Runner  # noqa: E402

import tasks  # noqa: F401,E402
from perception.dataset import Stage2DatasetWriter  # noqa: E402
from perception.dataset_collector import IsaacStage2DatasetCollector  # noqa: E402
from perception.perfect_gate_corner_sensor import PerfectGateCornerSensor  # noqa: E402
from perception.rigid_transform import RigidTransform  # noqa: E402
from perception.stage2_calibration import load_stage2_gate_geometry  # noqa: E402
from tasks.drone_racer.drone_racer_stage2_env_cfg import (  # noqa: E402
    stage2_reference_camera_cfg,
)


TRACE_FIELDS = [
    "step",
    "t_s",
    "profile",
    "active_gate_index",
    "truth_px",
    "truth_py",
    "truth_pz",
    "truth_vx",
    "truth_vy",
    "truth_vz",
    "truth_qw",
    "truth_qx",
    "truth_qy",
    "truth_qz",
    "imu_gx",
    "imu_gy",
    "imu_gz",
    "imu_ax",
    "imu_ay",
    "imu_az",
    "collective_thrust_n",
    "thrust_b_x",
    "thrust_b_y",
    "thrust_b_z",
    "policy_a0",
    "policy_a1",
    "policy_a2",
    "policy_a3",
    "ctbr_collective_accel_mps2",
    "ctbr_rate_x_radps",
    "ctbr_rate_y_radps",
    "ctbr_rate_z_radps",
    "speed_mps",
    "body_rate_norm_radps",
]


def _np(value) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    return np.asarray(value)


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


def _split_for_success(index: int, total: int, val: int, test: int) -> str:
    train = total - val - test
    if index < train:
        return "train"
    if index < train + val:
        return "val"
    return "test"


def _prepare_output(root: Path) -> None:
    if root.exists():
        shutil.rmtree(root)
    for split in ("train", "val", "test"):
        (root / "vision" / split / "images").mkdir(parents=True, exist_ok=True)
        (root / "vision" / split / "labels").mkdir(parents=True, exist_ok=True)
        (root / "traces" / split).mkdir(parents=True, exist_ok=True)
    (root / "_staging").mkdir(parents=True, exist_ok=True)


def _gate_pose_from_track(raw_env, gate_index: int) -> RigidTransform:
    track = raw_env.scene["track"]
    data = track.data
    pos_all = getattr(data, "object_pos_w", None)
    quat_all = getattr(data, "object_quat_w", None)
    if pos_all is None or quat_all is None:
        pos_all = getattr(data, "object_link_pos_w", None)
        quat_all = getattr(data, "object_link_quat_w", None)
    if pos_all is None or quat_all is None:
        raise RuntimeError("track does not expose actor/link gate poses")
    return RigidTransform.from_pose_wxyz(
        _np(pos_all[0, gate_index]),
        _np(quat_all[0, gate_index]),
        to_frame="W",
        from_frame="G",
    )


def _all_mapped_gate_labels(raw_env, geometry, snapshot) -> list[dict]:
    sensor = PerfectGateCornerSensor(geometry, snapshot.camera)
    command = raw_env.command_manager.get_term("target")
    output: list[dict] = []
    for gate_index in range(int(command.num_gates)):
        T_wg = _gate_pose_from_track(raw_env, gate_index)
        T_cg = snapshot.truth.T_wc.inverse() @ T_wg
        corners = sensor.measure(
            T_cg,
            timestamp_s=snapshot.truth.timestamp_s,
        )
        output.append(
            {
                "gate_index": int(gate_index),
                "corners_uv": np.asarray(corners.corners_uv).tolist(),
                "visible": np.asarray(corners.visible, dtype=bool).tolist(),
                "confidence": np.asarray(corners.confidence).tolist(),
                "T_wg": {
                    "to_frame": T_wg.to_frame,
                    "from_frame": T_wg.from_frame,
                    "matrix": T_wg.as_matrix().tolist(),
                },
            }
        )
    return output


def _capture_multigate_frame(
    collector: IsaacStage2DatasetCollector,
    raw_env,
    geometry,
    sample_id: str,
    *,
    extra: dict,
) -> None:
    _, label_path = collector.capture(sample_id, 0, extra=extra)
    payload = json.loads(label_path.read_text(encoding="utf-8"))
    snapshot = collector.adapter.snapshot(0)
    payload["mapped_gates"] = _all_mapped_gate_labels(
        raw_env,
        geometry,
        snapshot,
    )
    payload["schema"] = "isaac_drone_racer.racing_estimator_frame.v1"
    label_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _trace_row(raw_env, actions, ep_step: int, active_gate_index: int) -> dict:
    robot = raw_env.scene["robot"]
    imu = raw_env.scene["imu"]
    action_term = raw_env.action_manager.get_term("control_action")

    p = _np(robot.data.root_pos_w[0]).astype(np.float64)
    v = _np(robot.data.root_lin_vel_w[0]).astype(np.float64)
    q = _np(robot.data.root_quat_w[0]).astype(np.float64)
    rates = _np(robot.data.root_ang_vel_b[0]).astype(np.float64)
    gyro = _np(imu.data.ang_vel_b[0]).astype(np.float64)
    accel = _np(imu.data.lin_acc_b[0]).astype(np.float64)
    processed = _np(action_term.processed_actions[0]).astype(np.float64)
    ctbr = _np(action_term.ctbr_command[0]).astype(np.float64)
    policy = _np(actions[0]).astype(np.float64)

    mass_kg = float(action_term.cfg.vehicle_mass_kg)
    collective_force_n = float(processed[0])
    thrust_b = np.array(
        [0.0, 0.0, collective_force_n / mass_kg],
        dtype=np.float64,
    )

    return {
        "step": int(ep_step),
        "t_s": float(ep_step * raw_env.step_dt),
        "profile": "circular12_gt_racing",
        "active_gate_index": int(active_gate_index),
        "truth_px": p[0],
        "truth_py": p[1],
        "truth_pz": p[2],
        "truth_vx": v[0],
        "truth_vy": v[1],
        "truth_vz": v[2],
        "truth_qw": q[0],
        "truth_qx": q[1],
        "truth_qy": q[2],
        "truth_qz": q[3],
        "imu_gx": gyro[0],
        "imu_gy": gyro[1],
        "imu_gz": gyro[2],
        "imu_ax": accel[0],
        "imu_ay": accel[1],
        "imu_az": accel[2],
        "collective_thrust_n": collective_force_n,
        "thrust_b_x": thrust_b[0],
        "thrust_b_y": thrust_b[1],
        "thrust_b_z": thrust_b[2],
        "policy_a0": policy[0],
        "policy_a1": policy[1],
        "policy_a2": policy[2],
        "policy_a3": policy[3],
        "ctbr_collective_accel_mps2": ctbr[0],
        "ctbr_rate_x_radps": ctbr[1],
        "ctbr_rate_y_radps": ctbr[2],
        "ctbr_rate_z_radps": ctbr[3],
        "speed_mps": float(np.linalg.norm(v)),
        "body_rate_norm_radps": float(np.linalg.norm(rates)),
    }


def _write_trace(
    path: Path,
    rows: list[dict],
    *,
    split: str,
    accepted_episode_index: int,
    attempt_index: int,
    gates_passed: int,
    termination: str,
) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=TRACE_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    metadata = {
        "schema": "isaac_drone_racer.imo_supervised_trace.racing_v1",
        "trace_csv": path.name,
        "profile": "circular12_gt_racing",
        "dataset_split": split,
        "accepted_episode_index": int(accepted_episode_index),
        "attempt_index": int(attempt_index),
        "episode_gates_passed": int(gates_passed),
        "episode_termination": termination,
        "completed_steps": len(rows),
        "nominal_control_rate_hz": 100.0,
        "thrust_b_units": "m/s^2 (mass-normalized applied collective force)",
        "ground_truth_use": "supervised labels/evaluation only",
        "control_source": "frozen_gt_swift_ctbr_policy",
    }
    path.with_suffix(path.suffix + ".json").write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )


def _promote_vision_episode(
    staging: Path,
    split_root: Path,
    *,
    accepted_episode_index: int,
    attempt_index: int,
    gates_passed: int,
    termination: str,
) -> int:
    label_paths = sorted((staging / "labels").glob("*.json"))
    for frame_index, label_path in enumerate(label_paths):
        payload = json.loads(label_path.read_text(encoding="utf-8"))
        extra = dict(payload.get("extra") or {})
        new_id = (
            f"ep{accepted_episode_index:03d}_"
            f"attempt{attempt_index:03d}_frame{frame_index:05d}"
        )
        old_image = staging / "images" / f"{label_path.stem}.png"
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
    shutil.rmtree(staging)
    return len(label_paths)


def _write_vision_manifest(root: Path, split: str, episodes: list[dict]) -> None:
    labels = list((root / "vision" / split / "labels").glob("*.json"))
    active_hist = Counter()
    any_hist = Counter()
    speeds: list[float] = []
    rates: list[float] = []
    for path in labels:
        payload = json.loads(path.read_text(encoding="utf-8"))
        active_hist[int(sum(bool(v) for v in payload["visible"]))] += 1
        mapped = payload.get("mapped_gates") or []
        best = max(
            (int(sum(bool(v) for v in gate["visible"])) for gate in mapped),
            default=0,
        )
        any_hist[best] += 1
        extra = payload.get("extra") or {}
        speeds.append(float(extra.get("speed_mps", 0.0)))
        rates.append(float(extra.get("body_rate_norm_radps", 0.0)))

    manifest = {
        "schema": "isaac_drone_racer.racing_estimator_vision_split.v1",
        "split": split,
        "samples": len(labels),
        "episodes": episodes,
        "active_gate_visible_corner_histogram": {
            str(k): int(active_hist.get(k, 0)) for k in range(5)
        },
        "any_mapped_gate_max_visible_corner_histogram": {
            str(k): int(any_hist.get(k, 0)) for k in range(5)
        },
        "any_mapped_gate_ge2_fraction": float(
            sum(v for k, v in any_hist.items() if k >= 2) / max(len(labels), 1)
        ),
        "speed_mps": {
            "mean": float(np.mean(speeds)) if speeds else 0.0,
            "max": float(np.max(speeds)) if speeds else 0.0,
        },
        "body_rate_norm_radps": {
            "mean": float(np.mean(rates)) if rates else 0.0,
            "max": float(np.max(rates)) if rates else 0.0,
        },
    }
    (root / "vision" / split / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    total = int(args_cli.target_successful_episodes)
    if total <= 0:
        raise ValueError("--target-successful-episodes must be positive")
    if args_cli.val_episodes < 0 or args_cli.test_episodes < 0:
        raise ValueError("validation/test episode counts must be non-negative")
    if args_cli.val_episodes + args_cli.test_episodes >= total:
        raise ValueError("at least one accepted episode must remain for training")
    if args_cli.capture_every_steps <= 0:
        raise ValueError("--capture-every-steps must be positive")

    root = args_cli.output_dir.expanduser().resolve()
    _prepare_output(root)

    cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=1)
    cfg.scene.num_envs = 1
    cfg.seed = int(args_cli.seed)
    cfg.scene.tiled_camera = stage2_reference_camera_cfg(
        pitch_up_deg=float(args_cli.camera_pitch_up_deg)
    )
    cfg.scene.tiled_camera.width = int(args_cli.width)
    cfg.scene.tiled_camera.height = int(args_cli.height)
    cfg.scene.imu = ImuCfg(
        prim_path="{ENV_REGEX_NS}/Robot/body",
        debug_vis=False,
    )
    cfg.scene.collision_sensor.debug_vis = False
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
    print(f"[racing-estimator-data] loading checkpoint: {checkpoint}", flush=True)
    runner.agent.load(checkpoint)
    runner.agent.set_running_mode("eval")
    runner.agent.set_mode("eval")

    geometry = load_stage2_gate_geometry()
    command = raw_env.command_manager.get_term("target")
    robot = raw_env.scene["robot"]
    num_gates = int(command.num_gates)

    split_episodes = {"train": [], "val": [], "test": []}
    trace_entries: list[dict] = []
    episode_rows: list[dict] = []
    accepted = 0
    attempts = 0

    obs, _ = wrapped.reset()
    prev_gate_idx = int(command.next_gate_idx[0].item())

    try:
        while accepted < total and attempts < int(args_cli.max_attempts):
            attempts += 1
            ep_step = 0
            ep_gates = 0
            ep_return = 0.0
            trace_rows: list[dict] = []
            staging = root / "_staging" / f"attempt_{attempts:03d}"
            writer = Stage2DatasetWriter(staging)
            collector = IsaacStage2DatasetCollector(raw_env, geometry, writer)

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

                if not done:
                    trace_rows.append(
                        _trace_row(
                            raw_env,
                            actions,
                            ep_step,
                            current_gate_idx,
                        )
                    )

                if (
                    not done
                    and ep_step % int(args_cli.capture_every_steps) == 0
                ):
                    v = robot.data.root_lin_vel_w[0]
                    rates = robot.data.root_ang_vel_b[0]
                    extra = {
                        "attempt_index": int(attempts),
                        "episode_step": int(ep_step),
                        "gates_passed_so_far": int(ep_gates),
                        "active_gate_index": int(current_gate_idx),
                        "speed_mps": float(torch.linalg.vector_norm(v).item()),
                        "body_rate_norm_radps": float(
                            torch.linalg.vector_norm(rates).item()
                        ),
                    }
                    _capture_multigate_frame(
                        collector,
                        raw_env,
                        geometry,
                        f"attempt{attempts:03d}_step{ep_step:06d}",
                        extra=extra,
                    )

                if not done:
                    continue

                cause = _termination_cause(raw_env)
                keep = ep_gates >= int(args_cli.min_gates)
                row = {
                    "attempt": int(attempts),
                    "accepted": bool(keep),
                    "gates_passed": int(ep_gates),
                    "steps": int(ep_step),
                    "duration_s": float(ep_step * raw_env.step_dt),
                    "return": float(ep_return),
                    "termination": cause,
                }

                if keep:
                    split = _split_for_success(
                        accepted,
                        total,
                        int(args_cli.val_episodes),
                        int(args_cli.test_episodes),
                    )
                    sample_count = _promote_vision_episode(
                        staging,
                        root / "vision" / split,
                        accepted_episode_index=accepted,
                        attempt_index=attempts,
                        gates_passed=ep_gates,
                        termination=cause,
                    )
                    trace_name = (
                        f"ep{accepted:03d}_attempt{attempts:03d}.csv"
                    )
                    trace_path = root / "traces" / split / trace_name
                    _write_trace(
                        trace_path,
                        trace_rows,
                        split=split,
                        accepted_episode_index=accepted,
                        attempt_index=attempts,
                        gates_passed=ep_gates,
                        termination=cause,
                    )
                    trace_entries.append(
                        {
                            "name": trace_path.stem,
                            "path": str(trace_path.relative_to(root)),
                            "split": split,
                        }
                    )
                    row.update(
                        {
                            "split": split,
                            "vision_samples": int(sample_count),
                            "trace_samples": int(len(trace_rows)),
                        }
                    )
                    split_episodes[split].append(dict(row))
                    accepted += 1
                    print(
                        "[racing-estimator-data] KEEP "
                        f"attempt={attempts:02d} accepted={accepted}/{total} "
                        f"split={split} gates={ep_gates} "
                        f"vision={sample_count} trace={len(trace_rows)}",
                        flush=True,
                    )
                else:
                    shutil.rmtree(staging)
                    print(
                        "[racing-estimator-data] DROP "
                        f"attempt={attempts:02d} gates={ep_gates} "
                        f"steps={ep_step} cause={cause}",
                        flush=True,
                    )

                episode_rows.append(row)
                obs, _ = wrapped.reset()
                prev_gate_idx = int(command.next_gate_idx[0].item())
                break

        if accepted < total:
            raise RuntimeError(
                "unable to collect requested successful episodes: "
                f"accepted={accepted}, attempts={attempts}"
            )

        for split in ("train", "val", "test"):
            _write_vision_manifest(root, split, split_episodes[split])

        with (root / "episodes.csv").open("w", newline="", encoding="utf-8") as handle:
            fields = [
                "attempt",
                "accepted",
                "gates_passed",
                "steps",
                "duration_s",
                "return",
                "termination",
                "split",
                "vision_samples",
                "trace_samples",
            ]
            writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(episode_rows)

        imo_manifest = {
            "schema": "isaac_drone_racer.imo_dataset_manifest.v2",
            "path_base": ".",
            "description": (
                "High-speed Circular-12 GT-policy racing traces for learned "
                "inertial retraining; split by complete episode."
            ),
            "traces": trace_entries,
        }
        (root / "imo_manifest.json").write_text(
            json.dumps(imo_manifest, indent=2) + "\n",
            encoding="utf-8",
        )

        summary = {
            "schema": "isaac_drone_racer.racing_estimator_dataset.v1",
            "checkpoint": checkpoint,
            "task": args_cli.task,
            "accepted_episodes": int(accepted),
            "attempted_episodes": int(attempts),
            "minimum_gates_to_keep": int(args_cli.min_gates),
            "num_track_gates": int(num_gates),
            "control_rate_hz": float(1.0 / raw_env.step_dt),
            "camera_capture_rate_hz": float(
                1.0
                / (
                    float(raw_env.step_dt)
                    * int(args_cli.capture_every_steps)
                )
            ),
            "image_size": [int(args_cli.width), int(args_cli.height)],
            "camera_pitch_up_deg": float(args_cli.camera_pitch_up_deg),
            "camera_mount_contract": (
                "positive pitch-up tilts optical forward from body +X toward body +Z"
            ),
            "split_episode_counts": {
                k: len(v) for k, v in split_episodes.items()
            },
            "vision_labels": "all mapped gates per frame",
            "trace_manifest": "imo_manifest.json",
            "gate_geometry_source": geometry.source,
        }
        (root / "manifest.json").write_text(
            json.dumps(summary, indent=2) + "\n",
            encoding="utf-8",
        )
        print("=" * 96)
        print("RACING ESTIMATOR DATASET COMPLETE")
        print("=" * 96)
        print(json.dumps(summary, indent=2))
        print(f"[racing-estimator-data] dataset: {root}", flush=True)
    finally:
        wrapped.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
