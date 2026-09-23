"""Collect the formal diversified Circular-12 vision dataset.

This collector deliberately avoids the tumbling GT racing policy. The vehicle is
prescribed on physically coordinated circular trajectories while the production
camera renders the real high-contrast per-gate textures.

Fixed capture contract:
  * camera pitch-up: 20 deg by default
  * body height: 2.07 m by default
  * image: 256x256 at 25 Hz
  * exact labels for all 12 mapped gates, including Gate ID/color and 4 corners

Diversity comes from coordinated-turn speed, radial path offset, and sub-gate
sampling phase. Train/val/test are split by the whole (speed, radius) condition,
so phase-shifted versions of the same condition never leak across splits.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import shutil

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--output-dir",
    type=Path,
    default=Path("artifacts/racing_vision/circular12_color20_h207_diversified_v1"),
)
parser.add_argument("--camera-pitch-up-deg", type=float, default=20.0)
parser.add_argument("--body-height-m", type=float, default=2.07)
parser.add_argument("--fps", type=int, default=25)
parser.add_argument("--width", type=int, default=256)
parser.add_argument("--height", type=int, default=256)
parser.add_argument(
    "--annotated-every",
    type=int,
    default=25,
    help="Save one annotated audit frame every N captured frames per run; 0 disables.",
)
parser.add_argument(
    "--overwrite",
    action="store_true",
    help="Delete an existing output directory before collection.",
)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True
simulation_app = AppLauncher(args).app

import cv2  # noqa: E402
import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import isaaclab.utils.math as math_utils  # noqa: E402

import tasks  # noqa: F401,E402
from perception.gate_identity import identity_for_gate_index  # noqa: E402
from perception.isaac_adapter import IsaacStage2TruthAdapter  # noqa: E402
from perception.perfect_gate_corner_sensor import PerfectGateCornerSensor  # noqa: E402
from perception.rigid_transform import RigidTransform  # noqa: E402
from perception.stage2_calibration import load_stage2_gate_geometry  # noqa: E402
from tasks.drone_racer.drone_racer_stage2_env_cfg import (  # noqa: E402
    DroneRacerStage2DataEnvCfg,
    stage2_reference_camera_cfg,
)
from tasks.drone_racer.track_generator import (  # noqa: E402
    CIRCULAR_12_GATE_TRACK_CONFIG,
    generate_track,
)


SPEEDS_MPS = (14.0, 15.5, 17.0, 18.5, 20.0, 21.5)
PATH_RADII_M = (11.6, 12.0, 12.4)
PHASE_OFFSETS_DEG = (0.0, 7.5, 15.0)

# Hold out whole speed/radius conditions. All three phase offsets for a
# condition stay in the same split.
VAL_CONDITIONS = {(1, 0), (3, 1), (5, 2)}
TEST_CONDITIONS = {(0, 2), (2, 1), (4, 0)}

TRACK_CENTER_XY_M = (0.0, 12.0)
TRACK_GATE_RADIUS_M = 12.0
GRAVITY_MPS2 = 9.81
VEHICLE_MASS_KG = 0.6076


def _np(value) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    return np.asarray(value)


def _split_for_condition(speed_index: int, radius_index: int) -> str:
    key = (int(speed_index), int(radius_index))
    if key in VAL_CONDITIONS:
        return "val"
    if key in TEST_CONDITIONS:
        return "test"
    return "train"


def _gate_pose_from_track(raw_env, gate_index: int) -> RigidTransform:
    data = raw_env.scene["track"].data
    pos_all = getattr(data, "object_pos_w", None)
    quat_all = getattr(data, "object_quat_w", None)
    if pos_all is None or quat_all is None:
        pos_all = getattr(data, "object_link_pos_w", None)
        quat_all = getattr(data, "object_link_quat_w", None)
    if pos_all is None or quat_all is None:
        raise RuntimeError("track does not expose gate poses")
    return RigidTransform.from_pose_wxyz(
        _np(pos_all[0, gate_index]),
        _np(quat_all[0, gate_index]),
        to_frame="W",
        from_frame="G",
    )


def _mapped_gate_labels(raw_env, geometry, snapshot) -> list[dict]:
    sensor = PerfectGateCornerSensor(geometry, snapshot.camera)
    command = raw_env.command_manager.get_term("target")
    labels: list[dict] = []
    for gate_index in range(int(command.num_gates)):
        identity = identity_for_gate_index(gate_index)
        T_wg = _gate_pose_from_track(raw_env, gate_index)
        T_cg = snapshot.truth.T_wc.inverse() @ T_wg
        corners = sensor.measure(
            T_cg,
            timestamp_s=snapshot.truth.timestamp_s,
        )
        labels.append(
            {
                "gate_index": int(gate_index),
                "gate_id": int(identity.gate_id),
                "color_name": identity.color_name,
                "color_rgb": list(identity.rgb),
                "color_hex": identity.hex,
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
    return labels


def _annotate(rgb: np.ndarray, mapped_gates: list[dict]) -> np.ndarray:
    bgr = cv2.cvtColor(np.ascontiguousarray(rgb), cv2.COLOR_RGB2BGR)
    for gate in mapped_gates:
        corners = np.asarray(gate["corners_uv"], dtype=np.float64).reshape(4, 2)
        visible = np.asarray(gate["visible"], dtype=bool).reshape(4)
        if int(visible.sum()) < 1:
            continue
        rgb_color = tuple(int(v) for v in gate["color_rgb"])
        color = (rgb_color[2], rgb_color[1], rgb_color[0])
        points = corners[visible]
        for point in points:
            cv2.circle(
                bgr,
                (int(round(point[0])), int(round(point[1]))),
                3,
                color,
                -1,
                lineType=cv2.LINE_AA,
            )
        anchor = np.mean(points, axis=0)
        cv2.putText(
            bgr,
            f"G{int(gate['gate_id']):02d}",
            (int(round(anchor[0])) + 4, int(round(anchor[1])) - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            color,
            1,
            cv2.LINE_AA,
        )
    return bgr


def _prepare_output(root: Path) -> None:
    if root.exists():
        if not args.overwrite:
            raise FileExistsError(
                f"{root} already exists; pass --overwrite to replace it"
            )
        shutil.rmtree(root)

    for split in ("train", "val", "test"):
        (root / "vision" / split / "images").mkdir(parents=True, exist_ok=True)
        (root / "vision" / split / "labels").mkdir(parents=True, exist_ok=True)
        (root / "audit" / split).mkdir(parents=True, exist_ok=True)


def _build_profiles() -> list[dict]:
    profiles: list[dict] = []
    run_index = 0
    for speed_index, speed_mps in enumerate(SPEEDS_MPS):
        for radius_index, path_radius_m in enumerate(PATH_RADII_M):
            split = _split_for_condition(speed_index, radius_index)
            for phase_index, phase_offset_deg in enumerate(PHASE_OFFSETS_DEG):
                profiles.append(
                    {
                        "run_index": int(run_index),
                        "split": split,
                        "speed_index": int(speed_index),
                        "radius_index": int(radius_index),
                        "phase_index": int(phase_index),
                        "speed_mps": float(speed_mps),
                        "path_radius_m": float(path_radius_m),
                        "phase_offset_deg": float(phase_offset_deg),
                    }
                )
                run_index += 1
    return profiles


def main() -> None:
    if args.fps <= 0:
        raise ValueError("--fps must be positive")
    if args.width <= 0 or args.height <= 0:
        raise ValueError("image dimensions must be positive")
    if args.annotated_every < 0:
        raise ValueError("--annotated-every must be >= 0")

    root = args.output_dir.expanduser().resolve()
    _prepare_output(root)
    profiles = _build_profiles()

    cfg = DroneRacerStage2DataEnvCfg()
    cfg.scene.num_envs = 1
    cfg.scene.track = generate_track(CIRCULAR_12_GATE_TRACK_CONFIG)
    cfg.scene.tiled_camera = stage2_reference_camera_cfg(
        pitch_up_deg=float(args.camera_pitch_up_deg)
    )
    cfg.scene.tiled_camera.width = int(args.width)
    cfg.scene.tiled_camera.height = int(args.height)
    cfg.scene.collision_sensor.debug_vis = False
    cfg.commands.target.debug_vis = False
    cfg.events.push_robot = None
    cfg.seed = 7

    env = gym.make("Isaac-Drone-Racer-Stage2-Data-v0", cfg=cfg)
    raw = env.unwrapped
    env.reset()

    robot = raw.scene["robot"]
    adapter = IsaacStage2TruthAdapter(raw)
    geometry = load_stage2_gate_geometry()
    device = raw.device

    split_profiles: dict[str, list[dict]] = {"train": [], "val": [], "test": []}
    split_visible_hist = {
        split: Counter() for split in ("train", "val", "test")
    }
    split_sample_count = Counter()

    try:
        for profile in profiles:
            run_index = int(profile["run_index"])
            split = str(profile["split"])
            speed_mps = float(profile["speed_mps"])
            path_radius_m = float(profile["path_radius_m"])
            start_angle = (
                -0.5 * math.pi
                + math.radians(float(profile["phase_offset_deg"]))
            )
            omega_radps = speed_mps / path_radius_m
            bank_rad = -math.atan2(
                speed_mps**2 / path_radius_m,
                GRAVITY_MPS2,
            )
            collective_accel = math.hypot(
                GRAVITY_MPS2,
                speed_mps**2 / path_radius_m,
            )
            duration_s = 2.0 * math.pi / omega_radps  # exactly one lap
            frame_count = int(math.ceil(duration_s * args.fps)) + 1

            force_b = torch.tensor(
                [[[0.0, 0.0, VEHICLE_MASS_KG * collective_accel]]],
                dtype=torch.float32,
                device=device,
            )
            torque_b = torch.zeros_like(force_b)
            robot.set_external_force_and_torque(force_b, torque_b)

            run_record = dict(profile)
            run_record.update(
                {
                    "frames": int(frame_count),
                    "duration_s": float(duration_s),
                    "bank_deg": float(math.degrees(bank_rad)),
                    "yaw_rate_degps": float(math.degrees(omega_radps)),
                    "camera_pitch_up_deg": float(args.camera_pitch_up_deg),
                    "body_height_m": float(args.body_height_m),
                }
            )
            split_profiles[split].append(run_record)

            print(
                "[diversified-vision] "
                f"run={run_index:02d}/{len(profiles)-1:02d} "
                f"split={split:5s} speed={speed_mps:4.1f}m/s "
                f"radius={path_radius_m:4.1f}m "
                f"phase={profile['phase_offset_deg']:4.1f}deg "
                f"bank={math.degrees(bank_rad):6.2f}deg "
                f"frames={frame_count}",
                flush=True,
            )

            for frame_idx in range(frame_count):
                t_s = min(frame_idx / float(args.fps), duration_s)
                angle = start_angle + omega_radps * t_s
                yaw = angle + 0.5 * math.pi

                pos = torch.tensor(
                    [[
                        TRACK_CENTER_XY_M[0] + path_radius_m * math.cos(angle),
                        TRACK_CENTER_XY_M[1] + path_radius_m * math.sin(angle),
                        float(args.body_height_m),
                    ]],
                    dtype=torch.float32,
                    device=device,
                )
                orientation = math_utils.quat_from_euler_xyz(
                    torch.tensor([bank_rad], dtype=torch.float32, device=device),
                    torch.tensor([0.0], dtype=torch.float32, device=device),
                    torch.tensor([yaw], dtype=torch.float32, device=device),
                )
                velocity = torch.tensor(
                    [[
                        -speed_mps * math.sin(angle),
                        speed_mps * math.cos(angle),
                        0.0,
                        0.0,
                        0.0,
                        omega_radps,
                    ]],
                    dtype=torch.float32,
                    device=device,
                )

                robot.write_root_pose_to_sim(torch.cat((pos, orientation), dim=-1))
                robot.write_root_velocity_to_sim(velocity)
                raw.scene.write_data_to_sim()
                raw.sim.step(render=True)
                raw.scene.update(dt=raw.physics_dt)

                rgb = adapter.rgb(0)
                snapshot = adapter.snapshot(0)
                mapped_gates = _mapped_gate_labels(raw, geometry, snapshot)
                sample_id = f"run{run_index:03d}_frame{frame_idx:05d}"

                image_path = root / "vision" / split / "images" / f"{sample_id}.png"
                label_path = root / "vision" / split / "labels" / f"{sample_id}.json"
                cv2.imwrite(
                    str(image_path),
                    cv2.cvtColor(
                        np.ascontiguousarray(rgb),
                        cv2.COLOR_RGB2BGR,
                    ),
                )

                payload = {
                    "schema": "isaac_drone_racer.circular12_diversified_color_frame.v1",
                    "sample_id": sample_id,
                    "dataset_split": split,
                    "run_index": run_index,
                    "frame_index": int(frame_idx),
                    "t_s": float(t_s),
                    "camera_pitch_up_deg": float(args.camera_pitch_up_deg),
                    "body_height_m": float(args.body_height_m),
                    "speed_mps": speed_mps,
                    "path_radius_m": path_radius_m,
                    "phase_offset_deg": float(profile["phase_offset_deg"]),
                    "prescribed_bank_deg": float(math.degrees(bank_rad)),
                    "lap_fraction": float(
                        omega_radps * t_s / (2.0 * math.pi)
                    ),
                    "mapped_gates": mapped_gates,
                }
                label_path.write_text(
                    json.dumps(payload, indent=2) + "\n",
                    encoding="utf-8",
                )

                best_visible = max(
                    (
                        int(sum(bool(v) for v in gate["visible"]))
                        for gate in mapped_gates
                    ),
                    default=0,
                )
                split_visible_hist[split][best_visible] += 1
                split_sample_count[split] += 1

                if (
                    args.annotated_every > 0
                    and frame_idx % int(args.annotated_every) == 0
                ):
                    audit_path = (
                        root
                        / "audit"
                        / split
                        / f"{sample_id}.png"
                    )
                    cv2.imwrite(str(audit_path), _annotate(rgb, mapped_gates))

    finally:
        # Do not close SimulationApp here. In Isaac Sim 4.5 close() can tear
        # down the process before ordinary Python code after this block runs.
        # Manifests must be flushed first; the app is closed at the very end.
        env.close()

    for split in ("train", "val", "test"):
        manifest = {
            "schema": "isaac_drone_racer.circular12_diversified_color_split.v1",
            "split": split,
            "samples": int(split_sample_count[split]),
            "runs": split_profiles[split],
            "run_count": len(split_profiles[split]),
            "camera_pitch_up_deg": float(args.camera_pitch_up_deg),
            "body_height_m": float(args.body_height_m),
            "capture_rate_hz": int(args.fps),
            "image_width": int(args.width),
            "image_height": int(args.height),
            "any_mapped_gate_max_visible_corner_histogram": {
                str(k): int(split_visible_hist[split].get(k, 0))
                for k in range(5)
            },
            "split_contract": (
                "whole (speed_mps, path_radius_m) conditions; "
                "all phase offsets stay in one split"
            ),
        }
        (
            root / "vision" / split / "manifest.json"
        ).write_text(
            json.dumps(manifest, indent=2) + "\n",
            encoding="utf-8",
        )

    root_manifest = {
        "schema": "isaac_drone_racer.circular12_diversified_color_dataset.v1",
        "purpose": "12-class Gate-ID + 4-corner visual retraining",
        "control_source": "prescribed_coordinated_circle_not_policy",
        "track": "CIRCULAR_12_GATE_TRACK_CONFIG",
        "track_gate_radius_m": TRACK_GATE_RADIUS_M,
        "track_center_xy_m": list(TRACK_CENTER_XY_M),
        "camera_pitch_up_deg": float(args.camera_pitch_up_deg),
        "body_height_m": float(args.body_height_m),
        "speeds_mps": list(SPEEDS_MPS),
        "path_radii_m": list(PATH_RADII_M),
        "phase_offsets_deg": list(PHASE_OFFSETS_DEG),
        "total_runs": len(profiles),
        "split_run_counts": {
            split: len(split_profiles[split])
            for split in ("train", "val", "test")
        },
        "split_sample_counts": {
            split: int(split_sample_count[split])
            for split in ("train", "val", "test")
        },
        "leakage_guard": (
            "speed/radius condition groups are disjoint across train/val/test"
        ),
        "gate_identity": (
            "high-contrast gate texture -> supervised Gate ID 1..12; "
            "known-map reprojection remains runtime geometric consistency gate"
        ),
    }
    (root / "manifest.json").write_text(
        json.dumps(root_manifest, indent=2) + "\n",
        encoding="utf-8",
    )

    print("=" * 104)
    print("CIRCULAR-12 DIVERSIFIED COLOR VISION DATASET COMPLETE")
    print("=" * 104)
    print(json.dumps(root_manifest, indent=2))
    print(f"[diversified-vision] dataset: {root}", flush=True)
    simulation_app.close()


if __name__ == "__main__":
    main()
