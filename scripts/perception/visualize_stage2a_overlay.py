"""Render varied Stage2A views and overlay calibrated oracle gate corners."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", type=Path, default=Path("outputs/stage2a_overlay"))
parser.add_argument("--num_samples", type=int, default=32)
parser.add_argument("--width", type=int, default=256)
parser.add_argument("--height", type=int, default=256)
parser.add_argument("--seed", type=int, default=11)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if not args.enable_cameras:
    parser.error("Stage2A RGB overlays require --enable_cameras")
app = AppLauncher(args).app

sys.path.insert(0, str(Path(__file__).parents[2]))

import cv2
import gymnasium as gym
import numpy as np
import torch
import isaaclab.utils.math as math_utils

import tasks  # noqa: F401,E402
from perception.isaac_adapter import IsaacStage2TruthAdapter  # noqa: E402
from perception.perfect_gate_corner_sensor import PerfectGateCornerSensor  # noqa: E402
from perception.rigid_transform import rotation_error_rad, translation_error_m  # noqa: E402
from perception.stage2_calibration import (  # noqa: E402
    CAMERA_MODEL,
    CAMERA_OPTICAL_CONVENTION,
    GATE_KEYPOINT_CALIBRATION_PATH,
    load_stage2_gate_geometry,
    stage2_camera_to_body,
)
from tasks.drone_racer.drone_racer_stage2_env_cfg import DroneRacerStage2DataEnvCfg  # noqa: E402


CORNER_LABELS = ("LB", "RB", "RT", "LT")
CORNER_COLORS_BGR = ((0, 0, 255), (0, 220, 0), (255, 80, 0), (220, 0, 220))


def _pattern(values: tuple[float, ...], count: int, device: str) -> torch.Tensor:
    source = torch.tensor(values, dtype=torch.float32, device=device)
    return source[torch.arange(count, device=device) % len(values)]


def _draw_overlay(rgb: np.ndarray, corners_uv: np.ndarray, visible: np.ndarray, title: str) -> np.ndarray:
    image = cv2.cvtColor(np.asarray(rgb), cv2.COLOR_RGB2BGR).copy()
    points = np.rint(corners_uv).astype(np.int32)
    cv2.polylines(image, [points.reshape(-1, 1, 2)], True, (0, 255, 255), 1, cv2.LINE_AA)
    height, width = image.shape[:2]
    for label, color, point, is_visible in zip(CORNER_LABELS, CORNER_COLORS_BGR, points, visible):
        x, y = int(point[0]), int(point[1])
        if 0 <= x < width and 0 <= y < height:
            cv2.circle(image, (x, y), 6, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.circle(image, (x, y), 3, color, -1, cv2.LINE_AA)
            cv2.putText(
                image,
                label,
                (min(x + 5, width - 24), max(y - 5, 12)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.38,
                color,
                1,
                cv2.LINE_AA,
            )
        if not is_visible:
            cv2.putText(image, f"{label}:OUT", (4, 28 + 14 * CORNER_LABELS.index(label)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1, cv2.LINE_AA)
    cv2.rectangle(image, (0, 0), (width, 18), (255, 255, 255), -1)
    cv2.putText(image, title, (3, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.34, (0, 0, 0), 1, cv2.LINE_AA)
    return image


def _make_contact_sheet(images: list[np.ndarray], columns: int = 8) -> np.ndarray:
    height, width = images[0].shape[:2]
    rows = (len(images) + columns - 1) // columns
    sheet = np.full((rows * height, columns * width, 3), 235, dtype=np.uint8)
    for index, image in enumerate(images):
        row, column = divmod(index, columns)
        sheet[row * height : (row + 1) * height, column * width : (column + 1) * width] = image
    return sheet


def main() -> None:
    if args.num_samples < 20:
        raise ValueError("Physical validation requires at least 20 overlay samples")
    torch.manual_seed(args.seed)
    cfg = DroneRacerStage2DataEnvCfg()
    cfg.scene.num_envs = args.num_samples
    cfg.scene.tiled_camera.width = args.width
    cfg.scene.tiled_camera.height = args.height
    cfg.scene.collision_sensor.debug_vis = False
    cfg.commands.target.debug_vis = False
    cfg.events.push_robot = None
    cfg.seed = args.seed

    env = gym.make("Isaac-Drone-Racer-Stage2-Data-v0", cfg=cfg)
    unwrapped = env.unwrapped
    robot = unwrapped.scene["robot"]
    track = unwrapped.scene["track"]
    command = unwrapped.command_manager.get_term("target")
    env_ids = torch.arange(args.num_samples, device=unwrapped.device)
    gate_indices = (env_ids % track.num_objects).to(torch.int32)
    command.next_gate_idx.copy_(gate_indices)
    geometry = load_stage2_gate_geometry()
    configured_T_bc = stage2_camera_to_body()
    adapter = IsaacStage2TruthAdapter(unwrapped)

    gate_pos = track.data.object_pos_w[env_ids, gate_indices.long()]
    gate_quat = track.data.object_quat_w[env_ids, gate_indices.long()]
    distance = _pattern((2.5, 3.5, 5.0, 6.5, 8.0), args.num_samples, unwrapped.device)
    lateral = _pattern((-1.2, -0.6, 0.0, 0.6, 1.2), args.num_samples, unwrapped.device)
    vertical = _pattern((-0.7, 0.0, 0.7, 0.35, -0.35), args.num_samples, unwrapped.device)
    local_position = torch.stack(
        (-distance, lateral, torch.full_like(distance, float(geometry.center_g[2])) + vertical), dim=-1
    )
    world_position = gate_pos + math_utils.quat_apply(gate_quat, local_position)

    center_local = torch.as_tensor(geometry.center_g, dtype=torch.float32, device=unwrapped.device).expand(
        args.num_samples, -1
    )
    center_world = gate_pos + math_utils.quat_apply(gate_quat, center_local)
    direction = center_world - world_position
    horizontal = torch.linalg.vector_norm(direction[:, :2], dim=-1)
    yaw = torch.atan2(direction[:, 1], direction[:, 0])
    pitch = -torch.atan2(direction[:, 2], horizontal)
    yaw += _pattern((-0.18, -0.09, 0.0, 0.09, 0.18), args.num_samples, unwrapped.device)
    pitch += _pattern((-0.10, 0.0, 0.10, 0.05, -0.05), args.num_samples, unwrapped.device)
    roll = _pattern((-0.20, -0.10, 0.0, 0.10, 0.20), args.num_samples, unwrapped.device)
    orientation = math_utils.quat_from_euler_xyz(roll, pitch, yaw)

    args.output.mkdir(parents=True, exist_ok=True)
    records = []
    overlays = []
    try:
        robot.write_root_pose_to_sim(torch.cat((world_position, orientation), dim=-1), env_ids=env_ids)
        robot.write_root_velocity_to_sim(
            torch.zeros((args.num_samples, 6), device=unwrapped.device), env_ids=env_ids
        )
        unwrapped.scene.write_data_to_sim()
        unwrapped.sim.step(render=True)
        unwrapped.scene.update(dt=unwrapped.physics_dt)

        for env_id in range(args.num_samples):
            snapshot = adapter.snapshot(env_id)
            rgb = adapter.rgb(env_id)
            T_cg_truth = snapshot.truth.T_wc.inverse() @ snapshot.truth.T_wg
            observation = PerfectGateCornerSensor(geometry, snapshot.camera).measure(T_cg_truth)
            T_bc_truth = snapshot.truth.T_wb.inverse() @ snapshot.truth.T_wc
            extrinsic_translation_error = translation_error_m(configured_T_bc, T_bc_truth)
            extrinsic_rotation_error_deg = float(
                np.degrees(rotation_error_rad(configured_T_bc, T_bc_truth))
            )
            center_c = T_cg_truth.transform_points(geometry.center_g)
            bottom_width_px = float(np.linalg.norm(observation.corners_uv[1] - observation.corners_uv[0]))
            top_width_px = float(np.linalg.norm(observation.corners_uv[2] - observation.corners_uv[3]))
            title = (
                f"#{env_id:02d} G{snapshot.gate_index} d={np.linalg.norm(center_c):.2f}m "
                f"vis={int(observation.visible.sum())}/4"
            )
            overlay = _draw_overlay(rgb, observation.corners_uv, observation.visible, title)
            output_path = args.output / f"overlay_{env_id:03d}.png"
            if not cv2.imwrite(str(output_path), overlay):
                raise RuntimeError(f"Failed to write {output_path}")
            overlays.append(overlay)
            records.append(
                {
                    "sample_id": env_id,
                    "gate_index": snapshot.gate_index,
                    "distance_to_opening_center_m": float(np.linalg.norm(center_c)),
                    "projected_gate_width_px": 0.5 * (bottom_width_px + top_width_px),
                    "corners_uv": observation.corners_uv.tolist(),
                    "visible": observation.visible.tolist(),
                    "camera_K": snapshot.camera.K.tolist(),
                    "extrinsic_translation_error_m": extrinsic_translation_error,
                    "extrinsic_rotation_error_deg": extrinsic_rotation_error_deg,
                    "overlay_path": str(output_path),
                }
            )

        contact_sheet_path = args.output / "contact_sheet.png"
        if not cv2.imwrite(str(contact_sheet_path), _make_contact_sheet(overlays)):
            raise RuntimeError(f"Failed to write {contact_sheet_path}")
        report = {
            "schema": "isaac_drone_racer.stage2a_rgb_overlay.v1",
            "camera_model": CAMERA_MODEL,
            "camera_optical_convention": CAMERA_OPTICAL_CONVENTION,
            "gate_calibration": str(GATE_KEYPOINT_CALIBRATION_PATH),
            "samples": len(records),
            "complete_samples": sum(all(record["visible"]) for record in records),
            "gate_indices": sorted(set(record["gate_index"] for record in records)),
            "distance_range_m": [
                min(record["distance_to_opening_center_m"] for record in records),
                max(record["distance_to_opening_center_m"] for record in records),
            ],
            "projected_gate_width_range_px": [
                min(record["projected_gate_width_px"] for record in records),
                max(record["projected_gate_width_px"] for record in records),
            ],
            "configured_vs_truth_extrinsic_translation_error_m": {
                "mean": float(np.mean([record["extrinsic_translation_error_m"] for record in records])),
                "max": max(record["extrinsic_translation_error_m"] for record in records),
            },
            "configured_vs_truth_extrinsic_rotation_error_deg": {
                "mean": float(np.mean([record["extrinsic_rotation_error_deg"] for record in records])),
                "max": max(record["extrinsic_rotation_error_deg"] for record in records),
            },
            "records": records,
        }
        report_path = args.output / "report.json"
        report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(
            f"STAGE2A_OVERLAY_COMPLETE samples={len(records)} complete={report['complete_samples']} "
            f"contact_sheet={contact_sheet_path} report={report_path}",
            flush=True,
        )
    finally:
        env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        app.close()
