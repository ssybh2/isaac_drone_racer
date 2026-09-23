"""One-off prescribed-circle FPV/data-capture comparison; prescribed flight only.

The pose and velocity are forced to the ideal steady turn at every frame.
The corresponding 2.85 g body-up force is also set in the simulator, but
the rendered path is prescribed, not evidence of open-loop controllability.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent)
parser.add_argument("--speed-mps", type=float, default=17.712658128452922)
parser.add_argument("--fps", type=int, default=25)
parser.add_argument("--camera-pitch-up-deg", type=float, default=20.0)
parser.add_argument(
    "--capture-dataset",
    action="store_true",
    help=(
        "Save raw 256x256 RGB frames plus exact all-map gate corner/identity "
        "labels and annotated preview frames alongside the FPV video."
    ),
)
parser.add_argument(
    "--dataset-every-frames",
    type=int,
    default=1,
    help="Save every N rendered frames when --capture-dataset is enabled.",
)
parser.add_argument("--body-height-m", type=float, default=2.07)
parser.add_argument(
    "--multicolor-gates", action="store_true",
    help="Add visual-only, per-gate color panels without changing the gate USD.",
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
from multicolor_preview import gate_panels  # noqa: E402


def _np(value) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    return np.asarray(value)


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


def _annotate_gate_identities(rgb: np.ndarray, mapped_gates: list[dict]) -> np.ndarray:
    bgr = cv2.cvtColor(np.ascontiguousarray(rgb), cv2.COLOR_RGB2BGR)
    for gate in mapped_gates:
        corners = np.asarray(gate["corners_uv"], dtype=np.float64).reshape(4, 2)
        visible = np.asarray(gate["visible"], dtype=bool).reshape(4)
        if int(visible.sum()) < 1:
            continue
        color_rgb = tuple(int(v) for v in gate["color_rgb"])
        color_bgr = (color_rgb[2], color_rgb[1], color_rgb[0])
        points = corners[visible]
        for point in points:
            cv2.circle(
                bgr,
                (int(round(point[0])), int(round(point[1]))),
                3,
                color_bgr,
                -1,
                lineType=cv2.LINE_AA,
            )
        anchor = np.mean(points, axis=0)
        cv2.putText(
            bgr,
            f"G{int(gate['gate_id']):02d} {gate['color_name']}",
            (int(round(anchor[0])) + 4, int(round(anchor[1])) - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            color_bgr,
            1,
            cv2.LINE_AA,
        )
    return bgr


def _install_multicolor_gate_overlays(stage) -> None:
    """Add four non-colliding colored panels per gate after scene creation."""
    from pxr import Gf, Sdf, UsdGeom, UsdShade

    for gate_index, panels in gate_panels().items():
        gate_path = f"/World/envs/env_0/Gate_{gate_index}"
        if not stage.GetPrimAtPath(gate_path).IsValid():
            raise RuntimeError(f"Multicolor preview gate not found: {gate_path}")
        root_path = gate_path + "/PreviewColor"
        UsdGeom.Xform.Define(stage, root_path)
        material = UsdShade.Material.Define(stage, root_path + "/Material")
        shader = UsdShade.Shader.Define(stage, root_path + "/Material/Surface")
        shader.CreateIdAttr("UsdPreviewSurface")
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(
            Gf.Vec3f(*panels[0].color_rgb)
        )
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.7)
        shader.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).Set(
            Gf.Vec3f(*(0.15 * value for value in panels[0].color_rgb))
        )
        material.CreateSurfaceOutput().ConnectToSource(
            shader.CreateOutput("surface", Sdf.ValueTypeNames.Token)
        )
        for panel_index, panel in enumerate(panels):
            cube = UsdGeom.Cube.Define(stage, f"{root_path}/Panel_{panel_index}")
            cube.CreateSizeAttr(1.0)
            xform = UsdGeom.Xformable(cube)
            xform.AddTranslateOp().Set(Gf.Vec3d(*panel.center_g))
            xform.AddScaleOp().Set(Gf.Vec3f(*panel.size_m))
            UsdShade.MaterialBindingAPI.Apply(cube.GetPrim()).Bind(material)
    print("[ideal-bank] installed 12 distinct gate-color overlays", flush=True)


def main() -> None:
    radius_m = 12.0
    gravity_mps2 = 9.81
    mass_kg = 0.6076
    pitch_up_deg = float(args.camera_pitch_up_deg)
    laps = 3
    omega_radps = args.speed_mps / radius_m
    bank_rad = -math.atan2(args.speed_mps**2 / radius_m, gravity_mps2)
    collective_accel = math.hypot(gravity_mps2, args.speed_mps**2 / radius_m)
    duration_s = laps * 2.0 * math.pi / omega_radps
    frame_count = math.ceil(duration_s * args.fps) + 1

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if int(args.dataset_every_frames) < 1:
        raise ValueError("--dataset-every-frames must be positive")
    dataset_root = output_dir / "vision_smoke"
    dataset_images = dataset_root / "images"
    dataset_labels = dataset_root / "labels"
    dataset_annotated = dataset_root / "annotated"
    if args.capture_dataset:
        dataset_images.mkdir(parents=True, exist_ok=True)
        dataset_labels.mkdir(parents=True, exist_ok=True)
        dataset_annotated.mkdir(parents=True, exist_ok=True)
    video_path = output_dir / f"fpv_{pitch_up_deg:g}deg_ideal_bank_3laps.mp4"
    cfg = DroneRacerStage2DataEnvCfg()
    cfg.scene.num_envs = 1
    cfg.scene.track = generate_track(CIRCULAR_12_GATE_TRACK_CONFIG)
    cfg.scene.tiled_camera = stage2_reference_camera_cfg(pitch_up_deg=pitch_up_deg)
    cfg.scene.tiled_camera.width = 256
    cfg.scene.tiled_camera.height = 256
    cfg.scene.collision_sensor.debug_vis = False
    cfg.commands.target.debug_vis = False
    cfg.events.push_robot = None
    cfg.seed = 1

    env = gym.make("Isaac-Drone-Racer-Stage2-Data-v0", cfg=cfg)
    raw = env.unwrapped
    env.reset()
    if args.multicolor_gates:
        _install_multicolor_gate_overlays(raw.sim.stage)
    robot = raw.scene["robot"]
    adapter = IsaacStage2TruthAdapter(raw)
    geometry = load_stage2_gate_geometry()
    device = raw.device
    force_b = torch.tensor(
        [[[0.0, 0.0, mass_kg * collective_accel]]],
        dtype=torch.float32,
        device=device,
    )
    torque_b = torch.zeros_like(force_b)
    robot.set_external_force_and_torque(force_b, torque_b)
    writer = cv2.VideoWriter(
        str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (256, 256)
    )
    if not writer.isOpened():
        raise RuntimeError(f"Cannot open video writer: {video_path}")

    samples = []
    try:
        for frame_idx in range(frame_count):
            t_s = min(frame_idx / args.fps, duration_s)
            angle = -math.pi / 2.0 + omega_radps * t_s
            yaw = angle + math.pi / 2.0
            pos = torch.tensor(
                [[radius_m * math.cos(angle), 12.0 + radius_m * math.sin(angle), args.body_height_m]],
                dtype=torch.float32,
                device=device,
            )
            orientation = math_utils.quat_from_euler_xyz(
                torch.tensor([bank_rad], device=device),
                torch.tensor([0.0], device=device),
                torch.tensor([yaw], device=device),
            )
            velocity = torch.tensor(
                [[-args.speed_mps * math.sin(angle), args.speed_mps * math.cos(angle),
                  0.0, 0.0, 0.0, omega_radps]],
                dtype=torch.float32,
                device=device,
            )
            robot.write_root_pose_to_sim(torch.cat((pos, orientation), dim=-1))
            robot.write_root_velocity_to_sim(velocity)
            raw.scene.write_data_to_sim()
            raw.sim.step(render=True)
            raw.scene.update(dt=raw.physics_dt)
            rgb = adapter.rgb(0)
            writer.write(cv2.cvtColor(np.ascontiguousarray(rgb), cv2.COLOR_RGB2BGR))

            if (
                args.capture_dataset
                and frame_idx % int(args.dataset_every_frames) == 0
            ):
                snapshot = adapter.snapshot(0)
                mapped_gates = _mapped_gate_labels(raw, geometry, snapshot)
                sample_id = f"frame_{frame_idx:05d}"
                cv2.imwrite(
                    str(dataset_images / f"{sample_id}.png"),
                    cv2.cvtColor(np.ascontiguousarray(rgb), cv2.COLOR_RGB2BGR),
                )
                payload = {
                    "schema": "isaac_drone_racer.ideal_bank_color_frame.v1",
                    "sample_id": sample_id,
                    "camera_pitch_up_deg": pitch_up_deg,
                    "prescribed_bank_deg": math.degrees(bank_rad),
                    "speed_mps": float(args.speed_mps),
                    "lap_fraction": float(
                        omega_radps * t_s / (2.0 * math.pi)
                    ),
                    "mapped_gates": mapped_gates,
                }
                (dataset_labels / f"{sample_id}.json").write_text(
                    json.dumps(payload, indent=2) + "\n",
                    encoding="utf-8",
                )
                cv2.imwrite(
                    str(dataset_annotated / f"{sample_id}.png"),
                    _annotate_gate_identities(rgb, mapped_gates),
                )
            if frame_idx % args.fps == 0 or frame_idx == frame_count - 1:
                samples.append({
                    "frame": frame_idx,
                    "t_s": t_s,
                    "laps": omega_radps * t_s / (2.0 * math.pi),
                    "x_m": float(pos[0, 0]),
                    "y_m": float(pos[0, 1]),
                    "yaw_deg": math.degrees(yaw),
                })
                print(f"[ideal-bank] frame={frame_idx}/{frame_count - 1} "
                      f"laps={samples[-1]['laps']:.3f}", flush=True)
    finally:
        writer.release()
        env.close()

    metadata = {
        "experiment": "prescribed_steady_circular_turn_visual_reference",
        "not_a_free_flight_controller": True,
        "camera_pitch_up_relative_body_deg": pitch_up_deg,
        "multicolor_gate_preview": bool(args.multicolor_gates),
        "gate_colors_rgb": {
            str(index): list(panels[0].color_rgb)
            for index, panels in gate_panels().items()
        } if args.multicolor_gates else None,
        "body_height_m": float(args.body_height_m),
        "track_radius_m": radius_m,
        "track_center_xy_m": [0.0, 12.0],
        "speed_mps": args.speed_mps,
        "roll_deg": math.degrees(bank_rad),
        "pitch_deg": 0.0,
        "yaw_rate_degps": math.degrees(omega_radps),
        "collective_accel_mps2": collective_accel,
        "collective_accel_g": collective_accel / gravity_mps2,
        "laps": laps,
        "duration_s": duration_s,
        "fps": args.fps,
        "frames": frame_count,
        "samples": samples,
        "video": str(video_path),
        "dataset_capture_enabled": bool(args.capture_dataset),
        "dataset_root": str(dataset_root) if args.capture_dataset else None,
        "dataset_contract": (
            "smoke/visual-verification only; prescribed repeated laps are not "
            "an independent train/val/test split"
            if args.capture_dataset
            else None
        ),
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(f"[ideal-bank] saved {video_path}", flush=True)
    if args.capture_dataset:
        frame_total = len(list(dataset_labels.glob("*.json")))
        print(
            f"[ideal-bank] saved color-ID smoke dataset: "
            f"{dataset_root} frames={frame_total}",
            flush=True,
        )
        print(
            f"[ideal-bank] annotated verification frames: {dataset_annotated}",
            flush=True,
        )
    simulation_app.close()


if __name__ == "__main__":
    main()
