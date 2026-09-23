"""One-off prescribed-circle FPV comparison; never used by training.

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
parser.add_argument("--camera-pitch-up-deg", type=float, default=40.0)
parser.add_argument("--body-height-m", type=float, default=1.0)
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
from perception.isaac_adapter import IsaacStage2TruthAdapter  # noqa: E402
from tasks.drone_racer.drone_racer_stage2_env_cfg import (  # noqa: E402
    DroneRacerStage2DataEnvCfg,
    stage2_reference_camera_cfg,
)
from tasks.drone_racer.track_generator import (  # noqa: E402
    CIRCULAR_12_GATE_TRACK_CONFIG,
    generate_track,
)
from multicolor_preview import gate_panels  # noqa: E402


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
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(f"[ideal-bank] saved {video_path}", flush=True)
    simulation_app.close()


if __name__ == "__main__":
    main()
