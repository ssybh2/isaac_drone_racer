"""Open the real Circular-12 scene with all 12 gate textures bound.

This is a manual Isaac Sim viewport inspection tool. It does not add color
overlay geometry: each gate is spawned from its actual per-gate USD variant,
which differs from the source gate asset only by the bitmap texture binding.
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = False
simulation_app = AppLauncher(args).app

import gymnasium as gym  # noqa: E402

import tasks  # noqa: F401,E402
from tasks.drone_racer.drone_racer_stage2_env_cfg import DroneRacerStage2DataEnvCfg  # noqa: E402
from tasks.drone_racer.gate_texture_variants import validate_circular12_gate_usd_variants  # noqa: E402
from tasks.drone_racer.track_generator import (  # noqa: E402
    CIRCULAR_12_GATE_TRACK_CONFIG,
    generate_track,
)


def main() -> None:
    bindings = validate_circular12_gate_usd_variants()

    cfg = DroneRacerStage2DataEnvCfg()
    cfg.scene.num_envs = 1
    cfg.scene.env_spacing = 0.0
    cfg.scene.track = generate_track(CIRCULAR_12_GATE_TRACK_CONFIG)
    cfg.scene.collision_sensor.debug_vis = False
    cfg.commands.target.debug_vis = False
    cfg.events.push_robot = None

    env = gym.make("Isaac-Drone-Racer-Stage2-Data-v0", cfg=cfg)
    raw = env.unwrapped
    env.reset()

    print("=" * 96)
    print("CIRCULAR-12 REAL TEXTURE PREVIEW")
    print("=" * 96)
    for gate_id, gate_cfg in CIRCULAR_12_GATE_TRACK_CONFIG.items():
        bound = ", ".join(bindings[str(gate_id)])
        pos = gate_cfg["pos"]
        print(
            f"Gate {int(gate_id):02d} pos=({pos[0]: .4f}, {pos[1]: .4f}, {pos[2]: .1f}) "
            f"texture={bound}"
        )
    print()
    print("Use the Isaac Sim viewport to orbit/zoom around the track.")
    print("Close the Isaac Sim window or press Ctrl+C in this terminal when finished.")

    try:
        raw.sim.set_camera_view(
            eye=[30.0, -18.0, 24.0],
            target=[0.0, 12.0, 1.0],
        )
    except Exception as exc:
        print(f"[preview] camera-view preset unavailable: {exc}")

    try:
        while simulation_app.is_running():
            raw.scene.write_data_to_sim()
            raw.sim.step(render=True)
            raw.scene.update(dt=raw.physics_dt)
    except KeyboardInterrupt:
        pass
    finally:
        env.close()
        simulation_app.close()


if __name__ == "__main__":
    main()
