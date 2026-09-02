"""Isaac-backed configuration contract test. Run directly with --headless."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
launcher = AppLauncher(args)
simulation_app = launcher.app

import gymnasium as gym
from isaaclab.managers import ObservationTermCfg

import tasks  # noqa: F401, E402
from tasks.drone_racer.drone_racer_env_cfg import DroneRacerEnvCfg  # noqa: E402
from tasks.drone_racer.drone_racer_stage1_env_cfg import (  # noqa: E402
    DroneRacerStage1EnvCfg,
    DroneRacerStage1EnvCfg_PLAY,
)


def observation_terms(group) -> list[str]:
    return [name for name, value in vars(group).items() if isinstance(value, ObservationTermCfg)]


def main() -> None:
    stage0 = DroneRacerEnvCfg()
    stage1 = DroneRacerStage1EnvCfg()
    play = DroneRacerStage1EnvCfg_PLAY()

    assert observation_terms(stage0.observations.policy) == [
        "position",
        "attitude",
        "lin_vel",
        "ang_vel",
        "target_pos_b",
        "actions",
    ]
    assert observation_terms(stage1.observations.policy) == [
        "estimated_drone_state",
        "target_pos_b",
        "actions",
    ]
    assert stage1.observations.policy.target_pos_b.func is stage0.observations.policy.target_pos_b.func
    assert stage1.fake_sensors.profile == "clean"
    assert play.fake_sensors.profile == "clean"
    assert stage1.scene.imu is None and stage1.scene.tiled_camera is None
    assert play.scene.imu is None and play.scene.tiled_camera is None
    assert stage1.scene.num_envs == 4096
    assert gym.spec("Isaac-Drone-Racer-v0").entry_point == "isaaclab.envs:ManagerBasedRLEnv"
    assert gym.spec("Isaac-Drone-Racer-Stage1-v0").entry_point.endswith(
        ".stage1_env:Stage1DroneRacerEnv"
    )
    print("STAGE1_CONFIG_OK", flush=True)


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
