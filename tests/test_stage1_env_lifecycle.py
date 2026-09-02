"""Real Isaac Sim lifecycle smoke test for Stage 1.

Run directly so Isaac's application is created before importing task modules:
    python tests/test_stage1_env_lifecycle.py --headless
"""

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
import torch

import tasks  # noqa: F401, E402
from tasks.drone_racer.drone_racer_stage1_env_cfg import DroneRacerStage1EnvCfg  # noqa: E402


def main() -> None:
    cfg = DroneRacerStage1EnvCfg()
    cfg.scene.num_envs = 2
    cfg.seed = 42
    env = gym.make("Isaac-Drone-Racer-Stage1-v0", cfg=cfg)
    try:
        observation, _ = env.reset()
        unwrapped = env.unwrapped
        before = unwrapped.stage1_diagnostics.copy()
        action = torch.zeros(
            unwrapped.num_envs,
            unwrapped.action_manager.total_action_dim,
            device=unwrapped.device,
        )

        observation, _, _, _, _ = env.step(action)
        after = unwrapped.stage1_diagnostics

        assert after["imu_ingest_count"] - before["imu_ingest_count"] == cfg.decimation
        assert after["vio_ingest_count"] - before["vio_ingest_count"] == 1
        assert observation["policy"].shape == (2, 20)
        robot = unwrapped.scene["robot"]
        stage0_clean_state = torch.cat(
            (
                robot.data.root_pos_w,
                robot.data.root_quat_w,
                robot.data.root_lin_vel_b,
                robot.data.root_ang_vel_b,
            ),
            dim=-1,
        )
        torch.testing.assert_close(
            observation["policy"][:, :13],
            stage0_clean_state,
            atol=1.0e-6,
            rtol=1.0e-6,
        )
        expected_timestamp = unwrapped._sim_step_counter * unwrapped.physics_dt
        torch.testing.assert_close(
            unwrapped.stage1_state_estimate.publish_timestamp_s,
            torch.full((2,), expected_timestamp, device=unwrapped.device),
        )
        assert unwrapped.reward_manager.active_terms == [
            "terminating",
            "ang_vel_l2",
            "progress",
            "gate_passed",
            "lookat_next",
        ]
        assert unwrapped.termination_manager.active_terms == ["time_out", "flyaway", "collision"]
        for name in (
            "Stage1/position_error_mean_m",
            "Stage1/attitude_error_mean_rad",
            "Stage1/velocity_error_mean_mps",
            "Stage1/vio_age_mean_s",
            "Stage1/imu_age_mean_s",
            "Stage1/vio_fresh_fraction",
            "Stage1/imu_fresh_fraction",
        ):
            assert name in unwrapped.extras["log"]
        print("STAGE1_LIFECYCLE_OK")
    finally:
        env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
