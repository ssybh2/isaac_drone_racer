"""Validate clean Stage 1 observation and complete-checkpoint equivalence."""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--checkpoint", required=True, help="Complete Stage 0 SKRL checkpoint")
parser.add_argument("--num_envs", type=int, default=2)
parser.add_argument("--seed", type=int, default=1)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import gymnasium as gym
import isaaclab.utils.math as math_utils
import torch
from skrl.utils.runner.torch import Runner

import tasks  # noqa: F401, E402
from isaaclab_rl.skrl import SkrlVecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry  # noqa: E402
from tasks.drone_racer.drone_racer_stage1_env_cfg import DroneRacerStage1EnvCfg  # noqa: E402
from utils.checkpoint_validation import assert_complete_stage0_checkpoint  # noqa: E402


def maximum_error(left: torch.Tensor, right: torch.Tensor) -> float:
    return torch.max(torch.abs(left - right)).detach().cpu().item()


def assert_equivalent(name: str, left: torch.Tensor, right: torch.Tensor) -> float:
    error = maximum_error(left, right)
    if not torch.allclose(left, right, atol=1.0e-5, rtol=1.0e-5):
        raise AssertionError(f"{name} mismatch: maximum absolute error {error:.9g}")
    return error


def main() -> None:
    assert_complete_stage0_checkpoint(args.checkpoint)
    cfg = DroneRacerStage1EnvCfg()
    cfg.scene.num_envs = args.num_envs
    cfg.seed = args.seed
    agent_cfg = load_cfg_from_registry("Isaac-Drone-Racer-Stage1-v0", "skrl_cfg_entry_point")
    agent_cfg["seed"] = args.seed
    agent_cfg["trainer"]["close_environment_at_exit"] = False
    agent_cfg["agent"]["experiment"]["write_interval"] = 0
    agent_cfg["agent"]["experiment"]["checkpoint_interval"] = 0

    gym_env = gym.make("Isaac-Drone-Racer-Stage1-v0", cfg=cfg)
    raw_env = gym_env.unwrapped
    env = SkrlVecEnvWrapper(gym_env, ml_framework="torch")
    try:
        runner = Runner(env, agent_cfg)
        runner.agent.load(str(Path(args.checkpoint).expanduser().resolve()))
        runner.agent.set_running_mode("eval")
        stage1_observation, _ = env.reset()

        robot = raw_env.scene["robot"]
        target = raw_env.command_manager.get_term("target").command[:, :3]
        target_pos_b, _ = math_utils.subtract_frame_transforms(
            robot.data.root_pos_w, robot.data.root_quat_w, target
        )
        stage0_observation = torch.cat(
            (
                robot.data.root_pos_w,
                robot.data.root_quat_w,
                robot.data.root_lin_vel_b,
                robot.data.root_ang_vel_b,
                target_pos_b,
                raw_env.action_manager.action,
            ),
            dim=-1,
        )

        raw_error = assert_equivalent("raw observation", stage0_observation, stage1_observation)
        stage0_preprocessed = runner.agent._state_preprocessor(stage0_observation, train=False)
        stage1_preprocessed = runner.agent._state_preprocessor(stage1_observation, train=False)
        preprocessor_error = assert_equivalent(
            "preprocessed observation", stage0_preprocessed, stage1_preprocessed
        )
        with torch.inference_mode():
            _, _, stage0_policy = runner.agent.policy.act(
                {"states": stage0_preprocessed}, role="policy"
            )
            _, _, stage1_policy = runner.agent.policy.act(
                {"states": stage1_preprocessed}, role="policy"
            )
            action_error = assert_equivalent(
                "deterministic mean action",
                stage0_policy["mean_actions"],
                stage1_policy["mean_actions"],
            )
            stage0_value, _, _ = runner.agent.value.act(
                {"states": stage0_preprocessed}, role="value"
            )
            stage1_value, _, _ = runner.agent.value.act(
                {"states": stage1_preprocessed}, role="value"
            )
            value_error = assert_equivalent("value", stage0_value, stage1_value)

        print(
            "STAGE1_EQUIVALENCE_OK "
            f"raw={raw_error:.3g} preprocessed={preprocessor_error:.3g} "
            f"mean_action={action_error:.3g} value={value_error:.3g}",
            flush=True,
        )
    finally:
        env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
