"""Evaluate a complete checkpoint over fixed Stage 1 profile/seed sweeps."""

import argparse
import csv
import hashlib
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--profiles", default="clean,mild,nominal,stress")
parser.add_argument("--seeds", default="11,29,47")
parser.add_argument("--episodes_per_case", type=int, default=128)
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--output_dir", default=None)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import gymnasium as gym
import torch
from skrl.utils.runner.torch import Runner

import tasks  # noqa: F401, E402
from isaaclab_rl.skrl import SkrlVecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry  # noqa: E402
from tasks.drone_racer.drone_racer_stage1_env_cfg import DroneRacerStage1EnvCfg  # noqa: E402
from utils.checkpoint_validation import assert_complete_stage0_checkpoint  # noqa: E402
from utils.stage1_metrics import EpisodeRecord, Stage1EpisodeAccumulator  # noqa: E402


def _csv_row(
    profile: str, seed: int, index: int, record: EpisodeRecord, metadata: dict
) -> dict:
    return {
        **metadata,
        "profile": profile,
        "seed": seed,
        "episode": index,
        "completed": record.completed,
        "gates_passed": record.gates_passed,
        "collision": record.collision,
        "flyaway": record.flyaway,
        "return": record.episode_return,
        "duration_s": record.duration_s,
        "position_rmse_m": record.position_rmse_m,
        "attitude_rmse_rad": record.attitude_rmse_rad,
        "velocity_rmse_mps": record.velocity_rmse_mps,
    }


def evaluate_case(env, agent, profile: str, seed: int, episode_count: int):
    raw_env = env.unwrapped
    raw_env.set_fake_sensor_profile(profile, seed=seed)
    observation_dict, _ = raw_env.reset(seed=seed)
    observation = observation_dict["policy"]
    count = raw_env.num_envs
    device = raw_env.device
    returns = torch.zeros(count, device=device)
    steps = torch.zeros(count, dtype=torch.long, device=device)
    gates = torch.zeros(count, dtype=torch.long, device=device)
    position_squared_error = torch.zeros(count, device=device)
    attitude_squared_error = torch.zeros(count, device=device)
    velocity_squared_error = torch.zeros(count, device=device)
    history_length = raw_env.max_episode_length + 1
    vio_age_history = torch.zeros(history_length, count, device=device)
    imu_age_history = torch.zeros(history_length, count, device=device)
    vio_valid_history = torch.zeros(history_length, count, dtype=torch.bool, device=device)
    imu_valid_history = torch.zeros(history_length, count, dtype=torch.bool, device=device)
    env_ids = torch.arange(count, device=device)
    accumulator = Stage1EpisodeAccumulator()
    maximum_steps = max(10_000, episode_count * raw_env.max_episode_length * 2)

    for _ in range(maximum_steps):
        with torch.inference_mode():
            outputs = agent.act(observation, timestep=0, timesteps=0)
            actions = outputs[-1].get("mean_actions", outputs[0])
            observation, reward, terminated, truncated, _ = env.step(actions)

        snapshot = raw_env.stage1_evaluation_snapshot
        done = (terminated | truncated).reshape(-1)
        returns += reward.reshape(-1)
        steps += 1
        gates += snapshot["gate_passed"].long()
        position_squared_error += snapshot["position_error_m"].square()
        attitude_squared_error += snapshot["attitude_error_rad"].square()
        velocity_squared_error += snapshot["velocity_error_mps"].square()
        history_indices = (steps - 1).clamp_max(history_length - 1)
        vio_age_history[history_indices, env_ids] = snapshot["vio_age_s"]
        imu_age_history[history_indices, env_ids] = snapshot["imu_age_s"]
        vio_valid_history[history_indices, env_ids] = snapshot["vio_valid"]
        imu_valid_history[history_indices, env_ids] = snapshot["imu_valid"]

        done_ids = done.nonzero(as_tuple=False).flatten().detach().cpu().tolist()
        for env_index in done_ids:
            if len(accumulator.records) < episode_count:
                sample_count = max(1, int(steps[env_index].item()))
                history_slice = slice(0, sample_count)
                accumulator.add(
                    EpisodeRecord(
                        completed=int(gates[env_index].item())
                        >= raw_env.command_manager.get_term("target").num_gates,
                        gates_passed=int(gates[env_index].item()),
                        collision=bool(snapshot["collision"][env_index].item()),
                        flyaway=bool(snapshot["flyaway"][env_index].item()),
                        episode_return=float(returns[env_index].item()),
                        duration_s=sample_count * raw_env.step_dt,
                        position_rmse_m=float(
                            torch.sqrt(position_squared_error[env_index] / sample_count).item()
                        ),
                        attitude_rmse_rad=float(
                            torch.sqrt(attitude_squared_error[env_index] / sample_count).item()
                        ),
                        velocity_rmse_mps=float(
                            torch.sqrt(velocity_squared_error[env_index] / sample_count).item()
                        ),
                        vio_age_s=tuple(
                            vio_age_history[history_slice, env_index].detach().cpu().tolist()
                        ),
                        imu_age_s=tuple(
                            imu_age_history[history_slice, env_index].detach().cpu().tolist()
                        ),
                        vio_valid=tuple(
                            vio_valid_history[history_slice, env_index].detach().cpu().tolist()
                        ),
                        imu_valid=tuple(
                            imu_valid_history[history_slice, env_index].detach().cpu().tolist()
                        ),
                    )
                )
            returns[env_index] = 0.0
            steps[env_index] = 0
            gates[env_index] = 0
            position_squared_error[env_index] = 0.0
            attitude_squared_error[env_index] = 0.0
            velocity_squared_error[env_index] = 0.0
        if len(accumulator.records) >= episode_count:
            return accumulator
    raise RuntimeError(f"evaluation did not finish {episode_count} episodes within {maximum_steps} steps")


def main() -> None:
    checkpoint = Path(args.checkpoint).expanduser().resolve()
    assert_complete_stage0_checkpoint(checkpoint)
    checkpoint_hash = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    code_revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    row_metadata = {
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_hash,
        "code_revision": code_revision,
        "oracle_gate_relative_guidance": True,
    }
    profiles = [value.strip() for value in args.profiles.split(",") if value.strip()]
    seeds = [int(value.strip()) for value in args.seeds.split(",") if value.strip()]
    if not profiles or not seeds:
        raise ValueError("--profiles and --seeds must contain at least one value")
    if args.episodes_per_case <= 0 or args.num_envs <= 0:
        raise ValueError("--episodes_per_case and --num_envs must be positive")

    cfg = DroneRacerStage1EnvCfg()
    cfg.scene.num_envs = args.num_envs
    cfg.events.push_robot = None
    agent_cfg = load_cfg_from_registry("Isaac-Drone-Racer-Stage1-v0", "skrl_cfg_entry_point")
    agent_cfg["trainer"]["close_environment_at_exit"] = False
    agent_cfg["agent"]["experiment"]["write_interval"] = 0
    agent_cfg["agent"]["experiment"]["checkpoint_interval"] = 0

    gym_env = gym.make("Isaac-Drone-Racer-Stage1-v0", cfg=cfg)
    env = SkrlVecEnvWrapper(gym_env, ml_framework="torch")
    try:
        runner = Runner(env, agent_cfg)
        runner.agent.load(str(checkpoint))
        runner.agent.set_running_mode("eval")
        summaries = {}
        csv_rows = []
        for profile in profiles:
            for seed in seeds:
                print(f"[EVAL] profile={profile} seed={seed}", flush=True)
                result = evaluate_case(
                    env, runner.agent, profile, seed, args.episodes_per_case
                )
                key = f"{profile}/seed_{seed}"
                summaries[key] = result.summary()
                csv_rows.extend(
                    _csv_row(profile, seed, index, record, row_metadata)
                    for index, record in enumerate(result.records)
                )

        output_dir = (
            Path(args.output_dir).expanduser().resolve()
            if args.output_dir
            else PROJECT_ROOT
            / "logs"
            / "stage1_evaluations"
            / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        )
        output_dir.mkdir(parents=True, exist_ok=False)
        payload = {
            **row_metadata,
            "profiles": profiles,
            "seeds": seeds,
            "episodes_per_case": args.episodes_per_case,
            "summaries": summaries,
        }
        (output_dir / "summary.json").write_text(json.dumps(payload, indent=2) + "\n")
        with (output_dir / "episodes.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(csv_rows[0]))
            writer.writeheader()
            writer.writerows(csv_rows)
        print(f"STAGE1_EVALUATION_OK {output_dir}", flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
