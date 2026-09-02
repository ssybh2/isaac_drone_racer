"""Evaluate Stage 1 checkpoints with isolated profile processes and first-lap metrics."""

import argparse
import csv
import hashlib
import json
import os
import random
import subprocess
import sys
import traceback
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
parser.add_argument("--_stage1_worker", action="store_true", help=argparse.SUPPRESS)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()


def _split_profiles(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _split_seeds(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def _replace_cli_option(argv: list[str], option: str, value: str) -> list[str]:
    """Replace one argparse-style option while preserving all unrelated flags."""
    output: list[str] = []
    index = 0
    while index < len(argv):
        token = argv[index]
        if token == option:
            index += 2
            continue
        if token.startswith(option + "="):
            index += 1
            continue
        output.append(token)
        index += 1
    output.extend([option, value])
    return output


def _default_output_dir() -> Path:
    return (
        PROJECT_ROOT
        / "logs"
        / "stage1_evaluations"
        / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    )


def _run_isolated_profile_sweep(profiles: list[str]) -> None:
    """Run each profile in a fresh Isaac process and merge its artifacts."""
    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else _default_output_dir()
    )
    output_dir.mkdir(parents=True, exist_ok=False)

    combined_summaries: dict[str, object] = {}
    combined_rows: list[dict[str, str]] = []
    merged_metadata: dict[str, object] | None = None
    worker_dirs: dict[str, str] = {}

    for profile in profiles:
        worker_dir = output_dir / profile
        worker_dirs[profile] = str(worker_dir)
        child_args = list(sys.argv[1:])
        child_args = _replace_cli_option(child_args, "--profiles", profile)
        child_args = _replace_cli_option(child_args, "--output_dir", str(worker_dir))
        child_args.append("--_stage1_worker")

        child_env = os.environ.copy()
        existing_pythonpath = child_env.get("PYTHONPATH", "")
        child_env["PYTHONPATH"] = (
            str(PROJECT_ROOT)
            if not existing_pythonpath
            else f"{PROJECT_ROOT}:{existing_pythonpath}"
        )

        print(f"[ISOLATED] launching profile={profile}", flush=True)
        subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), *child_args],
            cwd=PROJECT_ROOT,
            env=child_env,
            check=True,
        )

        worker_summary = json.loads((worker_dir / "summary.json").read_text())
        if merged_metadata is None:
            merged_metadata = {
                key: worker_summary[key]
                for key in (
                    "checkpoint",
                    "checkpoint_sha256",
                    "code_revision",
                    "oracle_gate_relative_guidance",
                    "seeds",
                    "episodes_per_case",
                )
            }
        combined_summaries.update(worker_summary["summaries"])

        with (worker_dir / "episodes.csv").open(newline="") as stream:
            combined_rows.extend(csv.DictReader(stream))

    if merged_metadata is None or not combined_rows:
        raise RuntimeError("isolated Stage 1 sweep produced no evaluation rows")

    payload = {
        **merged_metadata,
        "profiles": profiles,
        "isolated_process_per_profile": True,
        "worker_output_dirs": worker_dirs,
        "summaries": combined_summaries,
    }
    (output_dir / "summary.json").write_text(json.dumps(payload, indent=2) + "\n")
    with (output_dir / "episodes.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(combined_rows[0]))
        writer.writeheader()
        writer.writerows(combined_rows)

    print(f"STAGE1_EVALUATION_OK {output_dir}", flush=True)


profiles_requested = _split_profiles(args.profiles)
if not profiles_requested:
    raise ValueError("--profiles must contain at least one value")

# Multi-profile evaluation must never reuse one Isaac environment. The parent
# process stays Isaac-free and launches one worker process per profile.
if len(profiles_requested) > 1 and not args._stage1_worker:
    try:
        _run_isolated_profile_sweep(profiles_requested)
    except BaseException:
        traceback.print_exc()
        raise
    sys.exit(0)

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import gymnasium as gym
import numpy as np
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
        "first_lap_completed": record.completed,
        "lap_time_s": record.lap_time_s,
        "failure_gate": record.failure_gate,
        "gates_passed": record.gates_passed,
        "collision": record.collision,
        "flyaway": record.flyaway,
        "return": record.episode_return,
        "duration_s": record.duration_s,
        "position_rmse_m": record.position_rmse_m,
        "attitude_rmse_rad": record.attitude_rmse_rad,
        "velocity_rmse_mps": record.velocity_rmse_mps,
    }


def _seed_case(seed: int) -> None:
    """Seed runtime RNGs without invoking Isaac Sim's host-side Warp RNG built-in."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def evaluate_case(env, agent, profile: str, seed: int, episode_count: int):
    """Evaluate first-lap completion while the underlying env keeps its normal lifecycle."""
    raw_env = env.unwrapped
    _seed_case(seed)
    raw_env.set_fake_sensor_profile(profile, seed=seed)
    observation, _ = env.reset()
    count = raw_env.num_envs
    device = raw_env.device
    num_gates = raw_env.command_manager.get_term("target").num_gates

    returns = torch.zeros(count, device=device)
    trial_steps = torch.zeros(count, dtype=torch.long, device=device)
    gates = torch.zeros(count, dtype=torch.long, device=device)
    first_lap_completed = torch.zeros(count, dtype=torch.bool, device=device)
    first_lap_steps = torch.zeros(count, dtype=torch.long, device=device)
    position_squared_error = torch.zeros(count, device=device)
    attitude_squared_error = torch.zeros(count, device=device)
    velocity_squared_error = torch.zeros(count, device=device)

    history_length = raw_env.max_episode_length + 1
    vio_age_history = torch.zeros(history_length, count, device=device)
    imu_age_history = torch.zeros(history_length, count, device=device)
    vio_valid_history = torch.zeros(history_length, count, dtype=torch.bool, device=device)
    imu_valid_history = torch.zeros(history_length, count, dtype=torch.bool, device=device)
    vio_dropout_history = torch.zeros(history_length, count, dtype=torch.bool, device=device)
    imu_dropout_history = torch.zeros(history_length, count, dtype=torch.bool, device=device)

    accumulator = Stage1EpisodeAccumulator()
    maximum_steps = max(10_000, episode_count * raw_env.max_episode_length * 2)

    for _ in range(maximum_steps):
        with torch.inference_mode():
            outputs = agent.act(observation, timestep=0, timesteps=0)
            actions = outputs[-1].get("mean_actions", outputs[0])
            observation, reward, terminated, truncated, _ = env.step(actions)

        snapshot = raw_env.stage1_evaluation_snapshot
        done = (terminated | truncated).reshape(-1)
        active_before_step = ~first_lap_completed
        active_ids = active_before_step.nonzero(as_tuple=False).flatten()

        if len(active_ids) > 0:
            returns[active_ids] += reward.reshape(-1)[active_ids]
            trial_steps[active_ids] += 1
            gates[active_ids] += snapshot["gate_passed"][active_ids].long()
            gates[active_ids] = gates[active_ids].clamp_max(num_gates)
            position_squared_error[active_ids] += snapshot["position_error_m"][active_ids].square()
            attitude_squared_error[active_ids] += snapshot["attitude_error_rad"][active_ids].square()
            velocity_squared_error[active_ids] += snapshot["velocity_error_mps"][active_ids].square()

            history_indices = (trial_steps[active_ids] - 1).clamp_max(history_length - 1)
            vio_age_history[history_indices, active_ids] = snapshot["vio_age_s"][active_ids]
            imu_age_history[history_indices, active_ids] = snapshot["imu_age_s"][active_ids]
            vio_valid_history[history_indices, active_ids] = snapshot["vio_valid"][active_ids]
            imu_valid_history[history_indices, active_ids] = snapshot["imu_valid"][active_ids]
            vio_dropout_history[history_indices, active_ids] = snapshot["vio_dropped"][active_ids]
            imu_dropout_history[history_indices, active_ids] = snapshot["imu_dropped"][active_ids]

        just_completed = (~first_lap_completed) & (gates >= num_gates)
        first_lap_steps = torch.where(just_completed, trial_steps, first_lap_steps)
        first_lap_completed |= just_completed

        done_ids = done.nonzero(as_tuple=False).flatten().detach().cpu().tolist()
        for env_index in done_ids:
            if len(accumulator.records) < episode_count:
                completed = bool(first_lap_completed[env_index].item())
                sample_count = max(1, int(trial_steps[env_index].item()))
                lap_time_s = (
                    float(first_lap_steps[env_index].item() * raw_env.step_dt)
                    if completed
                    else None
                )
                failure_gate = (
                    None
                    if completed
                    else int(snapshot["next_gate_index"][env_index].item()) + 1
                )
                history_slice = slice(0, sample_count)
                accumulator.add(
                    EpisodeRecord(
                        completed=completed,
                        gates_passed=int(gates[env_index].item()),
                        # A crash after the first lap no longer turns a successful
                        # first-lap trial into a collision failure.
                        collision=(
                            False
                            if completed
                            else bool(snapshot["collision"][env_index].item())
                        ),
                        flyaway=(
                            False
                            if completed
                            else bool(snapshot["flyaway"][env_index].item())
                        ),
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
                        lap_time_s=lap_time_s,
                        failure_gate=failure_gate,
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
                        vio_dropped=tuple(
                            vio_dropout_history[history_slice, env_index].detach().cpu().tolist()
                        ),
                        imu_dropped=tuple(
                            imu_dropout_history[history_slice, env_index].detach().cpu().tolist()
                        ),
                    )
                )

            returns[env_index] = 0.0
            trial_steps[env_index] = 0
            gates[env_index] = 0
            first_lap_completed[env_index] = False
            first_lap_steps[env_index] = 0
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
    profiles = _split_profiles(args.profiles)
    seeds = _split_seeds(args.seeds)
    if not profiles or not seeds:
        raise ValueError("--profiles and --seeds must contain at least one value")
    if args._stage1_worker and len(profiles) != 1:
        raise ValueError("Stage 1 worker process must evaluate exactly one profile")
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
            else _default_output_dir()
        )
        output_dir.mkdir(parents=True, exist_ok=False)
        payload = {
            **row_metadata,
            "profiles": profiles,
            "seeds": seeds,
            "episodes_per_case": args.episodes_per_case,
            "isolated_process_per_profile": bool(args._stage1_worker),
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
    except BaseException:
        traceback.print_exc()
        raise
    finally:
        simulation_app.close()
