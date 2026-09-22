"""Deterministic full-sensor evaluation for learned-inertial racing policies.

Unlike the legacy CSV logger, this evaluator records the pre-reset terminal
cause, truth/mission gate counts, estimator errors, and perception/fusion
availability for every episode.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--task",
    default="Isaac-Drone-Racer-Learned-Inertial-RL-v0",
)
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--episodes", type=int, default=20)
parser.add_argument("--seed", type=int, default=1)
parser.add_argument(
    "--legacy_hard_clip_policy",
    action="store_true",
    default=False,
    help=(
        "Evaluate a legacy checkpoint with its original unbounded Gaussian mean "
        "(output ACTIONS, clip_actions=False)."
    ),
)
parser.add_argument(
    "--legacy_executed_action_scale",
    type=float,
    default=1.0,
    help=(
        "Diagnostic scale applied after reproducing the legacy environment clamp. "
        "Only valid with --legacy_hard_clip_policy. 1.0 reproduces the historical "
        "behavior exactly; e.g. 0.95 tests sensitivity to a 5%% motor-action margin."
    ),
)
parser.add_argument("--output-dir", type=Path, required=True)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

args_cli.enable_cameras = True
if args_cli.episodes <= 0:
    parser.error("--episodes must be positive")

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from skrl.utils.runner.torch import Runner  # noqa: E402
from isaaclab_rl.skrl import SkrlVecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils import load_cfg_from_registry, parse_env_cfg  # noqa: E402

import tasks  # noqa: F401,E402


def _estimator_truth_error(raw_env) -> tuple[float, float, float]:
    robot = raw_env.scene["robot"]
    gt_p = robot.data.root_pos_w[0].detach().cpu().numpy().astype(np.float64)
    gt_v = robot.data.root_lin_vel_w[0].detach().cpu().numpy().astype(np.float64)
    gt_q = robot.data.root_quat_w[0].detach().cpu().numpy().astype(np.float64)
    state = raw_env.learned_inertial_state
    est_p = np.asarray(state.position_w_b, dtype=np.float64)
    est_v = np.asarray(state.linear_velocity_w_b, dtype=np.float64)
    est_q = np.asarray(state.orientation_w_b_wxyz, dtype=np.float64)

    gt_q /= np.linalg.norm(gt_q)
    est_q /= np.linalg.norm(est_q)
    q_dot = float(np.clip(abs(np.dot(gt_q, est_q)), 0.0, 1.0))
    orientation_error_deg = float(np.degrees(2.0 * np.arccos(q_dot)))

    return (
        float(np.linalg.norm(est_p - gt_p)),
        float(np.linalg.norm(est_v - gt_v)),
        orientation_error_deg,
    )


def _mean(values):
    return None if not values else float(sum(values) / len(values))


def main() -> None:
    torch.manual_seed(int(args_cli.seed))
    np.random.seed(int(args_cli.seed))
    if not (0.0 < float(args_cli.legacy_executed_action_scale) <= 1.0):
        raise ValueError("--legacy_executed_action_scale must be in (0, 1]")
    if (
        float(args_cli.legacy_executed_action_scale) != 1.0
        and not args_cli.legacy_hard_clip_policy
    ):
        raise ValueError(
            "--legacy_executed_action_scale requires --legacy_hard_clip_policy"
        )

    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=1,
    )
    env_cfg.scene.num_envs = 1
    env_cfg.seed = int(args_cli.seed)

    agent_cfg = load_cfg_from_registry(args_cli.task, "skrl_cfg_entry_point")
    if args_cli.legacy_hard_clip_policy:
        policy_cfg = agent_cfg["models"]["policy"]
        policy_cfg["clip_actions"] = False
        policy_cfg["clip_mean_actions"] = False
        policy_cfg["clip_log_std"] = True
        policy_cfg["min_log_std"] = -20.0
        policy_cfg["max_log_std"] = 2.0
        policy_cfg["initial_log_std"] = 0.0
        policy_cfg["output"] = "ACTIONS"
        print(
            "[eval] legacy action semantics enabled: "
            "raw Gaussian mean -> environment hard clamp[-1, 1]",
            flush=True,
        )
    agent_cfg["seed"] = int(args_cli.seed)
    agent_cfg["trainer"]["close_environment_at_exit"] = False
    agent_cfg["agent"]["experiment"]["write_interval"] = 0
    agent_cfg["agent"]["experiment"]["checkpoint_interval"] = 0

    env = gym.make(args_cli.task, cfg=env_cfg)
    raw_env = env.unwrapped
    wrapped = SkrlVecEnvWrapper(env, ml_framework="torch")
    runner = Runner(wrapped, agent_cfg)

    checkpoint = str(Path(args_cli.checkpoint).expanduser().resolve())
    print(f"[eval] loading checkpoint: {checkpoint}", flush=True)
    runner.agent.load(checkpoint)
    runner.agent.set_running_mode("eval")
    runner.agent.set_mode("eval")

    output_dir = args_cli.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    obs, _ = wrapped.reset()
    records: list[dict[str, object]] = []

    episode_return = 0.0
    position_sq_sum = 0.0
    velocity_sq_sum = 0.0
    orientation_sq_sum = 0.0
    error_samples = 0
    max_position_error = 0.0
    max_velocity_error = 0.0
    max_orientation_error = 0.0

    try:
        while len(records) < int(args_cli.episodes):
            p_err, v_err, r_err = _estimator_truth_error(raw_env)
            position_sq_sum += p_err * p_err
            velocity_sq_sum += v_err * v_err
            orientation_sq_sum += r_err * r_err
            error_samples += 1
            max_position_error = max(max_position_error, p_err)
            max_velocity_error = max(max_velocity_error, v_err)
            max_orientation_error = max(max_orientation_error, r_err)

            with torch.inference_mode():
                outputs = runner.agent.act(obs, timestep=0, timesteps=0)
                actions = outputs[-1].get("mean_actions", outputs[0])
                if args_cli.legacy_hard_clip_policy:
                    # Reproduce the historical environment-executed command
                    # explicitly so action-scale sensitivity can be tested
                    # independently of actor distillation.
                    actions = actions.clamp(-1.0, 1.0)
                    actions = actions * float(
                        args_cli.legacy_executed_action_scale
                    )
                obs, reward, terminated, truncated, _ = wrapped.step(actions)

            episode_return += float(reward.reshape(-1)[0].item())
            done = bool(terminated.reshape(-1)[0].item()) or bool(
                truncated.reshape(-1)[0].item()
            )
            if not done:
                continue

            terminal = raw_env.last_episode_diagnostic
            if terminal is None:
                raise RuntimeError(
                    "environment terminated without last_episode_diagnostic"
                )

            sample_count = max(1, error_samples)
            row = {
                "episode": len(records) + 1,
                "return": float(episode_return),
                **terminal,
                "position_rmse_m": float(
                    np.sqrt(position_sq_sum / sample_count)
                ),
                "velocity_rmse_mps": float(
                    np.sqrt(velocity_sq_sum / sample_count)
                ),
                "orientation_rmse_deg": float(
                    np.sqrt(orientation_sq_sum / sample_count)
                ),
                "position_max_error_m": float(max_position_error),
                "velocity_max_error_mps": float(max_velocity_error),
                "orientation_max_error_deg": float(max_orientation_error),
            }
            records.append(row)

            cause = (
                "collision"
                if row["collision"]
                else "flyaway"
                if row["flyaway"]
                else "timeout"
                if row["time_out"]
                else "other"
            )
            print(
                "[eval] "
                f"ep={row['episode']:02d}/{args_cli.episodes} "
                f"steps={row['steps']} "
                f"truth_gates={row['truth_gates_passed']} "
                f"mission_gates={row['mission_gates_passed']} "
                f"cause={cause} "
                f"return={row['return']:.3f} "
                f"p_rmse={row['position_rmse_m']:.3f}m "
                f"v_rmse={row['velocity_rmse_mps']:.3f}m/s "
                f"R_rmse={row['orientation_rmse_deg']:.2f}deg "
                f"visual={row['gate_updates']}/{row['gate_attempts']}",
                flush=True,
            )

            episode_return = 0.0
            position_sq_sum = 0.0
            velocity_sq_sum = 0.0
            orientation_sq_sum = 0.0
            error_samples = 0
            max_position_error = 0.0
            max_velocity_error = 0.0
            max_orientation_error = 0.0

        csv_path = output_dir / "episodes.csv"
        with csv_path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(records[0].keys()))
            writer.writeheader()
            writer.writerows(records)

        num_gates = int(records[0]["num_gates"])
        truth_gates = [int(r["truth_gates_passed"]) for r in records]
        mission_gates = [int(r["mission_gates_passed"]) for r in records]
        durations = [float(r["duration_s"]) for r in records]
        returns = [float(r["return"]) for r in records]
        p_rmse = [float(r["position_rmse_m"]) for r in records]
        v_rmse = [float(r["velocity_rmse_mps"]) for r in records]
        r_rmse = [float(r["orientation_rmse_deg"]) for r in records]

        summary = {
            "checkpoint": checkpoint,
            "action_semantics": (
                "legacy_explicit_hard_clip_then_scale"
                if args_cli.legacy_hard_clip_policy
                else "current_registered_policy"
            ),
            "legacy_executed_action_scale": (
                float(args_cli.legacy_executed_action_scale)
                if args_cli.legacy_hard_clip_policy
                else None
            ),
            "episodes": len(records),
            "num_gates": num_gates,
            "full_lap_completion_rate": float(
                sum(g >= num_gates for g in truth_gates) / len(records)
            ),
            "truth_gates_passed": {
                "mean": _mean(truth_gates),
                "median": float(np.median(truth_gates)),
                "min": int(min(truth_gates)),
                "max": int(max(truth_gates)),
                "histogram": {
                    str(k): int(sum(g == k for g in truth_gates))
                    for k in sorted(set(truth_gates))
                },
            },
            "mission_gates_passed": {
                "mean": _mean(mission_gates),
                "median": float(np.median(mission_gates)),
                "min": int(min(mission_gates)),
                "max": int(max(mission_gates)),
            },
            "termination_counts": {
                "collision": int(sum(bool(r["collision"]) for r in records)),
                "flyaway": int(sum(bool(r["flyaway"]) for r in records)),
                "timeout": int(sum(bool(r["time_out"]) for r in records)),
                "other": int(
                    sum(
                        not bool(r["collision"])
                        and not bool(r["flyaway"])
                        and not bool(r["time_out"])
                        for r in records
                    )
                ),
            },
            "duration_s": {
                "mean": _mean(durations),
                "median": float(np.median(durations)),
                "min": float(min(durations)),
                "max": float(max(durations)),
            },
            "return": {
                "mean": _mean(returns),
                "median": float(np.median(returns)),
            },
            "estimator": {
                "position_rmse_mean_m": _mean(p_rmse),
                "velocity_rmse_mean_mps": _mean(v_rmse),
                "orientation_rmse_mean_deg": _mean(r_rmse),
                "position_max_error_across_episodes_m": float(
                    max(float(r["position_max_error_m"]) for r in records)
                ),
                "velocity_max_error_across_episodes_mps": float(
                    max(float(r["velocity_max_error_mps"]) for r in records)
                ),
                "orientation_max_error_across_episodes_deg": float(
                    max(float(r["orientation_max_error_deg"]) for r in records)
                ),
            },
            "perception": {
                "gate_attempts_mean": _mean(
                    [int(r["gate_attempts"]) for r in records]
                ),
                "gate_updates_mean": _mean(
                    [int(r["gate_updates"]) for r in records]
                ),
                "gate_rejects_mean": _mean(
                    [int(r["gate_rejects"]) for r in records]
                ),
            },
            "learned_motion": {
                "updates_mean": _mean(
                    [int(r["learned_updates"]) for r in records]
                ),
                "fusions_mean": _mean(
                    [int(r["learned_fusions"]) for r in records]
                ),
                "skips_mean": _mean(
                    [int(r["learned_update_skips"]) for r in records]
                ),
            },
        }

        summary_path = output_dir / "summary.json"
        summary_path.write_text(json.dumps(summary, indent=2) + "\n")

        print("=" * 96)
        print("LEARNED-INERTIAL POLICY EVALUATION SUMMARY")
        print("=" * 96)
        print(json.dumps(summary, indent=2))
        print(f"[eval] wrote: {csv_path}")
        print(f"[eval] wrote: {summary_path}")
    finally:
        wrapped.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
