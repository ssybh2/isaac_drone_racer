"""Flight-quality audit for expert, BC/DAgger and skrl PPO controllers.

This is the hard gate before perception/estimator qualification. It measures
gate progress plus the failure mode that ordinary reward misses: complete or
near-complete body tumbling relative to the coordinated Circular-12 reference.
"""

from __future__ import annotations

import argparse
import sys
import csv
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--task",
    default="Isaac-Drone-Racer-Swift-CTBR-GT-Circular12-ExpertValidation-v0",
)
parser.add_argument(
    "--controller",
    choices=["expert", "bc", "skrl"],
    default="expert",
)
parser.add_argument("--checkpoint", type=Path, default=None)
parser.add_argument("--episodes", type=int, default=10)
parser.add_argument("--target-speed-mps", type=float, default=14.0)
parser.add_argument("--output-dir", type=Path, required=True)
parser.add_argument("--seed", type=int, default=1)
parser.add_argument(
    "--fail-on-tumble",
    action="store_true",
    help="Exit non-zero if any inversion/tumble event is observed.",
)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.controller in {"bc", "skrl"} and args.checkpoint is None:
    parser.error("--checkpoint is required for bc/skrl controllers")
if args.episodes <= 0:
    parser.error("--episodes must be positive")

simulation_app = AppLauncher(args).app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
import isaaclab.utils.math as math_utils  # noqa: E402
from isaaclab_rl.skrl import SkrlVecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils import load_cfg_from_registry, parse_env_cfg  # noqa: E402
from skrl.utils.runner.torch import Runner  # noqa: E402

import tasks  # noqa: F401,E402
from imitation.bc_policy import Circular12BCPolicy  # noqa: E402
from imitation.circular12_expert import (  # noqa: E402
    circular12_expert_action,
    config_from_ctbr_action_cfg,
)


def _termination_cause(raw_env) -> str:
    manager = raw_env.termination_manager
    fired = {
        name
        for name in manager.active_terms
        if bool(manager.get_term(name).reshape(-1)[0].item())
    }
    if "collision" in fired:
        return "collision"
    if "flyaway" in fired:
        return "flyaway"
    if "time_out" in fired:
        return "timeout"
    return "other"


def _skrl_runner(wrapped):
    cfg = load_cfg_from_registry(args.task, "skrl_cfg_entry_point")
    cfg["seed"] = int(args.seed)
    cfg["trainer"]["close_environment_at_exit"] = False
    cfg["agent"]["experiment"]["write_interval"] = 0
    cfg["agent"]["experiment"]["checkpoint_interval"] = 0
    runner = Runner(wrapped, cfg)
    runner.agent.load(str(args.checkpoint.expanduser().resolve()))
    runner.agent.set_running_mode("eval")
    runner.agent.set_mode("eval")
    return runner


def main() -> None:
    env_cfg = parse_env_cfg(
        args.task,
        device=args.device,
        num_envs=1,
    )
    env_cfg.scene.num_envs = 1
    env_cfg.seed = int(args.seed)
    if getattr(env_cfg.events, "reset_base", None) is not None:
        env_cfg.events.reset_base.params["target_speed_mps"] = float(
            args.target_speed_mps
        )
    env = gym.make(args.task, cfg=env_cfg)
    raw = env.unwrapped
    wrapped = SkrlVecEnvWrapper(env, ml_framework="torch")

    bc = None
    runner = None
    if args.controller == "bc":
        bc, bc_metadata = Circular12BCPolicy.load(
            args.checkpoint, map_location=raw.device
        )
        trained_speed = bc_metadata.get("target_speed_mps")
        if trained_speed is not None and abs(
            float(trained_speed) - float(args.target_speed_mps)
        ) > 1.0e-6:
            raise ValueError(
                "BC flight-quality target speed must match checkpoint metadata: "
                f"checkpoint={trained_speed} requested={args.target_speed_mps}"
            )
        bc = bc.to(raw.device).eval()
    elif args.controller == "skrl":
        runner = _skrl_runner(wrapped)

    expert_cfg = config_from_ctbr_action_cfg(
        raw.cfg.actions.control_action,
        target_speed_mps=float(args.target_speed_mps),
        radius_m=12.0,
        height_m=2.07,
    )

    out_dir = args.output_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    obs, _ = wrapped.reset()
    command = raw.command_manager.get_term("target")
    prev_gate = int(command.next_gate_idx[0].item())

    episode = 0
    ep_step = 0
    ep_gates = 0
    speed_samples: list[float] = []
    rate_samples: list[float] = []
    radius_error: list[float] = []
    height_error: list[float] = []
    attitude_error: list[float] = []
    action_abs: list[float] = []
    bc_expert_action_mae: list[float] = []
    bc_expert_action_abs_components: list[np.ndarray] = []
    obs_zmax: list[float] = []
    obs_group_zmax: dict[str, list[float]] = {
        "position": [],
        "velocity": [],
        "rotation": [],
        "gate_corners": [],
        "previous_action": [],
    }
    obs_clip_fraction: list[float] = []
    inverted_samples = 0
    inversion_events = 0
    gross_excursion_events = 0
    inverted_active = False
    gross_active = False

    try:
        while episode < int(args.episodes):
            robot = raw.scene["robot"]
            p_w = robot.data.root_pos_w
            v_w = robot.data.root_lin_vel_w
            R_wb = math_utils.matrix_from_quat(robot.data.root_quat_w)
            expert = circular12_expert_action(
                p_w, v_w, R_wb, expert_cfg
            )

            if args.controller == "expert":
                action = expert.action
            elif args.controller == "bc":
                with torch.inference_mode():
                    action = bc(obs)
                    standardized = (
                        obs - bc.observation_mean
                    ) / bc.observation_std
                action_abs_error = (action - expert.action).abs()
                bc_expert_action_mae.append(
                    float(action_abs_error.mean().item())
                )
                bc_expert_action_abs_components.append(
                    action_abs_error.detach()
                    .cpu()
                    .numpy()
                    .reshape(-1)
                    .astype(np.float64)
                )
                obs_zmax.append(
                    float(standardized.abs().max().item())
                )
                group_slices = {
                    "position": slice(0, 3),
                    "velocity": slice(3, 6),
                    "rotation": slice(6, 15),
                    "gate_corners": slice(15, 27),
                    "previous_action": slice(27, 31),
                }
                for group_name, group_slice in group_slices.items():
                    obs_group_zmax[group_name].append(
                        float(
                            standardized[
                                :, group_slice
                            ].abs().max().item()
                        )
                    )
                clip = bc.cfg.standardized_observation_clip
                if clip is not None:
                    obs_clip_fraction.append(
                        float(
                            (standardized.abs() > float(clip))
                            .float()
                            .mean()
                            .item()
                        )
                    )
            else:
                with torch.inference_mode():
                    outputs = runner.agent.act(
                        obs, timestep=0, timesteps=0
                    )
                    action = outputs[-1].get(
                        "mean_actions", outputs[0]
                    )

            speed_samples.append(
                float(torch.linalg.vector_norm(v_w[0, :2]).item())
            )
            rate_samples.append(
                float(
                    torch.linalg.vector_norm(
                        robot.data.root_ang_vel_b[0]
                    ).item()
                )
            )
            radius = float(
                torch.linalg.vector_norm(
                    p_w[0, :2]
                    - torch.tensor(
                        [0.0, 12.0],
                        dtype=p_w.dtype,
                        device=p_w.device,
                    )
                ).item()
            )
            radius_error.append(radius - 12.0)
            height_error.append(float(p_w[0, 2].item()) - 2.07)
            att_err = float(
                torch.linalg.vector_norm(
                    expert.attitude_error_rotvec_b[0]
                ).item()
            )
            attitude_error.append(att_err)

            # In a correct ~69-deg coordinated turn body +Z still has a positive
            # world-Z component. A negative value means the vehicle crossed
            # through a genuinely inverted attitude.
            inverted_now = bool((R_wb[0, 2, 2] < 0.0).item())
            gross_now = att_err > float(np.deg2rad(120.0))
            if inverted_now and not inverted_active:
                inversion_events += 1
            if gross_now and not gross_active:
                gross_excursion_events += 1
            inverted_active = inverted_now
            gross_active = gross_now
            inverted_samples += int(inverted_now)

            action_abs.extend(
                action.detach().abs().cpu().numpy().reshape(-1).tolist()
            )

            obs, _, terminated, truncated, _ = wrapped.step(action)
            ep_step += 1
            done = bool(terminated.reshape(-1)[0].item()) or bool(
                truncated.reshape(-1)[0].item()
            )

            current_gate = int(command.next_gate_idx[0].item())
            if not done and current_gate != prev_gate:
                ep_gates += int(
                    (current_gate - prev_gate) % int(command.num_gates)
                )
            prev_gate = current_gate
            if not done:
                continue

            row = {
                "episode": episode + 1,
                "steps": ep_step,
                "gates": ep_gates,
                "termination": _termination_cause(raw),
                "speed_mean_mps": float(np.mean(speed_samples)),
                "speed_p95_mps": float(np.percentile(speed_samples, 95)),
                "body_rate_mean_radps": float(np.mean(rate_samples)),
                "body_rate_p95_radps": float(np.percentile(rate_samples, 95)),
                "body_rate_max_radps": float(np.max(rate_samples)),
                "radius_rmse_m": float(
                    np.sqrt(np.mean(np.square(radius_error)))
                ),
                "height_rmse_m": float(
                    np.sqrt(np.mean(np.square(height_error)))
                ),
                "attitude_error_p95_deg": float(
                    np.degrees(np.percentile(attitude_error, 95))
                ),
                "attitude_error_max_deg": float(
                    np.degrees(np.max(attitude_error))
                ),
                "inverted_fraction": float(
                    inverted_samples / max(ep_step, 1)
                ),
                "inversion_events": int(inversion_events),
                "gross_attitude_excursion_events": int(
                    gross_excursion_events
                ),
                "action_abs_mean": float(np.mean(action_abs)),
                "action_saturation_fraction": float(
                    np.mean(np.asarray(action_abs) > 0.95)
                ),
                "bc_expert_action_mae_mean": (
                    float(np.mean(bc_expert_action_mae))
                    if bc_expert_action_mae else float("nan")
                ),
                "bc_expert_action_mae_p95": (
                    float(np.percentile(bc_expert_action_mae, 95))
                    if bc_expert_action_mae else float("nan")
                ),
                "obs_zmax_mean": (
                    float(np.mean(obs_zmax))
                    if obs_zmax else float("nan")
                ),
                "obs_zmax_p95": (
                    float(np.percentile(obs_zmax, 95))
                    if obs_zmax else float("nan")
                ),
                "obs_zmax_max": (
                    float(np.max(obs_zmax))
                    if obs_zmax else float("nan")
                ),
                "obs_clip_fraction_mean": (
                    float(np.mean(obs_clip_fraction))
                    if obs_clip_fraction else 0.0
                ),
                "obs_clip_fraction_p95": (
                    float(np.percentile(obs_clip_fraction, 95))
                    if obs_clip_fraction else 0.0
                ),
                **{
                    f"obs_{group_name}_zmax_p95": (
                        float(np.percentile(values, 95))
                        if values else float("nan")
                    )
                    for group_name, values in obs_group_zmax.items()
                },
                **{
                    f"obs_{group_name}_zmax_max": (
                        float(np.max(values))
                        if values else float("nan")
                    )
                    for group_name, values in obs_group_zmax.items()
                },
                **(
                    {
                        f"bc_expert_action_{name}_mae": float(
                            np.asarray(
                                bc_expert_action_abs_components
                            )[:, index].mean()
                        )
                        for index, name in enumerate(
                            ("thrust", "p", "q", "r")
                        )
                    }
                    if bc_expert_action_abs_components
                    else {}
                ),
            }
            rows.append(row)
            print(
                "[flight-quality] "
                f"ep={episode + 1:02d}/{args.episodes} "
                f"gates={ep_gates} cause={row['termination']} "
                f"inv={inversion_events} gross={gross_excursion_events} "
                f"rate95={row['body_rate_p95_radps']:.3f}",
                flush=True,
            )

            episode += 1
            ep_step = 0
            ep_gates = 0
            speed_samples = []
            rate_samples = []
            radius_error = []
            height_error = []
            attitude_error = []
            action_abs = []
            bc_expert_action_mae = []
            bc_expert_action_abs_components = []
            obs_zmax = []
            obs_group_zmax = {
                "position": [],
                "velocity": [],
                "rotation": [],
                "gate_corners": [],
                "previous_action": [],
            }
            obs_clip_fraction = []
            inverted_samples = 0
            inversion_events = 0
            gross_excursion_events = 0
            inverted_active = False
            gross_active = False
            prev_gate = int(command.next_gate_idx[0].item())

        total_inversions = int(
            sum(row["inversion_events"] for row in rows)
        )
        total_gross = int(
            sum(
                row["gross_attitude_excursion_events"]
                for row in rows
            )
        )
        gates = np.asarray([row["gates"] for row in rows], dtype=np.int32)
        summary = {
            "controller": args.controller,
            "checkpoint": (
                str(args.checkpoint.expanduser().resolve())
                if args.checkpoint is not None
                else None
            ),
            "episodes": int(args.episodes),
            "target_speed_mps": float(args.target_speed_mps),
            "gates_mean": float(gates.mean()),
            "gates_min": int(gates.min()),
            "gates_max": int(gates.max()),
            "full_lap_rate": float(
                np.mean(gates >= int(command.num_gates))
            ),
            "inversion_events_total": total_inversions,
            "gross_attitude_excursion_events_total": total_gross,
            "zero_tumble_gate": (
                total_inversions == 0 and total_gross == 0
            ),
            "body_rate_p95_mean_radps": float(
                np.mean(
                    [row["body_rate_p95_radps"] for row in rows]
                )
            ),
            "radius_rmse_mean_m": float(
                np.mean([row["radius_rmse_m"] for row in rows])
            ),
            "height_rmse_mean_m": float(
                np.mean([row["height_rmse_m"] for row in rows])
            ),
            "speed_mean_mps": float(
                np.mean([row["speed_mean_mps"] for row in rows])
            ),
            "bc_expert_action_mae_mean": float(
                np.nanmean(
                    [row["bc_expert_action_mae_mean"] for row in rows]
                )
            ),
            "bc_expert_action_mae_p95_mean": float(
                np.nanmean(
                    [row["bc_expert_action_mae_p95"] for row in rows]
                )
            ),
            "obs_zmax_p95_mean": float(
                np.nanmean([row["obs_zmax_p95"] for row in rows])
            ),
            "obs_zmax_max": float(
                np.nanmax([row["obs_zmax_max"] for row in rows])
            ),
            "obs_clip_fraction_mean": float(
                np.mean(
                    [row["obs_clip_fraction_mean"] for row in rows]
                )
            ),
            "obs_clip_fraction_p95_mean": float(
                np.mean(
                    [row["obs_clip_fraction_p95"] for row in rows]
                )
            ),
            **{
                f"obs_{group_name}_zmax_p95_mean": float(
                    np.nanmean(
                        [
                            row[f"obs_{group_name}_zmax_p95"]
                            for row in rows
                        ]
                    )
                )
                for group_name in (
                    "position",
                    "velocity",
                    "rotation",
                    "gate_corners",
                    "previous_action",
                )
            },
            **{
                f"obs_{group_name}_zmax_max": float(
                    np.nanmax(
                        [
                            row[f"obs_{group_name}_zmax_max"]
                            for row in rows
                        ]
                    )
                )
                for group_name in (
                    "position",
                    "velocity",
                    "rotation",
                    "gate_corners",
                    "previous_action",
                )
            },
            **{
                f"bc_expert_action_{name}_mae_mean": float(
                    np.nanmean(
                        [
                            row.get(
                                f"bc_expert_action_{name}_mae",
                                float("nan"),
                            )
                            for row in rows
                        ]
                    )
                )
                for name in ("thrust", "p", "q", "r")
            },
        }

        with (out_dir / "episodes.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        (out_dir / "summary.json").write_text(
            json.dumps(summary, indent=2) + "\n"
        )
        print(json.dumps(summary, indent=2), flush=True)

        if args.fail_on_tumble and not summary["zero_tumble_gate"]:
            raise RuntimeError(
                "flight-quality gate failed: inversion/gross tumble observed"
            )
    finally:
        wrapped.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
