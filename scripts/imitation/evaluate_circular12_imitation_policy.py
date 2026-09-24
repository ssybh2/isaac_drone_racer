"""Free-flight quality audit for the Circular-12 expert or BC/DAgger student.

This evaluator never writes simulator pose/velocity. Both controller modes act
through the existing normalized Swift CTBR action contract and therefore use
body-rate PID, allocation, motors and physics.
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
    default="Isaac-Drone-Racer-Swift-CTBR-GT-Circular12-ImitationFineTune-v0",
)
parser.add_argument(
    "--student",
    type=Path,
    default=None,
    help="BC/DAgger checkpoint. Omit to evaluate the analytic GT CTBR expert.",
)
parser.add_argument("--episodes", type=int, default=10)
parser.add_argument("--target-speed-mps", type=float, default=14.0)
parser.add_argument("--output-dir", type=Path, required=True)
parser.add_argument("--seed", type=int, default=1)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.episodes <= 0:
    parser.error("--episodes must be positive")
if args.target_speed_mps <= 0.0:
    parser.error("--target-speed-mps must be positive")

simulation_app = AppLauncher(args).app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
import isaaclab.utils.math as math_utils  # noqa: E402
from isaaclab_rl.skrl import SkrlVecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

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
    for name in ("collision", "flyaway", "time_out"):
        if name in fired:
            return "timeout" if name == "time_out" else name
    return "other"


def _rmse(values: list[float]) -> float:
    if not values:
        return float("nan")
    array = np.asarray(values, dtype=np.float64)
    return float(np.sqrt(np.mean(np.square(array))))


def main() -> None:
    cfg = parse_env_cfg(args.task, device=args.device, num_envs=1)
    cfg.scene.num_envs = 1
    cfg.seed = int(args.seed)
    env = gym.make(args.task, cfg=cfg)
    raw = env.unwrapped
    wrapped = SkrlVecEnvWrapper(env, ml_framework="torch")

    student = None
    if args.student is not None:
        student, _ = Circular12BCPolicy.load(
            args.student, map_location=raw.device
        )
        student = student.to(raw.device).eval()

    expert_cfg = config_from_ctbr_action_cfg(
        raw.cfg.actions.control_action,
        target_speed_mps=float(args.target_speed_mps),
        radius_m=12.0,
        height_m=2.07,
    )

    out_dir = args.output_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    obs, _ = wrapped.reset()
    command = raw.command_manager.get_term("target")
    previous_gate = int(command.next_gate_idx[0].item())

    rows: list[dict] = []
    episode = 0
    step = 0
    gates = 0
    radius_errors: list[float] = []
    height_errors: list[float] = []
    speed_errors: list[float] = []
    body_rates: list[float] = []
    attitude_errors: list[float] = []
    inverted_samples = 0
    inversion_entries = 0
    was_inverted = False
    action_abs: list[float] = []

    try:
        while episode < int(args.episodes):
            robot = raw.scene["robot"]
            p_w = robot.data.root_pos_w
            v_w = robot.data.root_lin_vel_w
            R_wb = math_utils.matrix_from_quat(robot.data.root_quat_w)
            expert = circular12_expert_action(
                p_w, v_w, R_wb, expert_cfg
            )

            if student is None:
                action = expert.action
            else:
                with torch.inference_mode():
                    action = student(obs)

            radius = torch.linalg.vector_norm(
                p_w[0, :2]
                - torch.tensor(
                    [0.0, 12.0], dtype=p_w.dtype, device=p_w.device
                )
            )
            speed = torch.linalg.vector_norm(v_w[0, :2])
            body_rate = torch.linalg.vector_norm(
                robot.data.root_ang_vel_b[0]
            )
            attitude_error = torch.linalg.vector_norm(
                expert.attitude_error_rotvec_b[0]
            )

            # body +Z expressed in world is the third column of R_wb.
            body_z_up = float(R_wb[0, 2, 2].item())
            inverted = body_z_up < 0.0
            if inverted and not was_inverted:
                inversion_entries += 1
            was_inverted = inverted
            inverted_samples += int(inverted)

            radius_errors.append(float(radius.item()) - 12.0)
            height_errors.append(float(p_w[0, 2].item()) - 2.07)
            speed_errors.append(float(speed.item()) - float(args.target_speed_mps))
            body_rates.append(float(body_rate.item()))
            attitude_errors.append(float(attitude_error.item()))
            action_abs.extend(
                action.detach().abs().cpu().numpy().reshape(-1).tolist()
            )

            obs, _, terminated, truncated, _ = wrapped.step(action)
            step += 1
            done = bool(terminated.reshape(-1)[0].item()) or bool(
                truncated.reshape(-1)[0].item()
            )

            current_gate = int(command.next_gate_idx[0].item())
            if not done and current_gate != previous_gate:
                gates += int(
                    (current_gate - previous_gate) % int(command.num_gates)
                )
            previous_gate = current_gate

            if not done:
                continue

            count = max(step, 1)
            row = {
                "episode": episode + 1,
                "controller": "expert" if student is None else "student",
                "steps": step,
                "duration_s": step * float(raw.step_dt),
                "gates_passed": gates,
                "termination": _termination_cause(raw),
                "radius_rmse_m": _rmse(radius_errors),
                "height_rmse_m": _rmse(height_errors),
                "speed_rmse_mps": _rmse(speed_errors),
                "body_rate_mean_radps": float(np.mean(body_rates)),
                "body_rate_p95_radps": float(np.percentile(body_rates, 95)),
                "attitude_error_rmse_deg": float(
                    np.degrees(_rmse(attitude_errors))
                ),
                "inverted_fraction": float(inverted_samples / count),
                "inversion_entries": int(inversion_entries),
                "action_abs_mean": float(np.mean(action_abs)),
                "action_saturation_fraction": float(
                    np.mean(np.asarray(action_abs) > 0.95)
                ),
            }
            rows.append(row)
            print(
                "[imitation-audit] "
                f"ep={episode + 1:02d}/{args.episodes} "
                f"gates={gates} cause={row['termination']} "
                f"r_rmse={row['radius_rmse_m']:.3f}m "
                f"h_rmse={row['height_rmse_m']:.3f}m "
                f"rate95={row['body_rate_p95_radps']:.3f}rad/s "
                f"inversions={inversion_entries}",
                flush=True,
            )

            episode += 1
            step = 0
            gates = 0
            radius_errors = []
            height_errors = []
            speed_errors = []
            body_rates = []
            attitude_errors = []
            inverted_samples = 0
            inversion_entries = 0
            was_inverted = False
            action_abs = []
            previous_gate = int(command.next_gate_idx[0].item())

        with (out_dir / "episodes.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

        summary = {
            "controller": "expert" if student is None else "student",
            "student_checkpoint": (
                None if args.student is None else str(args.student)
            ),
            "episodes": int(args.episodes),
            "target_speed_mps": float(args.target_speed_mps),
            "gates_mean": float(np.mean([r["gates_passed"] for r in rows])),
            "full_lap_rate": float(
                np.mean(
                    [
                        r["gates_passed"] >= int(command.num_gates)
                        for r in rows
                    ]
                )
            ),
            "radius_rmse_m_mean": float(
                np.mean([r["radius_rmse_m"] for r in rows])
            ),
            "height_rmse_m_mean": float(
                np.mean([r["height_rmse_m"] for r in rows])
            ),
            "speed_rmse_mps_mean": float(
                np.mean([r["speed_rmse_mps"] for r in rows])
            ),
            "body_rate_p95_radps_mean": float(
                np.mean([r["body_rate_p95_radps"] for r in rows])
            ),
            "attitude_error_rmse_deg_mean": float(
                np.mean([r["attitude_error_rmse_deg"] for r in rows])
            ),
            "total_inversion_entries": int(
                sum(r["inversion_entries"] for r in rows)
            ),
            "inverted_fraction_mean": float(
                np.mean([r["inverted_fraction"] for r in rows])
            ),
            "action_saturation_fraction_mean": float(
                np.mean(
                    [r["action_saturation_fraction"] for r in rows]
                )
            ),
        }
        (out_dir / "summary.json").write_text(
            json.dumps(summary, indent=2) + "\n"
        )
        print(json.dumps(summary, indent=2), flush=True)
    finally:
        wrapped.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
