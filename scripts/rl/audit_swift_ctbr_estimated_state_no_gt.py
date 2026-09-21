"""Audit that the deployed Swift CTBR racing policy receives no simulator GT.

The successful GT policy is allowed to keep simulator truth only in the
reward/evaluation side channel. Actor observations and mission progression must
come from the learned-inertial estimator plus the known gate map.
"""

from __future__ import annotations

import argparse
import inspect
import json
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--task",
    default="Isaac-Drone-Racer-Learned-Inertial-Swift-CTBR-GTPolicy-v0",
)
parser.add_argument("--output-json", type=Path, default=None)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

import tasks  # noqa: F401,E402
from tasks.drone_racer import mdp  # noqa: E402


FORBIDDEN_ACTOR_TOKENS = (
    "root_pos_w",
    "root_lin_vel_w",
    "root_quat_w",
    "root_state_w",
)


def _assert_no_gt_in_function(func, label: str) -> None:
    source = inspect.getsource(func)
    hits = [token for token in FORBIDDEN_ACTOR_TOKENS if token in source]
    if hits:
        raise RuntimeError(
            f"{label} reads simulator GT actor state via {hits}"
        )


def main() -> None:
    cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=1)
    cfg.scene.num_envs = 1

    policy_cfg = cfg.observations.policy
    platform_func = policy_cfg.platform_state.func
    gate_func = policy_cfg.next_gate_corners.func
    prev_action_func = policy_cfg.previous_action.func
    command_cfg = cfg.commands.target
    command_type = command_cfg.class_type

    if platform_func is not mdp.learned_inertial_swift_state:
        raise RuntimeError(
            "deployment platform_state is not estimator-backed "
            "learned_inertial_swift_state"
        )
    if gate_func is not mdp.learned_next_gate_corners_relative_w:
        raise RuntimeError(
            "deployment next-gate geometry is not estimator-backed"
        )
    if prev_action_func is not mdp.last_action:
        raise RuntimeError("deployment previous-action observation changed")
    if command_type is not mdp.EstimatedStateGateTargetingCommand:
        raise RuntimeError(
            "mission progression is not EstimatedStateGateTargetingCommand"
        )

    _assert_no_gt_in_function(platform_func, "platform_state")
    _assert_no_gt_in_function(gate_func, "next_gate_corners")
    _assert_no_gt_in_function(
        command_type._estimated_position_w,
        "mission estimated position",
    )

    command_update_source = inspect.getsource(command_type._update_command)
    required_mission_line = (
        "self.next_gate_idx[self._mission_gate_passed] += 1"
    )
    forbidden_gt_line = "self.next_gate_idx[self._gt_gate_passed] += 1"
    if required_mission_line not in command_update_source:
        raise RuntimeError(
            "mission next_gate_idx is not advanced from estimator-derived pass"
        )
    if forbidden_gt_line in command_update_source:
        raise RuntimeError(
            "mission next_gate_idx is advanced from simulator-truth gate pass"
        )

    oracle_flags = {
        "gate_debug_gt_diagnostics": bool(
            getattr(cfg, "gate_debug_gt_diagnostics", False)
        ),
        "learned_debug_oracle_residual_fusion": bool(
            getattr(cfg, "learned_debug_oracle_residual_fusion", False)
        ),
        "learned_debug_oracle_uzh_displacement_fusion": bool(
            getattr(cfg, "learned_debug_oracle_uzh_displacement_fusion", False)
        ),
        "learned_debug_oracle_body_end_displacement_fusion": bool(
            getattr(
                cfg,
                "learned_debug_oracle_body_end_displacement_fusion",
                False,
            )
        ),
        "learned_debug_oracle_second_difference_fusion": bool(
            getattr(
                cfg,
                "learned_debug_oracle_second_difference_fusion",
                False,
            )
        ),
        "learned_debug_oracle_delta_velocity_fusion": bool(
            getattr(cfg, "learned_debug_oracle_delta_velocity_fusion", False)
        ),
        "learned_debug_oracle_body_end_delta_velocity_fusion": bool(
            getattr(
                cfg,
                "learned_debug_oracle_body_end_delta_velocity_fusion",
                False,
            )
        ),
        "learned_debug_truth_orientation_for_features": bool(
            getattr(
                cfg,
                "learned_debug_truth_orientation_for_features",
                False,
            )
        ),
    }
    enabled_oracles = [name for name, enabled in oracle_flags.items() if enabled]
    if enabled_oracles:
        raise RuntimeError(
            f"deployment task has GT/oracle estimator inputs enabled: {enabled_oracles}"
        )

    env = gym.make(args_cli.task, cfg=cfg)
    raw_env = env.unwrapped
    try:
        observation, _ = env.reset()
        policy_obs = (
            observation["policy"]
            if isinstance(observation, dict)
            else observation
        )
        if tuple(policy_obs.shape) != (1, 31):
            raise RuntimeError(
                f"deployment policy observation must be (1, 31), got {tuple(policy_obs.shape)}"
            )
        if int(raw_env.action_manager.total_action_dim) != 4:
            raise RuntimeError("deployment policy action dimension must be 4")

        command = raw_env.command_manager.get_term("target")
        if type(command) is not mdp.EstimatedStateGateTargetingCommand:
            raise RuntimeError(
                "runtime mission command is not estimator-driven"
            )

        summary = {
            "task": args_cli.task,
            "policy_observation_shape": list(policy_obs.shape),
            "action_dim": int(raw_env.action_manager.total_action_dim),
            "platform_state_source": platform_func.__name__,
            "next_gate_source": gate_func.__name__,
            "mission_command_type": type(command).__name__,
            "oracle_flags": oracle_flags,
            "actor_reads_runtime_simulator_gt": False,
            "mission_progression_reads_runtime_simulator_gt": False,
            "simulator_gt_allowed_for": [
                "reward",
                "evaluation diagnostics",
                "known fixed reset anchor",
            ],
        }

        print("=" * 96)
        print("SWIFT CTBR ESTIMATED-STATE NO-GT AUDIT")
        print("=" * 96)
        print(json.dumps(summary, indent=2))

        if args_cli.output_json is not None:
            output = args_cli.output_json.expanduser().resolve()
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps(summary, indent=2) + "\n")
            print(f"[no-gt-audit] wrote: {output}")
    finally:
        env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
