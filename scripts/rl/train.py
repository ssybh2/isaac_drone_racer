# Copyright (c) 2025, Kousheek Chakraborty
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# This project uses the IsaacLab framework (https://github.com/isaac-sim/IsaacLab),
# which is licensed under the BSD-3-Clause License.

import argparse
import sys

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with skrl.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument(
    "--distributed", action="store_true", default=False, help="Run training with multiple GPUs or nodes."
)
parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint to resume training.")
parser.add_argument(
    "--imitation_bc_checkpoint",
    type=str,
    default=None,
    help=(
        "Initialize the Circular-12 256^3 ELU CTBR actor and observation "
        "scaler from a behavior-cloning checkpoint before PPO fine-tuning."
    ),
)
parser.add_argument(
    "--imitation_bc_action_std",
    type=float,
    default=0.05,
    help="Initial Gaussian exploration std after BC -> PPO transfer.",
)
parser.add_argument(
    "--fake_sensor_profile",
    type=str,
    choices=["clean", "mild", "nominal", "mixed", "stress"],
    default=None,
    help="Named Stage 1 Fake VIO/Fake IMU profile.",
)
parser.add_argument(
    "--post_load_learning_rate",
    type=float,
    default=None,
    help="Reset only the current optimizer/scheduler learning rate after loading a checkpoint.",
)
parser.add_argument(
    "--post_load_max_action_std",
    type=float,
    default=None,
    help="Clamp the loaded Gaussian policy exploration standard deviation before continuation.",
)
parser.add_argument(
    "--entropy_loss_scale",
    type=float,
    default=None,
    help="Override the PPO entropy loss scale for this run.",
)
parser.add_argument(
    "--adaptive_lr_max",
    type=float,
    default=None,
    help="Override the KL-adaptive scheduler maximum learning rate for this run.",
)
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")
parser.add_argument(
    "--recalibrate_legacy_actor",
    action="store_true",
    default=False,
    help=(
        "One-time migration for legacy hard-clipped learned-inertial checkpoints: "
        "distill the old environment-executed motor actions into a bounded tanh "
        "actor while preserving hidden layers and normalization by default."
    ),
)
parser.add_argument(
    "--legacy_actor_calibration_samples",
    type=int,
    default=4096,
    help="Synthetic standardized samples used for one-time legacy actor calibration.",
)
parser.add_argument(
    "--legacy_behavior_action_limit",
    type=float,
    default=0.95,
    help=(
        "Maximum absolute bounded action used when distilling the old hard-clipped "
        "policy behavior into the tanh actor."
    ),
)
parser.add_argument(
    "--legacy_preprocessor_count_cap",
    type=float,
    default=None,
    help=(
        "Optional RunningStandardScaler pseudo-count cap during migration. "
        "Omit by default to preserve the source policy observation coordinates."
    ),
)
parser.add_argument(
    "--legacy_transfer_only_path",
    type=str,
    default=None,
    help=(
        "If set, save the migrated checkpoint to this path and exit before PPO training. "
        "Use this to evaluate transfer quality before committing to a long run."
    ),
)
parser.add_argument(
    "--legacy_transfer_dataset",
    type=str,
    default=None,
    help=(
        "Optional .npz collected by collect_legacy_policy_transfer_dataset.py. "
        "When supplied, actor-head distillation uses real on-policy standardized "
        "observations instead of synthetic Gaussian probes."
    ),
)
parser.add_argument(
    "--ml_framework",
    type=str,
    default="torch",
    choices=["torch", "jax", "jax-numpy"],
    help="The ML framework used for training the skrl agent.",
)
parser.add_argument(
    "--algorithm",
    type=str,
    default="PPO",
    choices=["AMP", "PPO", "IPPO", "MAPPO"],
    help="The RL algorithm used for training the skrl agent.",
)

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()

# The deployment-faithful learned-inertial task always requires its onboard
# RTX camera even in headless training. Keep this task-specific requirement
# explicit so forgetting --enable_cameras cannot silently disable perception.
CAMERA_REQUIRED_TASK_PREFIXES = (
    "Isaac-Drone-Racer-Learned-Inertial-",
)
if args_cli.task is not None and args_cli.task.startswith(CAMERA_REQUIRED_TASK_PREFIXES):
    args_cli.enable_cameras = True

# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import math
import os
import random
from datetime import datetime

import gymnasium as gym
import numpy as np
import skrl
import torch
from packaging import version

# check for minimum supported skrl version
SKRL_VERSION = "1.4.2"
if version.parse(skrl.__version__) < version.parse(SKRL_VERSION):
    skrl.logger.error(
        f"Unsupported skrl version: {skrl.__version__}. "
        f"Install supported version using 'pip install skrl>={SKRL_VERSION}'"
    )
    exit()

if args_cli.ml_framework.startswith("torch"):
    from skrl.utils.runner.torch import Runner
elif args_cli.ml_framework.startswith("jax"):
    from skrl.utils.runner.jax import Runner

import isaaclab_tasks  # noqa: F401
from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_pickle, dump_yaml
from isaaclab_rl.skrl import SkrlVecEnvWrapper
from isaaclab_tasks.utils.hydra import hydra_task_config

import tasks  # noqa: F401
from utils.imitation_initialization import initialize_skrl_policy_from_bc
from utils.training_overrides import (
    apply_post_load_training_overrides,
    cap_running_scaler_count,
    distill_legacy_clamped_actor_output,
    reset_optimizer_parameter_state,
)

LEARNED_INERTIAL_RL_TASK = "Isaac-Drone-Racer-Learned-Inertial-RL-v0"
SWIFT_CTBR_POLICY0_TASKS = {
    "Isaac-Drone-Racer-Swift-CTBR-Train-v0",
    "Isaac-Drone-Racer-Swift-CTBR-Train-PassState-v0",
}
SWIFT_CTBR_GT_RACING_TASKS = {
    "Isaac-Drone-Racer-Swift-CTBR-GT-Racing-v0",
    "Isaac-Drone-Racer-Swift-CTBR-GT-Circular12-v0",
    "Isaac-Drone-Racer-Swift-CTBR-GT-Circular12-StableHeading-v0",
    "Isaac-Drone-Racer-Swift-CTBR-GT-Circular12-StableMultiGate-v0",
    "Isaac-Drone-Racer-Swift-CTBR-GT-Circular12-ImitationFineTune-v0",
    "Isaac-Drone-Racer-Swift-CTBR-GT-PerceptionAware-v0",
    "Isaac-Drone-Racer-Swift-CTBR-GT-PerceptionAwareV2-v0",
    "Isaac-Drone-Racer-Swift-CTBR-GT-PerceptionAwareV3-v0",
}
LEARNED_INERTIAL_SWIFT_CTBR_TASK = (
    "Isaac-Drone-Racer-Learned-Inertial-Swift-CTBR-v0"
)


def _audit_learned_inertial_bounded_cfg(env, agent_cfg: dict) -> None:
    """Fail closed if the PPO/action contract regresses to the old unbounded path."""
    if args_cli.task != LEARNED_INERTIAL_RL_TASK:
        return

    action_space = env.unwrapped.single_action_space
    low = float(action_space.low.min())
    high = float(action_space.high.max())
    if abs(low + 1.0) > 1.0e-6 or abs(high - 1.0) > 1.0e-6:
        raise RuntimeError(
            "learned-inertial PPO requires a finite [-1, 1] action space; "
            f"got [{low}, {high}]"
        )

    policy_cfg = agent_cfg["models"]["policy"]
    expected_output = "tanh(ACTIONS)"
    if str(policy_cfg.get("output")) != expected_output:
        raise RuntimeError(
            "learned-inertial PPO mean must use the bounded contract "
            f"{expected_output!r}"
        )
    if not bool(policy_cfg.get("clip_actions", False)):
        raise RuntimeError("learned-inertial PPO must clip sampled actions before storage/execution")

    max_std = math.exp(float(policy_cfg["max_log_std"]))
    if max_std > 0.0500001:
        raise RuntimeError(f"configured learned-inertial action std cap is too large: {max_std}")

    print("[INFO] Learned-inertial bounded PPO contract:")
    print(f"  action_space               : [{low:.1f}, {high:.1f}]")
    print(f"  policy_mean                : {expected_output}")
    print(f"  sampled_action_clipping    : {policy_cfg['clip_actions']}")
    print(f"  configured_std_cap         : {max_std:.6f}")
    print(f"  rollouts                   : {agent_cfg['agent']['rollouts']}")
    print(f"  mini_batches               : {agent_cfg['agent']['mini_batches']}")
    print(f"  configured_learning_rate   : {agent_cfg['agent']['learning_rate']}")
    print(f"  entropy_loss_scale         : {agent_cfg['agent']['entropy_loss_scale']}")


def _audit_swift_ctbr_policy0_cfg(env, agent_cfg: dict) -> None:
    """Fail closed if the fresh Swift policy-0 contract drifts."""
    if args_cli.task not in SWIFT_CTBR_POLICY0_TASKS:
        return

    action_space = env.unwrapped.single_action_space
    low = float(action_space.low.min())
    high = float(action_space.high.max())
    if abs(low + 1.0) > 1.0e-6 or abs(high - 1.0) > 1.0e-6:
        raise RuntimeError(
            "Swift CTBR policy-0 requires finite [-1, 1]^4 actions; "
            f"got [{low}, {high}]"
        )
    if int(action_space.shape[0]) != 4:
        raise RuntimeError(
            f"Swift CTBR policy-0 requires 4 actions, got {action_space.shape}"
        )

    models = agent_cfg["models"]
    policy_cfg = models["policy"]
    value_cfg = models["value"]

    if not bool(models.get("separate", False)):
        raise RuntimeError("Swift policy-0 requires separate actor and critic")
    if str(policy_cfg.get("output")) != "tanh(ACTIONS)":
        raise RuntimeError("Swift policy-0 actor mean must be tanh(ACTIONS)")
    if not bool(policy_cfg.get("clip_actions", False)):
        raise RuntimeError("Swift policy-0 must clip sampled CTBR actions")

    expected_layers = [128, 128]
    for name, cfg in (("policy", policy_cfg), ("value", value_cfg)):
        network = cfg.get("network", [])
        if len(network) != 1 or list(network[0].get("layers", [])) != expected_layers:
            raise RuntimeError(
                f"Swift {name} network must use 2x128 hidden layers"
            )
        if str(network[0].get("activations")) != "leaky_relu":
            raise RuntimeError(
                f"Swift {name} network must use leaky_relu activations"
            )

    agent = agent_cfg["agent"]
    if abs(float(agent["learning_rate"]) - 3.0e-4) > 1.0e-12:
        raise RuntimeError("Swift policy-0 learning rate must start at 3e-4")
    if abs(float(agent["discount_factor"]) - 0.99) > 1.0e-12:
        raise RuntimeError("Swift policy-0 gamma must be 0.99")
    if abs(float(agent["ratio_clip"]) - 0.2) > 1.0e-12:
        raise RuntimeError("Swift policy-0 PPO ratio clip must be 0.2")

    print("[INFO] Swift CTBR policy-0 contract:")
    print(f"  num_envs                   : {env.unwrapped.num_envs}")
    print(f"  action_space               : [{low:.1f}, {high:.1f}]^4")
    print("  actor / critic             : separate 2x128 leaky_relu")
    print(f"  policy_mean                : {policy_cfg['output']}")
    print(f"  sampled_action_clipping    : {policy_cfg['clip_actions']}")
    print(f"  rollouts                   : {agent['rollouts']}")
    print(f"  learning_rate              : {agent['learning_rate']}")
    print(f"  gamma / PPO clip           : {agent['discount_factor']} / {agent['ratio_clip']}")


def _audit_swift_ctbr_gt_racing_cfg(env, agent_cfg: dict) -> None:
    """Fail closed on the upstream-inspired GT racing training contract."""
    if args_cli.task not in SWIFT_CTBR_GT_RACING_TASKS:
        return

    action_space = env.unwrapped.single_action_space
    low = float(action_space.low.min())
    high = float(action_space.high.max())
    if abs(low + 1.0) > 1.0e-6 or abs(high - 1.0) > 1.0e-6:
        raise RuntimeError(
            "GT racing requires bounded [-1, 1]^4 CTBR actions; "
            f"got [{low}, {high}]"
        )
    if int(action_space.shape[0]) != 4:
        raise RuntimeError(
            f"GT racing requires four CTBR actions, got {action_space.shape}"
        )

    if int(env.unwrapped.num_envs) != 4096:
        raise RuntimeError(
            "GT racing baseline is intentionally fixed at 4096 environments; "
            f"got {env.unwrapped.num_envs}"
        )

    models = agent_cfg["models"]
    policy_cfg = models["policy"]
    value_cfg = models["value"]

    if bool(models.get("separate", True)):
        raise RuntimeError(
            "GT racing baseline must use the upstream-inspired shared model "
            "(models.separate=False)"
        )
    if str(policy_cfg.get("output")) != "tanh(ACTIONS)":
        raise RuntimeError("GT racing CTBR actor mean must be tanh(ACTIONS)")
    if not bool(policy_cfg.get("clip_actions", False)):
        raise RuntimeError("GT racing must clip sampled CTBR actions")

    expected_layers = [256, 256, 256]
    for name, cfg in (("policy", policy_cfg), ("value", value_cfg)):
        network = cfg.get("network", [])
        if len(network) != 1 or list(network[0].get("layers", [])) != expected_layers:
            raise RuntimeError(
                f"GT racing {name} must use 256x256x256 hidden layers"
            )
        if str(network[0].get("activations")) != "elu":
            raise RuntimeError(f"GT racing {name} must use ELU activations")

    agent = agent_cfg["agent"]
    expected = {
        "rollouts": 24,
        "learning_epochs": 5,
        "mini_batches": 4,
    }
    for key, value in expected.items():
        if int(agent[key]) != value:
            raise RuntimeError(
                f"GT racing requires {key}={value}, got {agent[key]}"
            )

    if abs(float(agent["learning_rate"]) - 1.0e-4) > 1.0e-12:
        raise RuntimeError("GT racing learning rate must be 1e-4")
    if abs(float(agent["discount_factor"]) - 0.99) > 1.0e-12:
        raise RuntimeError("GT racing gamma must be 0.99")
    if abs(float(agent["ratio_clip"]) - 0.2) > 1.0e-12:
        raise RuntimeError("GT racing PPO ratio clip must be 0.2")
    if abs(float(agent["rewards_shaper_scale"]) - 0.6) > 1.0e-12:
        raise RuntimeError("GT racing reward shaper scale must be 0.6")

    print("[INFO] Swift CTBR GT racing contract:")
    print(f"  num_envs                   : {env.unwrapped.num_envs}")
    print(f"  action_space               : [{low:.1f}, {high:.1f}]^4")
    print("  actor / critic             : shared 256x256x256 ELU")
    print(f"  policy_mean                : {policy_cfg['output']}")
    print(f"  rollouts                   : {agent['rollouts']}")
    print(f"  learning_epochs            : {agent['learning_epochs']}")
    print(f"  mini_batches               : {agent['mini_batches']}")
    print(f"  learning_rate              : {agent['learning_rate']}")
    print(f"  reward_shaper              : {agent['rewards_shaper_scale']}")


def _audit_loaded_policy_std(agent, max_std: float) -> None:
    log_std = getattr(agent.policy, "log_std_parameter", None)
    if log_std is None:
        raise RuntimeError("learned-inertial Gaussian policy has no log_std_parameter")
    std = log_std.detach().exp()
    std_max = float(std.max().item())
    if std_max > float(max_std) + 1.0e-6:
        raise RuntimeError(
            f"loaded policy action std {std_max:.6f} exceeds cap {max_std:.6f}"
        )
    print(f"[INFO] Loaded policy action std max: {std_max:.6f}")


def _sample_legacy_actor_features(
    agent,
    *,
    samples: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Probe the loaded legacy actor in its standardized observation coordinates."""
    if samples < 32:
        raise ValueError("--legacy_actor_calibration_samples must be at least 32")

    policy = agent.policy
    trunk = getattr(policy, "net_container", None)
    layer = getattr(policy, "policy_layer", None)
    if trunk is None or not isinstance(layer, torch.nn.Linear):
        raise RuntimeError(
            "legacy actor recalibration expects skrl shared net_container + policy_layer"
        )

    first_linear = next(
        (module for module in trunk.modules() if isinstance(module, torch.nn.Linear)),
        None,
    )
    if first_linear is None:
        raise RuntimeError("could not infer actor standardized input dimension")

    generator = torch.Generator(device=layer.weight.device)
    generator.manual_seed(int(seed))
    standardized = torch.randn(
        int(samples),
        int(first_linear.in_features),
        generator=generator,
        device=layer.weight.device,
        dtype=layer.weight.dtype,
    ).clamp_(-3.0, 3.0)
    zero = torch.zeros(
        1,
        int(first_linear.in_features),
        device=layer.weight.device,
        dtype=layer.weight.dtype,
    )

    with torch.inference_mode():
        hidden = trunk(standardized)
        raw = layer(hidden)
        zero_hidden = trunk(zero)
        zero_raw = layer(zero_hidden)
    return (
        hidden.detach(),
        raw.detach(),
        zero_hidden.detach(),
        zero_raw.detach(),
    )


def _migrate_legacy_learned_inertial_checkpoint(agent) -> dict:
    """Distill old environment-clipped behavior into a trainable tanh actor."""
    dataset_path = None
    if args_cli.legacy_transfer_dataset is not None:
        dataset_path = os.path.abspath(
            os.path.expanduser(args_cli.legacy_transfer_dataset)
        )
        if not os.path.isfile(dataset_path):
            raise FileNotFoundError(
                f"legacy transfer dataset not found: {dataset_path}"
            )

    if dataset_path is None:
        hidden, raw_means, zero_hidden, zero_raw = _sample_legacy_actor_features(
            agent,
            samples=int(args_cli.legacy_actor_calibration_samples),
            seed=int(args_cli.seed if args_cli.seed is not None else 1) + 1701,
        )
        dataset_mode = "synthetic_standardized_gaussian"
        dataset_samples = int(raw_means.shape[0])
    else:
        data = np.load(dataset_path)
        required = (
            "standardized_observation",
            "legacy_raw_mean",
            "legacy_executed_action",
        )
        missing = [key for key in required if key not in data]
        if missing:
            raise ValueError(
                "legacy transfer dataset missing arrays: " + ", ".join(missing)
            )

        std_obs = torch.as_tensor(
            data["standardized_observation"],
            dtype=torch.float32,
            device=agent.policy.device,
        )
        raw_means = torch.as_tensor(
            data["legacy_raw_mean"],
            dtype=torch.float32,
            device=agent.policy.device,
        )
        expected_executed = torch.as_tensor(
            data["legacy_executed_action"],
            dtype=torch.float32,
            device=agent.policy.device,
        )
        if std_obs.ndim != 2 or raw_means.ndim != 2:
            raise ValueError(
                "legacy transfer dataset arrays must be rank-2"
            )
        if std_obs.shape[0] != raw_means.shape[0]:
            raise ValueError(
                "legacy standardized observations/raw means have different sample counts"
            )
        if expected_executed.shape != raw_means.shape:
            raise ValueError(
                "legacy executed action shape does not match raw mean shape"
            )

        reconstructed = raw_means.clamp(-1.0, 1.0)
        reconstruction_mae = float(
            (reconstructed - expected_executed).abs().mean().item()
        )
        if reconstruction_mae > 1.0e-6:
            raise RuntimeError(
                "legacy transfer dataset executed-action semantics are inconsistent: "
                f"MAE={reconstruction_mae:.8f}"
            )

        trunk = getattr(agent.policy, "net_container", None)
        layer = getattr(agent.policy, "policy_layer", None)
        if trunk is None or not isinstance(layer, torch.nn.Linear):
            raise RuntimeError(
                "behavior transfer expects skrl net_container + policy_layer"
            )
        with torch.inference_mode():
            hidden = trunk(std_obs)
            first_linear = next(
                (
                    module
                    for module in trunk.modules()
                    if isinstance(module, torch.nn.Linear)
                ),
                None,
            )
            if first_linear is None:
                raise RuntimeError(
                    "could not infer actor standardized input dimension"
                )
            zero_std = torch.zeros(
                1,
                int(first_linear.in_features),
                dtype=std_obs.dtype,
                device=std_obs.device,
            )
            zero_hidden = trunk(zero_std)
            zero_raw = layer(zero_hidden)

        dataset_mode = "real_on_policy_legacy_trajectory"
        dataset_samples = int(raw_means.shape[0])

    metadata = distill_legacy_clamped_actor_output(
        agent.policy,
        hidden,
        raw_means,
        anchor_hidden=zero_hidden,
        anchor_raw_mean=zero_raw,
        action_limit=float(args_cli.legacy_behavior_action_limit),
        ridge=1.0e-4,
        anchor_repeats=64,
    )
    metadata["legacy_transfer_dataset_mode"] = dataset_mode
    metadata["legacy_transfer_dataset_samples"] = dataset_samples
    if dataset_path is not None:
        metadata["legacy_transfer_dataset_path"] = dataset_path

    # Preserve source-policy normalization by default. The source checkpoint
    # already ran on the Easy curriculum, so changing scaler statistics during
    # transfer would move the actor input coordinates at the same time as the
    # output parameterization. A cap remains available as an explicit ablation.
    if args_cli.legacy_preprocessor_count_cap is not None:
        if args_cli.legacy_preprocessor_count_cap <= 0.0:
            raise ValueError("--legacy_preprocessor_count_cap must be positive")

        seen = set()
        for label, attr in (
            ("state", "_state_preprocessor"),
            ("observation", "_observation_preprocessor"),
            ("value", "_value_preprocessor"),
        ):
            preprocessor = getattr(agent, attr, None)
            if preprocessor is None or id(preprocessor) in seen:
                continue
            seen.add(id(preprocessor))
            scaler_metadata = cap_running_scaler_count(
                preprocessor,
                float(args_cli.legacy_preprocessor_count_cap),
            )
            for key, value in scaler_metadata.items():
                metadata[f"{label}_{key}"] = value

    actor_layer = getattr(agent.policy, "policy_layer", None)
    if not isinstance(actor_layer, torch.nn.Linear):
        raise RuntimeError("expected skrl policy_layer after behavior distillation")

    reset_parameters = [actor_layer.weight, actor_layer.bias]
    log_std = getattr(agent.policy, "log_std_parameter", None)
    if isinstance(log_std, torch.nn.Parameter):
        reset_parameters.append(log_std)

    metadata.update(
        reset_optimizer_parameter_state(agent.optimizer, reset_parameters)
    )

    print("[INFO] One-time behavior-preserving legacy actor transfer:")
    print(f"  dataset mode                : {dataset_mode}")
    print(f"  dataset samples             : {dataset_samples}")
    print(
        "  old executed saturation    : "
        f"{metadata['legacy_executed_saturation_fraction']:.4f}"
    )
    print(
        "  action behavior MAE/RMSE   : "
        f"{metadata['distilled_behavior_mae']:.4f} / "
        f"{metadata['distilled_behavior_rmse']:.4f}"
    )
    print(
        "  action sign agreement      : "
        f"{metadata['distilled_sign_agreement']:.4f}"
    )
    print(
        "  tanh dead derivative frac  : "
        f"{metadata['distilled_tanh_dead_fraction']:.4f}"
    )
    print(
        "  mean tanh derivative       : "
        f"{metadata['distilled_tanh_derivative_mean']:.4f}"
    )
    print(
        "  legacy zero-input executed : "
        f"{metadata.get('legacy_zero_input_executed_action')}"
    )
    print(
        "  distilled zero-input action: "
        f"{metadata.get('distilled_zero_input_action')}"
    )
    print(
        "  zero-input behavior MAE    : "
        f"{metadata.get('distilled_zero_input_mae', float('nan')):.4f}"
    )
    print(
        "  optimizer states cleared   : "
        f"{metadata['optimizer_parameter_states_cleared']}"
    )

    max_mae = 0.20 if dataset_mode == "synthetic_standardized_gaussian" else 0.12
    if metadata["distilled_behavior_mae"] > max_mae:
        raise RuntimeError(
            "behavior-preserving actor transfer is too inaccurate: "
            f"MAE={metadata['distilled_behavior_mae']:.4f} "
            f"(limit={max_mae:.4f}, mode={dataset_mode})"
        )
    if metadata["distilled_sign_agreement"] < 0.97:
        raise RuntimeError(
            "behavior-preserving actor transfer changed too many action signs: "
            f"agreement={metadata['distilled_sign_agreement']:.4f}"
        )

    return metadata

# config shortcuts
algorithm = args_cli.algorithm.lower()
agent_cfg_entry_point = "skrl_cfg_entry_point" if algorithm in ["ppo"] else f"skrl_{algorithm}_cfg_entry_point"


@hydra_task_config(args_cli.task, agent_cfg_entry_point)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: dict):
    """Train with skrl agent."""
    # override configurations with non-hydra CLI arguments
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    if (
        args_cli.task == "Isaac-Drone-Racer-Learned-Inertial-RL-v0"
        and int(env_cfg.scene.num_envs) != 1
    ):
        raise ValueError(
            "Isaac-Drone-Racer-Learned-Inertial-RL-v0 currently requires "
            "--num_envs 1 because the validated detector/estimator runtime is "
            "single-stream. Vectorize it deliberately before scaling PPO."
        )
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    if args_cli.fake_sensor_profile is not None:
        if not hasattr(env_cfg, "fake_sensors"):
            raise ValueError("--fake_sensor_profile requires a Stage 1 task")
        from estimation.fake_sensor_cfg import FakeSensorPipelineCfg

        env_cfg.fake_sensors = FakeSensorPipelineCfg.from_profile(args_cli.fake_sensor_profile)
        agent_cfg["agent"]["experiment"]["experiment_name"] = (
            f"stage1_{args_cli.fake_sensor_profile}"
        )
    if args_cli.entropy_loss_scale is not None:
        if args_cli.entropy_loss_scale < 0.0:
            raise ValueError("--entropy_loss_scale must be non-negative")
        agent_cfg["agent"]["entropy_loss_scale"] = args_cli.entropy_loss_scale
    if args_cli.adaptive_lr_max is not None:
        if args_cli.adaptive_lr_max <= 0.0:
            raise ValueError("--adaptive_lr_max must be positive")
        agent_cfg["agent"]["learning_rate_scheduler_kwargs"]["max_lr"] = (
            args_cli.adaptive_lr_max
        )
    if (
        args_cli.post_load_learning_rate is not None
        and args_cli.adaptive_lr_max is not None
        and args_cli.post_load_learning_rate > args_cli.adaptive_lr_max
    ):
        raise ValueError("--post_load_learning_rate cannot exceed --adaptive_lr_max")

    # multi-gpu training config
    if args_cli.distributed:
        env_cfg.sim.device = f"cuda:{app_launcher.local_rank}"
    # max iterations for training
    if args_cli.max_iterations:
        agent_cfg["trainer"]["timesteps"] = args_cli.max_iterations * agent_cfg["agent"]["rollouts"]
    agent_cfg["trainer"]["close_environment_at_exit"] = False
    # configure the ML framework into the global skrl variable
    if args_cli.ml_framework.startswith("jax"):
        skrl.config.jax.backend = "jax" if args_cli.ml_framework == "jax" else "numpy"

    # randomly sample a seed if seed = -1
    if args_cli.seed == -1:
        args_cli.seed = random.randint(0, 10000)

    # set the agent and environment seed from command line
    # note: certain randomization occur in the environment initialization so we set the seed here
    agent_cfg["seed"] = args_cli.seed if args_cli.seed is not None else agent_cfg["seed"]
    env_cfg.seed = agent_cfg["seed"]

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "skrl", agent_cfg["agent"]["experiment"]["directory"])
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    # specify directory for logging runs: {time-stamp}_{run_name}
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + f"_{algorithm}_{args_cli.ml_framework}"
    print(f"Exact experiment name requested from command line {log_dir}")
    if agent_cfg["agent"]["experiment"]["experiment_name"]:
        log_dir += f'_{agent_cfg["agent"]["experiment"]["experiment_name"]}'
    # set directory into agent config
    agent_cfg["agent"]["experiment"]["directory"] = log_root_path
    agent_cfg["agent"]["experiment"]["experiment_name"] = log_dir
    # update log_dir
    log_dir = os.path.join(log_root_path, log_dir)

    # dump the configuration into log-directory
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
    dump_pickle(os.path.join(log_dir, "params", "env.pkl"), env_cfg)
    dump_pickle(os.path.join(log_dir, "params", "agent.pkl"), agent_cfg)

    # get checkpoint path (to resume training)
    resume_path = retrieve_file_path(args_cli.checkpoint) if args_cli.checkpoint else None
    if args_cli.imitation_bc_checkpoint is not None and resume_path is not None:
        raise ValueError(
            "--imitation_bc_checkpoint and --checkpoint are mutually exclusive"
        )
    if (
        args_cli.imitation_bc_checkpoint is not None
        and args_cli.task
        != "Isaac-Drone-Racer-Swift-CTBR-GT-Circular12-ImitationFineTune-v0"
    ):
        raise ValueError(
            "--imitation_bc_checkpoint is only valid for the Circular12 "
            "imitation fine-tune task"
        )
    if (
        args_cli.post_load_learning_rate is not None
        or args_cli.post_load_max_action_std is not None
        or args_cli.recalibrate_legacy_actor
        or args_cli.legacy_transfer_only_path is not None
        or args_cli.legacy_transfer_dataset is not None
    ) and resume_path is None:
        raise ValueError(
            "post-load overrides / legacy actor recalibration require --checkpoint"
        )
    if args_cli.legacy_transfer_dataset is not None and not args_cli.recalibrate_legacy_actor:
        raise ValueError(
            "--legacy_transfer_dataset requires --recalibrate_legacy_actor"
        )
    if args_cli.legacy_transfer_only_path is not None and not args_cli.recalibrate_legacy_actor:
        raise ValueError(
            "--legacy_transfer_only_path requires --recalibrate_legacy_actor"
        )
    if args_cli.recalibrate_legacy_actor and args_cli.task != LEARNED_INERTIAL_RL_TASK:
        raise ValueError(
            "--recalibrate_legacy_actor is only valid for the learned-inertial RL task"
        )

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    _audit_learned_inertial_bounded_cfg(env, agent_cfg)
    _audit_swift_ctbr_policy0_cfg(env, agent_cfg)
    _audit_swift_ctbr_gt_racing_cfg(env, agent_cfg)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv) and algorithm in ["ppo"]:
        env = multi_agent_to_single_agent(env)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for skrl
    env = SkrlVecEnvWrapper(env, ml_framework=args_cli.ml_framework)  # same as: `wrap_env(env, wrapper="auto")`

    # configure and instantiate the skrl runner
    # https://skrl.readthedocs.io/en/latest/api/utils/runner.html
    runner = Runner(env, agent_cfg)

    if args_cli.imitation_bc_checkpoint is not None:
        bc_metadata = initialize_skrl_policy_from_bc(
            runner.agent,
            args_cli.imitation_bc_checkpoint,
            action_std=float(args_cli.imitation_bc_action_std),
        )
        dump_yaml(
            os.path.join(log_dir, "params", "bc_initialization.yaml"),
            bc_metadata,
        )
        print("[INFO] Initialized PPO actor from Circular-12 BC checkpoint")
        print_dict(bc_metadata, nesting=4)

    # load checkpoint (if specified)
    if resume_path:
        print(f"[INFO] Loading model checkpoint from: {resume_path}")
        # runner.agent.load is intentionally strict with the model state. The
        # tanh mean changes only the forward expression and adds no trainable
        # parameters, so legacy checkpoints remain state-dict compatible.
        runner.agent.load(resume_path)

        migration_metadata = {}
        if args_cli.recalibrate_legacy_actor:
            migration_metadata = _migrate_legacy_learned_inertial_checkpoint(
                runner.agent
            )

        continuation_lr = args_cli.post_load_learning_rate
        continuation_std = args_cli.post_load_max_action_std
        if args_cli.task == LEARNED_INERTIAL_RL_TASK:
            if continuation_lr is None:
                continuation_lr = float(agent_cfg["agent"]["learning_rate"])
            if continuation_std is None:
                continuation_std = math.exp(
                    float(agent_cfg["models"]["policy"]["max_log_std"])
                )

        continuation_metadata = apply_post_load_training_overrides(
            runner.agent,
            learning_rate=continuation_lr,
            max_action_std=continuation_std,
        )
        if args_cli.task == LEARNED_INERTIAL_RL_TASK:
            _audit_loaded_policy_std(runner.agent, max_std=continuation_std)
        if continuation_metadata or migration_metadata:
            continuation_metadata = {
                "source_checkpoint": resume_path,
                "legacy_actor_recalibrated": bool(args_cli.recalibrate_legacy_actor),
                **migration_metadata,
                **continuation_metadata,
            }
            dump_yaml(
                os.path.join(log_dir, "params", "continuation_overrides.yaml"),
                continuation_metadata,
            )
            print_dict(continuation_metadata, nesting=4)

    if args_cli.legacy_transfer_only_path is not None:
        output_path = os.path.abspath(
            os.path.expanduser(args_cli.legacy_transfer_only_path)
        )
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        runner.agent.save(output_path)
        print(f"[INFO] Saved transfer-only checkpoint: {output_path}")
        env.close()
        return

    # run training
    runner.run()

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
