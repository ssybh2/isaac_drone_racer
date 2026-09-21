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
        "rescale the actor output layer into the tanh trainable region, cap stale "
        "RunningStandardScaler pseudo-counts, and clear stale optimizer moments."
    ),
)
parser.add_argument(
    "--legacy_actor_calibration_samples",
    type=int,
    default=4096,
    help="Synthetic standardized samples used for one-time legacy actor calibration.",
)
parser.add_argument(
    "--legacy_actor_target_pretanh_abs",
    type=float,
    default=1.25,
    help="Target representative absolute pre-tanh actor output after migration.",
)
parser.add_argument(
    "--legacy_preprocessor_count_cap",
    type=float,
    default=4096.0,
    help="Maximum retained effective sample count for loaded running scalers during migration.",
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
if args_cli.task == "Isaac-Drone-Racer-Learned-Inertial-RL-v0":
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
from utils.training_overrides import (
    apply_post_load_training_overrides,
    cap_running_scaler_count,
    recalibrate_legacy_actor_output,
    reset_optimizer_state,
)

LEARNED_INERTIAL_RL_TASK = "Isaac-Drone-Racer-Learned-Inertial-RL-v0"


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


def _sample_legacy_actor_raw_means(
    agent,
    *,
    samples: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Probe the loaded actor on standardized synthetic inputs before migration."""
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
        raw = layer(trunk(standardized))
        zero_raw = layer(trunk(zero))
    return raw.detach(), zero_raw.detach()


def _migrate_legacy_learned_inertial_checkpoint(agent) -> dict:
    """Perform the one-time old-hard-clamp to bounded-tanh transfer."""
    raw_means, zero_raw = _sample_legacy_actor_raw_means(
        agent,
        samples=int(args_cli.legacy_actor_calibration_samples),
        seed=int(args_cli.seed if args_cli.seed is not None else 1) + 1701,
    )

    metadata = recalibrate_legacy_actor_output(
        agent.policy,
        raw_means,
        target_pretanh_abs=float(args_cli.legacy_actor_target_pretanh_abs),
        reference_quantile=0.75,
    )

    row_scale = torch.tensor(
        metadata["actor_output_row_scale"],
        device=zero_raw.device,
        dtype=zero_raw.dtype,
    ).view(1, -1)
    zero_raw_after = zero_raw * row_scale
    metadata["actor_zero_input_raw_before"] = zero_raw.cpu().reshape(-1).tolist()
    metadata["actor_zero_input_raw_after"] = zero_raw_after.cpu().reshape(-1).tolist()
    metadata["actor_zero_input_mean_after"] = (
        torch.tanh(zero_raw_after).cpu().reshape(-1).tolist()
    )

    obs_scaler = cap_running_scaler_count(
        getattr(agent, "_observation_preprocessor", None),
        float(args_cli.legacy_preprocessor_count_cap),
    )
    value_scaler = cap_running_scaler_count(
        getattr(agent, "_value_preprocessor", None),
        float(args_cli.legacy_preprocessor_count_cap),
    )
    for key, value in obs_scaler.items():
        metadata[f"observation_{key}"] = value
    for key, value in value_scaler.items():
        metadata[f"value_{key}"] = value

    metadata.update(reset_optimizer_state(agent.optimizer))

    print("[INFO] One-time legacy actor transfer calibration:")
    print(
        "  raw RMS                   : "
        f"{metadata['actor_raw_rms_before']:.4f} -> "
        f"{metadata['actor_raw_rms_after']:.4f}"
    )
    print(
        "  dead tanh derivative frac : "
        f"{metadata['actor_tanh_dead_fraction_before']:.4f} -> "
        f"{metadata['actor_tanh_dead_fraction_after']:.4f}"
    )
    print(
        "  mean tanh derivative      : "
        f"{metadata['actor_tanh_derivative_mean_after']:.4f}"
    )
    print(f"  output row scales          : {metadata['actor_output_row_scale']}")
    print(
        "  zero-input raw before      : "
        f"{metadata['actor_zero_input_raw_before']}"
    )
    print(
        "  zero-input mean after      : "
        f"{metadata['actor_zero_input_mean_after']}"
    )
    if "observation_scaler_count_before" in metadata:
        print(
            "  observation scaler count   : "
            f"{metadata['observation_scaler_count_before']:.1f} -> "
            f"{metadata['observation_scaler_count_after']:.1f}"
        )
    print(
        "  optimizer states reset     : "
        f"{metadata['optimizer_state_entries_before_reset']} -> "
        f"{metadata['optimizer_state_entries_after_reset']}"
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
    if (
        args_cli.post_load_learning_rate is not None
        or args_cli.post_load_max_action_std is not None
        or args_cli.recalibrate_legacy_actor
    ) and resume_path is None:
        raise ValueError(
            "post-load overrides / legacy actor recalibration require --checkpoint"
        )
    if args_cli.recalibrate_legacy_actor and args_cli.task != LEARNED_INERTIAL_RL_TASK:
        raise ValueError(
            "--recalibrate_legacy_actor is only valid for the learned-inertial RL task"
        )

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    _audit_learned_inertial_bounded_cfg(env, agent_cfg)

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

    # load checkpoint (if specified)
    if resume_path:
        print(f"[INFO] Loading model checkpoint from: {resume_path}")
        # runner.agent.load is intentionally strict with the model state. The
        # 0.8*tanh mean changes only the forward expression and adds no trainable
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

    # run training
    runner.run()

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
