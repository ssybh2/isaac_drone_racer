"""Privileged Circular-12 imitation/coordinated-flight PPO shaping terms."""

from __future__ import annotations

import torch
import isaaclab.utils.math as math_utils
from isaaclab.managers import SceneEntityCfg

from imitation.circular12_expert import (
    circular12_expert_action,
    config_from_ctbr_action_cfg,
)


def _expert_output(
    env,
    *,
    target_speed_mps: float,
    radius_m: float = 12.0,
    height_m: float = 2.07,
    action_name: str = "control_action",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Compute/cache the privileged expert once per environment control step."""
    counter = int(getattr(env, "common_step_counter", -1))
    key = (
        counter,
        float(target_speed_mps),
        float(radius_m),
        float(height_m),
        action_name,
    )
    cached = getattr(env, "_circular12_expert_reward_cache", None)
    if cached is not None and cached[0] == key:
        return cached[1]

    robot = env.scene[asset_cfg.name]
    action_term = env.action_manager.get_term(action_name)
    action_cfg = getattr(action_term, "cfg", env.cfg.actions.control_action)
    cfg = config_from_ctbr_action_cfg(
        action_cfg,
        target_speed_mps=float(target_speed_mps),
        radius_m=float(radius_m),
        height_m=float(height_m),
    )
    output = circular12_expert_action(
        robot.data.root_pos_w,
        robot.data.root_lin_vel_w,
        math_utils.matrix_from_quat(robot.data.root_quat_w),
        cfg,
    )
    env._circular12_expert_reward_cache = (key, output)
    return output


def circular12_expert_anchor_l2(
    env,
    target_speed_mps: float = 17.712658128452922,
    start_scale: float = 4.0,
    end_scale: float = 0.25,
    anneal_steps: int = 24000,
    action_name: str = "control_action",
) -> torch.Tensor:
    """Scheduled normalized-action imitation loss used as a PPO safety anchor."""
    if anneal_steps <= 0:
        raise ValueError("anneal_steps must be positive")
    expert = _expert_output(
        env,
        target_speed_mps=target_speed_mps,
        action_name=action_name,
    )
    term = env.action_manager.get_term(action_name)
    raw_action = getattr(term, "raw_actions", None)
    if raw_action is None:
        raise AttributeError(f"{action_name!r} does not expose raw_actions")

    step = max(0, int(getattr(env, "common_step_counter", 0)))
    fraction = min(float(step) / float(anneal_steps), 1.0)
    scale = float(start_scale) + (
        float(end_scale) - float(start_scale)
    ) * fraction
    return scale * torch.sum(torch.square(raw_action - expert.action), dim=-1)


def circular12_coordinated_attitude_l2(
    env,
    target_speed_mps: float = 17.712658128452922,
) -> torch.Tensor:
    expert = _expert_output(env, target_speed_mps=target_speed_mps)
    return torch.sum(
        torch.square(expert.attitude_error_rotvec_b), dim=-1
    )


def circular12_coordinated_body_rate_l2(
    env,
    target_speed_mps: float = 17.712658128452922,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    expert = _expert_output(env, target_speed_mps=target_speed_mps)
    robot = env.scene[asset_cfg.name]
    return torch.sum(
        torch.square(
            robot.data.root_ang_vel_b - expert.desired_body_rate_b
        ),
        dim=-1,
    )


def circular12_radius_error_l2(
    env,
    radius_m: float = 12.0,
    center_xy: tuple[float, float] = (0.0, 12.0),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    robot = env.scene[asset_cfg.name]
    center = torch.tensor(
        center_xy,
        dtype=robot.data.root_pos_w.dtype,
        device=robot.device,
    )
    radius = torch.linalg.vector_norm(
        robot.data.root_pos_w[:, :2] - center, dim=-1
    )
    return torch.square(radius - float(radius_m))


def circular12_height_error_l2(
    env,
    height_m: float = 2.07,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    robot = env.scene[asset_cfg.name]
    return torch.square(robot.data.root_pos_w[:, 2] - float(height_m))


def circular12_speed_error_l2(
    env,
    target_speed_mps: float = 17.712658128452922,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    robot = env.scene[asset_cfg.name]
    speed_xy = torch.linalg.vector_norm(
        robot.data.root_lin_vel_w[:, :2], dim=-1
    )
    return torch.square(speed_xy - float(target_speed_mps))
