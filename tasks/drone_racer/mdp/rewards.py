# Copyright (c) 2025, Kousheek Chakraborty
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# This project uses the IsaacLab framework (https://github.com/isaac-sim/IsaacLab),
# which is licensed under the BSD-3-Clause License.

from __future__ import annotations

from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
import torch
from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def pos_error_l2(
    env: ManagerBasedRLEnv,
    command_name: str,
    target_pos: list | None = None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize asset pos from its target pos using L2 squared kernel."""

    asset: RigidObject = env.scene[asset_cfg.name]

    if target_pos is None:
        target_pos = env.command_manager.get_term(command_name).command
        target_pos_tensor = target_pos[:, :3]
    else:
        target_pos_tensor = (
            torch.tensor(target_pos, dtype=torch.float32, device=asset.device).repeat(env.num_envs, 1)
            + env.scene.env_origins
        )

    return torch.sum(torch.square(asset.data.root_pos_w - target_pos_tensor), dim=1)


def pos_error_tanh(
    env: ManagerBasedRLEnv,
    std: float,
    command_name: str | None = None,
    target_pos: list | None = None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize asset pos from its target pos using L2 squared kernel."""

    asset: RigidObject = env.scene[asset_cfg.name]

    if target_pos is None:
        target_pos = env.command_manager.get_term(command_name).command
        target_pos_tensor = target_pos[:, :3]
    else:
        target_pos_tensor = (
            torch.tensor(target_pos, dtype=torch.float32, device=asset.device).repeat(env.num_envs, 1)
            + env.scene.env_origins
        )

    distance = torch.norm(asset.data.root_pos_w - target_pos_tensor, dim=1)
    return 1 - torch.tanh(distance / std)


def progress(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward reduction in distance to the active gate."""

    asset: RigidObject = env.scene[asset_cfg.name]

    target_pos = env.command_manager.get_term(command_name).command[:, :3]
    previous_pos = env.command_manager.get_term(command_name).previous_pos
    current_pos = asset.data.root_pos_w

    prev_distance = torch.norm(previous_pos - target_pos, dim=1)
    current_distance = torch.norm(current_pos - target_pos, dim=1)

    return prev_distance - current_distance


def gate_passed(
    env: ManagerBasedRLEnv,
    command_name: str | None = None,
) -> torch.Tensor:
    """Reward for passing a gate."""
    missed = (-1.0) * env.command_manager.get_term(command_name).gate_missed
    passed = (1.0) * env.command_manager.get_term(command_name).gate_passed
    return missed + passed


def lookat_next_gate(
    env: ManagerBasedRLEnv,
    std: float,
    command_name: str | None = None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Legacy reward for looking at the next gate."""

    asset: RigidObject = env.scene[asset_cfg.name]

    drone_pos = asset.data.root_pos_w
    drone_att = asset.data.root_quat_w
    next_gate_pos = env.command_manager.get_term(command_name).command[:, :3]

    vec_to_gate = math_utils.normalize(next_gate_pos - drone_pos)

    x_axis = torch.tensor([1.0, 0.0, 0.0], device=asset.device).expand(env.num_envs, 3)
    drone_x_axis = math_utils.normalize(math_utils.quat_apply(drone_att, x_axis))

    dot = (drone_x_axis * vec_to_gate).sum(dim=1).clamp(-1.0, 1.0)
    angle = torch.acos(dot)
    return torch.exp(-angle / std)


def swift_perception_awareness(
    env: ManagerBasedRLEnv,
    command_name: str,
    lambda_3: float = -10.0,
    camera_pos_b: tuple[float, float, float] = (0.14, 0.0, 0.05),
    camera_optical_axis_b: tuple[float, float, float] = (1.0, 0.0, 0.0),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Swift-2023 perception-aware reward without the external lambda_2 weight.

    Kaufmann et al. use

        r_perc = lambda_2 * exp(lambda_3 * delta_cam**4)

    where ``delta_cam`` is the angle between the camera optical axis and the
    center of the next gate. IsaacLab's RewardTerm applies ``lambda_2`` as the
    term weight, so this function returns only ``exp(lambda_3*delta_cam**4)``.

    The Stage2 calibrated camera optical +Z axis maps to body +X, hence the
    default body-frame optical axis is ``(1, 0, 0)``. The camera-center offset
    is included instead of measuring the angle from the vehicle origin.
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    device = asset.device

    body_pos_w = asset.data.root_pos_w
    body_quat_w = asset.data.root_quat_w
    gate_center_w = env.command_manager.get_term(command_name).command[:, :3]

    camera_offset_b = torch.tensor(camera_pos_b, dtype=body_pos_w.dtype, device=device).expand(env.num_envs, 3)
    optical_axis_b = torch.tensor(
        camera_optical_axis_b, dtype=body_pos_w.dtype, device=device
    ).expand(env.num_envs, 3)

    camera_pos_w = body_pos_w + math_utils.quat_apply(body_quat_w, camera_offset_b)
    optical_axis_w = math_utils.normalize(math_utils.quat_apply(body_quat_w, optical_axis_b))
    to_gate_w = math_utils.normalize(gate_center_w - camera_pos_w)

    dot = torch.sum(optical_axis_w * to_gate_w, dim=1).clamp(-1.0, 1.0)
    delta_cam = torch.acos(dot)
    return torch.exp(float(lambda_3) * torch.pow(delta_cam, 4))


def ang_vel_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize base angular velocity using L2 squared kernel."""
    asset: RigidObject = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.root_ang_vel_b), dim=1)
