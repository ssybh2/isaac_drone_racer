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
from isaaclab.assets import Articulation, RigidObject

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def reset_after_prev_gate(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    gate_pose: torch.Tensor,
    pose_range: dict[str, tuple[float, float]],
    velocity_range: dict[str, tuple[float, float]],
    asset_cfg_name: str = "robot",
):
    """Reset the asset right after a random gate."""

    # extract the used quantities (to enable type-hinting)
    asset: RigidObject | Articulation = env.scene[asset_cfg_name]

    # get default root state
    root_states = asset.data.default_root_state[env_ids].clone()

    # poses
    range_list = [pose_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
    ranges = torch.tensor(range_list, device=asset.device)
    rand_samples = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=asset.device)

    gate_pos = gate_pose[env_ids, :3]
    gate_quat = gate_pose[env_ids, 3:7]
    offset = torch.tensor([1.0, 0.0, 0.0], device=asset.device).expand(len(env_ids), 3)
    offset_world = math_utils.quat_apply(gate_quat, offset)
    pos_after_prev_gate = gate_pos + offset_world

    positions = root_states[:, 0:3] + env.scene.env_origins[env_ids] + pos_after_prev_gate + rand_samples[:, 0:3]
    orientations_delta = math_utils.quat_from_euler_xyz(rand_samples[:, 3], rand_samples[:, 4], rand_samples[:, 5])
    orientations = math_utils.quat_mul(root_states[:, 3:7], orientations_delta)

    # velocities
    range_list = [velocity_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
    ranges = torch.tensor(range_list, device=asset.device)
    rand_samples = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=asset.device)

    velocities = root_states[:, 7:13] + rand_samples

    # set into the physics simulation
    asset.write_root_pose_to_sim(torch.cat([positions, orientations], dim=-1), env_ids=env_ids)
    asset.write_root_velocity_to_sim(velocities, env_ids=env_ids)



def reset_circular12_coordinated_state(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    target_speed_mps: float = 14.0,
    radius_m: float = 12.0,
    center_xy: tuple[float, float] = (0.0, 12.0),
    height_m: float = 2.07,
    phase_rad: float = -7.0 * torch.pi / 12.0,
    gravity_mps2: float = 9.81,
    asset_cfg_name: str = "robot",
):
    """Reset once onto a physically consistent coordinated Circular-12 state.

    This is only an episode initialization event for expert/BC validation and
    demonstration collection. After reset, no pose or velocity is prescribed;
    the vehicle flies through Swift CTBR -> rate PID -> mixer -> motors ->
    physics.

    The default phase (-105 deg) starts before Gate 1 (-90 deg), so the first
    mission target can remain Gate 1 while the vehicle approaches it along the
    radius-12 reference circle.
    """
    if target_speed_mps <= 0.0 or radius_m <= 0.0 or gravity_mps2 <= 0.0:
        raise ValueError("speed, radius and gravity must be positive")

    asset: RigidObject | Articulation = env.scene[asset_cfg_name]
    n = len(env_ids)
    dtype = asset.data.root_pos_w.dtype
    device = asset.device

    phase = torch.full(
        (n,), float(phase_rad), dtype=dtype, device=device
    )
    c = torch.cos(phase)
    s = torch.sin(phase)
    tangent = torch.stack(
        (-s, c, torch.zeros_like(c)), dim=-1
    )

    positions = torch.stack(
        (
            float(center_xy[0]) + float(radius_m) * c,
            float(center_xy[1]) + float(radius_m) * s,
            torch.full_like(c, float(height_m)),
        ),
        dim=-1,
    )
    positions = positions + env.scene.env_origins[env_ids]

    speed = float(target_speed_mps)
    omega = speed / float(radius_m)
    bank = -torch.atan2(
        torch.tensor(
            speed * speed / float(radius_m),
            dtype=dtype,
            device=device,
        ),
        torch.tensor(float(gravity_mps2), dtype=dtype, device=device),
    )
    yaw = phase + torch.pi / 2.0
    quat = math_utils.quat_from_euler_xyz(
        bank.expand_as(phase),
        torch.zeros_like(phase),
        yaw,
    )

    velocity = torch.zeros((n, 6), dtype=dtype, device=device)
    velocity[:, :3] = speed * tangent
    # Isaac root angular velocity is world-frame. A steady coordinated circle
    # rotates the body frame about world +Z at speed/radius.
    velocity[:, 5] = omega

    asset.write_root_pose_to_sim(
        torch.cat((positions, quat), dim=-1), env_ids=env_ids
    )
    asset.write_root_velocity_to_sim(velocity, env_ids=env_ids)

    # Avoid a false first-step gate-plane crossing caused by stale command
    # history from the pre-reset pose.
    try:
        command = env.command_manager.get_term("target")
        command.prev_robot_pos_w = positions.clone()
    except (AttributeError, KeyError):
        pass
