"""Policy observations backed by the standalone learned-inertial estimator.

No Isaac root pose is read here.  The world-frame gate command is part of the
known track map; the drone pose comes from LearnedInertialOdometry.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import isaaclab.utils.math as math_utils

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def learned_inertial_drone_state(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Return [p_w(3), q_wb(4), v_b(3), omega_b(3)] from estimator/sensors."""
    state = getattr(env, "learned_inertial_state", None)
    if state is None:
        return torch.zeros(env.num_envs, 13, device=env.device)

    p = torch.as_tensor(state.position_w_b, dtype=torch.float32, device=env.device).view(1, 3)
    q = torch.as_tensor(
        state.orientation_w_b_wxyz, dtype=torch.float32, device=env.device
    ).view(1, 4)
    v_w = torch.as_tensor(
        state.linear_velocity_w_b, dtype=torch.float32, device=env.device
    ).view(1, 3)
    v_b = math_utils.quat_rotate_inverse(q, v_w)

    imu = env.scene["imu"]
    omega_b = imu.data.ang_vel_b
    if omega_b.shape[0] != env.num_envs:
        omega_b = omega_b[:1]
    return torch.cat((p, q, v_b, omega_b), dim=-1)


def learned_target_pos_b(env: ManagerBasedRLEnv, command_name: str = "target") -> torch.Tensor:
    """Known mapped target position expressed in the estimated body frame."""
    state = getattr(env, "learned_inertial_state", None)
    if state is None:
        return torch.zeros(env.num_envs, 3, device=env.device)

    target_w = env.command_manager.get_term(command_name).command[:, :3]
    p = torch.as_tensor(state.position_w_b, dtype=torch.float32, device=env.device).view(1, 3)
    q = torch.as_tensor(
        state.orientation_w_b_wxyz, dtype=torch.float32, device=env.device
    ).view(1, 4)
    return math_utils.quat_rotate_inverse(q, target_w - p)
