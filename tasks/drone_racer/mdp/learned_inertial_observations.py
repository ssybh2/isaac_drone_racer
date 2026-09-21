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

    # Use the same synthetic onboard gyro sample consumed by the estimator
    # when available. Falling back to the Isaac IMU keeps the observation
    # usable before the first propagated estimator sample.
    gyro_meas = getattr(env, "_last_imu_gyro_b_meas", None)
    if gyro_meas is not None:
        omega_b = torch.as_tensor(
            gyro_meas, dtype=torch.float32, device=env.device
        ).view(1, 3)
    else:
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


def learned_inertial_swift_state(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Return Swift-style estimated platform state [p_w, v_w, R_wb].

    Dimensions are 3 + 3 + 9 = 15. This intentionally mirrors the deployed
    policy input described for Swift: position, velocity and attitude encoded
    as a rotation matrix. No simulator root pose is read here.
    """
    state = getattr(env, "learned_inertial_state", None)
    if state is None:
        return torch.zeros(env.num_envs, 15, device=env.device)

    p_w = torch.as_tensor(
        state.position_w_b,
        dtype=torch.float32,
        device=env.device,
    ).view(1, 3)
    v_w = torch.as_tensor(
        state.linear_velocity_w_b,
        dtype=torch.float32,
        device=env.device,
    ).view(1, 3)
    q_wb = torch.as_tensor(
        state.orientation_w_b_wxyz,
        dtype=torch.float32,
        device=env.device,
    ).view(1, 4)
    R_wb = math_utils.matrix_from_quat(q_wb).reshape(1, 9)
    return torch.cat((p_w, v_w, R_wb), dim=-1)


def learned_next_gate_corners_relative_w(
    env: ManagerBasedRLEnv,
    command_name: str = "target",
) -> torch.Tensor:
    """Return the next mapped gate's four corner vectors relative to the vehicle.

    The four gate corners are generated from the known map pose. Their relative
    positions are expressed in the world frame, matching the rotation-matrix
    state representation used by the Swift-style policy. Output dimension is
    4 * 3 = 12. Simulator truth is never read.
    """
    state = getattr(env, "learned_inertial_state", None)
    if state is None:
        return torch.zeros(env.num_envs, 12, device=env.device)

    command = env.command_manager.get_term(command_name)
    gate_pose_w = command.command
    gate_center_w = gate_pose_w[:, :3]
    gate_quat_w = gate_pose_w[:, 3:7]
    half = float(command.gate_size) / 2.0

    local_corners = torch.tensor(
        [
            [0.0, -half, -half],
            [0.0, +half, -half],
            [0.0, +half, +half],
            [0.0, -half, +half],
        ],
        dtype=torch.float32,
        device=env.device,
    )
    local_corners = local_corners.unsqueeze(0).expand(env.num_envs, -1, -1)

    q = gate_quat_w.unsqueeze(1).expand(-1, 4, -1).reshape(-1, 4)
    corners_w = math_utils.quat_apply(
        q,
        local_corners.reshape(-1, 3),
    ).reshape(env.num_envs, 4, 3)
    corners_w = corners_w + gate_center_w.unsqueeze(1)

    p_w = torch.as_tensor(
        state.position_w_b,
        dtype=torch.float32,
        device=env.device,
    ).view(1, 1, 3)
    relative_w = corners_w - p_w
    return relative_w.reshape(env.num_envs, 12)

