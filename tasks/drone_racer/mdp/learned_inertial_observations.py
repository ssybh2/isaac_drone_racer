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



def _quat_slerp_wxyz(
    q0: torch.Tensor,
    q1: torch.Tensor,
    alpha: float,
) -> torch.Tensor:
    """Shortest-arc batched quaternion SLERP for the GT->estimator curriculum."""
    if not 0.0 <= float(alpha) <= 1.0:
        raise ValueError("blend alpha must be in [0, 1]")
    q0 = q0 / torch.linalg.vector_norm(q0, dim=-1, keepdim=True).clamp_min(1.0e-8)
    q1 = q1 / torch.linalg.vector_norm(q1, dim=-1, keepdim=True).clamp_min(1.0e-8)
    dot = torch.sum(q0 * q1, dim=-1, keepdim=True)
    q1 = torch.where(dot < 0.0, -q1, q1)
    dot = torch.abs(dot).clamp(0.0, 1.0)

    blend = torch.full_like(dot, float(alpha))
    linear = dot > 0.9995
    theta = torch.acos(dot.clamp_max(1.0 - 1.0e-7))
    sin_theta = torch.sin(theta).clamp_min(1.0e-7)
    s0 = torch.sin((1.0 - blend) * theta) / sin_theta
    s1 = torch.sin(blend * theta) / sin_theta
    spherical = s0 * q0 + s1 * q1
    lerped = (1.0 - blend) * q0 + blend * q1
    result = torch.where(linear, lerped, spherical)
    return result / torch.linalg.vector_norm(
        result, dim=-1, keepdim=True
    ).clamp_min(1.0e-8)


def _blended_platform_components(
    env: ManagerBasedRLEnv,
    blend_alpha: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return curriculum p/v/q. GT is used only by explicit blend tasks."""
    if not 0.0 <= float(blend_alpha) <= 1.0:
        raise ValueError("blend_alpha must be in [0, 1]")

    robot = env.scene["robot"]
    p_gt = robot.data.root_pos_w
    v_gt = robot.data.root_lin_vel_w
    q_gt = robot.data.root_quat_w

    state = getattr(env, "learned_inertial_state", None)
    if state is None:
        # Partial blend stages deliberately retain GT until the estimator has
        # produced its first state. The 100% estimator stage must never
        # silently leak simulator truth: match learned_inertial_swift_state's
        # fail-closed semantics until the estimator is initialized.
        if float(blend_alpha) >= 1.0 - 1.0e-12:
            p_zero = torch.zeros_like(p_gt)
            v_zero = torch.zeros_like(v_gt)
            q_identity = torch.zeros_like(q_gt)
            q_identity[:, 0] = 1.0
            return p_zero, v_zero, q_identity
        return p_gt, v_gt, q_gt

    p_est = torch.as_tensor(
        state.position_w_b, dtype=p_gt.dtype, device=env.device
    ).view(1, 3)
    v_est = torch.as_tensor(
        state.linear_velocity_w_b, dtype=v_gt.dtype, device=env.device
    ).view(1, 3)
    q_est = torch.as_tensor(
        state.orientation_w_b_wxyz, dtype=q_gt.dtype, device=env.device
    ).view(1, 4)

    alpha = float(blend_alpha)
    p = (1.0 - alpha) * p_gt + alpha * p_est
    v = (1.0 - alpha) * v_gt + alpha * v_est
    q = _quat_slerp_wxyz(q_gt, q_est, alpha)
    return p, v, q


def blended_inertial_swift_state(
    env: ManagerBasedRLEnv,
    blend_alpha: float,
) -> torch.Tensor:
    """Swift [p,v,R] with p/v lerp and attitude SLERP."""
    p, v, q = _blended_platform_components(env, blend_alpha)
    R = math_utils.matrix_from_quat(q).reshape(env.num_envs, 9)
    return torch.cat((p, v, R), dim=-1)


def blended_truth_next_gate_corners_relative_w(
    env: ManagerBasedRLEnv,
    blend_alpha: float,
    command_name: str = "target",
) -> torch.Tensor:
    """Truth mission gate with corner vectors relative to blended position."""
    p, _, _ = _blended_platform_components(env, blend_alpha)
    command = env.command_manager.get_term(command_name)
    if not hasattr(command, "gt_next_gate_idx"):
        raise AttributeError(
            "blended truth-mission observation requires gt_next_gate_idx"
        )

    gate_indices = command.gt_next_gate_idx.to(dtype=torch.long)
    env_ids = torch.arange(env.num_envs, device=env.device)
    track_data = command.track.data
    gate_center_w = track_data.object_com_pos_w[env_ids, gate_indices]
    gate_quat_w = track_data.object_quat_w[env_ids, gate_indices]
    half = float(command.gate_size) / 2.0
    local_corners = torch.tensor(
        [
            [0.0, -half, -half],
            [0.0, +half, -half],
            [0.0, +half, +half],
            [0.0, -half, +half],
        ],
        dtype=p.dtype,
        device=env.device,
    ).unsqueeze(0).expand(env.num_envs, -1, -1)
    q = gate_quat_w.unsqueeze(1).expand(-1, 4, -1).reshape(-1, 4)
    corners_w = math_utils.quat_apply(
        q, local_corners.reshape(-1, 3)
    ).reshape(env.num_envs, 4, 3)
    corners_w = corners_w + gate_center_w.unsqueeze(1)
    return (corners_w - p.unsqueeze(1)).reshape(env.num_envs, 12)



def _noisy_gt_components(
    env: ManagerBasedRLEnv,
    *,
    position_std_m: float,
    velocity_std_mps: float,
    attitude_std_deg: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return one internally consistent noisy-GT sample per control step.

    This is Stage C robustness training only. It deliberately uses simulator
    truth plus synthetic residuals and must never be confused with the final
    no-GT estimator observation path.
    """
    if min(position_std_m, velocity_std_mps, attitude_std_deg) < 0.0:
        raise ValueError("GT-noise standard deviations must be non-negative")

    step = int(getattr(env, "common_step_counter", -1))
    key = (
        step,
        float(position_std_m),
        float(velocity_std_mps),
        float(attitude_std_deg),
    )
    cached = getattr(env, "_circular12_noisy_gt_cache", None)
    if cached is not None and cached[0] == key:
        return cached[1]

    robot = env.scene["robot"]
    p_gt = robot.data.root_pos_w
    v_gt = robot.data.root_lin_vel_w
    q_gt = robot.data.root_quat_w

    p = p_gt + float(position_std_m) * torch.randn_like(p_gt)
    v = v_gt + float(velocity_std_mps) * torch.randn_like(v_gt)

    sigma_rad = float(attitude_std_deg) * torch.pi / 180.0
    rotvec = sigma_rad * torch.randn(
        env.num_envs, 3, dtype=q_gt.dtype, device=q_gt.device
    )
    angle = torch.linalg.vector_norm(rotvec, dim=-1, keepdim=True)
    axis = rotvec / angle.clamp_min(1.0e-8)
    half = 0.5 * angle
    dq = torch.cat(
        (torch.cos(half), axis * torch.sin(half)),
        dim=-1,
    )
    tiny = angle.squeeze(-1) < 1.0e-8
    if torch.any(tiny):
        dq[tiny] = torch.tensor(
            [1.0, 0.0, 0.0, 0.0],
            dtype=q_gt.dtype,
            device=q_gt.device,
        )
    q = math_utils.quat_mul(q_gt, dq)
    q = q / torch.linalg.vector_norm(q, dim=-1, keepdim=True).clamp_min(1.0e-8)

    result = (p, v, q)
    env._circular12_noisy_gt_cache = (key, result)
    return result


def noisy_gt_swift_state(
    env: ManagerBasedRLEnv,
    position_std_m: float = 0.08,
    velocity_std_mps: float = 0.08,
    attitude_std_deg: float = 1.0,
) -> torch.Tensor:
    """31D-policy platform state with configurable synthetic estimator residuals."""
    p, v, q = _noisy_gt_components(
        env,
        position_std_m=position_std_m,
        velocity_std_mps=velocity_std_mps,
        attitude_std_deg=attitude_std_deg,
    )
    R = math_utils.matrix_from_quat(q).reshape(env.num_envs, 9)
    return torch.cat((p, v, R), dim=-1)


def noisy_gt_next_gate_corners_relative_w(
    env: ManagerBasedRLEnv,
    command_name: str = "target",
    position_std_m: float = 0.08,
    velocity_std_mps: float = 0.08,
    attitude_std_deg: float = 1.0,
) -> torch.Tensor:
    """Truth mission gate geometry relative to the same noisy position sample."""
    p, _, _ = _noisy_gt_components(
        env,
        position_std_m=position_std_m,
        velocity_std_mps=velocity_std_mps,
        attitude_std_deg=attitude_std_deg,
    )
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
        dtype=p.dtype,
        device=env.device,
    ).unsqueeze(0).expand(env.num_envs, -1, -1)
    q = gate_quat_w.unsqueeze(1).expand(-1, 4, -1).reshape(-1, 4)
    corners_w = math_utils.quat_apply(
        q, local_corners.reshape(-1, 3)
    ).reshape(env.num_envs, 4, 3)
    corners_w = corners_w + gate_center_w.unsqueeze(1)
    return (corners_w - p.unsqueeze(1)).reshape(env.num_envs, 12)


def learned_truth_next_gate_corners_relative_w(
    env: ManagerBasedRLEnv,
    command_name: str = "target",
) -> torch.Tensor:
    """Estimator-relative gate corners driven by truth-only mission progression.

    This Stage-E diagnostic does not read simulator root pose. GT is used only
    to choose which known-map gate is currently the truth mission target.
    """
    state = getattr(env, "learned_inertial_state", None)
    if state is None:
        return torch.zeros(env.num_envs, 12, device=env.device)

    command = env.command_manager.get_term(command_name)
    if not hasattr(command, "gt_next_gate_idx"):
        raise AttributeError(
            "truth-mission estimator observation requires gt_next_gate_idx"
        )

    gate_indices = command.gt_next_gate_idx.to(dtype=torch.long)
    env_ids = torch.arange(env.num_envs, device=env.device)
    track_data = command.track.data
    gate_center_w = track_data.object_com_pos_w[env_ids, gate_indices]
    gate_quat_w = track_data.object_quat_w[env_ids, gate_indices]
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
    ).unsqueeze(0).expand(env.num_envs, -1, -1)

    q = gate_quat_w.unsqueeze(1).expand(-1, 4, -1).reshape(-1, 4)
    corners_w = math_utils.quat_apply(
        q, local_corners.reshape(-1, 3)
    ).reshape(env.num_envs, 4, 3)
    corners_w = corners_w + gate_center_w.unsqueeze(1)

    p_est = torch.as_tensor(
        state.position_w_b,
        dtype=torch.float32,
        device=env.device,
    ).view(1, 1, 3)
    return (corners_w - p_est).reshape(env.num_envs, 12)
