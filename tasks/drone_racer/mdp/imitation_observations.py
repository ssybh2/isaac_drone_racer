"""Training-only GT observation corruption for estimator-robust policy tuning.

This stage stays vectorized and sensor-free. It injects estimator-scale
position, velocity and attitude residuals into the unchanged 31-D policy
contract before real estimator state is allowed to close the control loop.
"""

from __future__ import annotations

import math

import torch
import isaaclab.utils.math as math_utils


def _rotvec_to_quat_wxyz(rotvec: torch.Tensor) -> torch.Tensor:
    angle = torch.linalg.vector_norm(rotvec, dim=-1, keepdim=True)
    half = 0.5 * angle
    scale = torch.where(
        angle > 1.0e-7,
        torch.sin(half) / angle.clamp_min(1.0e-7),
        0.5 - angle.square() / 48.0,
    )
    quat = torch.cat((torch.cos(half), rotvec * scale), dim=-1)
    return quat / torch.linalg.vector_norm(
        quat, dim=-1, keepdim=True
    ).clamp_min(1.0e-8)


def _noisy_gt_components(
    env,
    *,
    position_std_m: float,
    velocity_std_mps: float,
    attitude_std_deg: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if position_std_m < 0.0 or velocity_std_mps < 0.0 or attitude_std_deg < 0.0:
        raise ValueError("residual-noise standard deviations must be non-negative")

    counter = int(getattr(env, "common_step_counter", -1))
    key = (
        counter,
        float(position_std_m),
        float(velocity_std_mps),
        float(attitude_std_deg),
    )
    cached = getattr(env, "_imitation_noisy_gt_cache", None)
    if cached is not None and cached[0] == key:
        return cached[1]

    robot = env.scene["robot"]
    p_gt = robot.data.root_pos_w
    v_gt = robot.data.root_lin_vel_w
    q_gt = robot.data.root_quat_w

    p = p_gt + float(position_std_m) * torch.randn_like(p_gt)
    v = v_gt + float(velocity_std_mps) * torch.randn_like(v_gt)
    attitude_std_rad = math.radians(float(attitude_std_deg))
    rotvec = attitude_std_rad * torch.randn_like(p_gt)
    q_delta = _rotvec_to_quat_wxyz(rotvec)
    q = math_utils.quat_mul(q_gt, q_delta)
    q = q / torch.linalg.vector_norm(
        q, dim=-1, keepdim=True
    ).clamp_min(1.0e-8)

    result = (p, v, q)
    env._imitation_noisy_gt_cache = (key, result)
    return result


def swift_gt_state_with_residual_noise(
    env,
    position_std_m: float = 0.08,
    velocity_std_mps: float = 0.06,
    attitude_std_deg: float = 0.5,
) -> torch.Tensor:
    """Return noisy [p_w(3), v_w(3), R_wb(9)] with the same 15-D layout."""
    p, v, q = _noisy_gt_components(
        env,
        position_std_m=position_std_m,
        velocity_std_mps=velocity_std_mps,
        attitude_std_deg=attitude_std_deg,
    )
    R = math_utils.matrix_from_quat(q).reshape(env.num_envs, 9)
    return torch.cat((p, v, R), dim=-1)


def swift_gt_next_gate_corners_relative_noisy_w(
    env,
    command_name: str = "target",
    position_std_m: float = 0.08,
    velocity_std_mps: float = 0.06,
    attitude_std_deg: float = 0.5,
) -> torch.Tensor:
    """Known-map next gate corners relative to the exact same noisy position."""
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
        device=p.device,
    ).unsqueeze(0).expand(env.num_envs, -1, -1)
    q = gate_quat_w.unsqueeze(1).expand(-1, 4, -1).reshape(-1, 4)
    corners_w = math_utils.quat_apply(
        q,
        local_corners.reshape(-1, 3),
    ).reshape(env.num_envs, 4, 3)
    corners_w = corners_w + gate_center_w.unsqueeze(1)
    return (corners_w - p.unsqueeze(1)).reshape(env.num_envs, 12)
