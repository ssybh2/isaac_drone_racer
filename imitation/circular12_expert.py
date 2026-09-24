"""Vectorized GT CTBR expert used only for imitation labels and diagnostics.

The expert outputs the same normalized four-dimensional command consumed by
SwiftCTBRAction. The vehicle therefore still flies through the existing
body-rate PID, allocation, motors and physics; no pose or velocity is written
by this controller.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class Circular12ExpertConfig:
    radius_m: float = 12.0
    center_x_m: float = 0.0
    center_y_m: float = 12.0
    height_m: float = 2.07
    target_speed_mps: float = 17.712658128452922
    gravity_mps2: float = 9.81
    body_rate_max_radps: tuple[float, float, float] = (4.0, 4.0, 2.0)
    max_collective_accel_mps2: float = 45.0
    radial_kp: float = 2.5
    height_kp: float = 4.0
    horizontal_kd: float = 2.0
    vertical_kd: float = 2.5
    attitude_kp: float = 3.0

    def __post_init__(self) -> None:
        if self.radius_m <= 0.0:
            raise ValueError("radius_m must be positive")
        if self.target_speed_mps <= 0.0:
            raise ValueError("target_speed_mps must be positive")
        if self.max_collective_accel_mps2 <= self.gravity_mps2:
            raise ValueError("maximum collective acceleration must exceed gravity")
        if any(v <= 0.0 for v in self.body_rate_max_radps):
            raise ValueError("body-rate limits must be positive")


@dataclass
class Circular12ExpertOutput:
    action: torch.Tensor
    reference_position_w: torch.Tensor
    reference_velocity_w: torch.Tensor
    desired_rotation_wb: torch.Tensor
    desired_body_rate_b: torch.Tensor
    attitude_error_rotvec_b: torch.Tensor
    collective_accel_mps2: torch.Tensor
    phase_rad: torch.Tensor


def _normalize(v: torch.Tensor, eps: float = 1.0e-8) -> torch.Tensor:
    return v / torch.linalg.vector_norm(v, dim=-1, keepdim=True).clamp_min(eps)


def _so3_log(rotation: torch.Tensor) -> torch.Tensor:
    """Return a stable batched SO(3) logarithm as a body-frame rotation vector."""
    trace = torch.diagonal(rotation, dim1=-2, dim2=-1).sum(dim=-1)
    cos_theta = ((trace - 1.0) * 0.5).clamp(-1.0 + 1.0e-7, 1.0 - 1.0e-7)
    theta = torch.acos(cos_theta)
    vee_half = 0.5 * torch.stack(
        (
            rotation[..., 2, 1] - rotation[..., 1, 2],
            rotation[..., 0, 2] - rotation[..., 2, 0],
            rotation[..., 1, 0] - rotation[..., 0, 1],
        ),
        dim=-1,
    )
    sin_theta = torch.sin(theta)
    scale = torch.where(
        theta < 1.0e-4,
        torch.ones_like(theta),
        theta / sin_theta.clamp_min(1.0e-6),
    )
    return vee_half * scale.unsqueeze(-1)


def _inverse_collective_map(
    collective_accel: torch.Tensor,
    *,
    gravity_mps2: float,
    max_collective_accel_mps2: float,
) -> torch.Tensor:
    """Invert the hover-centred thrust map implemented by SwiftCTBRAction."""
    gravity = float(gravity_mps2)
    maximum = float(max_collective_accel_mps2)
    above = (collective_accel - gravity) / (maximum - gravity)
    below = (collective_accel - gravity) / gravity
    return torch.where(collective_accel >= gravity, above, below).clamp(-1.0, 1.0)


def circular12_reference_from_state(
    position_w: torch.Tensor,
    velocity_w: torch.Tensor,
    cfg: Circular12ExpertConfig,
) -> tuple[torch.Tensor, ...]:
    """Build a nearest-phase coordinated reference plus recovery acceleration."""
    if position_w.ndim != 2 or position_w.shape[-1] != 3:
        raise ValueError("position_w must have shape [N, 3]")
    if velocity_w.shape != position_w.shape:
        raise ValueError("velocity_w must match position_w")

    dtype, device = position_w.dtype, position_w.device
    center = torch.tensor(
        [cfg.center_x_m, cfg.center_y_m], dtype=dtype, device=device
    )
    rel_xy = position_w[:, :2] - center
    phase = torch.atan2(rel_xy[:, 1], rel_xy[:, 0])
    c, s = torch.cos(phase), torch.sin(phase)

    radial = torch.stack((c, s, torch.zeros_like(c)), dim=-1)
    tangent = torch.stack((-s, c, torch.zeros_like(c)), dim=-1)
    inward = -radial

    reference_position = torch.stack(
        (
            cfg.center_x_m + cfg.radius_m * c,
            cfg.center_y_m + cfg.radius_m * s,
            torch.full_like(c, float(cfg.height_m)),
        ),
        dim=-1,
    )
    reference_velocity = float(cfg.target_speed_mps) * tangent
    feedforward_accel = (
        float(cfg.target_speed_mps) ** 2 / float(cfg.radius_m)
    ) * inward

    position_error = reference_position - position_w
    velocity_error = reference_velocity - velocity_w
    correction = torch.zeros_like(position_w)
    correction[:, :2] = (
        float(cfg.radial_kp) * position_error[:, :2]
        + float(cfg.horizontal_kd) * velocity_error[:, :2]
    )
    correction[:, 2] = (
        float(cfg.height_kp) * position_error[:, 2]
        + float(cfg.vertical_kd) * velocity_error[:, 2]
    )
    desired_accel = feedforward_accel + correction

    gravity = torch.tensor(
        [0.0, 0.0, float(cfg.gravity_mps2)], dtype=dtype, device=device
    ).expand_as(desired_accel)
    thrust_accel_w = desired_accel + gravity
    z_des = _normalize(thrust_accel_w)
    y_des = _normalize(torch.linalg.cross(z_des, tangent, dim=-1))
    x_des = _normalize(torch.linalg.cross(y_des, z_des, dim=-1))
    desired_rotation = torch.stack((x_des, y_des, z_des), dim=-1)

    omega = float(cfg.target_speed_mps) / float(cfg.radius_m)
    desired_omega_w = torch.tensor(
        [0.0, 0.0, omega], dtype=dtype, device=device
    ).expand_as(position_w)
    desired_body_rate = torch.einsum(
        "nji,nj->ni", desired_rotation, desired_omega_w
    )

    return (
        phase,
        reference_position,
        reference_velocity,
        desired_accel,
        thrust_accel_w,
        desired_rotation,
        desired_body_rate,
    )


def circular12_expert_action(
    position_w: torch.Tensor,
    velocity_w: torch.Tensor,
    rotation_wb: torch.Tensor,
    cfg: Circular12ExpertConfig,
) -> Circular12ExpertOutput:
    """Compute a bounded normalized [collective, p, q, r] expert command."""
    if rotation_wb.ndim != 3 or rotation_wb.shape[-2:] != (3, 3):
        raise ValueError("rotation_wb must have shape [N, 3, 3]")
    if rotation_wb.shape[0] != position_w.shape[0]:
        raise ValueError("rotation batch size must match position")

    (
        phase,
        reference_position,
        reference_velocity,
        _desired_accel,
        thrust_accel_w,
        desired_rotation,
        desired_body_rate,
    ) = circular12_reference_from_state(position_w, velocity_w, cfg)

    error_rotation = torch.matmul(
        rotation_wb.transpose(-1, -2), desired_rotation
    )
    attitude_error = _so3_log(error_rotation)

    omega = float(cfg.target_speed_mps) / float(cfg.radius_m)
    desired_omega_w = torch.tensor(
        [0.0, 0.0, omega],
        dtype=position_w.dtype,
        device=position_w.device,
    ).expand_as(position_w)
    feedforward_rate_b = torch.einsum(
        "nji,nj->ni", rotation_wb, desired_omega_w
    )
    rate_command = feedforward_rate_b + float(cfg.attitude_kp) * attitude_error

    rate_limit = torch.tensor(
        cfg.body_rate_max_radps,
        dtype=position_w.dtype,
        device=position_w.device,
    ).view(1, 3)
    rate_command = torch.maximum(
        torch.minimum(rate_command, rate_limit), -rate_limit
    )

    collective_accel = torch.linalg.vector_norm(
        thrust_accel_w, dim=-1
    ).clamp(0.0, float(cfg.max_collective_accel_mps2))
    thrust_action = _inverse_collective_map(
        collective_accel,
        gravity_mps2=cfg.gravity_mps2,
        max_collective_accel_mps2=cfg.max_collective_accel_mps2,
    )
    normalized_rate = rate_command / rate_limit
    action = torch.cat(
        (thrust_action.unsqueeze(-1), normalized_rate), dim=-1
    ).clamp(-1.0, 1.0)

    return Circular12ExpertOutput(
        action=action,
        reference_position_w=reference_position,
        reference_velocity_w=reference_velocity,
        desired_rotation_wb=desired_rotation,
        desired_body_rate_b=desired_body_rate,
        attitude_error_rotvec_b=attitude_error,
        collective_accel_mps2=collective_accel,
        phase_rad=phase,
    )


def max_collective_accel_from_ctbr_cfg(action_cfg) -> float:
    """Mirror SwiftCTBRAction's static-thrust calculation from its config."""
    return (
        4.0
        * float(action_cfg.thrust_coef)
        * float(action_cfg.omega_max) ** 2
        / float(action_cfg.vehicle_mass_kg)
    )


def config_from_ctbr_action_cfg(
    action_cfg,
    *,
    target_speed_mps: float,
    radius_m: float = 12.0,
    height_m: float = 2.07,
) -> Circular12ExpertConfig:
    """Create an expert config that exactly matches the environment action scale."""
    return Circular12ExpertConfig(
        radius_m=float(radius_m),
        height_m=float(height_m),
        target_speed_mps=float(target_speed_mps),
        gravity_mps2=float(action_cfg.gravity_mps2),
        body_rate_max_radps=tuple(
            float(v) for v in action_cfg.body_rate_max_radps
        ),
        max_collective_accel_mps2=max_collective_accel_from_ctbr_cfg(action_cfg),
    )
