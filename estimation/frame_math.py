"""Batched scalar-first quaternion and rigid-transform operations."""

from __future__ import annotations

import torch


def normalize_quaternion(quaternion: torch.Tensor) -> torch.Tensor:
    """Normalize scalar-first quaternions without leaving their device."""
    epsilon = torch.finfo(quaternion.dtype).eps
    return quaternion / torch.linalg.vector_norm(quaternion, dim=-1, keepdim=True).clamp_min(epsilon)


def _quaternion_multiply_raw(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    lw, lx, ly, lz = left.unbind(dim=-1)
    rw, rx, ry, rz = right.unbind(dim=-1)
    return torch.stack(
        (
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ),
        dim=-1,
    )


def quaternion_multiply(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    """Compose two scalar-first rotations and normalize the result."""
    return normalize_quaternion(_quaternion_multiply_raw(left, right))


def quaternion_conjugate(quaternion: torch.Tensor) -> torch.Tensor:
    result = quaternion.clone()
    result[..., 1:] = -result[..., 1:]
    return result


def quaternion_from_rotation_vector(rotation_vector: torch.Tensor) -> torch.Tensor:
    """Convert axis-angle rotation vectors to scalar-first quaternions."""
    angle = torch.linalg.vector_norm(rotation_vector, dim=-1, keepdim=True)
    half_angle = 0.5 * angle
    epsilon = torch.finfo(rotation_vector.dtype).eps
    scale = torch.where(
        angle > epsilon,
        torch.sin(half_angle) / angle.clamp_min(epsilon),
        0.5 - angle.square() / 48.0,
    )
    return normalize_quaternion(torch.cat((torch.cos(half_angle), rotation_vector * scale), dim=-1))


def rotate_vector(quaternion: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
    """Rotate a vector with an active scalar-first quaternion."""
    q = normalize_quaternion(quaternion)
    xyz = q[..., 1:]
    uv = torch.cross(xyz, vector, dim=-1)
    uuv = torch.cross(xyz, uv, dim=-1)
    return vector + 2.0 * (q[..., :1] * uv + uuv)


def compose_transform_w_b(
    position_w_v: torch.Tensor,
    orientation_w_v: torch.Tensor,
    position_v_b: torch.Tensor,
    orientation_v_b: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compose ``T_WB = T_WV * T_VB``."""
    position_w_b = position_w_v + rotate_vector(orientation_w_v, position_v_b)
    orientation_w_b = quaternion_multiply(orientation_w_v, orientation_v_b)
    return position_w_b, orientation_w_b


def rotate_world_to_body(orientation_w_b: torch.Tensor, vector_w: torch.Tensor) -> torch.Tensor:
    """Express a world-frame vector in body frame B."""
    return rotate_vector(quaternion_conjugate(normalize_quaternion(orientation_w_b)), vector_w)
