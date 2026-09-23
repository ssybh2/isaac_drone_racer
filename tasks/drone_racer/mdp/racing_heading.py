"""Torch-only shaping used by the Circular-12 GT heading reward."""

import torch


def forward_velocity_heading_error(
    forward_w: torch.Tensor,
    velocity_w: torch.Tensor,
    min_speed_mps: float,
) -> torch.Tensor:
    """Return 1-cos(heading error) above the speed threshold, else zero."""
    speed = torch.linalg.vector_norm(velocity_w, dim=-1)
    direction = velocity_w / speed.clamp_min(1.0e-6).unsqueeze(-1)
    cosine = torch.sum(forward_w * direction, dim=-1).clamp(-1.0, 1.0)
    return torch.where(speed >= min_speed_mps, 1.0 - cosine, torch.zeros_like(speed))
