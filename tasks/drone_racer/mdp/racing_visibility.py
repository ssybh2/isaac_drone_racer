"""Torch-only image and temporal metrics for multi-gate racing visibility."""

import torch


def best_gate_visibility(
    corners_c: torch.Tensor,
    *,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    image_width: int,
    image_height: int,
    margin_px: float,
    min_depth_m: float = 0.05,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return best mapped-gate image quality and whether any gate is usable.

    ``corners_c`` has shape [environment, gate, four corners, xyz]. A usable
    gate has two corners inside the safety margin in the same camera frame.
    """
    z = corners_c[..., 2]
    front = z > min_depth_m
    safe_z = z.clamp_min(min_depth_m)
    u = fx * corners_c[..., 0] / safe_z + cx
    v = fy * corners_c[..., 1] / safe_z + cy

    inside = front & (u >= 0) & (u < image_width) & (v >= 0) & (v < image_height)
    safe = (
        front
        & (u >= margin_px)
        & (u < image_width - margin_px)
        & (v >= margin_px)
        & (v < image_height - margin_px)
    )
    usable = (safe.sum(dim=-1) >= 2).any(dim=-1)

    coverage = inside.float().mean(dim=-1)
    border_distance = torch.minimum(
        torch.minimum(u, image_width - 1 - u),
        torch.minimum(v, image_height - 1 - v),
    )
    margin = (border_distance / margin_px).clamp(0.0, 1.0)
    margin_score = (margin * inside).mean(dim=-1)

    center = corners_c.mean(dim=-2)
    center_z = center[..., 2]
    center_safe_z = center_z.clamp_min(min_depth_m)
    norm_u = fx * center[..., 0] / center_safe_z / (image_width / 2)
    norm_v = fy * center[..., 1] / center_safe_z / (image_height / 2)
    center_score = torch.exp(-0.5 * (norm_u.square() + norm_v.square()))
    center_score = center_score * (center_z > min_depth_m)

    per_gate_score = 0.35 * center_score + 0.35 * coverage + 0.30 * margin_score
    return per_gate_score.max(dim=-1).values, usable


def advance_blackout(
    previous_frames: torch.Tensor,
    usable: torch.Tensor,
    *,
    capture_due: bool,
    max_frames: int,
) -> torch.Tensor:
    """Count consecutive unusable camera frames independently per environment."""
    if not capture_due:
        return previous_frames
    return torch.where(
        usable,
        torch.zeros_like(previous_frames),
        (previous_frames + 1).clamp_max(max_frames),
    )
