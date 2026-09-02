"""Read-only conversion from StateEstimate to the legacy policy contract."""

from __future__ import annotations

import torch

from .frame_math import rotate_world_to_body
from .state_estimate import StateEstimate


def policy_drone_state(estimate: StateEstimate) -> torch.Tensor:
    """Return Stage 0's 13 drone-state values without simulator access.

    The order is position W (3), scalar-first orientation W<-B (4), linear
    velocity B (3), and gyroscope angular velocity B (3).
    """
    linear_velocity_b = rotate_world_to_body(
        estimate.orientation_w_b, estimate.linear_velocity_w_b
    )
    return torch.cat(
        (
            estimate.position_w_b,
            estimate.orientation_w_b,
            linear_velocity_b,
            estimate.angular_velocity_b,
        ),
        dim=-1,
    )
