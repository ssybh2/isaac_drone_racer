"""Observation terms backed exclusively by the Stage 1 estimator contract."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from estimation.policy_adapter import policy_drone_state

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def estimated_drone_state(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Return the 13-value drone state without reading Isaac asset state."""
    estimate = getattr(env, "stage1_state_estimate", None)
    if estimate is None:
        # ObservationManager probes term dimensions during base construction,
        # before the Stage 1 provider pipeline can be initialized.
        return torch.zeros(env.num_envs, 13, device=env.device)
    return policy_drone_state(estimate)
