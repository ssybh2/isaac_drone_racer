"""Controlled optimizer and exploration overrides for checkpoint continuation."""

from __future__ import annotations

import math
from typing import Any

import torch


def apply_post_load_training_overrides(
    agent: Any,
    *,
    learning_rate: float | None = None,
    max_action_std: float | None = None,
) -> dict[str, Any]:
    """Apply explicit continuation controls without discarding checkpoint state.

    The optimizer object and its moment estimates remain intact. Only the
    optimizer's current learning rate and/or the policy's Gaussian log-standard
    deviation parameter are changed.
    """
    if learning_rate is not None and learning_rate <= 0.0:
        raise ValueError("post-load learning rate must be positive")
    if max_action_std is not None and max_action_std <= 0.0:
        raise ValueError("post-load maximum action standard deviation must be positive")

    metadata: dict[str, Any] = {}
    if learning_rate is not None:
        optimizer = agent.optimizer
        previous = [float(group["lr"]) for group in optimizer.param_groups]
        for group in optimizer.param_groups:
            group["lr"] = learning_rate

        scheduler = getattr(agent, "scheduler", None)
        if scheduler is not None:
            if hasattr(scheduler, "base_lrs"):
                scheduler.base_lrs = [learning_rate] * len(optimizer.param_groups)
            if hasattr(scheduler, "_last_lr"):
                scheduler._last_lr = [learning_rate] * len(optimizer.param_groups)
        metadata["optimizer_learning_rate_before"] = previous
        metadata["optimizer_learning_rate_after"] = [learning_rate] * len(previous)

    if max_action_std is not None:
        log_std = getattr(agent.policy, "log_std_parameter", None)
        if log_std is None:
            raise ValueError("policy does not expose log_std_parameter")
        before = log_std.detach().exp().cpu().tolist()
        with torch.no_grad():
            log_std.clamp_(max=math.log(max_action_std))
        metadata["policy_action_std_before"] = before
        metadata["policy_action_std_after"] = log_std.detach().exp().cpu().tolist()

    return metadata
