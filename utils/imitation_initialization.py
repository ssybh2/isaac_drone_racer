"""Initialize the existing skrl CTBR PPO actor from a Circular-12 BC checkpoint."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import torch

from imitation.bc_policy import Circular12BCPolicy


def _set_running_standard_scaler(
    scaler: Any,
    mean: torch.Tensor,
    std: torch.Tensor,
    *,
    sample_count: float,
) -> dict[str, float]:
    running_mean = getattr(scaler, "running_mean", None)
    running_variance = getattr(scaler, "running_variance", None)
    if running_variance is None:
        running_variance = getattr(scaler, "running_var", None)
    if not torch.is_tensor(running_mean) or not torch.is_tensor(running_variance):
        raise RuntimeError(
            "BC warm-start requires skrl RunningStandardScaler buffers "
            "running_mean and running_variance/running_var"
        )

    device, dtype = running_mean.device, running_mean.dtype
    mean = mean.to(device=device, dtype=dtype).reshape_as(running_mean)
    variance = std.to(device=device, dtype=dtype).square().reshape_as(
        running_variance
    )
    with torch.no_grad():
        running_mean.copy_(mean)
        running_variance.copy_(variance.clamp_min(1.0e-8))
        current_count = getattr(scaler, "current_count", None)
        if torch.is_tensor(current_count):
            current_count.fill_(max(float(sample_count), 1.0))
    return {
        "observation_scaler_sample_count": float(sample_count),
        "observation_scaler_std_min": float(std.min().item()),
        "observation_scaler_std_max": float(std.max().item()),
    }


def initialize_skrl_policy_from_bc(
    agent: Any,
    checkpoint: str | Path,
    *,
    action_std: float = 0.05,
) -> dict[str, Any]:
    """Copy BC trunk/head and normalization into the existing 256^3 ELU actor."""
    if action_std <= 0.0:
        raise ValueError("action_std must be positive")

    model, bc_metadata = Circular12BCPolicy.load(
        checkpoint, map_location=agent.policy.device
    )
    model = model.to(agent.policy.device).eval()
    if model.cfg.standardized_observation_clip is not None:
        raise RuntimeError(
            "BC checkpoint uses standardized-observation clipping, but the "
            "current skrl warm-start path does not yet reproduce that input "
            "contract. Keep PPO disabled until equivalent clipping is added."
        )
    bc_layers = model.linear_layers()
    if len(bc_layers) != 4:
        raise RuntimeError("Circular12 BC checkpoint must contain four Linear layers")

    policy = agent.policy
    trunk = getattr(policy, "net_container", None)
    head = getattr(policy, "policy_layer", None)
    if trunk is None or not isinstance(head, torch.nn.Linear):
        raise RuntimeError(
            "BC warm-start expects generated skrl net_container + policy_layer"
        )
    trunk_linears = [
        module for module in trunk.modules()
        if isinstance(module, torch.nn.Linear)
    ]
    if len(trunk_linears) != 3:
        raise RuntimeError(
            f"expected 3 actor trunk Linear layers, got {len(trunk_linears)}"
        )

    destination = [*trunk_linears, head]
    source_shapes = [tuple(layer.weight.shape) for layer in bc_layers]
    destination_shapes = [tuple(layer.weight.shape) for layer in destination]
    if source_shapes != destination_shapes:
        raise RuntimeError(
            f"BC/skrl actor topology mismatch: source={source_shapes} "
            f"destination={destination_shapes}"
        )

    with torch.no_grad():
        for source, target in zip(bc_layers, destination):
            target.weight.copy_(source.weight.to(target.weight))
            target.bias.copy_(source.bias.to(target.bias))

    seen: set[int] = set()
    scaler_metadata: dict[str, float] = {}
    sample_count = float(bc_metadata.get("samples", 1.0))
    for name in ("_state_preprocessor", "_observation_preprocessor"):
        scaler = getattr(agent, name, None)
        if scaler is None or id(scaler) in seen:
            continue
        seen.add(id(scaler))
        scaler_metadata.update(
            _set_running_standard_scaler(
                scaler,
                model.observation_mean.detach(),
                model.observation_std.detach(),
                sample_count=sample_count,
            )
        )

    log_std = getattr(policy, "log_std_parameter", None)
    if not isinstance(log_std, torch.nn.Parameter):
        raise RuntimeError("skrl Gaussian actor does not expose log_std_parameter")
    with torch.no_grad():
        log_std.fill_(math.log(float(action_std)))

    return {
        "bc_checkpoint": str(Path(checkpoint).expanduser().resolve()),
        "bc_dataset_samples": int(bc_metadata.get("samples", 0)),
        "bc_best_epoch": int(bc_metadata.get("best_epoch", -1)),
        "bc_test_mae": float(
            bc_metadata.get("test", {}).get("mae", float("nan"))
        ),
        "bc_test_rmse": float(
            bc_metadata.get("test", {}).get("rmse", float("nan"))
        ),
        "initialized_action_std": float(action_std),
        **scaler_metadata,
    }
