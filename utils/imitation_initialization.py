"""Initialize the existing skrl CTBR PPO actor from a Circular-12 BC checkpoint."""

from __future__ import annotations

import math
from pathlib import Path
from types import MethodType
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


def freeze_skrl_observation_normalization(agent: Any) -> dict[str, float]:
    """Freeze the BC observation coordinates during PPO fine-tuning.

    skrl's PPO intentionally calls RunningStandardScaler(..., train=True) on
    the first epoch of every rollout. That behavior is appropriate for PPO
    trained from scratch, but it would overwrite the semantic std floors that
    are part of the validated Circular-12 BC policy contract. Keep the loaded
    mean/variance in the checkpoint while turning the scaler's variance update
    into a no-op.
    """

    seen: set[int] = set()
    frozen = 0
    metadata: dict[str, float] = {}
    for name in ("_state_preprocessor", "_observation_preprocessor"):
        scaler = getattr(agent, name, None)
        if scaler is None or id(scaler) in seen:
            continue
        seen.add(id(scaler))

        running_mean = getattr(scaler, "running_mean", None)
        running_variance = getattr(scaler, "running_variance", None)
        current_count = getattr(scaler, "current_count", None)
        parallel_variance = getattr(scaler, "_parallel_variance", None)
        if (
            not torch.is_tensor(running_mean)
            or not torch.is_tensor(running_variance)
            or not callable(parallel_variance)
        ):
            raise RuntimeError(
                "imitation PPO requires a skrl RunningStandardScaler-like "
                "observation preprocessor that can be frozen"
            )

        def _frozen_parallel_variance(
            self: Any,
            input_mean: torch.Tensor,
            input_var: torch.Tensor,
            input_count: int,
        ) -> None:
            del self, input_mean, input_var, input_count
            return None

        scaler._parallel_variance = MethodType(
            _frozen_parallel_variance, scaler
        )
        setattr(scaler, "_imitation_updates_frozen", True)
        frozen += 1

        std = torch.sqrt(running_variance.detach().float())
        metadata.update(
            {
                "frozen_observation_scaler_std_min": float(std.min().item()),
                "frozen_observation_scaler_std_max": float(std.max().item()),
                "frozen_observation_scaler_sample_count": float(
                    current_count.item()
                    if torch.is_tensor(current_count)
                    else float("nan")
                ),
            }
        )

    if frozen == 0:
        raise RuntimeError(
            "imitation PPO did not expose an observation preprocessor to freeze"
        )
    metadata["frozen_observation_scaler_instances"] = float(frozen)
    return metadata


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
    # skrl names the head differently for shared and separate generated
    # models: shared.py uses policy_layer, while gaussian.py uses output_layer.
    head = getattr(policy, "policy_layer", None)
    if not isinstance(head, torch.nn.Linear):
        head = getattr(policy, "output_layer", None)
    if trunk is None or not isinstance(head, torch.nn.Linear):
        raise RuntimeError(
            "BC warm-start expects generated skrl net_container plus a "
            "Linear policy_layer/output_layer"
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
