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


def _policy_output_layer(policy: Any) -> torch.nn.Linear:
    """Return the trainable actor mean layer for skrl generated models."""
    layer = getattr(policy, "policy_layer", None)
    if layer is None:
        layer = getattr(policy, "output_layer", None)
    if not isinstance(layer, torch.nn.Linear):
        raise ValueError(
            "expected skrl policy to expose a Linear policy_layer/output_layer"
        )
    return layer


def recalibrate_legacy_actor_output(
    policy: Any,
    raw_means: torch.Tensor,
    *,
    anchor_raw_mean: torch.Tensor | None = None,
    target_pretanh_abs: float = 1.25,
    reference_quantile: float = 0.75,
    min_reference_abs: float = 0.25,
    min_scale: float = 0.02,
    max_scale: float = 1.0,
) -> dict[str, Any]:
    """Rescale a legacy saturated actor into a trainable tanh operating region.

    The old policy learned under an environment-side hard clamp, so its final
    linear layer can emit values with magnitudes far above one. Applying tanh
    directly to such a checkpoint destroys the gradient. This migration keeps
    the hidden representation and per-action sign structure, but rescales each
    output row so a representative absolute raw mean lands near the requested
    pre-tanh target.

    raw_means must be collected with the loaded legacy checkpoint and its
    original observation preprocessor before this function is called.
    """
    if raw_means.ndim != 2:
        raise ValueError("raw_means must have shape [samples, actions]")
    if raw_means.shape[0] < 1:
        raise ValueError("raw_means must contain at least one sample")
    if target_pretanh_abs <= 0.0:
        raise ValueError("target_pretanh_abs must be positive")
    if not (0.0 < reference_quantile <= 1.0):
        raise ValueError("reference_quantile must be in (0, 1]")

    layer = _policy_output_layer(policy)
    if layer.out_features != raw_means.shape[1]:
        raise ValueError(
            "raw mean action dimension does not match actor output layer: "
            f"{raw_means.shape[1]} vs {layer.out_features}"
        )

    raw = raw_means.detach().to(device=layer.weight.device, dtype=layer.weight.dtype)
    reference_abs = torch.quantile(raw.abs(), reference_quantile, dim=0)
    if anchor_raw_mean is not None:
        anchor = anchor_raw_mean.detach().to(
            device=layer.weight.device, dtype=layer.weight.dtype
        ).reshape(-1, raw.shape[1])
        anchor_abs = anchor.abs().amax(dim=0)
        reference_abs = torch.maximum(reference_abs, anchor_abs)
    safe_reference = reference_abs.clamp_min(float(min_reference_abs))
    scale = (float(target_pretanh_abs) / safe_reference).clamp(
        min=float(min_scale), max=float(max_scale)
    )

    before_rms = float(torch.sqrt(torch.mean(raw.square())).item())
    before_dead_fraction = float(
        ((1.0 - torch.tanh(raw).square()) < 1.0e-3).float().mean().item()
    )

    with torch.no_grad():
        layer.weight.mul_(scale.view(-1, 1))
        if layer.bias is not None:
            layer.bias.mul_(scale)

    migrated_raw = raw * scale.view(1, -1)
    migrated_mean = torch.tanh(migrated_raw)
    migrated_derivative = 1.0 - migrated_mean.square()

    return {
        "actor_output_reference_quantile": float(reference_quantile),
        "actor_output_target_pretanh_abs": float(target_pretanh_abs),
        "actor_output_reference_abs_before": reference_abs.cpu().tolist(),
        "actor_output_row_scale": scale.cpu().tolist(),
        "actor_raw_rms_before": before_rms,
        "actor_raw_rms_after": float(torch.sqrt(torch.mean(migrated_raw.square())).item()),
        "actor_tanh_dead_fraction_before": before_dead_fraction,
        "actor_tanh_dead_fraction_after": float(
            (migrated_derivative < 1.0e-3).float().mean().item()
        ),
        "actor_tanh_derivative_mean_after": float(migrated_derivative.mean().item()),
        "actor_abs_mean_action_after": float(migrated_mean.abs().mean().item()),
    }


def cap_running_scaler_count(preprocessor: Any, max_count: float) -> dict[str, Any]:
    """Reduce only a RunningStandardScaler effective historical sample count.

    Mean and variance are preserved, so loading the checkpoint does not cause
    an instantaneous observation-coordinate jump. Lowering the pseudo-count
    lets data from the new curriculum update those statistics on a useful time
    scale instead of being dominated by hundreds of millions of old samples.
    """
    if max_count <= 0.0:
        raise ValueError("scaler max_count must be positive")
    count = getattr(preprocessor, "current_count", None)
    if count is None or not torch.is_tensor(count):
        return {}

    before = float(count.detach().item())
    after = min(before, float(max_count))
    with torch.no_grad():
        count.fill_(after)
    return {
        "scaler_count_before": before,
        "scaler_count_after": after,
    }


def reset_optimizer_state(optimizer: Any) -> dict[str, Any]:
    """Discard stale Adam moments after a deliberate policy reparameterization."""
    state_entries_before = len(optimizer.state)
    optimizer.state.clear()
    return {
        "optimizer_state_entries_before_reset": int(state_entries_before),
        "optimizer_state_entries_after_reset": int(len(optimizer.state)),
    }

def distill_legacy_clamped_actor_output(
    policy: Any,
    hidden_features: torch.Tensor,
    raw_means: torch.Tensor,
    *,
    anchor_hidden: torch.Tensor | None = None,
    anchor_raw_mean: torch.Tensor | None = None,
    action_limit: float = 0.95,
    ridge: float = 1.0e-4,
    anchor_repeats: int = 64,
) -> dict[str, Any]:
    """Fit the bounded tanh actor head to the legacy environment-executed action.

    The legacy policy was trained with an unbounded Gaussian mean while the
    environment hard-clamped actions to [-1, 1]. Directly rescaling output rows
    changes the motor behavior. Instead, this migration reconstructs the old
    executed action, clips only to action_limit to keep finite atanh targets,
    and refits only the final actor Linear layer by ridge regression.

    Hidden layers and observation coordinates are preserved exactly.
    """
    if hidden_features.ndim != 2 or raw_means.ndim != 2:
        raise ValueError("hidden_features/raw_means must both be rank-2")
    if hidden_features.shape[0] != raw_means.shape[0]:
        raise ValueError("hidden_features and raw_means must share sample count")
    if not (0.0 < action_limit < 1.0):
        raise ValueError("action_limit must be in (0, 1)")
    if ridge < 0.0:
        raise ValueError("ridge must be non-negative")
    if anchor_repeats < 0:
        raise ValueError("anchor_repeats must be non-negative")

    layer = _policy_output_layer(policy)
    if layer.in_features != hidden_features.shape[1]:
        raise ValueError(
            "hidden feature dimension does not match actor output layer: "
            f"{hidden_features.shape[1]} vs {layer.in_features}"
        )
    if layer.out_features != raw_means.shape[1]:
        raise ValueError(
            "raw action dimension does not match actor output layer: "
            f"{raw_means.shape[1]} vs {layer.out_features}"
        )

    device = layer.weight.device
    dtype = layer.weight.dtype
    hidden = hidden_features.detach().to(device=device, dtype=dtype)
    raw = raw_means.detach().to(device=device, dtype=dtype)

    old_executed = raw.clamp(-1.0, 1.0)
    bounded_target = old_executed.clamp(-float(action_limit), float(action_limit))
    pretanh_target = torch.atanh(bounded_target)

    fit_hidden = hidden
    fit_target = pretanh_target

    if anchor_hidden is not None and anchor_raw_mean is not None and anchor_repeats > 0:
        ah = anchor_hidden.detach().to(device=device, dtype=dtype).reshape(1, -1)
        ar = anchor_raw_mean.detach().to(device=device, dtype=dtype).reshape(1, -1)
        anchor_exec = ar.clamp(-1.0, 1.0)
        anchor_target = torch.atanh(
            anchor_exec.clamp(-float(action_limit), float(action_limit))
        )
        fit_hidden = torch.cat((fit_hidden, ah.expand(int(anchor_repeats), -1)), dim=0)
        fit_target = torch.cat((fit_target, anchor_target.expand(int(anchor_repeats), -1)), dim=0)

    x = fit_hidden.detach().cpu().to(torch.float64)
    y = fit_target.detach().cpu().to(torch.float64)
    ones = torch.ones((x.shape[0], 1), dtype=x.dtype)
    xa = torch.cat((x, ones), dim=1)
    gram = xa.T @ xa
    reg = torch.eye(gram.shape[0], dtype=gram.dtype) * float(ridge)
    reg[-1, -1] = 0.0
    beta = torch.linalg.solve(gram + reg, xa.T @ y)

    new_weight = beta[:-1].T.to(device=device, dtype=dtype)
    new_bias = beta[-1].to(device=device, dtype=dtype)

    with torch.no_grad():
        layer.weight.copy_(new_weight)
        if layer.bias is None:
            raise ValueError("behavior-preserving actor distillation requires a bias")
        layer.bias.copy_(new_bias)

    with torch.inference_mode():
        new_raw = layer(hidden)
        new_action = torch.tanh(new_raw)
        derivative = 1.0 - new_action.square()
        old_zero = None
        new_zero = None
        if anchor_hidden is not None and anchor_raw_mean is not None:
            ah = anchor_hidden.detach().to(device=device, dtype=dtype).reshape(1, -1)
            ar = anchor_raw_mean.detach().to(device=device, dtype=dtype).reshape(1, -1)
            old_zero = ar.clamp(-1.0, 1.0)
            new_zero = torch.tanh(layer(ah))

    behavior_error = new_action - old_executed
    sign_mask = old_executed.abs() > 0.1
    sign_agreement = torch.ones_like(old_executed, dtype=torch.bool)
    sign_agreement[sign_mask] = (
        torch.sign(new_action[sign_mask]) == torch.sign(old_executed[sign_mask])
    )

    metadata: dict[str, Any] = {
        "legacy_behavior_action_limit": float(action_limit),
        "legacy_behavior_ridge": float(ridge),
        "legacy_behavior_anchor_repeats": int(anchor_repeats),
        "legacy_raw_rms": float(torch.sqrt(torch.mean(raw.square())).item()),
        "legacy_executed_abs_mean": float(old_executed.abs().mean().item()),
        "legacy_executed_saturation_fraction": float((old_executed.abs() > 0.999).float().mean().item()),
        "distilled_action_abs_mean": float(new_action.abs().mean().item()),
        "distilled_behavior_mae": float(behavior_error.abs().mean().item()),
        "distilled_behavior_rmse": float(torch.sqrt(torch.mean(behavior_error.square())).item()),
        "distilled_sign_agreement": float(sign_agreement.float().mean().item()),
        "distilled_tanh_dead_fraction": float((derivative < 1.0e-3).float().mean().item()),
        "distilled_tanh_derivative_mean": float(derivative.mean().item()),
    }

    if old_zero is not None and new_zero is not None:
        metadata["legacy_zero_input_executed_action"] = old_zero.detach().cpu().reshape(-1).tolist()
        metadata["distilled_zero_input_action"] = new_zero.detach().cpu().reshape(-1).tolist()
        metadata["distilled_zero_input_mae"] = float((new_zero - old_zero).abs().mean().item())

    return metadata


def reset_optimizer_parameter_state(
    optimizer: Any,
    parameters: list[torch.nn.Parameter],
) -> dict[str, Any]:
    """Clear Adam moments only for deliberately reparameterized tensors."""
    before = len(optimizer.state)
    cleared = 0
    for parameter in parameters:
        if parameter in optimizer.state:
            del optimizer.state[parameter]
            cleared += 1
    return {
        "optimizer_state_entries_before_selective_reset": int(before),
        "optimizer_parameter_states_cleared": int(cleared),
        "optimizer_state_entries_after_selective_reset": int(len(optimizer.state)),
    }
