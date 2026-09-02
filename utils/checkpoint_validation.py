"""Safety checks for complete Stage 0 -> Stage 1 SKRL warm starts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


REQUIRED_SKRL_MODULES = {
    "policy",
    "value",
    "optimizer",
    "state_preprocessor",
    "value_preprocessor",
}
REQUIRED_SCALER_STATE = {"current_count", "running_mean", "running_variance"}


def required_skrl_modules(checkpoint: dict[str, Any]) -> set[str]:
    """Return the complete-agent module names found in a checkpoint."""
    return REQUIRED_SKRL_MODULES.intersection(checkpoint)


def assert_complete_stage0_checkpoint(path: str | Path) -> dict[str, Any]:
    """Load a trusted local checkpoint and reject incomplete warm-start state."""
    checkpoint_path = Path(path).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Stage 0 checkpoint does not exist: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict):
        raise ValueError("Stage 0 checkpoint root must be a dictionary")

    missing_modules = REQUIRED_SKRL_MODULES.difference(checkpoint)
    if missing_modules:
        names = ", ".join(sorted(missing_modules))
        raise ValueError(f"Stage 0 checkpoint is missing required module(s): {names}")

    for scaler_name in ("state_preprocessor", "value_preprocessor"):
        scaler_state = checkpoint[scaler_name]
        if not isinstance(scaler_state, dict):
            raise ValueError(f"{scaler_name} state must be a dictionary")
        missing_state = REQUIRED_SCALER_STATE.difference(scaler_state)
        if missing_state:
            names = ", ".join(sorted(missing_state))
            raise ValueError(f"{scaler_name} is missing statistic(s): {names}")
    return checkpoint
