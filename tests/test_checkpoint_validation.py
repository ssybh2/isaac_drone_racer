from pathlib import Path

import pytest
import torch

from utils.checkpoint_validation import (
    REQUIRED_SKRL_MODULES,
    assert_complete_stage0_checkpoint,
    required_skrl_modules,
)


BEST_STAGE0 = Path(
    "/home/donglei/isaac_projects/isaac_drone_racer/logs/skrl/drone_racer/"
    "2026-09-01_22-09-01_ppo_torch/checkpoints/best_agent.pt"
)


def scaler_state() -> dict[str, torch.Tensor]:
    return {
        "current_count": torch.tensor(12.0),
        "running_mean": torch.zeros(20),
        "running_variance": torch.ones(20),
    }


def complete_checkpoint() -> dict:
    return {
        "policy": {},
        "value": {},
        "optimizer": {},
        "state_preprocessor": scaler_state(),
        "value_preprocessor": scaler_state(),
    }


def test_required_module_set_includes_both_running_scalers():
    assert required_skrl_modules(complete_checkpoint()) == REQUIRED_SKRL_MODULES
    assert "state_preprocessor" in REQUIRED_SKRL_MODULES
    assert "value_preprocessor" in REQUIRED_SKRL_MODULES


@pytest.mark.parametrize("missing", ["state_preprocessor", "value_preprocessor", "policy", "value"])
def test_incomplete_checkpoint_names_missing_module(tmp_path: Path, missing: str):
    checkpoint = complete_checkpoint()
    checkpoint.pop(missing)
    path = tmp_path / "incomplete.pt"
    torch.save(checkpoint, path)

    with pytest.raises(ValueError, match=missing):
        assert_complete_stage0_checkpoint(path)


@pytest.mark.parametrize("field", ["current_count", "running_mean", "running_variance"])
def test_incomplete_scaler_names_missing_statistic(tmp_path: Path, field: str):
    checkpoint = complete_checkpoint()
    checkpoint["state_preprocessor"].pop(field)
    path = tmp_path / "incomplete_scaler.pt"
    torch.save(checkpoint, path)

    with pytest.raises(ValueError, match=field):
        assert_complete_stage0_checkpoint(path)


def test_real_stage0_best_checkpoint_is_complete():
    assert BEST_STAGE0.is_file()

    checkpoint = assert_complete_stage0_checkpoint(BEST_STAGE0)

    assert required_skrl_modules(checkpoint) == REQUIRED_SKRL_MODULES
