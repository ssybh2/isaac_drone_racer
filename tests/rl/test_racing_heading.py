"""Behavior checks for the GT racing heading penalty."""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import torch


MODULE_PATH = Path(__file__).resolve().parents[2] / "tasks/drone_racer/mdp/racing_heading.py"


def _heading_error(forward: torch.Tensor, velocity: torch.Tensor) -> torch.Tensor:
    spec = spec_from_file_location("racing_heading", MODULE_PATH)
    module = module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module.forward_velocity_heading_error(forward, velocity, min_speed_mps=5.0)


def test_forward_velocity_heading_error_penalizes_backward_flight():
    forward = torch.tensor([[1.0, 0.0, 0.0]] * 3)
    velocity = torch.tensor([[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [-10.0, 0.0, 0.0]])

    torch.testing.assert_close(_heading_error(forward, velocity), torch.tensor([0.0, 1.0, 2.0]))


def test_forward_velocity_heading_error_ignores_low_speed_launch():
    forward = torch.tensor([[1.0, 0.0, 0.0]] * 2)
    velocity = torch.tensor([[-4.99, 0.0, 0.0], [0.0, 0.0, 0.0]])

    torch.testing.assert_close(_heading_error(forward, velocity), torch.zeros(2))
