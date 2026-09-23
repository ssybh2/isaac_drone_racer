"""Geometry and continuity behavior for the multi-gate camera reward."""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import torch


MODULE_PATH = Path(__file__).resolve().parents[2] / "tasks/drone_racer/mdp/racing_visibility.py"


def _module():
    spec = spec_from_file_location("racing_visibility", MODULE_PATH)
    module = module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_any_mapped_gate_can_supply_two_safe_corners():
    corners = torch.tensor(
        [
            [
                [[-0.2, -0.2, 2.0], [0.2, -0.2, 2.0], [0.2, 0.2, 2.0], [-0.2, 0.2, 2.0]],
                [[-0.2, -0.2, -2.0], [0.2, -0.2, -2.0], [0.2, 0.2, -2.0], [-0.2, 0.2, -2.0]],
            ],
            [
                [[4.0, -0.2, 2.0], [4.4, -0.2, 2.0], [4.4, 0.2, 2.0], [4.0, 0.2, 2.0]],
                [[-0.2, -0.2, 2.0], [0.2, -0.2, 2.0], [0.2, 0.2, 2.0], [-0.2, 0.2, 2.0]],
            ],
        ]
    )
    score, usable = _module().best_gate_visibility(
        corners, fx=293.2, fy=293.2, cx=128.0, cy=128.0,
        image_width=256, image_height=256, margin_px=16.0,
    )

    assert usable.tolist() == [True, True]
    assert torch.all(score > 0.5)


def test_visible_corners_on_image_edge_do_not_count_as_safe():
    corners = torch.tensor(
        [[[[0.83, -0.05, 2.0], [0.85, -0.05, 2.0], [0.85, 0.05, 2.0], [0.83, 0.05, 2.0]]]]
    )
    _, usable = _module().best_gate_visibility(
        corners, fx=293.2, fy=293.2, cx=128.0, cy=128.0,
        image_width=256, image_height=256, margin_px=16.0,
    )

    assert usable.tolist() == [False]


def test_blackout_cost_grows_on_capture_frames_and_resets_on_visibility():
    step = _module().advance_blackout
    count = torch.zeros(2, dtype=torch.long)
    missing = torch.tensor([False, False])
    count = step(count, missing, capture_due=True, max_frames=25)
    torch.testing.assert_close(count, torch.ones(2, dtype=torch.long))
    torch.testing.assert_close(step(count, missing, capture_due=False, max_frames=25), count)
    count = step(count, torch.tensor([True, False]), capture_due=True, max_frames=25)
    torch.testing.assert_close(count, torch.tensor([0, 2]))
