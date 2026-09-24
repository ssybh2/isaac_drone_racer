"""Behavior tests for DAgger's gate-frame clearance intervention."""

import ast
from pathlib import Path

import isaaclab.utils.math as math_utils
import torch


SOURCE = Path(__file__).resolve().parents[2] / "scripts/imitation/collect_circular12_dagger_dataset.py"


def _clearance_metrics(position_w, gate_pose_w):
    tree = ast.parse(SOURCE.read_text())
    function = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_gate_clearance_metrics"
    )
    namespace = {"torch": torch, "math_utils": math_utils}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(SOURCE), "exec"), namespace)
    return namespace["_gate_clearance_metrics"](
        position_w, gate_pose_w, plane_distance_m=1.5, center_limit_m=0.4
    )


def test_gate_clearance_triggers_only_near_plane_and_outside_center():
    gate = torch.tensor([[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]])
    positions = torch.tensor([
        [1.5, 0.401, 0.0],
        [1.501, 0.8, 0.0],
        [0.0, 0.4, 0.4],
        [0.0, 0.0, -0.401],
    ])
    plane, lateral, vertical, unsafe = _clearance_metrics(
        positions, gate.expand(4, -1)
    )

    torch.testing.assert_close(plane, torch.tensor([1.5, 1.501, 0.0, 0.0]))
    torch.testing.assert_close(lateral, torch.tensor([0.401, 0.8, 0.4, 0.0]))
    torch.testing.assert_close(vertical, torch.tensor([0.0, 0.0, 0.4, 0.401]), atol=1e-6, rtol=0)
    assert unsafe.tolist() == [True, False, False, True]


def test_gate_clearance_uses_gate_rotation_not_world_axes():
    half_sqrt = 2 ** -0.5
    gate = torch.tensor([[0.0, 0.0, 2.07, half_sqrt, 0.0, 0.0, half_sqrt]])
    positions = torch.tensor([
        [0.45, 0.0, 2.07],
        [0.0, 0.45, 2.07],
    ])
    plane, lateral, vertical, unsafe = _clearance_metrics(
        positions, gate.expand(2, -1)
    )

    torch.testing.assert_close(plane, torch.tensor([0.0, 0.45]), atol=1e-6, rtol=0)
    torch.testing.assert_close(lateral, torch.tensor([0.45, 0.0]), atol=1e-6, rtol=0)
    torch.testing.assert_close(vertical, torch.zeros(2))
    assert unsafe.tolist() == [True, False]
