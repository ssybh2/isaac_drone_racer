from __future__ import annotations

import ast
from pathlib import Path


def _trace_fields() -> list[str]:
    script = (
        Path(__file__).resolve().parents[2]
        / "scripts"
        / "estimation"
        / "run_hybrid_openvins_evaluation.py"
    )
    tree = ast.parse(script.read_text(encoding="utf-8"))

    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        if any(isinstance(target, ast.Name) and target.id == "fields" for target in node.targets):
            return ast.literal_eval(node.value)
    raise AssertionError("hybrid evaluator fields list not found")


def test_raw_jump_vector_columns_match_put_vector_prefix() -> None:
    fields = _trace_fields()
    for axis in "xyz":
        assert f"raw_jump_d_{axis}" in fields


def test_position_residual_slew_diagnostics_are_in_trace_schema() -> None:
    fields = _trace_fields()
    assert "learned_position_injection_norm_m" in fields
    assert "learned_position_release_norm_m" in fields
    assert "learned_position_pending_norm_m" in fields
