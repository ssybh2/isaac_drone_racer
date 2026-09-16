from __future__ import annotations

import ast
from pathlib import Path


def _script_tree() -> ast.Module:
    script = (
        Path(__file__).resolve().parents[2]
        / "scripts"
        / "estimation"
        / "run_hybrid_openvins_evaluation.py"
    )
    return ast.parse(script.read_text(encoding="utf-8"))


def _trace_fields() -> list[str]:
    for node in ast.walk(_script_tree()):
        if not isinstance(node, ast.Assign):
            continue
        if any(isinstance(target, ast.Name) and target.id == "fields" for target in node.targets):
            return ast.literal_eval(node.value)
    raise AssertionError("hybrid evaluator fields list not found")


def _profile_choices() -> tuple[str, ...]:
    for node in ast.walk(_script_tree()):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        if node.func.attr != "add_argument" or not node.args:
            continue
        try:
            first_arg = ast.literal_eval(node.args[0])
        except (ValueError, TypeError):
            continue
        if first_arg != "--profile":
            continue
        for keyword in node.keywords:
            if keyword.arg == "choices":
                return tuple(ast.literal_eval(keyword.value))
    raise AssertionError("--profile choices not found")


def _target_xy_source() -> str:
    for node in _script_tree().body:
        if isinstance(node, ast.FunctionDef) and node.name == "_target_xy":
            return ast.unparse(node)
    raise AssertionError("_target_xy not found")


def test_raw_jump_vector_columns_match_put_vector_prefix() -> None:
    fields = _trace_fields()
    for axis in "xyz":
        assert f"raw_jump_d_{axis}" in fields


def test_position_residual_slew_diagnostics_are_in_trace_schema() -> None:
    fields = _trace_fields()
    assert "learned_position_injection_norm_m" in fields
    assert "learned_position_release_norm_m" in fields
    assert "learned_position_pending_norm_m" in fields


def test_circle_profile_is_supported_by_hybrid_evaluator() -> None:
    assert "circle" in _profile_choices()


def test_circle_profile_has_a_dedicated_circular_target_branch() -> None:
    source = _target_xy_source()
    assert "args_cli.profile == 'circle'" in source
    assert "np.cos" in source
