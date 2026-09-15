from __future__ import annotations

import ast
from pathlib import Path


def test_raw_jump_vector_columns_match_put_vector_prefix() -> None:
    script = (
        Path(__file__).resolve().parents[2]
        / "scripts"
        / "estimation"
        / "run_hybrid_openvins_evaluation.py"
    )
    tree = ast.parse(script.read_text(encoding="utf-8"))

    fields = None
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        if any(isinstance(target, ast.Name) and target.id == "fields" for target in node.targets):
            fields = ast.literal_eval(node.value)
            break

    assert fields is not None
    for axis in "xyz":
        assert f"raw_jump_d_{axis}" in fields
