"""Regression checks for texture paths in generated Circular-12 USD gates."""

import importlib.util
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = ROOT / "tasks/drone_racer/gate_texture_variants.py"
SPEC = importlib.util.spec_from_file_location("gate_texture_variants", MODULE_PATH)
gate_texture_variants = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(gate_texture_variants)


def test_generated_texture_asset_path_is_absolute_and_exists():
    texture = (
        gate_texture_variants.REPO_ROOT
        / "assets/gate/textures/circular12/bitmap_gate_02_red.png"
    )

    asset_path = gate_texture_variants._runtime_asset_path(texture)

    assert Path(asset_path).is_absolute()
    assert Path(asset_path).is_file()
    assert asset_path == str(texture.resolve())


def test_generated_gate_reference_path_is_absolute_and_exists():
    source_gate = gate_texture_variants.SOURCE_GATE_USD

    asset_path = gate_texture_variants._runtime_asset_path(source_gate)

    assert Path(asset_path).is_absolute()
    assert Path(asset_path).is_file()
    assert asset_path == str(source_gate.resolve())


def test_variant_rebuilds_when_builder_code_is_newer(tmp_path, monkeypatch):
    builder_mtime = MODULE_PATH.stat().st_mtime_ns
    old_mtime = builder_mtime - 2_000_000_000
    variant_mtime = builder_mtime - 1_000_000_000
    source = tmp_path / "gate.usd"
    mapping = tmp_path / "map.json"
    texture = tmp_path / "texture.png"
    variant = tmp_path / "variant.usd"
    for path in (source, mapping, texture, variant):
        path.touch()
        os.utime(path, ns=(old_mtime, old_mtime))
    os.utime(variant, ns=(variant_mtime, variant_mtime))
    monkeypatch.setattr(gate_texture_variants, "SOURCE_GATE_USD", source)
    monkeypatch.setattr(gate_texture_variants, "TEXTURE_MAP", mapping)

    assert not gate_texture_variants._variant_is_fresh(variant, texture)
