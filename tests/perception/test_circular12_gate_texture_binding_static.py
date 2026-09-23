"""Static integration checks for the real Circular-12 gate texture binding."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_track_generator_uses_per_gate_circular12_usd_variants():
    text = (ROOT / "tasks/drone_racer/track_generator.py").read_text(encoding="utf-8")
    assert "ensure_circular12_gate_usd_variants" in text
    assert "track_config is CIRCULAR_12_GATE_TRACK_CONFIG" in text
    assert "gate_usd_paths[str(gate_id)]" in text


def test_preview_uses_real_track_not_multicolor_overlay_geometry():
    text = (
        ROOT / "scripts/perception/preview_circular12_gate_colors.py"
    ).read_text(encoding="utf-8")
    assert "generate_track(CIRCULAR_12_GATE_TRACK_CONFIG)" in text
    assert "validate_circular12_gate_usd_variants" in text
    assert "gate_panels" not in text
    assert "UsdGeom.Cube" not in text


def test_generated_variant_naming_is_ignored_by_git():
    ignore = (ROOT / ".gitignore").read_text(encoding="utf-8")
    assert "assets/gate/gate_circular12_*.usd" in ignore


def test_gate_variant_builder_reloads_cached_source_layer():
    text = (
        ROOT / "tasks/drone_racer/gate_texture_variants.py"
    ).read_text(encoding="utf-8")
    assert "stage.Reload()" in text
    assert "basename.startswith(\"bitmap_gate_\")" in text
