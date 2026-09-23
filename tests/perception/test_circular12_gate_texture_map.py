"""Static contract for Circular-12 color-coded gate textures."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
MAP_PATH = ROOT / "assets/gate/textures/circular12_gate_texture_map.json"


def test_circular12_texture_map_has_twelve_unique_colors_and_positions():
    data = json.loads(MAP_PATH.read_text(encoding="utf-8"))
    gates = data["gates"]

    assert len(gates) == 12
    assert [gate["gate_id"] for gate in gates] == list(range(1, 13))
    assert len({tuple(gate["rgb"]) for gate in gates}) == 12
    assert len({gate["hex"] for gate in gates}) == 12
    assert len({tuple(gate["position_m"]) for gate in gates}) == 12
    assert all(gate["position_m"][2] == 1.0 for gate in gates)


def test_circular12_texture_map_matches_track_geometry():
    data = json.loads(MAP_PATH.read_text(encoding="utf-8"))
    expected = {
        1:  ((0.0, 0.0, 1.0), 0.0),
        2:  ((6.0, 1.60769515, 1.0), 30.0),
        3:  ((10.39230485, 6.0, 1.0), 60.0),
        4:  ((12.0, 12.0, 1.0), 90.0),
        5:  ((10.39230485, 18.0, 1.0), 120.0),
        6:  ((6.0, 22.39230485, 1.0), 150.0),
        7:  ((0.0, 24.0, 1.0), 180.0),
        8:  ((-6.0, 22.39230485, 1.0), -150.0),
        9:  ((-10.39230485, 18.0, 1.0), -120.0),
        10: ((-12.0, 12.0, 1.0), -90.0),
        11: ((-10.39230485, 6.0, 1.0), -60.0),
        12: ((-6.0, 1.60769515, 1.0), -30.0),
    }

    for gate in data["gates"]:
        position, yaw_deg = expected[gate["gate_id"]]
        assert tuple(gate["position_m"]) == position
        assert gate["yaw_deg"] == yaw_deg


def test_circular12_adjacent_gate_hues_are_widely_separated():
    data = json.loads(MAP_PATH.read_text(encoding="utf-8"))
    hues = [float(gate["hue_deg"]) for gate in data["gates"]]

    separations = []
    for index, hue_a in enumerate(hues):
        hue_b = hues[(index + 1) % len(hues)]
        raw = abs(hue_a - hue_b) % 360.0
        separations.append(min(raw, 360.0 - raw))

    assert data["adjacent_hue_separation_deg"] == 150
    assert separations == [150.0] * 12


def test_circular12_palette_names_match_generated_texture_names():
    data = json.loads(MAP_PATH.read_text(encoding="utf-8"))
    for gate in data["gates"]:
        expected = f"bitmap_gate_{int(gate['gate_id']):02d}_{gate['color_name']}.png"
        assert Path(gate["texture"]).name == expected
