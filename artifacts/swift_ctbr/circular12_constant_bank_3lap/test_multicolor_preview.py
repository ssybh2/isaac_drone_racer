"""Contracts for the isolated multicolor-gate FPV preview."""

from multicolor_preview import gate_panels


def test_twelve_gates_have_distinct_visible_panel_colors():
    panels = gate_panels()
    assert set(panels) == set(range(1, 13))
    colors = [panels[index][0].color_rgb for index in range(1, 13)]
    assert len(set(colors)) == 12
    assert all(max(color) >= 0.8 for color in colors)
    assert all(min(color) <= 0.42 for color in colors)


def test_panels_cover_only_uprights_away_from_calibrated_corners():
    all_panels = gate_panels()
    assert len(all_panels) == 12
    for panels in all_panels.values():
        assert len(panels) == 4
        assert len({panel.color_rgb for panel in panels}) == 1
        for panel in panels:
            x, y, z = panel.center_g
            sx, sy, sz = panel.size_m
            assert abs(x) - sx / 2 > 0.06614135
            assert 0.762 < abs(y) - sy / 2
            assert abs(y) + sy / 2 < 1.0668
            assert z - sz / 2 > 0.45
            assert z + sz / 2 < 1.68
