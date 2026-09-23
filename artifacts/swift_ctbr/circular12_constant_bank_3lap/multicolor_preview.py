"""Visual-only, corner-safe gate color panels for the isolated FPV preview."""

from dataclasses import dataclass


@dataclass(frozen=True)
class GatePanel:
    center_g: tuple[float, float, float]
    size_m: tuple[float, float, float]
    color_rgb: tuple[float, float, float]


# High-saturation colors ordered so every neighbouring Circular-12 gate differs
# by 150 degrees in hue. This visual-only preview mirrors the real texture map.
GATE_COLORS_RGB = (
    (1.00, 0.00, 0.00),  # 1 red
    (0.00, 1.00, 0.50),  # 2 spring green
    (1.00, 0.00, 1.00),  # 3 magenta
    (0.50, 1.00, 0.00),  # 4 chartreuse
    (0.00, 0.00, 1.00),  # 5 blue
    (1.00, 0.50, 0.00),  # 6 orange
    (0.00, 1.00, 1.00),  # 7 cyan
    (1.00, 0.00, 0.50),  # 8 rose
    (0.00, 1.00, 0.00),  # 9 green
    (0.50, 0.00, 1.00),  # 10 violet
    (1.00, 1.00, 0.00),  # 11 yellow
    (0.00, 0.50, 1.00),  # 12 azure
)


def gate_panels():
    """Return four front/back upright panels per gate, avoiding corner markers."""
    panels = {}
    for gate_index, color in enumerate(GATE_COLORS_RGB, start=1):
        panels[gate_index] = tuple(
            GatePanel(
                center_g=(x_side * 0.069, y_side * 0.9144, 1.0668),
                size_m=(0.003, 0.28, 1.10),
                color_rgb=color,
            )
            for x_side in (-1, 1)
            for y_side in (-1, 1)
        )
    return panels
