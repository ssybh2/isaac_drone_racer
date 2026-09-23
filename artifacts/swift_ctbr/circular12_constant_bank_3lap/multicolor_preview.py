"""Visual-only, corner-safe gate color panels for the isolated FPV preview."""

from dataclasses import dataclass


@dataclass(frozen=True)
class GatePanel:
    center_g: tuple[float, float, float]
    size_m: tuple[float, float, float]
    color_rgb: tuple[float, float, float]


# Distinct, saturated colors are legible in the small 256-pixel racing camera.
# The original blue/black texture and white corner checkerboards remain intact.
GATE_COLORS_RGB = (
    (0.08, 0.30, 0.95),  # 1 blue
    (0.95, 0.08, 0.08),  # 2 red
    (1.00, 0.86, 0.02),  # 3 yellow
    (0.04, 0.83, 0.12),  # 4 green
    (1.00, 0.38, 0.02),  # 5 orange
    (0.02, 0.82, 0.96),  # 6 cyan
    (0.58, 0.18, 0.95),  # 7 violet
    (0.96, 0.08, 0.63),  # 8 magenta
    (0.60, 0.96, 0.02),  # 9 lime
    (0.03, 0.82, 0.54),  # 10 turquoise
    (1.00, 0.41, 0.65),  # 11 pink
    (1.00, 0.65, 0.04),  # 12 amber
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
