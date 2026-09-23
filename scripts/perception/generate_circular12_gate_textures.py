"""Generate 12 color-coded gate textures from assets/gate/textures/bitmap.png.

Only pixels belonging to the original blue paint are hue-shifted. Black/white
checkerboards, alpha, shading and value are preserved.

The palette is deliberately ordered for the Circular-12 topology: every pair
of neighbouring gates, including Gate 12 -> Gate 1, differs by 150 degrees in
HSV hue. Saturation is floored in the recoloured paint so the 12 identities
remain vivid in the small racing camera and under motion blur.
"""

from __future__ import annotations

import argparse
import colorsys
import json
from pathlib import Path

import numpy as np
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = REPO_ROOT / "assets/gate/textures/bitmap.png"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "assets/gate/textures/circular12"
DEFAULT_MAP = REPO_ROOT / "assets/gate/textures/circular12_gate_texture_map.json"

# High-contrast cyclic ordering. Hues are chosen from a 30-degree wheel and
# permuted so every adjacent pair on the circular track is exactly 150 degrees
# apart in hue: [0, 150, 300, 90, 240, 30, 180, 330, 120, 270, 60, 210].
# Keep this palette synchronized with
# artifacts/swift_ctbr/circular12_constant_bank_3lap/multicolor_preview.py.
GATE_PALETTE = (
    ("red",          (255,   0,   0)),
    ("spring_green", (  0, 255, 128)),
    ("magenta",      (255,   0, 255)),
    ("chartreuse",   (128, 255,   0)),
    ("blue",         (  0,   0, 255)),
    ("orange",       (255, 128,   0)),
    ("cyan",         (  0, 255, 255)),
    ("rose",         (255,   0, 128)),
    ("green",        (  0, 255,   0)),
    ("violet",       (128,   0, 255)),
    ("yellow",       (255, 255,   0)),
    ("azure",        (  0, 128, 255)),
)

SATURATION_FLOOR_U8 = 235


def _target_hue_u8(rgb: tuple[int, int, int]) -> int:
    r, g, b = (channel / 255.0 for channel in rgb)
    hue, _, _ = colorsys.rgb_to_hsv(r, g, b)
    return int(round(hue * 255.0)) % 256


def _blue_mask(rgb: np.ndarray) -> np.ndarray:
    """Select saturated blue paint while leaving black/white markings alone."""
    r = rgb[..., 0].astype(np.int16)
    g = rgb[..., 1].astype(np.int16)
    b = rgb[..., 2].astype(np.int16)
    maximum = np.maximum(np.maximum(r, g), b)
    minimum = np.minimum(np.minimum(r, g), b)
    chroma = maximum - minimum
    dominance = b - np.maximum(r, g)
    return (b >= 40) & (chroma >= 20) & (dominance >= 16)


def generate(source: Path, output_dir: Path, mapping_path: Path) -> None:
    if not source.is_file():
        raise FileNotFoundError(source)
    if not mapping_path.is_file():
        raise FileNotFoundError(mapping_path)

    mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
    gates = mapping["gates"]
    if len(gates) != 12:
        raise RuntimeError(f"expected 12 mapped gates, got {len(gates)}")

    image = Image.open(source).convert("RGBA")
    rgba = np.asarray(image).copy()
    rgb = rgba[..., :3]
    mask = _blue_mask(rgb)
    mask_fraction = float(mask.mean())
    if not (0.005 <= mask_fraction <= 0.95):
        raise RuntimeError(
            f"blue mask coverage {mask_fraction:.4%} is implausible; "
            "inspect bitmap.png before generating variants"
        )

    hsv = np.asarray(image.convert("RGB").convert("HSV")).copy()

    output_dir.mkdir(parents=True, exist_ok=True)

    # Remove obsolete palette filenames so the directory always contains
    # exactly the current 12 identities.
    for stale in output_dir.glob("bitmap_gate_*.png"):
        stale.unlink()

    generated = []
    for gate, (color_name, target_rgb) in zip(gates, GATE_PALETTE, strict=True):
        gate_id = int(gate["gate_id"])
        if gate["color_name"] != color_name:
            raise RuntimeError(
                f"mapping palette mismatch at gate {gate_id}: "
                f"{gate['color_name']!r} != {color_name!r}"
            )

        filename = f"bitmap_gate_{gate_id:02d}_{color_name}.png"
        path = output_dir / filename

        variant_hsv = hsv.copy()
        variant_hsv[..., 0][mask] = _target_hue_u8(target_rgb)
        variant_hsv[..., 1][mask] = np.maximum(
            variant_hsv[..., 1][mask],
            SATURATION_FLOOR_U8,
        )
        variant_rgb = np.asarray(
            Image.fromarray(variant_hsv, mode="HSV").convert("RGB")
        )
        variant_rgba = rgba.copy()
        variant_rgba[..., :3][mask] = variant_rgb[mask]
        Image.fromarray(variant_rgba, mode="RGBA").save(
            path, optimize=True, compress_level=9
        )

        generated.append(path)
        print(
            f"[gate-texture] gate={gate_id:02d} "
            f"color={color_name:10s} file={path.relative_to(REPO_ROOT)}"
        )

    print(f"[gate-texture] source={source.relative_to(REPO_ROOT)}")
    print(f"[gate-texture] blue-mask coverage={mask_fraction:.3%}")
    print(f"[gate-texture] generated={len(generated)} textures")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--mapping", type=Path, default=DEFAULT_MAP)
    args = parser.parse_args()
    generate(args.source.resolve(), args.output_dir.resolve(), args.mapping.resolve())


if __name__ == "__main__":
    main()
