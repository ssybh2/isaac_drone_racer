"""Generate 12 color-coded gate textures from assets/gate/textures/bitmap.png.

Only pixels belonging to the original blue paint are hue-shifted. Black/white
checkerboards, alpha, shading and value are preserved. Gate 1 is an exact copy
of the original blue texture.

The palette intentionally matches the 12-color FPV preview palette introduced
for Circular-12 diagnostics.
"""

from __future__ import annotations

import argparse
import colorsys
import json
import shutil
from pathlib import Path

import numpy as np
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = REPO_ROOT / "assets/gate/textures/bitmap.png"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "assets/gate/textures/circular12"
DEFAULT_MAP = REPO_ROOT / "assets/gate/textures/circular12_gate_texture_map.json"

# Keep this palette synchronized with
# artifacts/swift_ctbr/circular12_constant_bank_3lap/multicolor_preview.py.
GATE_PALETTE = (
    ("blue",      (20,  77, 242)),
    ("red",       (242, 20,  20)),
    ("yellow",    (255, 219, 5)),
    ("green",     (10,  212, 31)),
    ("orange",    (255, 97,  5)),
    ("cyan",      (5,   209, 245)),
    ("violet",    (148, 46,  242)),
    ("magenta",   (245, 20,  161)),
    ("lime",      (153, 245, 5)),
    ("turquoise", (8,   209, 138)),
    ("pink",      (255, 105, 166)),
    ("amber",     (255, 166, 10)),
)


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

        if gate_id == 1:
            # Preserve the source blue texture byte-for-byte for Gate 1.
            shutil.copyfile(source, path)
        else:
            variant_hsv = hsv.copy()
            variant_hsv[..., 0][mask] = _target_hue_u8(target_rgb)
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
