# Circular-12 Gate Color Map

The Circular-12 track uses one distinct texture color per physical gate. The
geometry is the existing `CIRCULAR_12_GATE_TRACK_CONFIG`; only the blue paint
in the source texture is hue-shifted. Black/white checkerboards and shading are
preserved.

| Gate | Color | RGB | Hex | Position (x, y, z) m | Yaw |
|---:|---|---:|---|---|---:|
| 1 | Blue | (20, 77, 242) | `#144DF2` | (0.00000000, 0.00000000, 1.0) | 0° |
| 2 | Red | (242, 20, 20) | `#F21414` | (6.00000000, 1.60769515, 1.0) | +30° |
| 3 | Yellow | (255, 219, 5) | `#FFDB05` | (10.39230485, 6.00000000, 1.0) | +60° |
| 4 | Green | (10, 212, 31) | `#0AD41F` | (12.00000000, 12.00000000, 1.0) | +90° |
| 5 | Orange | (255, 97, 5) | `#FF6105` | (10.39230485, 18.00000000, 1.0) | +120° |
| 6 | Cyan | (5, 209, 245) | `#05D1F5` | (6.00000000, 22.39230485, 1.0) | +150° |
| 7 | Violet | (148, 46, 242) | `#942EF2` | (0.00000000, 24.00000000, 1.0) | 180° |
| 8 | Magenta | (245, 20, 161) | `#F514A1` | (-6.00000000, 22.39230485, 1.0) | -150° |
| 9 | Lime | (153, 245, 5) | `#99F505` | (-10.39230485, 18.00000000, 1.0) | -120° |
| 10 | Turquoise | (8, 209, 138) | `#08D18A` | (-12.00000000, 12.00000000, 1.0) | -90° |
| 11 | Pink | (255, 105, 166) | `#FF69A6` | (-10.39230485, 6.00000000, 1.0) | -60° |
| 12 | Amber | (255, 166, 10) | `#FFA60A` | (-6.00000000, 1.60769515, 1.0) | -30° |

The machine-readable source of truth is
`assets/gate/textures/circular12_gate_texture_map.json`.

Generated filenames are:

```text
bitmap_gate_01_blue.png
bitmap_gate_02_red.png
bitmap_gate_03_yellow.png
bitmap_gate_04_green.png
bitmap_gate_05_orange.png
bitmap_gate_06_cyan.png
bitmap_gate_07_violet.png
bitmap_gate_08_magenta.png
bitmap_gate_09_lime.png
bitmap_gate_10_turquoise.png
bitmap_gate_11_pink.png
bitmap_gate_12_amber.png
```

Regenerate from the original mother texture with:

```bash
python scripts/perception/generate_circular12_gate_textures.py
```

The generated images are written to
`assets/gate/textures/circular12/`.
