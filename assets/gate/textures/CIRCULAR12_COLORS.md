# Circular-12 Gate Color Map

The Circular-12 track uses one distinct, high-saturation texture color per physical gate.
Only the blue paint in the source texture is recoloured; black/white checkerboards,
alpha, shading and value remain unchanged.

The palette is deliberately **not** ordered by hue. It is permuted around the circular
track so every neighbouring pair—including Gate 12 -> Gate 1—has a **150° HSV hue
separation**. This makes adjacent landmarks visually very different in the FPV camera.

| Gate | Color | RGB | Hex | Hue | Position (x, y, z) m | Yaw |
|---:|---|---:|---|---:|---|---:|
| 1 | Red | (255, 0, 0) | `#FF0000` | 0° | (0.00000000, 0.00000000, 1.0) | 0° |
| 2 | Spring Green | (0, 255, 128) | `#00FF80` | 150° | (6.00000000, 1.60769515, 1.0) | +30° |
| 3 | Magenta | (255, 0, 255) | `#FF00FF` | 300° | (10.39230485, 6.00000000, 1.0) | +60° |
| 4 | Chartreuse | (128, 255, 0) | `#80FF00` | 90° | (12.00000000, 12.00000000, 1.0) | +90° |
| 5 | Blue | (0, 0, 255) | `#0000FF` | 240° | (10.39230485, 18.00000000, 1.0) | +120° |
| 6 | Orange | (255, 128, 0) | `#FF8000` | 30° | (6.00000000, 22.39230485, 1.0) | +150° |
| 7 | Cyan | (0, 255, 255) | `#00FFFF` | 180° | (0.00000000, 24.00000000, 1.0) | +180° |
| 8 | Rose | (255, 0, 128) | `#FF0080` | 330° | (-6.00000000, 22.39230485, 1.0) | -150° |
| 9 | Green | (0, 255, 0) | `#00FF00` | 120° | (-10.39230485, 18.00000000, 1.0) | -120° |
| 10 | Violet | (128, 0, 255) | `#8000FF` | 270° | (-12.00000000, 12.00000000, 1.0) | -90° |
| 11 | Yellow | (255, 255, 0) | `#FFFF00` | 60° | (-10.39230485, 6.00000000, 1.0) | -60° |
| 12 | Azure | (0, 128, 255) | `#0080FF` | 210° | (-6.00000000, 1.60769515, 1.0) | -30° |

The machine-readable source of truth is
`assets/gate/textures/circular12_gate_texture_map.json`.

Generated filenames are:

```text
bitmap_gate_01_red.png
bitmap_gate_02_spring_green.png
bitmap_gate_03_magenta.png
bitmap_gate_04_chartreuse.png
bitmap_gate_05_blue.png
bitmap_gate_06_orange.png
bitmap_gate_07_cyan.png
bitmap_gate_08_rose.png
bitmap_gate_09_green.png
bitmap_gate_10_violet.png
bitmap_gate_11_yellow.png
bitmap_gate_12_azure.png
```

Regenerate from the original mother texture with:

```bash
python scripts/perception/generate_circular12_gate_textures.py
```

The generator also applies a saturation floor to the recoloured paint and removes
obsolete `bitmap_gate_*.png` files before writing the current 12 textures.

The generated images are written to
`assets/gate/textures/circular12/`.
