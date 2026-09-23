"""Build per-gate USD variants for the Circular-12 color-coded track.

The repository keeps one authoritative geometric gate asset at
assets/gate/gate.usd. This helper opens that asset with USD, changes only
shader asset inputs that resolve to bitmap.png, and exports one sibling USD
variant per Circular-12 gate. Physics, mesh geometry, collision, UVs and all
other material parameters are inherited unchanged from the source asset.

The generated gate_circular12_*.usd files are local build products. They are
regenerated from the checked-in source asset and texture map instead of being
hand-edited binary USD files.
"""

from __future__ import annotations

import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
GATE_DIR = REPO_ROOT / "assets/gate"
SOURCE_GATE_USD = GATE_DIR / "gate.usd"
TEXTURE_MAP = GATE_DIR / "textures/circular12_gate_texture_map.json"


def _load_gate_map() -> list[dict]:
    data = json.loads(TEXTURE_MAP.read_text(encoding="utf-8"))
    gates = data.get("gates", [])
    if len(gates) != 12:
        raise RuntimeError(f"expected 12 Circular-12 gate identities, got {len(gates)}")
    ids = [int(item["gate_id"]) for item in gates]
    if ids != list(range(1, 13)):
        raise RuntimeError(f"unexpected gate IDs in texture map: {ids}")
    return gates


def variant_path_for_gate(gate: dict) -> Path:
    gate_id = int(gate["gate_id"])
    color = str(gate["color_name"])
    return GATE_DIR / f"gate_circular12_{gate_id:02d}_{color}.usd"


def _replace_bitmap_asset(stage, replacement_asset_path: str) -> list[str]:
    """Replace all shader asset inputs whose source basename is bitmap.png."""
    from pxr import Sdf, UsdShade

    changed: list[str] = []
    for prim in stage.Traverse():
        if not prim.IsA(UsdShade.Shader):
            continue
        shader = UsdShade.Shader(prim)
        for shader_input in shader.GetInputs():
            value = shader_input.Get()
            if not isinstance(value, Sdf.AssetPath):
                continue
            source = value.path or value.resolvedPath
            if not source:
                continue
            normalized = source.replace("\\", "/")
            if normalized.rsplit("/", 1)[-1] != "bitmap.png":
                continue
            shader_input.Set(Sdf.AssetPath(replacement_asset_path))
            changed.append(f"{prim.GetPath()}.{shader_input.GetBaseName()}")
    return changed


def _variant_is_fresh(destination: Path, texture: Path) -> bool:
    if not destination.is_file():
        return False
    newest_input = max(
        SOURCE_GATE_USD.stat().st_mtime_ns,
        TEXTURE_MAP.stat().st_mtime_ns,
        texture.stat().st_mtime_ns,
    )
    return destination.stat().st_mtime_ns >= newest_input


def ensure_circular12_gate_usd_variants(*, force: bool = False) -> dict[str, str]:
    """Return gate-ID -> USD path, generating stale variants when necessary."""
    if not SOURCE_GATE_USD.is_file():
        raise FileNotFoundError(SOURCE_GATE_USD)
    if not TEXTURE_MAP.is_file():
        raise FileNotFoundError(TEXTURE_MAP)

    from pxr import Usd

    result: dict[str, str] = {}
    for gate in _load_gate_map():
        gate_id = str(int(gate["gate_id"]))
        texture = REPO_ROOT / str(gate["texture"])
        destination = variant_path_for_gate(gate)
        if not texture.is_file():
            raise FileNotFoundError(texture)

        if force or not _variant_is_fresh(destination, texture):
            stage = Usd.Stage.Open(str(SOURCE_GATE_USD))
            if stage is None:
                raise RuntimeError(f"failed to open source gate USD: {SOURCE_GATE_USD}")

            replacement = f"./textures/circular12/{texture.name}"
            changed = _replace_bitmap_asset(stage, replacement)
            if not changed:
                raise RuntimeError(
                    "gate.usd contains no UsdShade asset input ending in "
                    "'bitmap.png'; refusing to generate an unbound color variant"
                )
            stage.GetRootLayer().Export(str(destination))
            print(
                f"[gate-usd] gate={int(gate_id):02d} "
                f"color={gate['color_name']:10s} "
                f"texture={texture.name} shader_inputs={len(changed)}"
            )

        result[gate_id] = str(destination.relative_to(REPO_ROOT))

    return result


def validate_circular12_gate_usd_variants() -> dict[str, list[str]]:
    """Open generated variants and report their bound bitmap asset paths."""
    from pxr import Sdf, Usd, UsdShade

    paths = ensure_circular12_gate_usd_variants()
    report: dict[str, list[str]] = {}

    for gate_id, relative_usd in paths.items():
        stage = Usd.Stage.Open(str(REPO_ROOT / relative_usd))
        if stage is None:
            raise RuntimeError(f"failed to open generated gate USD: {relative_usd}")

        found: list[str] = []
        for prim in stage.Traverse():
            if not prim.IsA(UsdShade.Shader):
                continue
            shader = UsdShade.Shader(prim)
            for shader_input in shader.GetInputs():
                value = shader_input.Get()
                if isinstance(value, Sdf.AssetPath):
                    path = value.path or value.resolvedPath
                    if path and "circular12/bitmap_gate_" in path.replace("\\", "/"):
                        found.append(path)

        if not found:
            raise RuntimeError(
                f"gate {gate_id} variant has no Circular-12 texture binding: {relative_usd}"
            )
        report[gate_id] = found

    return report
