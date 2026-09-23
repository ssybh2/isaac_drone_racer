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


def _runtime_asset_path(asset: Path) -> str:
    """Give nested USD references and MDL paths that survive scene instancing.

    These generated USD wrappers are local build products, so an absolute path
    is preferable to a wrapper-relative path that Isaac's MDL loader later
    resolves from the parent scene instead.
    """
    return str(asset.resolve(strict=True))


def _is_gate_bitmap_asset(value) -> bool:
    """Return True for the original or generated gate bitmap asset slot."""
    from pxr import Sdf

    if not isinstance(value, Sdf.AssetPath):
        return False
    source = value.path or value.resolvedPath
    if not source:
        return False
    basename = source.replace("\\", "/").rsplit("/", 1)[-1]
    return basename == "bitmap.png" or (
        basename.startswith("bitmap_gate_") and basename.endswith(".png")
    )


def _build_reference_wrapper_variant(destination: Path, replacement_asset_path: str) -> list[str]:
    """Create a tiny USD wrapper that references gate.usd and overrides only its texture.

    Copy/exporting the composed binary gate stage proved fragile in Isaac Sim:
    it could leave the referenced MDL material unresolved and render a white
    gate even though the edited asset input looked correct in USD inspection.
    A reference wrapper keeps the original gate.usd as the authoritative
    geometry/material/physics layer and authors only the texture override in a
    stronger sibling layer.
    """
    from pxr import Sdf, Usd, UsdShade

    source_stage = Usd.Stage.Open(str(SOURCE_GATE_USD))
    if source_stage is None:
        raise RuntimeError(f"failed to open source gate USD: {SOURCE_GATE_USD}")
    source_stage.Reload()

    source_default = source_stage.GetDefaultPrim()
    if not source_default or not source_default.IsValid():
        raise RuntimeError(f"source gate USD has no valid default prim: {SOURCE_GATE_USD}")

    root_path = source_default.GetPath()
    root_name = root_path.name
    root_type = source_default.GetTypeName() or "Xform"

    if destination.exists():
        destination.unlink()

    wrapper = Usd.Stage.CreateNew(str(destination))
    wrapper_root = wrapper.DefinePrim(f"/{root_name}", root_type)
    wrapper.SetDefaultPrim(wrapper_root)
    wrapper_root.GetReferences().AddReference(_runtime_asset_path(SOURCE_GATE_USD))

    # The reference is composed immediately. Setting an input below the
    # referenced prim authors a stronger override into this wrapper layer; it
    # does not mutate gate.usd.
    changed: list[str] = []
    for prim in wrapper.Traverse():
        if not prim.IsA(UsdShade.Shader):
            continue
        shader = UsdShade.Shader(prim)
        for shader_input in shader.GetInputs():
            value = shader_input.Get()
            if not _is_gate_bitmap_asset(value):
                continue
            shader_input.Set(Sdf.AssetPath(replacement_asset_path))
            changed.append(f"{prim.GetPath()}.{shader_input.GetBaseName()}")

    if not changed:
        raise RuntimeError(
            "referenced gate.usd contains no replaceable gate bitmap asset input; "
            "refusing to generate an unbound color variant"
        )

    wrapper.GetRootLayer().Save()
    return changed


def _variant_is_fresh(destination: Path, texture: Path) -> bool:
    if not destination.is_file():
        return False
    newest_input = max(
        Path(__file__).stat().st_mtime_ns,
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

    result: dict[str, str] = {}
    for gate in _load_gate_map():
        gate_id = str(int(gate["gate_id"]))
        texture = REPO_ROOT / str(gate["texture"])
        destination = variant_path_for_gate(gate)
        if not texture.is_file():
            raise FileNotFoundError(texture)

        if force or not _variant_is_fresh(destination, texture):
            replacement = _runtime_asset_path(texture)
            changed = _build_reference_wrapper_variant(destination, replacement)
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
