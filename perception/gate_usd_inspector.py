"""USD inspection helpers used to calibrate the gate opening keypoints.

This is deliberately diagnostic code. A full-asset bounding box is useful for
finding the scale/origin, but it is not automatically treated as the opening
rectangle because the visible gate frame has thickness.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class UsdBounds:
    minimum: tuple[float, float, float]
    maximum: tuple[float, float, float]


@dataclass(frozen=True)
class GateUsdInspection:
    usd_path: str
    default_prim: str | None
    bounds: UsdBounds
    prim_paths: tuple[str, ...]


def inspect_gate_usd(usd_path: str | Path) -> GateUsdInspection:
    """Open a USD with pxr and report actor-space bounds and prim paths."""
    try:
        from pxr import Usd, UsdGeom
    except ImportError as exc:  # pragma: no cover - available in Isaac Sim
        raise ImportError("pxr/USD is required to inspect gate.usd") from exc

    usd_path = Path(usd_path)
    stage = Usd.Stage.Open(str(usd_path))
    if stage is None:
        raise RuntimeError(f"Unable to open USD stage: {usd_path}")

    default_prim = stage.GetDefaultPrim()
    root = default_prim if default_prim and default_prim.IsValid() else stage.GetPseudoRoot()

    cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(),
        [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy],
        useExtentsHint=True,
    )
    aligned = cache.ComputeWorldBound(root).ComputeAlignedBox()
    minimum = tuple(float(v) for v in aligned.GetMin())
    maximum = tuple(float(v) for v in aligned.GetMax())
    prim_paths = tuple(str(prim.GetPath()) for prim in stage.Traverse())

    return GateUsdInspection(
        usd_path=str(usd_path),
        default_prim=str(default_prim.GetPath()) if default_prim and default_prim.IsValid() else None,
        bounds=UsdBounds(minimum=minimum, maximum=maximum),
        prim_paths=prim_paths,
    )
