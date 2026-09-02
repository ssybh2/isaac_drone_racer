from pathlib import Path

from pxr import Usd


def test_gate_texture_assets_resolve_from_the_project_runtime_directory():
    project_root = Path(__file__).parents[1]
    gate_usd = project_root / "assets" / "gate" / "gate.usd"
    stage = Usd.Stage.Open(str(gate_usd))

    assert stage is not None

    texture_assets = []
    for prim in stage.Traverse():
        for attribute in prim.GetAttributes():
            if attribute.GetName() != "inputs:texture":
                continue
            asset = attribute.Get()
            resolved_path = Path(asset.path)
            if not resolved_path.is_absolute():
                resolved_path = project_root / resolved_path
            texture_assets.append((str(prim.GetPath()), asset.path, resolved_path))

    assert texture_assets, "gate.usd must contain at least one texture asset"
    assert all(Path(resolved).is_file() for _, _, resolved in texture_assets), texture_assets
