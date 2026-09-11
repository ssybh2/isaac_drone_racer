"""Gate USD calibration boundary for Stage 2.

The old Stage2A branch silently assumed a 1 m x 1 m opening. This module
intentionally has no numeric fallback: calibrated PnP keypoints must come from
an explicit file or be provided by the caller.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np

from .gate_geometry import CORNER_NAMES, GateGeometry


@dataclass(frozen=True)
class GateUsdConfig:
    usd_path: str = "assets/gate/gate.usd"
    keypoint_calibration_path: str = "assets/gate/gate_keypoints.json"
    actor_frame: str = "gate_actor"
    expected_normal_axis: str = "+X"


DEFAULT_GATE_USD = GateUsdConfig()


def load_gate_keypoint_calibration(path: str | Path) -> GateGeometry:
    """Load four opening corners measured in the gate actor frame."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Gate keypoint calibration is missing: {path}. "
            "Run the USD inspection tool and save the actual opening corners; "
            "Stage2A must not fall back to guessed dimensions."
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    frame = payload.get("frame")
    if frame != "gate_actor":
        raise ValueError(f"Expected calibration frame 'gate_actor', got {frame!r}")
    names = tuple(payload.get("corner_order", CORNER_NAMES))
    if names != CORNER_NAMES:
        raise ValueError(f"Expected corner_order {CORNER_NAMES}, got {names}")
    points = np.asarray(payload["object_points_gate_actor_m"], dtype=np.float64)
    return GateGeometry(points, source=str(path))


def save_gate_keypoint_calibration(
    path: str | Path,
    geometry: GateGeometry,
    *,
    usd_path: str = DEFAULT_GATE_USD.usd_path,
    notes: str = "",
) -> None:
    """Write the explicit calibration artifact consumed by Stage2A/Stage2B."""
    payload = {
        "usd_path": usd_path,
        "frame": "gate_actor",
        "corner_order": list(CORNER_NAMES),
        "object_points_gate_actor_m": geometry.object_points_g.tolist(),
        "opening_width_m": geometry.opening_width_m,
        "opening_height_m": geometry.opening_height_m,
        "notes": notes,
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
