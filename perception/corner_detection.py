"""Stage 2 corner-observation contract.

Stage2A and Stage2B intentionally meet at this boundary:
- Stage2A supplies simulator-perfect projected corners.
- Stage2B supplies detector-predicted corners from RGB.

Everything downstream (PnP, camera/body transforms, metrics) is shared.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


@dataclass(frozen=True)
class CornerObservation:
    corners_uv: np.ndarray
    visible: np.ndarray | None = None
    confidence: np.ndarray | None = None
    timestamp_s: float = 0.0
    source: str = "unknown"
    # Optional global landmark identity. New Circular-12 color-aware detectors
    # use gate IDs 1..12; legacy class-agnostic detectors leave this unset.
    gate_id: int | None = None
    gate_id_confidence: float | None = None

    def __post_init__(self) -> None:
        corners = np.asarray(self.corners_uv, dtype=np.float64)
        if corners.shape != (4, 2):
            raise ValueError("corners_uv must have shape (4, 2)")
        if not np.all(np.isfinite(corners)):
            raise ValueError("Corner observation contains non-finite values")
        visible = np.ones(4, dtype=bool) if self.visible is None else np.asarray(self.visible, dtype=bool).reshape(4)
        confidence = (
            np.ones(4, dtype=np.float64)
            if self.confidence is None
            else np.asarray(self.confidence, dtype=np.float64).reshape(4)
        )
        object.__setattr__(self, "corners_uv", corners)
        object.__setattr__(self, "visible", visible)
        object.__setattr__(self, "confidence", confidence)
        if self.gate_id is not None:
            gate_id = int(self.gate_id)
            if gate_id < 1:
                raise ValueError("gate_id must be positive when provided")
            object.__setattr__(self, "gate_id", gate_id)
        if self.gate_id_confidence is not None:
            gate_id_confidence = float(self.gate_id_confidence)
            if not 0.0 <= gate_id_confidence <= 1.0:
                raise ValueError("gate_id_confidence must be in [0, 1]")
            object.__setattr__(
                self, "gate_id_confidence", gate_id_confidence
            )

    @property
    def complete(self) -> bool:
        return bool(np.all(self.visible))


class GateCornerDetector(Protocol):
    """Stage2B detector interface."""

    def detect(self, rgb_image: np.ndarray, *, timestamp_s: float = 0.0) -> CornerObservation:
        ...
