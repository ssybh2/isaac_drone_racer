"""Compose a precise corner detector with an independent visibility guard."""

from __future__ import annotations

import numpy as np

from .corner_detection import CornerObservation, GateCornerDetector


class VisibilityGuardedGateCornerDetector:
    """Use one model's corners and another model's learned visibility scores.

    The coordinate detector remains responsible for instance detection and
    geometric sanity checks.  The guard is deliberately not allowed to replace
    its coordinates: it can only turn individual corners off.
    """

    def __init__(
        self,
        coordinate_detector: GateCornerDetector,
        visibility_detector: GateCornerDetector,
        *,
        visibility_threshold: float = 0.75,
    ) -> None:
        if not 0.0 <= visibility_threshold <= 1.0:
            raise ValueError("visibility_threshold must be in [0, 1]")
        self.coordinate_detector = coordinate_detector
        self.visibility_detector = visibility_detector
        self.visibility_threshold = float(visibility_threshold)

    def detect(self, rgb_image: np.ndarray, *, timestamp_s: float = 0.0) -> CornerObservation:
        coordinate = self.coordinate_detector.detect(rgb_image, timestamp_s=timestamp_s)
        guard = self.visibility_detector.detect(rgb_image, timestamp_s=timestamp_s)
        guard_confidence = np.asarray(guard.confidence, dtype=np.float64).reshape(4)
        visible = np.asarray(coordinate.visible, dtype=bool) & (
            guard_confidence >= self.visibility_threshold
        )
        return CornerObservation(
            corners_uv=coordinate.corners_uv,
            visible=visible,
            confidence=guard_confidence,
            timestamp_s=timestamp_s,
            source=f"{coordinate.source}+visibility_guard:{guard.source}",
        )
