"""Stage2A oracle corner source.

This class exposes *pixels only*. Simulator world truth stays behind the
Stage2A validation boundary.
"""

from __future__ import annotations

import numpy as np

from .camera_model import CameraCalibration, project_visible_with_calibration
from .corner_detection import CornerObservation
from .gate_geometry import GateGeometry
from .rigid_transform import RigidTransform


class PerfectGateCornerSensor:
    def __init__(self, geometry: GateGeometry, camera: CameraCalibration):
        self.geometry = geometry
        self.camera = camera

    def measure(self, T_cg: RigidTransform, *, timestamp_s: float = 0.0) -> CornerObservation:
        points_c = T_cg.transform_points(self.geometry.object_points_g)
        corners_uv, visible = project_visible_with_calibration(points_c, self.camera)
        return CornerObservation(
            corners_uv=np.asarray(corners_uv, dtype=np.float64),
            visible=visible,
            confidence=visible.astype(np.float64),
            timestamp_s=timestamp_s,
            source="isaac_oracle_projection",
        )
