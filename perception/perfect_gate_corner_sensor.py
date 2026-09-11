"""Stage2A oracle corner source.

This class exposes *pixels only*. Simulator world truth stays behind the
Stage2A validation boundary.
"""

from __future__ import annotations

import numpy as np

from .camera_model import CameraCalibration, project_with_calibration
from .corner_detection import CornerObservation
from .gate_geometry import GateGeometry
from .rigid_transform import RigidTransform


class PerfectGateCornerSensor:
    def __init__(self, geometry: GateGeometry, camera: CameraCalibration):
        self.geometry = geometry
        self.camera = camera

    def measure(self, T_cg: RigidTransform, *, timestamp_s: float = 0.0) -> CornerObservation:
        points_c = T_cg.transform_points(self.geometry.object_points_g)
        corners_uv = project_with_calibration(points_c, self.camera)
        visible = (points_c[:, 2] > 0.0) & self.camera.in_image(corners_uv)
        return CornerObservation(
            corners_uv=np.asarray(corners_uv, dtype=np.float64),
            visible=visible,
            confidence=np.ones(4, dtype=np.float64),
            timestamp_s=timestamp_s,
            source="isaac_oracle_projection",
        )
