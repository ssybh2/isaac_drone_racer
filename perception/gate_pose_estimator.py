"""Compatibility facade for direct gate PnP estimation."""

from __future__ import annotations

from .camera_model import CameraCalibration
from .corner_detection import CornerObservation
from .gate_geometry import GateGeometry
from .planar_pnp import OpenCvPlanarPnP, PnPBackend, PnPResult


class GatePoseEstimator:
    """Estimate ``T_cg`` from four gate pixels.

    New code should usually use :class:`GatePoseRecovery`, which also applies
    camera/body extrinsics and optional world-map recovery.
    """

    def __init__(
        self,
        geometry: GateGeometry,
        camera: CameraCalibration,
        *,
        backend: PnPBackend | None = None,
    ):
        self.geometry = geometry
        self.camera = camera
        self.backend = backend or OpenCvPlanarPnP()

    def estimate(self, corners_uv) -> PnPResult:
        observation = (
            corners_uv
            if isinstance(corners_uv, CornerObservation)
            else CornerObservation(corners_uv, source="external")
        )
        if not observation.complete:
            raise ValueError("Four visible gate corners are required by the Stage2 reference PnP backend")
        return self.backend.solve(
            self.geometry.object_points_g,
            observation.corners_uv,
            self.camera.K,
            self.camera.dist_coeffs,
        )
