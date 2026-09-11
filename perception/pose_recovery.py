"""Shared 2D-corner -> PnP -> camera/body/world pose recovery."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .camera_model import CameraCalibration
from .corner_detection import CornerObservation
from .gate_geometry import GateGeometry
from .planar_pnp import OpenCvPlanarPnP, PnPBackend, PnPResult
from .rigid_transform import RigidTransform


@dataclass(frozen=True)
class GatePoseSolution:
    """Pose solution shared by Stage2A and Stage2B."""

    observation: CornerObservation
    pnp: PnPResult
    T_cg: RigidTransform
    T_gc: RigidTransform
    T_bg: RigidTransform
    target_pos_b: np.ndarray
    T_wg: RigidTransform | None = None
    T_wc_est: RigidTransform | None = None
    T_wb_est: RigidTransform | None = None


class GatePoseRecovery:
    """Recover gate/camera/body relationships from four image corners.

    ``T_bc`` is the calibrated camera optical-frame pose in the drone body
    frame: it maps camera coordinates into body coordinates.
    """

    def __init__(
        self,
        geometry: GateGeometry,
        camera: CameraCalibration,
        T_bc: RigidTransform,
        *,
        pnp_backend: PnPBackend | None = None,
    ):
        if T_bc.to_frame and T_bc.to_frame != "B":
            raise ValueError("T_bc.to_frame must be 'B'")
        if T_bc.from_frame and T_bc.from_frame != "C":
            raise ValueError("T_bc.from_frame must be 'C'")
        self.geometry = geometry
        self.camera = camera
        self.T_bc = T_bc
        self.pnp_backend = pnp_backend or OpenCvPlanarPnP()

    def recover(
        self,
        observation: CornerObservation,
        *,
        T_wg: RigidTransform | None = None,
    ) -> GatePoseSolution:
        if not observation.complete:
            raise ValueError("Reference PnP requires all four gate corners")

        pnp = self.pnp_backend.solve(
            self.geometry.object_points_g,
            observation.corners_uv,
            self.camera.K,
            self.camera.dist_coeffs,
        )
        if not pnp.success or pnp.T_cg is None:
            raise RuntimeError(pnp.message or "PnP failed")

        T_cg = pnp.T_cg
        T_gc = T_cg.inverse()
        T_bg = self.T_bc @ T_cg
        target_pos_b = T_bg.t.copy()

        T_wc_est = None
        T_wb_est = None
        if T_wg is not None:
            if T_wg.to_frame and T_wg.to_frame != "W":
                raise ValueError("T_wg.to_frame must be 'W'")
            if T_wg.from_frame and T_wg.from_frame != "G":
                raise ValueError("T_wg.from_frame must be 'G'")
            T_wc_est = T_wg @ T_gc
            T_wb_est = T_wg @ T_bg.inverse()

        return GatePoseSolution(
            observation=observation,
            pnp=pnp,
            T_cg=T_cg,
            T_gc=T_gc,
            T_bg=T_bg,
            target_pos_b=target_pos_b,
            T_wg=T_wg,
            T_wc_est=T_wc_est,
            T_wb_est=T_wb_est,
        )
