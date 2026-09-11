"""Stage2A: simulator truth -> perfect pixels -> PnP -> drone pose.

Stage2A is an oracle *corner source*, not an oracle policy observation. World
gate/camera/body truth is used only to synthesize exact pixel labels and to
score the recovered pose. The policy-facing quantity is recovered through PnP.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .camera_model import CameraCalibration, project_visible_with_calibration, project_with_calibration
from .corner_detection import CornerObservation
from .gate_geometry import GateGeometry
from .pose_recovery import GatePoseRecovery, GatePoseSolution
from .rigid_transform import RigidTransform, rotation_error_rad, translation_error_m


@dataclass(frozen=True)
class Stage2ATruth:
    """One simulator frame expressed with explicit transform semantics."""

    T_wg: RigidTransform
    T_wc: RigidTransform
    T_wb: RigidTransform | None = None
    timestamp_s: float = 0.0


@dataclass(frozen=True)
class Stage2AMetrics:
    corner_reprojection_rmse_px: float
    gate_translation_error_m: float
    gate_rotation_error_rad: float
    camera_translation_error_m: float
    camera_rotation_error_rad: float
    body_translation_error_m: float | None
    body_rotation_error_rad: float | None
    extrinsic_translation_error_m: float | None
    extrinsic_rotation_error_rad: float | None


@dataclass(frozen=True)
class Stage2AResult:
    truth: Stage2ATruth
    oracle_corners: CornerObservation
    solution: GatePoseSolution
    T_cg_truth: RigidTransform
    metrics: Stage2AMetrics


class Stage2APerceptionPipeline:
    """Reference Stage2A validation pipeline."""

    def __init__(
        self,
        geometry: GateGeometry,
        camera: CameraCalibration,
        T_bc: RigidTransform,
        *,
        pose_recovery: GatePoseRecovery | None = None,
    ):
        self.geometry = geometry
        self.camera = camera
        self.T_bc = T_bc
        self.pose_recovery = pose_recovery or GatePoseRecovery(geometry, camera, T_bc)

    def project_perfect_corners(self, T_cg: RigidTransform, *, timestamp_s: float = 0.0) -> CornerObservation:
        points_c = T_cg.transform_points(self.geometry.object_points_g)
        uv, visible = project_visible_with_calibration(points_c, self.camera)
        return CornerObservation(
            uv,
            visible=visible,
            confidence=visible.astype(np.float64),
            timestamp_s=timestamp_s,
            source="isaac_oracle_projection",
        )

    def estimate_pose(self, corners_uv) -> GatePoseSolution:
        observation = (
            corners_uv
            if isinstance(corners_uv, CornerObservation)
            else CornerObservation(corners_uv, source="external")
        )
        return self.pose_recovery.recover(observation)

    def process_truth(self, truth: Stage2ATruth) -> Stage2AResult:
        # Camera optical frame C from gate actor frame G.
        T_cg_truth = truth.T_wc.inverse() @ truth.T_wg
        oracle_corners = self.project_perfect_corners(T_cg_truth, timestamp_s=truth.timestamp_s)
        if not oracle_corners.complete:
            raise ValueError(
                "Gate is not fully visible in the calibrated image. "
                "Stage2A four-point PnP equivalence requires four visible corners."
            )

        solution = self.pose_recovery.recover(oracle_corners, T_wg=truth.T_wg)

        # Reproject the PnP solution with the exact same calibrated camera model.
        uv_est = project_with_calibration(
            solution.T_cg.transform_points(self.geometry.object_points_g),
            self.camera,
        )
        residual = uv_est - oracle_corners.corners_uv
        corner_rmse = float(np.sqrt(np.mean(np.sum(residual * residual, axis=1))))

        body_t_err = body_r_err = None
        extrinsic_t_err = extrinsic_r_err = None
        if truth.T_wb is not None:
            body_t_err = translation_error_m(solution.T_wb_est, truth.T_wb)
            body_r_err = rotation_error_rad(solution.T_wb_est, truth.T_wb)

            # Simulator-derived extrinsic is an independent check of the
            # configured mount/convention: T_bc_truth = T_bw * T_wc.
            T_bc_truth = truth.T_wb.inverse() @ truth.T_wc
            extrinsic_t_err = translation_error_m(self.T_bc, T_bc_truth)
            extrinsic_r_err = rotation_error_rad(self.T_bc, T_bc_truth)

        metrics = Stage2AMetrics(
            corner_reprojection_rmse_px=corner_rmse,
            gate_translation_error_m=translation_error_m(solution.T_cg, T_cg_truth),
            gate_rotation_error_rad=rotation_error_rad(solution.T_cg, T_cg_truth),
            camera_translation_error_m=translation_error_m(solution.T_wc_est, truth.T_wc),
            camera_rotation_error_rad=rotation_error_rad(solution.T_wc_est, truth.T_wc),
            body_translation_error_m=body_t_err,
            body_rotation_error_rad=body_r_err,
            extrinsic_translation_error_m=extrinsic_t_err,
            extrinsic_rotation_error_rad=extrinsic_r_err,
        )

        return Stage2AResult(
            truth=truth,
            oracle_corners=oracle_corners,
            solution=solution,
            T_cg_truth=T_cg_truth,
            metrics=metrics,
        )
