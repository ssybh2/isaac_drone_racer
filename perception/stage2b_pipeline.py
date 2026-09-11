"""Stage2B: RGB -> learned corner detector -> shared PnP/pose backend."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .corner_detection import CornerObservation, GateCornerDetector
from .pose_recovery import GatePoseRecovery, GatePoseSolution
from .rigid_transform import RigidTransform, rotation_error_rad, translation_error_m


@dataclass(frozen=True)
class Stage2BMetrics:
    corner_rmse_px: float | None = None
    body_translation_error_m: float | None = None
    body_rotation_error_rad: float | None = None


@dataclass(frozen=True)
class Stage2BResult:
    detector_output: CornerObservation
    solution: GatePoseSolution
    metrics: Stage2BMetrics


class Stage2BPerceptionPipeline:
    """Keep the detector replaceable while reusing the Stage2A pose backend."""

    def __init__(self, detector: GateCornerDetector, pose_recovery: GatePoseRecovery):
        self.detector = detector
        self.pose_recovery = pose_recovery

    def process(
        self,
        rgb_image: np.ndarray,
        *,
        timestamp_s: float = 0.0,
        T_wg: RigidTransform | None = None,
        oracle_corners: CornerObservation | None = None,
        T_wb_truth: RigidTransform | None = None,
    ) -> Stage2BResult:
        detected = self.detector.detect(rgb_image, timestamp_s=timestamp_s)
        solution = self.pose_recovery.recover(detected, T_wg=T_wg)

        corner_rmse = None
        if oracle_corners is not None:
            residual = detected.corners_uv - oracle_corners.corners_uv
            corner_rmse = float(np.sqrt(np.mean(np.sum(residual * residual, axis=1))))

        body_t_err = body_r_err = None
        if T_wb_truth is not None:
            if solution.T_wb_est is None:
                raise ValueError("T_wg is required to score world/body pose")
            body_t_err = translation_error_m(solution.T_wb_est, T_wb_truth)
            body_r_err = rotation_error_rad(solution.T_wb_est, T_wb_truth)

        return Stage2BResult(
            detector_output=detected,
            solution=solution,
            metrics=Stage2BMetrics(
                corner_rmse_px=corner_rmse,
                body_translation_error_m=body_t_err,
                body_rotation_error_rad=body_r_err,
            ),
        )
