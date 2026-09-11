"""Reference planar PnP backend for gate pose recovery.

The OpenCV backend is intentionally the calibration/reference implementation.
It is not the final 4096-environment training backend; a batched Torch/GPU PnP
implementation can later satisfy the same interface.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from .rigid_transform import RigidTransform

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None


@dataclass(frozen=True)
class PnPResult:
    success: bool
    T_cg: RigidTransform | None
    reprojection_rmse_px: float
    method: str
    candidate_count: int
    message: str = ""


class PnPBackend(Protocol):
    def solve(self, object_points_g, image_points_uv, K, dist_coeffs=None) -> PnPResult:
        ...


def _candidate_rmse(object_points, image_points, K, dist, rvec, tvec) -> float:
    projected, _ = cv2.projectPoints(object_points, rvec, tvec, K, dist)
    residual = projected.reshape(-1, 2) - image_points
    return float(np.sqrt(np.mean(np.sum(residual * residual, axis=1))))


class OpenCvPlanarPnP:
    """Planar IPPE reference backend with ambiguity/cheirality handling."""

    def __init__(self, min_depth_m: float = 1.0e-5):
        self.min_depth_m = float(min_depth_m)

    def solve(self, object_points_g, image_points_uv, K, dist_coeffs=None) -> PnPResult:
        if cv2 is None:
            raise ImportError("opencv-python is required for the OpenCV PnP backend")

        object_points = np.asarray(object_points_g, dtype=np.float64).reshape(-1, 3)
        image_points = np.asarray(image_points_uv, dtype=np.float64).reshape(-1, 2)
        K = np.asarray(K, dtype=np.float64).reshape(3, 3)
        dist = None if dist_coeffs is None else np.asarray(dist_coeffs, dtype=np.float64).reshape(-1)

        if object_points.shape[0] != image_points.shape[0]:
            raise ValueError("3D/2D point counts do not match")
        if object_points.shape[0] < 4:
            raise ValueError("PnP requires at least four correspondences")

        response = cv2.solvePnPGeneric(
            object_points,
            image_points,
            K,
            dist,
            flags=cv2.SOLVEPNP_IPPE,
        )
        ok = bool(response[0])
        rvecs = list(response[1]) if len(response) > 1 else []
        tvecs = list(response[2]) if len(response) > 2 else []

        if not ok or not rvecs:
            # IPPE should solve this calibrated planar problem. ITERATIVE is a
            # diagnostic fallback, not a hidden change of model.
            ok_fallback, rvec, tvec = cv2.solvePnP(
                object_points,
                image_points,
                K,
                dist,
                flags=cv2.SOLVEPNP_ITERATIVE,
            )
            if not ok_fallback:
                return PnPResult(False, None, float("inf"), "IPPE", 0, "PnP failed")
            rvecs, tvecs = [rvec], [tvec]

        candidates = []
        for rvec, tvec in zip(rvecs, tvecs):
            R, _ = cv2.Rodrigues(rvec)
            t = np.asarray(tvec, dtype=np.float64).reshape(3)
            points_c = object_points @ R.T + t
            positive_depth = bool(np.all(points_c[:, 2] > self.min_depth_m))
            rmse = _candidate_rmse(object_points, image_points, K, dist, rvec, tvec)
            candidates.append((not positive_depth, rmse, R, t))

        # Positive-depth candidates sort ahead of invalid ones, then by RMSE.
        candidates.sort(key=lambda item: (item[0], item[1]))
        invalid_depth, rmse, R, t = candidates[0]
        if invalid_depth:
            return PnPResult(
                False,
                None,
                rmse,
                "IPPE",
                len(candidates),
                "All PnP candidates put at least one gate corner behind the camera",
            )

        return PnPResult(
            True,
            RigidTransform(R, t, to_frame="C", from_frame="G"),
            rmse,
            "IPPE",
            len(candidates),
        )


def solve_gate_pnp(object_points_g, image_points_uv, K, dist_coeffs=None):
    """Compatibility wrapper returning ``(R_cg, t_cg)``."""
    result = OpenCvPlanarPnP().solve(object_points_g, image_points_uv, K, dist_coeffs)
    if not result.success or result.T_cg is None:
        raise RuntimeError(result.message or "PnP failed")
    return result.T_cg.R, result.T_cg.t
