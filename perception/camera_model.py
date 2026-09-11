"""Camera calibration and projection primitives shared by Stage 2A and Stage 2B."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover - torch exists in Isaac Lab runtime
    torch = None


@dataclass(frozen=True)
class CameraCalibration:
    """OpenCV/ROS-optical camera calibration.

    Camera frame ``C`` follows the OpenCV/ROS optical convention:
    +X right, +Y down, +Z forward.
    """

    K: np.ndarray
    image_width: int
    image_height: int
    distortion: np.ndarray | None = None
    model: str = "pinhole"

    def __post_init__(self) -> None:
        K = np.asarray(self.K, dtype=np.float64).reshape(3, 3)
        if not np.all(np.isfinite(K)):
            raise ValueError("K contains non-finite values")
        if K[0, 0] <= 0.0 or K[1, 1] <= 0.0:
            raise ValueError("Camera focal lengths must be positive")
        if self.image_width <= 0 or self.image_height <= 0:
            raise ValueError("Image dimensions must be positive")
        if self.model != "pinhole":
            raise ValueError(
                f"Unsupported projection model {self.model!r}. "
                "Calibrate/undistort non-pinhole cameras before PnP."
            )
        distortion = None
        if self.distortion is not None:
            distortion = np.asarray(self.distortion, dtype=np.float64).reshape(-1)
        object.__setattr__(self, "K", K)
        object.__setattr__(self, "distortion", distortion)

    @property
    def dist_coeffs(self) -> np.ndarray | None:
        return self.distortion

    def in_image(self, uv, margin_px: float = 0.0) -> np.ndarray:
        uv = np.asarray(uv)
        return (
            (uv[..., 0] >= -margin_px)
            & (uv[..., 0] < self.image_width + margin_px)
            & (uv[..., 1] >= -margin_px)
            & (uv[..., 1] < self.image_height + margin_px)
        )


def _project_numpy(points_c: np.ndarray, K: np.ndarray) -> np.ndarray:
    points_c = np.asarray(points_c, dtype=np.float64)
    if points_c.shape[-1] != 3:
        raise ValueError("points_c must have shape (..., 3)")
    z = points_c[..., 2]
    if np.any(z <= 0.0):
        raise ValueError("Cannot pinhole-project points with non-positive camera Z")
    u = K[0, 0] * (points_c[..., 0] / z) + K[0, 2]
    v = K[1, 1] * (points_c[..., 1] / z) + K[1, 2]
    return np.stack((u, v), axis=-1)


def _project_torch(points_c, K):
    if points_c.shape[-1] != 3:
        raise ValueError("points_c must have shape (..., 3)")
    z = points_c[..., 2]
    if torch.any(z <= 0.0):
        raise ValueError("Cannot pinhole-project points with non-positive camera Z")
    u = K[..., 0, 0] * (points_c[..., 0] / z) + K[..., 0, 2]
    v = K[..., 1, 1] * (points_c[..., 1] / z) + K[..., 1, 2]
    return torch.stack((u, v), dim=-1)


def project_points_pinhole(points_c, K, distortion=None):
    """Project camera-frame 3D points to pixels.

    Supports NumPy and Torch for the zero-distortion path. Distorted projection
    intentionally stays in the OpenCV reference backend so the simulator,
    dataset labels and PnP all use one explicit calibration model.
    """
    if torch is not None and isinstance(points_c, torch.Tensor):
        if distortion is not None:
            dist = torch.as_tensor(distortion, device=points_c.device)
            if torch.any(dist != 0):
                raise NotImplementedError(
                    "Torch distorted projection is intentionally not implemented. "
                    "Use undistorted/pinhole Stage2A or the OpenCV reference path."
                )
        K_t = K if isinstance(K, torch.Tensor) else torch.as_tensor(K, dtype=points_c.dtype, device=points_c.device)
        return _project_torch(points_c, K_t)

    K_np = np.asarray(K, dtype=np.float64).reshape(3, 3)
    dist = None if distortion is None else np.asarray(distortion, dtype=np.float64).reshape(-1)
    if dist is None or not np.any(dist):
        return _project_numpy(np.asarray(points_c), K_np)

    try:
        import cv2
    except ImportError as exc:  # pragma: no cover
        raise ImportError("OpenCV is required for distorted pinhole projection") from exc

    points = np.asarray(points_c, dtype=np.float64).reshape(-1, 3)
    if np.any(points[:, 2] <= 0.0):
        raise ValueError("Cannot pinhole-project points with non-positive camera Z")
    uv, _ = cv2.projectPoints(
        points,
        np.zeros(3, dtype=np.float64),
        np.zeros(3, dtype=np.float64),
        K_np,
        dist,
    )
    return uv.reshape(np.asarray(points_c).shape[:-1] + (2,))


def transform_points(points, R, t):
    """Compatibility helper: apply ``p_out = R @ p_in + t``."""
    if torch is not None and isinstance(points, torch.Tensor):
        return points @ R.transpose(-1, -2) + t
    points = np.asarray(points)
    return points @ np.asarray(R).T + np.asarray(t)


def project_with_calibration(points_c, calibration: CameraCalibration) -> np.ndarray:
    return project_points_pinhole(
        points_c,
        calibration.K,
        distortion=calibration.dist_coeffs,
    )
