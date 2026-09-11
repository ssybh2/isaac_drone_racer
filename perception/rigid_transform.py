"""Small, dependency-light rigid-transform utilities for Stage 2 perception.

Convention
----------
``T_ab`` maps coordinates expressed in frame ``b`` into frame ``a``::

    p_a = R_ab @ p_b + t_ab

Composition therefore follows normal matrix multiplication:

``T_ab @ T_bc == T_ac``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def quat_wxyz_to_matrix(quat_wxyz) -> np.ndarray:
    """Convert a ``(w, x, y, z)`` quaternion to a 3x3 rotation matrix."""
    q = np.asarray(quat_wxyz, dtype=np.float64).reshape(4)
    n = np.linalg.norm(q)
    if n <= 0.0:
        raise ValueError("Quaternion norm must be positive")
    w, x, y, z = q / n
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


@dataclass(frozen=True)
class RigidTransform:
    """Rigid transform with explicit source/destination frame names."""

    R: np.ndarray
    t: np.ndarray
    to_frame: str = ""
    from_frame: str = ""

    def __post_init__(self) -> None:
        R = np.asarray(self.R, dtype=np.float64).reshape(3, 3)
        t = np.asarray(self.t, dtype=np.float64).reshape(3)
        if not np.all(np.isfinite(R)) or not np.all(np.isfinite(t)):
            raise ValueError("RigidTransform contains non-finite values")
        if not np.allclose(R.T @ R, np.eye(3), atol=1e-6):
            raise ValueError("Rotation matrix is not orthonormal")
        if np.linalg.det(R) < 0.0:
            raise ValueError("Rotation matrix must be right-handed")
        object.__setattr__(self, "R", R)
        object.__setattr__(self, "t", t)

    @classmethod
    def identity(cls, frame: str = "") -> "RigidTransform":
        return cls(np.eye(3), np.zeros(3), to_frame=frame, from_frame=frame)

    @classmethod
    def from_pose_wxyz(
        cls,
        position,
        quaternion_wxyz,
        *,
        to_frame: str = "",
        from_frame: str = "",
    ) -> "RigidTransform":
        return cls(
            quat_wxyz_to_matrix(quaternion_wxyz),
            np.asarray(position, dtype=np.float64),
            to_frame=to_frame,
            from_frame=from_frame,
        )

    def inverse(self) -> "RigidTransform":
        R_inv = self.R.T
        return RigidTransform(
            R_inv,
            -(R_inv @ self.t),
            to_frame=self.from_frame,
            from_frame=self.to_frame,
        )

    def compose(self, other: "RigidTransform") -> "RigidTransform":
        if self.from_frame and other.to_frame and self.from_frame != other.to_frame:
            raise ValueError(
                f"Cannot compose T_{self.to_frame}{self.from_frame} with "
                f"T_{other.to_frame}{other.from_frame}"
            )
        return RigidTransform(
            self.R @ other.R,
            self.R @ other.t + self.t,
            to_frame=self.to_frame,
            from_frame=other.from_frame,
        )

    def __matmul__(self, other: "RigidTransform") -> "RigidTransform":
        return self.compose(other)

    def transform_points(self, points) -> np.ndarray:
        points = np.asarray(points, dtype=np.float64)
        if points.shape[-1] != 3:
            raise ValueError("Points must have shape (..., 3)")
        return points @ self.R.T + self.t

    def as_matrix(self) -> np.ndarray:
        matrix = np.eye(4, dtype=np.float64)
        matrix[:3, :3] = self.R
        matrix[:3, 3] = self.t
        return matrix


def rotation_error_rad(a: RigidTransform, b: RigidTransform) -> float:
    """Geodesic SO(3) angle between two transforms."""
    relative = a.R @ b.R.T
    cosine = np.clip((np.trace(relative) - 1.0) * 0.5, -1.0, 1.0)
    return float(np.arccos(cosine))


def translation_error_m(a: RigidTransform, b: RigidTransform) -> float:
    return float(np.linalg.norm(a.t - b.t))
