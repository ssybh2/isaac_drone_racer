"""Gate keypoint geometry.

The PnP object points must be defined in the *same gate actor frame* whose pose
is read from Isaac Lab. Do not substitute the rigid-body COM frame.

Repository convention for the rectangular helper:
- +X: gate normal / nominal flight-through direction at yaw=0
- +Y: left when looking through the gate along +X
- +Z: up

The semantic keypoint order is:
left-bottom, right-bottom, right-top, left-top as viewed from the approach side
(negative X looking toward +X).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


CORNER_NAMES = ("left_bottom", "right_bottom", "right_top", "left_top")


@dataclass(frozen=True)
class GateGeometry:
    """Four calibrated gate opening keypoints in the gate actor frame."""

    object_points_g: np.ndarray
    corner_names: tuple[str, str, str, str] = CORNER_NAMES
    source: str = "explicit"

    def __post_init__(self) -> None:
        points = np.asarray(self.object_points_g, dtype=np.float64)
        if points.shape != (4, 3):
            raise ValueError("GateGeometry.object_points_g must have shape (4, 3)")
        if not np.all(np.isfinite(points)):
            raise ValueError("GateGeometry contains non-finite values")
        centered = points - points.mean(axis=0, keepdims=True)
        if np.linalg.matrix_rank(centered, tol=1e-9) != 2:
            raise ValueError("Gate PnP keypoints must be coplanar and non-collinear")
        object.__setattr__(self, "object_points_g", points)

    @classmethod
    def rectangular_x_normal(
        cls,
        opening_width_m: float,
        opening_height_m: float,
        *,
        x_offset_m: float = 0.0,
        source: str = "rectangular_x_normal",
    ) -> "GateGeometry":
        """Build a rectangular opening in the YZ plane.

        This helper is appropriate only after confirming the gate asset actor
        frame uses the repository convention described in this module.
        """
        if opening_width_m <= 0.0 or opening_height_m <= 0.0:
            raise ValueError("Gate opening dimensions must be positive")
        w = opening_width_m * 0.5
        h = opening_height_m * 0.5
        return cls(
            np.array(
                [
                    [x_offset_m, +w, -h],
                    [x_offset_m, -w, -h],
                    [x_offset_m, -w, +h],
                    [x_offset_m, +w, +h],
                ],
                dtype=np.float64,
            ),
            source=source,
        )

    def corners_g(self, device="cpu", dtype=None):
        """Return Torch points for legacy/Isaac call sites."""
        import torch

        dtype = torch.float32 if dtype is None else dtype
        return torch.as_tensor(self.object_points_g, dtype=dtype, device=device)

    @property
    def center_g(self) -> np.ndarray:
        return self.object_points_g.mean(axis=0)

    @property
    def opening_width_m(self) -> float:
        bottom = np.linalg.norm(self.object_points_g[1] - self.object_points_g[0])
        top = np.linalg.norm(self.object_points_g[2] - self.object_points_g[3])
        return float(0.5 * (bottom + top))

    @property
    def opening_height_m(self) -> float:
        left = np.linalg.norm(self.object_points_g[3] - self.object_points_g[0])
        right = np.linalg.norm(self.object_points_g[2] - self.object_points_g[1])
        return float(0.5 * (left + right))
