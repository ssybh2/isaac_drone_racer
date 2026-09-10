"""Stage2A perception pipeline.

Flow:
Isaac gate truth -> 3D gate corners -> camera projection -> PnP -> gate pose.

This module intentionally does not expose oracle target_pos_b.
"""
from dataclasses import dataclass
import torch

from .gate_geometry import GateGeometry
from .camera_model import project_points
from .planar_pnp import solve_gate_pnp


@dataclass
class GateObservation:
    corners_uv: torch.Tensor
    timestamp: float = 0.0


class Stage2APerceptionPipeline:
    def __init__(self, geometry=None):
        self.geometry = geometry or GateGeometry()

    def project_perfect_corners(self, T_cg, K):
        corners_g = self.geometry.corners_g()
        return project_points(corners_g, T_cg, K)

    def estimate_pose(self, corners_uv, K):
        return solve_gate_pnp(
            self.geometry.corners_g().cpu().numpy(),
            corners_uv,
            K,
        )
