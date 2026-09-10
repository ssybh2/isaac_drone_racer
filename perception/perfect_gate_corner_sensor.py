import torch
from .camera_model import project_points_pinhole
from .gate_geometry import GateGeometry


class PerfectGateCornerSensor:
    """Stage2A ideal detector.

    Input: simulator truth pose.
    Output: only image corners.
    Policy never receives world gate truth directly.
    """

    def __init__(self, geometry: GateGeometry, K):
        self.geometry = geometry
        self.K = K

    def measure(self, R_cg, t_cg):
        corners_g = self.geometry.corners_g()
        corners_c = (R_cg @ corners_g.T).T + t_cg
        uv = project_points_pinhole(corners_c, self.K)
        return uv
