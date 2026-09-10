import numpy as np
from .planar_pnp import solve_gate_pnp


class GatePoseEstimator:
    def __init__(self, geometry, K):
        self.geometry = geometry
        self.K = K

    def estimate(self, corners_uv):
        R, t = solve_gate_pnp(
            self.geometry.corners_g().cpu().numpy(),
            np.asarray(corners_uv),
            self.K,
        )
        return R, t
