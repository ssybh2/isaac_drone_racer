"""Swift 2023 translational VIO-drift Kalman filter.

The paper does not replace VIO with gate PnP.  VIO supplies the high-rate state;
gate/IPPE observations are mapped landmarks that periodically measure VIO
translation drift.  Only translation and velocity drift are corrected; VIO
orientation is retained.

State:
    x = [p_d, v_d] in R^6

Prediction follows Kaufmann et al. (Nature 2023), equations (12)-(13):
    F = [[I, dt I], [0, I]]
    Q = diag(sigma_pos I, sigma_vel I)
with sigma_pos=0.05 and sigma_vel=0.1 by default.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _to_numpy(value) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    return np.asarray(value)


@dataclass(frozen=True)
class VioWorldEstimate:
    """World-aligned VIO state consumed by the Swift drift filter."""

    position_w_b: np.ndarray
    linear_velocity_w_b: np.ndarray
    orientation_w_b_wxyz: np.ndarray
    timestamp_s: float

    def __post_init__(self) -> None:
        position = np.asarray(self.position_w_b, dtype=np.float64).reshape(3)
        velocity = np.asarray(self.linear_velocity_w_b, dtype=np.float64).reshape(3)
        orientation = np.asarray(self.orientation_w_b_wxyz, dtype=np.float64).reshape(4)
        norm = np.linalg.norm(orientation)
        if norm <= 0.0:
            raise ValueError("VIO orientation quaternion must have positive norm")
        orientation = orientation / norm
        if not np.all(np.isfinite(position)) or not np.all(np.isfinite(velocity)):
            raise ValueError("VIO state contains non-finite values")
        object.__setattr__(self, "position_w_b", position)
        object.__setattr__(self, "linear_velocity_w_b", velocity)
        object.__setattr__(self, "orientation_w_b_wxyz", orientation)
        object.__setattr__(self, "timestamp_s", float(self.timestamp_s))

    @classmethod
    def from_stage1_state_estimate(cls, state_estimate, env_id: int = 0) -> "VioWorldEstimate":
        """Adapt the repository's existing Fake-VIO-backed Stage1 state.

        This is the simulation stand-in for Swift's real VIO.  A future hardware
        VIO adapter can satisfy the same contract without changing the filter.
        """

        return cls(
            position_w_b=_to_numpy(state_estimate.position_w_b[env_id]),
            linear_velocity_w_b=_to_numpy(state_estimate.linear_velocity_w_b[env_id]),
            orientation_w_b_wxyz=_to_numpy(state_estimate.orientation_w_b[env_id]),
            timestamp_s=float(_to_numpy(state_estimate.publish_timestamp_s[env_id]).item()),
        )


@dataclass(frozen=True)
class FusedWorldEstimate:
    """VIO state after subtracting the estimated translational drift."""

    position_w_b: np.ndarray
    linear_velocity_w_b: np.ndarray
    orientation_w_b_wxyz: np.ndarray
    estimated_position_drift_w: np.ndarray
    estimated_velocity_drift_w: np.ndarray
    drift_covariance: np.ndarray
    timestamp_s: float


class VioDriftKalmanFilter:
    """Single-vehicle reference implementation of Swift's drift estimator."""

    def __init__(self, *, sigma_pos: float = 0.05, sigma_vel: float = 0.1) -> None:
        if sigma_pos < 0.0 or sigma_vel < 0.0:
            raise ValueError("process-noise terms must be non-negative")
        self.sigma_pos = float(sigma_pos)
        self.sigma_vel = float(sigma_vel)
        self.x = np.zeros(6, dtype=np.float64)
        self.P = np.zeros((6, 6), dtype=np.float64)
        self.last_timestamp_s: float | None = None

    def reset(self, timestamp_s: float | None = None) -> None:
        self.x.fill(0.0)
        self.P.fill(0.0)
        self.last_timestamp_s = None if timestamp_s is None else float(timestamp_s)

    def predict(self, timestamp_s: float) -> None:
        timestamp_s = float(timestamp_s)
        if self.last_timestamp_s is None:
            self.last_timestamp_s = timestamp_s
            return
        dt = timestamp_s - self.last_timestamp_s
        if dt < -1.0e-9:
            raise ValueError("VIO drift filter timestamps must be monotonic")
        dt = max(0.0, dt)

        I3 = np.eye(3, dtype=np.float64)
        F = np.block([[I3, dt * I3], [np.zeros((3, 3)), I3]])
        Q = np.block(
            [
                [self.sigma_pos * I3, np.zeros((3, 3))],
                [np.zeros((3, 3)), self.sigma_vel * I3],
            ]
        )
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q
        self.P = 0.5 * (self.P + self.P.T)
        self.last_timestamp_s = timestamp_s

    def update_from_gate_positions(
        self,
        vio: VioWorldEstimate,
        gate_positions_w_b,
        position_covariances_w,
    ) -> None:
        """Correct VIO drift from one or more mapped gate-derived positions.

        For each gate observation, the drift measurement is

            z_i = p_vio - p_gate_i

        because the corrected position is ``p_vio - p_d``.  Multiple visible
        gates are stacked into one Kalman update, matching the paper.
        """

        positions = [np.asarray(p, dtype=np.float64).reshape(3) for p in gate_positions_w_b]
        covariances = [
            np.asarray(R, dtype=np.float64).reshape(3, 3) for R in position_covariances_w
        ]
        if len(positions) != len(covariances):
            raise ValueError("gate position/covariance counts do not match")
        if not positions:
            return

        self.predict(vio.timestamp_s)

        H_one = np.concatenate((np.eye(3), np.zeros((3, 3))), axis=1)
        H = np.concatenate([H_one for _ in positions], axis=0)
        z = np.concatenate([vio.position_w_b - position for position in positions], axis=0)

        count = len(covariances)
        R = np.zeros((3 * count, 3 * count), dtype=np.float64)
        for i, covariance in enumerate(covariances):
            covariance = 0.5 * (covariance + covariance.T)
            R[3 * i : 3 * (i + 1), 3 * i : 3 * (i + 1)] = covariance

        innovation = z - H @ self.x
        S = H @ self.P @ H.T + R
        PHt = self.P @ H.T
        try:
            K = np.linalg.solve(S.T, PHt.T).T
        except np.linalg.LinAlgError as exc:
            raise RuntimeError("Gate/VIO Kalman innovation covariance is singular") from exc

        self.x = self.x + K @ innovation
        I6 = np.eye(6, dtype=np.float64)
        self.P = (I6 - K @ H) @ self.P
        self.P = 0.5 * (self.P + self.P.T)

    def corrected(self, vio: VioWorldEstimate) -> FusedWorldEstimate:
        """Return the current corrected state without changing VIO orientation."""

        return FusedWorldEstimate(
            position_w_b=vio.position_w_b - self.x[:3],
            linear_velocity_w_b=vio.linear_velocity_w_b - self.x[3:],
            orientation_w_b_wxyz=vio.orientation_w_b_wxyz.copy(),
            estimated_position_drift_w=self.x[:3].copy(),
            estimated_velocity_drift_w=self.x[3:].copy(),
            drift_covariance=self.P.copy(),
            timestamp_s=vio.timestamp_s,
        )

    def step(
        self,
        vio: VioWorldEstimate,
        *,
        gate_positions_w_b=(),
        position_covariances_w=(),
    ) -> FusedWorldEstimate:
        """Advance to one VIO timestamp and optionally apply gate corrections."""

        positions = list(gate_positions_w_b)
        covariances = list(position_covariances_w)
        if positions:
            self.update_from_gate_positions(vio, positions, covariances)
        else:
            self.predict(vio.timestamp_s)
        return self.corrected(vio)
