"""Swift 2023 translational VIO-drift Kalman filter.

VIO supplies the high-rate pose. Mapped gate observations estimate only
translation/velocity drift; VIO attitude is retained. Gate corrections are
protected by a per-measurement Mahalanobis gate and the covariance update uses
the Joseph form for numerical robustness.
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
    position_w_b: np.ndarray
    linear_velocity_w_b: np.ndarray
    orientation_w_b_wxyz: np.ndarray
    timestamp_s: float

    def __post_init__(self) -> None:
        position = np.asarray(self.position_w_b, dtype=np.float64).reshape(3)
        velocity = np.asarray(self.linear_velocity_w_b, dtype=np.float64).reshape(3)
        orientation = np.asarray(self.orientation_w_b_wxyz, dtype=np.float64).reshape(4)
        timestamp = float(self.timestamp_s)
        norm = np.linalg.norm(orientation)
        if norm <= 0.0 or not np.isfinite(norm):
            raise ValueError("VIO orientation quaternion must have positive finite norm")
        orientation = orientation / norm
        if not np.all(np.isfinite(position)) or not np.all(np.isfinite(velocity)):
            raise ValueError("VIO state contains non-finite values")
        if not np.isfinite(timestamp):
            raise ValueError("VIO timestamp must be finite")
        object.__setattr__(self, "position_w_b", position)
        object.__setattr__(self, "linear_velocity_w_b", velocity)
        object.__setattr__(self, "orientation_w_b_wxyz", orientation)
        object.__setattr__(self, "timestamp_s", timestamp)

    @classmethod
    def from_stage1_state_estimate(cls, state_estimate, env_id: int = 0) -> "VioWorldEstimate":
        return cls(
            position_w_b=_to_numpy(state_estimate.position_w_b[env_id]),
            linear_velocity_w_b=_to_numpy(state_estimate.linear_velocity_w_b[env_id]),
            orientation_w_b_wxyz=_to_numpy(state_estimate.orientation_w_b[env_id]),
            timestamp_s=float(_to_numpy(state_estimate.publish_timestamp_s[env_id]).item()),
        )


@dataclass(frozen=True)
class FusedWorldEstimate:
    position_w_b: np.ndarray
    linear_velocity_w_b: np.ndarray
    orientation_w_b_wxyz: np.ndarray
    estimated_position_drift_w: np.ndarray
    estimated_velocity_drift_w: np.ndarray
    drift_covariance: np.ndarray
    timestamp_s: float


@dataclass(frozen=True)
class KalmanUpdateDiagnostics:
    attempted: int = 0
    accepted: int = 0
    rejected: int = 0
    mahalanobis2: tuple[float, ...] = ()


class VioDriftKalmanFilter:
    """Single-vehicle Swift translational drift estimator.

    State is ``x=[p_d, v_d]``. ``innovation_gate_chi2`` is a squared
    Mahalanobis threshold for each 3-D gate-position measurement. The default
    16.266 corresponds approximately to a 99.9% chi-square threshold for 3 DoF.
    Set it to ``None`` only for controlled diagnostics.
    """

    def __init__(
        self,
        *,
        sigma_pos: float = 0.05,
        sigma_vel: float = 0.1,
        innovation_gate_chi2: float | None = 16.26623619623813,
    ) -> None:
        if sigma_pos < 0.0 or sigma_vel < 0.0:
            raise ValueError("process-noise terms must be non-negative")
        if innovation_gate_chi2 is not None and innovation_gate_chi2 <= 0.0:
            raise ValueError("innovation_gate_chi2 must be positive or None")
        self.sigma_pos = float(sigma_pos)
        self.sigma_vel = float(sigma_vel)
        self.innovation_gate_chi2 = (
            None if innovation_gate_chi2 is None else float(innovation_gate_chi2)
        )
        self.x = np.zeros(6, dtype=np.float64)
        self.P = np.zeros((6, 6), dtype=np.float64)
        self.last_timestamp_s: float | None = None
        self.last_update_diagnostics = KalmanUpdateDiagnostics()

    def reset(self, timestamp_s: float | None = None) -> None:
        self.x.fill(0.0)
        self.P.fill(0.0)
        self.last_timestamp_s = None if timestamp_s is None else float(timestamp_s)
        self.last_update_diagnostics = KalmanUpdateDiagnostics()

    def predict(self, timestamp_s: float) -> None:
        timestamp_s = float(timestamp_s)
        if not np.isfinite(timestamp_s):
            raise ValueError("VIO drift filter timestamp must be finite")
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

    @staticmethod
    def _validated_covariance(covariance) -> np.ndarray:
        R = np.asarray(covariance, dtype=np.float64).reshape(3, 3)
        if not np.all(np.isfinite(R)):
            raise ValueError("Gate position covariance contains non-finite values")
        R = 0.5 * (R + R.T)
        eigenvalues = np.linalg.eigvalsh(R)
        if eigenvalues[0] < -1.0e-9:
            raise ValueError("Gate position covariance must be positive semidefinite")
        if eigenvalues[0] < 1.0e-12:
            R = R + np.eye(3) * (1.0e-12 - eigenvalues[0])
        return R

    def update_from_gate_positions(
        self,
        vio: VioWorldEstimate,
        gate_positions_w_b,
        position_covariances_w,
    ) -> int:
        """Correct drift using mapped gate-derived body positions.

        Each candidate is first tested independently with a 3-D normalized
        innovation squared (NIS) gate. Only accepted candidates are stacked into
        the Kalman update. Returns the number of accepted measurements.
        """
        positions = [np.asarray(p, dtype=np.float64).reshape(3) for p in gate_positions_w_b]
        covariances = [self._validated_covariance(R) for R in position_covariances_w]
        if len(positions) != len(covariances):
            raise ValueError("gate position/covariance counts do not match")
        if not positions:
            self.last_update_diagnostics = KalmanUpdateDiagnostics()
            self.predict(vio.timestamp_s)
            return 0
        if not all(np.all(np.isfinite(p)) for p in positions):
            raise ValueError("Gate-derived positions must be finite")

        self.predict(vio.timestamp_s)
        H_one = np.concatenate((np.eye(3), np.zeros((3, 3))), axis=1)

        accepted_positions: list[np.ndarray] = []
        accepted_covariances: list[np.ndarray] = []
        mahalanobis2: list[float] = []
        for position, covariance in zip(positions, covariances):
            z_i = vio.position_w_b - position
            innovation_i = z_i - H_one @ self.x
            S_i = H_one @ self.P @ H_one.T + covariance
            try:
                solved = np.linalg.solve(S_i, innovation_i)
            except np.linalg.LinAlgError as exc:
                raise RuntimeError("Gate/VIO innovation covariance is singular") from exc
            d2 = float(innovation_i.T @ solved)
            mahalanobis2.append(d2)
            if self.innovation_gate_chi2 is not None and d2 > self.innovation_gate_chi2:
                continue
            accepted_positions.append(position)
            accepted_covariances.append(covariance)

        attempted = len(positions)
        accepted = len(accepted_positions)
        self.last_update_diagnostics = KalmanUpdateDiagnostics(
            attempted=attempted,
            accepted=accepted,
            rejected=attempted - accepted,
            mahalanobis2=tuple(mahalanobis2),
        )
        if not accepted_positions:
            return 0

        H = np.concatenate([H_one for _ in accepted_positions], axis=0)
        z = np.concatenate(
            [vio.position_w_b - position for position in accepted_positions], axis=0
        )
        R = np.zeros((3 * accepted, 3 * accepted), dtype=np.float64)
        for i, covariance in enumerate(accepted_covariances):
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
        A = I6 - K @ H
        # Joseph form preserves symmetry/PSD better than (I-KH)P.
        self.P = A @ self.P @ A.T + K @ R @ K.T
        self.P = 0.5 * (self.P + self.P.T)
        return accepted

    def corrected(self, vio: VioWorldEstimate) -> FusedWorldEstimate:
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
        positions = list(gate_positions_w_b)
        covariances = list(position_covariances_w)
        if positions:
            self.update_from_gate_positions(vio, positions, covariances)
        else:
            self.predict(vio.timestamp_s)
            self.last_update_diagnostics = KalmanUpdateDiagnostics()
        return self.corrected(vio)
