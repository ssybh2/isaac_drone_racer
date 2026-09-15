"""Relative learned-motion correction for world-aligned VIO.

The filter estimates translational VIO drift only. A learned motion model
provides a relative displacement measurement over a finite time window; the
measurement is compared with raw VIO displacement over the same interval and
used to estimate the drift accumulated during that interval. Optional absolute
position measurements observe the current position drift directly and are used
only by explicit absolute-anchor sources such as gate PnP or simulator-oracle
diagnostics.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .swift_vio_drift import VioWorldEstimate


@dataclass(frozen=True)
class LearnedDisplacementMeasurement:
    """One learned relative-displacement constraint in world coordinates."""

    displacement_w: np.ndarray
    covariance_w: np.ndarray
    start_timestamp_s: float
    end_timestamp_s: float

    def __post_init__(self) -> None:
        displacement = np.asarray(self.displacement_w, dtype=np.float64).reshape(3)
        covariance = np.asarray(self.covariance_w, dtype=np.float64).reshape(3, 3)
        covariance = 0.5 * (covariance + covariance.T)
        start = float(self.start_timestamp_s)
        end = float(self.end_timestamp_s)
        if not np.all(np.isfinite(displacement)):
            raise ValueError("Learned displacement must be finite")
        if not np.all(np.isfinite(covariance)):
            raise ValueError("Learned displacement covariance must be finite")
        if not np.isfinite(start) or not np.isfinite(end) or end <= start:
            raise ValueError("Learned displacement timestamps must be finite and increasing")
        eig = np.linalg.eigvalsh(covariance)
        if eig[0] < -1.0e-10:
            raise ValueError("Learned displacement covariance must be positive semidefinite")
        if eig[0] < 1.0e-12:
            covariance = covariance + np.eye(3) * (1.0e-12 - eig[0])
        object.__setattr__(self, "displacement_w", displacement)
        object.__setattr__(self, "covariance_w", covariance)
        object.__setattr__(self, "start_timestamp_s", start)
        object.__setattr__(self, "end_timestamp_s", end)


@dataclass(frozen=True)
class LearnedDriftUpdateDiagnostics:
    attempted: int = 0
    accepted: int = 0
    rejected: int = 0
    mahalanobis2: float | None = None
    innovation_w: tuple[float, float, float] | None = None


class LearnedVioDriftFilter:
    """Estimate translational VIO drift from learned relative displacement.

    State ordering is ``[p_d_anchor, p_d_current, v_d_current]``. The anchor
    clone is fixed during propagation. A learned displacement over the same VIO
    interval observes ``p_d_current - p_d_anchor``. An optional absolute world
    position observes ``p_d_current`` through ``p_vio - p_absolute``.
    """

    def __init__(
        self,
        *,
        sigma_position: float = 0.05,
        sigma_velocity: float = 0.1,
        innovation_gate_chi2: float | None = 16.26623619623813,
        nominal_rate_hz: float = 100.0,
        timestamp_tolerance_s: float = 2.0e-3,
    ) -> None:
        if sigma_position < 0.0 or sigma_velocity < 0.0:
            raise ValueError("process-noise terms must be non-negative")
        if innovation_gate_chi2 is not None and innovation_gate_chi2 <= 0.0:
            raise ValueError("innovation_gate_chi2 must be positive or None")
        if nominal_rate_hz <= 0.0 or not np.isfinite(nominal_rate_hz):
            raise ValueError("nominal_rate_hz must be positive and finite")
        if timestamp_tolerance_s < 0.0 or not np.isfinite(timestamp_tolerance_s):
            raise ValueError("timestamp_tolerance_s must be non-negative and finite")
        self.sigma_position = float(sigma_position)
        self.sigma_velocity = float(sigma_velocity)
        self.innovation_gate_chi2 = (
            None if innovation_gate_chi2 is None else float(innovation_gate_chi2)
        )
        self.nominal_rate_hz = float(nominal_rate_hz)
        self.nominal_dt_s = 1.0 / self.nominal_rate_hz
        self.timestamp_tolerance_s = float(timestamp_tolerance_s)
        self.x = np.zeros(9, dtype=np.float64)
        self.P = np.zeros((9, 9), dtype=np.float64)
        self.last_timestamp_s: float | None = None
        self.anchor_timestamp_s: float | None = None
        self.anchor_vio_position_w: np.ndarray | None = None
        self.last_update_diagnostics = LearnedDriftUpdateDiagnostics()
        self.last_absolute_update_diagnostics = LearnedDriftUpdateDiagnostics()

    @property
    def anchor_position_drift_w(self) -> np.ndarray:
        return self.x[0:3].copy()

    @property
    def current_position_drift_w(self) -> np.ndarray:
        return self.x[3:6].copy()

    @property
    def current_velocity_drift_w(self) -> np.ndarray:
        return self.x[6:9].copy()

    def reset(
        self,
        timestamp_s: float | None = None,
        *,
        anchor_vio_position_w: np.ndarray | None = None,
    ) -> None:
        if timestamp_s is not None and not np.isfinite(float(timestamp_s)):
            raise ValueError("timestamp_s must be finite")
        if anchor_vio_position_w is not None:
            anchor = np.asarray(anchor_vio_position_w, dtype=np.float64).reshape(3)
            if not np.all(np.isfinite(anchor)):
                raise ValueError("anchor_vio_position_w must be finite")
            self.anchor_vio_position_w = anchor.copy()
        else:
            self.anchor_vio_position_w = None
        self.x.fill(0.0)
        self.P.fill(0.0)
        self.last_timestamp_s = None if timestamp_s is None else float(timestamp_s)
        self.anchor_timestamp_s = None if timestamp_s is None else float(timestamp_s)
        self.last_update_diagnostics = LearnedDriftUpdateDiagnostics()
        self.last_absolute_update_diagnostics = LearnedDriftUpdateDiagnostics()

    def _ensure_initialized(self, vio: VioWorldEstimate) -> None:
        if self.anchor_vio_position_w is None:
            self.reset(vio.timestamp_s, anchor_vio_position_w=vio.position_w_b)
        elif self.anchor_timestamp_s is None:
            self.anchor_timestamp_s = float(vio.timestamp_s)
        if self.last_timestamp_s is None:
            self.last_timestamp_s = float(vio.timestamp_s)

    @staticmethod
    def _validated_covariance(covariance_w: np.ndarray, *, name: str) -> np.ndarray:
        covariance = np.asarray(covariance_w, dtype=np.float64).reshape(3, 3)
        covariance = 0.5 * (covariance + covariance.T)
        if not np.all(np.isfinite(covariance)):
            raise ValueError(f"{name} covariance must be finite")
        eig = np.linalg.eigvalsh(covariance)
        if eig[0] < -1.0e-10:
            raise ValueError(f"{name} covariance must be positive semidefinite")
        if eig[0] < 1.0e-12:
            covariance = covariance + np.eye(3) * (1.0e-12 - eig[0])
        return covariance

    def predict(self, timestamp_s: float) -> None:
        timestamp = float(timestamp_s)
        if not np.isfinite(timestamp):
            raise ValueError("VIO drift filter timestamp must be finite")
        if self.last_timestamp_s is None:
            self.last_timestamp_s = timestamp
            return
        dt = timestamp - self.last_timestamp_s
        if dt < -1.0e-9:
            raise ValueError("VIO drift filter timestamps must be monotonic")
        dt = max(0.0, dt)
        if dt <= 1.0e-12:
            self.last_timestamp_s = timestamp
            return

        I3 = np.eye(3, dtype=np.float64)
        F = np.eye(9, dtype=np.float64)
        F[3:6, 6:9] = dt * I3

        scale = dt / self.nominal_dt_s
        q_pos = self.sigma_position * scale
        q_vel = self.sigma_velocity * scale
        # Correlate current position and velocity drift so a displacement
        # update can also inform drift velocity on the first learned window.
        q_pv = 0.5 * dt * q_vel
        Q = np.zeros((9, 9), dtype=np.float64)
        Q[3:6, 3:6] = (q_pos + 0.25 * dt * dt * q_vel) * I3
        Q[3:6, 6:9] = q_pv * I3
        Q[6:9, 3:6] = q_pv * I3
        Q[6:9, 6:9] = q_vel * I3

        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q
        self.P = 0.5 * (self.P + self.P.T)
        self.last_timestamp_s = timestamp

    def _measurement_update(
        self,
        vio: VioWorldEstimate,
        measurement: LearnedDisplacementMeasurement,
    ) -> bool:
        if self.anchor_vio_position_w is None or self.anchor_timestamp_s is None:
            raise RuntimeError("Learned VIO drift filter is not anchored")
        if abs(measurement.start_timestamp_s - self.anchor_timestamp_s) > self.timestamp_tolerance_s:
            raise ValueError(
                "Learned displacement start timestamp does not match drift-filter anchor"
            )
        if abs(measurement.end_timestamp_s - float(vio.timestamp_s)) > self.timestamp_tolerance_s:
            raise ValueError("Learned displacement end timestamp does not match VIO timestamp")

        dp_vio = vio.position_w_b - self.anchor_vio_position_w
        z = dp_vio - measurement.displacement_w
        H = np.zeros((3, 9), dtype=np.float64)
        H[:, 0:3] = -np.eye(3)
        H[:, 3:6] = np.eye(3)
        innovation = z - H @ self.x
        R = measurement.covariance_w
        S = H @ self.P @ H.T + R
        try:
            solved = np.linalg.solve(S, innovation)
        except np.linalg.LinAlgError as exc:
            raise RuntimeError("Learned displacement innovation covariance is singular") from exc
        d2 = float(innovation.T @ solved)

        if self.innovation_gate_chi2 is not None and d2 > self.innovation_gate_chi2:
            self.last_update_diagnostics = LearnedDriftUpdateDiagnostics(
                attempted=1,
                accepted=0,
                rejected=1,
                mahalanobis2=d2,
                innovation_w=tuple(float(v) for v in innovation),
            )
            return False

        PHt = self.P @ H.T
        try:
            K = np.linalg.solve(S.T, PHt.T).T
        except np.linalg.LinAlgError as exc:
            raise RuntimeError("Learned displacement Kalman gain solve failed") from exc
        self.x = self.x + K @ innovation
        I9 = np.eye(9, dtype=np.float64)
        A = I9 - K @ H
        self.P = A @ self.P @ A.T + K @ R @ K.T
        self.P = 0.5 * (self.P + self.P.T)
        self.last_update_diagnostics = LearnedDriftUpdateDiagnostics(
            attempted=1,
            accepted=1,
            rejected=0,
            mahalanobis2=d2,
            innovation_w=tuple(float(v) for v in innovation),
        )
        return True

    def apply_absolute_position(
        self,
        vio: VioWorldEstimate,
        *,
        position_w_b: np.ndarray,
        covariance_w: np.ndarray,
    ) -> VioWorldEstimate:
        """Fuse one absolute world-position anchor into the current drift state.

        The measurement model is ``p_vio - p_absolute = p_d_current``. This
        method deliberately does not re-anchor the learned-displacement window,
        so sparse absolute updates can arrive between learned window boundaries
        without invalidating the relative measurement timestamp contract.
        """
        self._ensure_initialized(vio)
        self.predict(vio.timestamp_s)
        position = np.asarray(position_w_b, dtype=np.float64).reshape(3)
        if not np.all(np.isfinite(position)):
            raise ValueError("Absolute position must be finite")
        R = self._validated_covariance(covariance_w, name="Absolute position")

        z = np.asarray(vio.position_w_b, dtype=np.float64) - position
        H = np.zeros((3, 9), dtype=np.float64)
        H[:, 3:6] = np.eye(3)
        innovation = z - H @ self.x
        S = H @ self.P @ H.T + R
        try:
            solved = np.linalg.solve(S, innovation)
        except np.linalg.LinAlgError as exc:
            raise RuntimeError("Absolute position innovation covariance is singular") from exc
        d2 = float(innovation.T @ solved)

        PHt = self.P @ H.T
        try:
            K = np.linalg.solve(S.T, PHt.T).T
        except np.linalg.LinAlgError as exc:
            raise RuntimeError("Absolute position Kalman gain solve failed") from exc
        self.x = self.x + K @ innovation
        I9 = np.eye(9, dtype=np.float64)
        A = I9 - K @ H
        self.P = A @ self.P @ A.T + K @ R @ K.T
        self.P = 0.5 * (self.P + self.P.T)
        self.last_absolute_update_diagnostics = LearnedDriftUpdateDiagnostics(
            attempted=1,
            accepted=1,
            rejected=0,
            mahalanobis2=d2,
            innovation_w=tuple(float(v) for v in innovation),
        )
        return self.corrected(vio)

    def reanchor(self, vio: VioWorldEstimate) -> None:
        """Clone current drift/current raw VIO position as a new window anchor."""
        # Linear transform x'=[p_current, p_current, v_current].
        T = np.zeros((9, 9), dtype=np.float64)
        T[0:3, 3:6] = np.eye(3)
        T[3:6, 3:6] = np.eye(3)
        T[6:9, 6:9] = np.eye(3)
        self.x = T @ self.x
        self.P = T @ self.P @ T.T
        self.P = 0.5 * (self.P + self.P.T)
        self.anchor_vio_position_w = np.asarray(vio.position_w_b, dtype=np.float64).copy()
        self.anchor_timestamp_s = float(vio.timestamp_s)

    def corrected(self, vio: VioWorldEstimate) -> VioWorldEstimate:
        return VioWorldEstimate(
            position_w_b=vio.position_w_b - self.x[3:6],
            linear_velocity_w_b=vio.linear_velocity_w_b - self.x[6:9],
            orientation_w_b_wxyz=vio.orientation_w_b_wxyz.copy(),
            timestamp_s=vio.timestamp_s,
        )

    def step(
        self,
        vio: VioWorldEstimate,
        *,
        measurement: LearnedDisplacementMeasurement | None = None,
    ) -> VioWorldEstimate:
        self._ensure_initialized(vio)
        self.predict(vio.timestamp_s)
        self.last_update_diagnostics = LearnedDriftUpdateDiagnostics()
        accepted = False
        if measurement is not None:
            accepted = self._measurement_update(vio, measurement)
        corrected = self.corrected(vio)
        if accepted:
            self.reanchor(vio)
        return corrected
