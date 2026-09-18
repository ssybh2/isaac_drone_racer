"""Standalone learned inertial odometry (IMO-style) estimator.

This module intentionally has no OpenVINS dependency. It follows the structure
of Cioffi et al. (RAL 2023):

* IMU propagation estimates attitude, velocity, position and IMU biases.
* Learned 0.5 s relative displacements are injected as Kalman measurements.
* Timestamped velocity+position clones retain the cross-covariances required
  for overlapping fixed-lag relative-motion updates.
* Optional mapped-gate PnP measurements provide absolute pose anchors.

The current error state is [dtheta, dv, dp, dba, dbg] (15 states). Each
historical kinematic clone appends [dv_clone, dp_clone] (6 states).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _skew(v: np.ndarray) -> np.ndarray:
    x, y, z = np.asarray(v, dtype=np.float64).reshape(3)
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]], dtype=np.float64)


def _exp_so3(phi: np.ndarray) -> np.ndarray:
    phi = np.asarray(phi, dtype=np.float64).reshape(3)
    theta = float(np.linalg.norm(phi))
    if theta < 1.0e-10:
        return np.eye(3) + _skew(phi)
    axis = phi / theta
    K = _skew(axis)
    return np.eye(3) + np.sin(theta) * K + (1.0 - np.cos(theta)) * (K @ K)


def _log_so3(R: np.ndarray) -> np.ndarray:
    R = np.asarray(R, dtype=np.float64).reshape(3, 3)
    cos_theta = float(np.clip((np.trace(R) - 1.0) * 0.5, -1.0, 1.0))
    theta = float(np.arccos(cos_theta))
    if theta < 1.0e-10:
        return 0.5 * np.array(
            [R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]],
            dtype=np.float64,
        )
    return theta / (2.0 * np.sin(theta)) * np.array(
        [R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]],
        dtype=np.float64,
    )


def quat_wxyz_to_rotmat(q) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64).reshape(4)
    norm = float(np.linalg.norm(q))
    if norm <= 0.0 or not np.isfinite(norm):
        raise ValueError("quaternion must be finite and non-zero")
    w, x, y, z = q / norm
    return np.array(
        [
            [1 - 2 * (y*y + z*z), 2 * (x*y - z*w), 2 * (x*z + y*w)],
            [2 * (x*y + z*w), 1 - 2 * (x*x + z*z), 2 * (y*z - x*w)],
            [2 * (x*z - y*w), 2 * (y*z + x*w), 1 - 2 * (x*x + y*y)],
        ],
        dtype=np.float64,
    )


def rotmat_to_quat_wxyz(R: np.ndarray) -> np.ndarray:
    R = np.asarray(R, dtype=np.float64).reshape(3, 3)
    t = float(np.trace(R))
    if t > 0.0:
        s = np.sqrt(t + 1.0) * 2.0
        q = np.array([0.25*s, (R[2,1]-R[1,2])/s, (R[0,2]-R[2,0])/s, (R[1,0]-R[0,1])/s])
    else:
        i = int(np.argmax(np.diag(R)))
        if i == 0:
            s = np.sqrt(1.0 + R[0,0] - R[1,1] - R[2,2]) * 2.0
            q = np.array([(R[2,1]-R[1,2])/s, 0.25*s, (R[0,1]+R[1,0])/s, (R[0,2]+R[2,0])/s])
        elif i == 1:
            s = np.sqrt(1.0 + R[1,1] - R[0,0] - R[2,2]) * 2.0
            q = np.array([(R[0,2]-R[2,0])/s, (R[0,1]+R[1,0])/s, 0.25*s, (R[1,2]+R[2,1])/s])
        else:
            s = np.sqrt(1.0 + R[2,2] - R[0,0] - R[1,1]) * 2.0
            q = np.array([(R[1,0]-R[0,1])/s, (R[0,2]+R[2,0])/s, (R[1,2]+R[2,1])/s, 0.25*s])
    q = q.astype(np.float64)
    q /= np.linalg.norm(q)
    if q[0] < 0.0:
        q = -q
    return q


def protected_displacement_covariance(
    covariance_w,
    *,
    sigma_floor_xyz_m=(0.10, 0.10, 0.01),
    covariance_scale: float = 1.25,
) -> np.ndarray:
    """Protect the EKF from learned-uncertainty collapse."""
    covariance = np.asarray(covariance_w, dtype=np.float64).reshape(3, 3)
    floor = np.asarray(sigma_floor_xyz_m, dtype=np.float64).reshape(3)
    scale = float(covariance_scale)
    if not np.all(np.isfinite(covariance)):
        raise ValueError("learned displacement covariance must be finite")
    if np.any(floor <= 0.0) or not np.all(np.isfinite(floor)):
        raise ValueError("sigma_floor_xyz_m must be positive and finite")
    if scale <= 0.0 or not np.isfinite(scale):
        raise ValueError("covariance_scale must be positive and finite")
    diagonal = np.diag(covariance)
    if np.any(diagonal < 0.0):
        raise ValueError("learned displacement covariance diagonal must be non-negative")
    predicted_sigma = np.sqrt(diagonal)
    sigma_used = np.maximum(predicted_sigma, floor)
    return np.diag((scale * sigma_used) ** 2)


@dataclass(frozen=True)
class LearnedInertialState:
    timestamp_s: float
    position_w_b: np.ndarray
    linear_velocity_w_b: np.ndarray
    orientation_w_b_wxyz: np.ndarray
    accel_bias_b: np.ndarray
    gyro_bias_b: np.ndarray
    covariance: np.ndarray

    def __post_init__(self) -> None:
        object.__setattr__(self, "position_w_b", np.asarray(self.position_w_b, dtype=np.float64).reshape(3))
        object.__setattr__(self, "linear_velocity_w_b", np.asarray(self.linear_velocity_w_b, dtype=np.float64).reshape(3))
        object.__setattr__(self, "orientation_w_b_wxyz", np.asarray(self.orientation_w_b_wxyz, dtype=np.float64).reshape(4))
        object.__setattr__(self, "accel_bias_b", np.asarray(self.accel_bias_b, dtype=np.float64).reshape(3))
        object.__setattr__(self, "gyro_bias_b", np.asarray(self.gyro_bias_b, dtype=np.float64).reshape(3))
        object.__setattr__(self, "covariance", np.asarray(self.covariance, dtype=np.float64))


class LearnedInertialOdometry:
    """Fixed-lag error-state EKF with timestamped velocity+position clones."""

    _CURRENT_DIM = 15
    _CLONE_DIM = 6

    def __init__(
        self,
        *,
        gravity_w=(0.0, 0.0, -9.81),
        accel_noise_sigma=0.01,
        gyro_noise_sigma=0.001,
        accel_bias_rw_sigma=0.001,
        gyro_bias_rw_sigma=0.0001,
        max_position_clones: int = 11,
    ) -> None:
        if int(max_position_clones) < 1:
            raise ValueError("max_position_clones must be at least 1")
        self.gravity_w = np.asarray(gravity_w, dtype=np.float64).reshape(3)
        self.accel_noise_sigma = float(accel_noise_sigma)
        self.gyro_noise_sigma = float(gyro_noise_sigma)
        self.accel_bias_rw_sigma = float(accel_bias_rw_sigma)
        self.gyro_bias_rw_sigma = float(gyro_bias_rw_sigma)
        self.max_position_clones = int(max_position_clones)
        self.reset()

    def reset(
        self,
        *,
        timestamp_s: float = 0.0,
        position_w_b=(0.0, 0.0, 0.0),
        linear_velocity_w_b=(0.0, 0.0, 0.0),
        orientation_w_b_wxyz=(1.0, 0.0, 0.0, 0.0),
        accel_bias_b=(0.0, 0.0, 0.0),
        gyro_bias_b=(0.0, 0.0, 0.0),
        initial_covariance: np.ndarray | None = None,
    ) -> None:
        self.timestamp_s = float(timestamp_s)
        self.p = np.asarray(position_w_b, dtype=np.float64).reshape(3).copy()
        self.v = np.asarray(linear_velocity_w_b, dtype=np.float64).reshape(3).copy()
        self.R = quat_wxyz_to_rotmat(orientation_w_b_wxyz)
        self.ba = np.asarray(accel_bias_b, dtype=np.float64).reshape(3).copy()
        self.bg = np.asarray(gyro_bias_b, dtype=np.float64).reshape(3).copy()
        self.P = (
            np.eye(self._CURRENT_DIM, dtype=np.float64) * 1.0e-3
            if initial_covariance is None
            else np.asarray(initial_covariance, dtype=np.float64)
            .reshape(self._CURRENT_DIM, self._CURRENT_DIM)
            .copy()
        )
        self._clone_velocities: list[np.ndarray] = []
        self._clone_positions: list[np.ndarray] = []
        self._clone_timestamps_s: list[float] = []

    @property
    def clone_count(self) -> int:
        return len(self._clone_positions)

    @property
    def clone_timestamps_s(self) -> tuple[float, ...]:
        return tuple(self._clone_timestamps_s)

    @property
    def clone_velocities_w_b(self) -> tuple[np.ndarray, ...]:
        return tuple(velocity.copy() for velocity in self._clone_velocities)

    @property
    def clone_positions_w_b(self) -> tuple[np.ndarray, ...]:
        return tuple(position.copy() for position in self._clone_positions)

    @property
    def has_displacement_clone(self) -> bool:
        return self.clone_count > 0

    def state(self) -> LearnedInertialState:
        return LearnedInertialState(
            timestamp_s=self.timestamp_s,
            position_w_b=self.p.copy(),
            linear_velocity_w_b=self.v.copy(),
            orientation_w_b_wxyz=rotmat_to_quat_wxyz(self.R),
            accel_bias_b=self.ba.copy(),
            gyro_bias_b=self.bg.copy(),
            covariance=self.P.copy(),
        )

    def _clone_slice(self, clone_index: int) -> slice:
        start = self._CURRENT_DIM + self._CLONE_DIM * int(clone_index)
        return slice(start, start + self._CLONE_DIM)

    def _clone_velocity_slice(self, clone_index: int) -> slice:
        sl = self._clone_slice(clone_index)
        return slice(sl.start, sl.start + 3)

    def _clone_position_slice(self, clone_index: int) -> slice:
        sl = self._clone_slice(clone_index)
        return slice(sl.start + 3, sl.start + 6)

    def _find_clone_index(self, timestamp_s: float, *, tolerance_s: float = 1.0e-6) -> int:
        if not self._clone_timestamps_s:
            raise RuntimeError("no learned-motion kinematic clone is available")
        timestamp = float(timestamp_s)
        tolerance = float(tolerance_s)
        if tolerance < 0.0 or not np.isfinite(tolerance):
            raise ValueError("clone timestamp tolerance must be finite and non-negative")
        times = np.asarray(self._clone_timestamps_s, dtype=np.float64)
        index = int(np.argmin(np.abs(times - timestamp)))
        error = abs(float(times[index]) - timestamp)
        if error > tolerance:
            raise KeyError(
                f"no kinematic clone within {tolerance:.6f}s of {timestamp:.6f}s "
                f"(nearest={times[index]:.6f}s)"
            )
        return index

    def clone_current_position(self) -> float:
        """Append current velocity+position with full cross-covariance.

        The historical public method name is preserved for compatibility with
        the existing fixed-lag scheduler. Each clone is now [v, p], enabling a
        statistically consistent measurement of

            dp_residual = (p_t - p_s) - v_s * dt.
        """
        timestamp = float(self.timestamp_s)
        if self._clone_timestamps_s and timestamp <= self._clone_timestamps_s[-1] + 1.0e-12:
            raise ValueError("kinematic clone timestamps must be strictly increasing")

        while self.clone_count >= self.max_position_clones:
            self.marginalize_clone(0)

        old_dim = self.P.shape[0]
        P_aug = np.zeros(
            (old_dim + self._CLONE_DIM, old_dim + self._CLONE_DIM),
            dtype=np.float64,
        )
        P_aug[:old_dim, :old_dim] = self.P

        J = np.zeros((self._CLONE_DIM, old_dim), dtype=np.float64)
        J[0:3, 3:6] = np.eye(3)
        J[3:6, 6:9] = np.eye(3)
        cross = J @ self.P
        P_aug[old_dim:, :old_dim] = cross
        P_aug[:old_dim, old_dim:] = cross.T
        P_aug[old_dim:, old_dim:] = J @ self.P @ J.T

        self.P = 0.5 * (P_aug + P_aug.T)
        self._clone_velocities.append(self.v.copy())
        self._clone_positions.append(self.p.copy())
        self._clone_timestamps_s.append(timestamp)
        return timestamp

    def begin_displacement_window(self) -> None:
        """Backward-compatible alias for cloning a window-start kinematic state."""
        self.clone_current_position()

    def marginalize_clone(self, clone_index: int) -> None:
        """Remove one historical clone and its covariance rows/columns."""
        index = int(clone_index)
        if index < 0:
            index += self.clone_count
        if index < 0 or index >= self.clone_count:
            raise IndexError("kinematic clone index out of range")
        sl = self._clone_slice(index)
        keep = np.ones(self.P.shape[0], dtype=bool)
        keep[sl] = False
        self.P = self.P[np.ix_(keep, keep)].copy()
        del self._clone_velocities[index]
        del self._clone_positions[index]
        del self._clone_timestamps_s[index]
        self.P = 0.5 * (self.P + self.P.T)

    def marginalize_clone_at_timestamp(
        self,
        timestamp_s: float,
        *,
        tolerance_s: float = 1.0e-6,
    ) -> None:
        """Remove the clone nearest the requested timestamp within tolerance."""
        clone_index = self._find_clone_index(timestamp_s, tolerance_s=tolerance_s)
        self.marginalize_clone(clone_index)

    def marginalize_clones_before(self, timestamp_s: float, *, inclusive: bool = False) -> int:
        """Drop clones older than a fixed-lag cutoff."""
        threshold = float(timestamp_s)
        removed = 0
        while self._clone_timestamps_s:
            oldest = self._clone_timestamps_s[0]
            expired = oldest <= threshold if inclusive else oldest < threshold
            if not expired:
                break
            self.marginalize_clone(0)
            removed += 1
        return removed

    def predicted_relative_displacement(
        self,
        *,
        start_timestamp_s: float | None = None,
        clone_tolerance_s: float = 1.0e-6,
    ) -> np.ndarray:
        """Return p_current - p_historical for the selected clone."""
        if self.clone_count == 0:
            raise RuntimeError("a learned-motion kinematic clone is required")
        if start_timestamp_s is None:
            clone_index = 0
        else:
            clone_index = self._find_clone_index(
                start_timestamp_s,
                tolerance_s=clone_tolerance_s,
            )
        return self.p - self._clone_positions[clone_index]

    def relative_displacement_jacobian(
        self,
        *,
        start_timestamp_s: float | None = None,
        clone_tolerance_s: float = 1.0e-6,
    ) -> np.ndarray:
        """Build H for a current-minus-historical position measurement."""
        if self.clone_count == 0:
            raise RuntimeError("a learned-motion kinematic clone is required")
        if start_timestamp_s is None:
            clone_index = 0
        else:
            clone_index = self._find_clone_index(
                start_timestamp_s,
                tolerance_s=clone_tolerance_s,
            )
        H = np.zeros((3, self.P.shape[0]), dtype=np.float64)
        H[:, 6:9] = np.eye(3)
        H[:, self._clone_position_slice(clone_index)] = -np.eye(3)
        return H


    def predicted_kinematic_residual(
        self,
        *,
        start_timestamp_s: float | None = None,
        clone_tolerance_s: float = 1.0e-6,
    ) -> np.ndarray:
        """Return (p_t - p_s) - v_s * (t - s) for a historical clone."""
        if self.clone_count == 0:
            raise RuntimeError("a learned-motion kinematic clone is required")
        if start_timestamp_s is None:
            clone_index = 0
        else:
            clone_index = self._find_clone_index(
                start_timestamp_s,
                tolerance_s=clone_tolerance_s,
            )
        dt = float(self.timestamp_s - self._clone_timestamps_s[clone_index])
        if dt <= 0.0 or not np.isfinite(dt):
            raise ValueError("kinematic-residual window duration must be positive")
        return (
            self.p
            - self._clone_positions[clone_index]
            - self._clone_velocities[clone_index] * dt
        )

    def kinematic_residual_jacobian(
        self,
        *,
        start_timestamp_s: float | None = None,
        clone_tolerance_s: float = 1.0e-6,
    ) -> np.ndarray:
        """Build H for (p_t - p_s) - v_s * dt."""
        if self.clone_count == 0:
            raise RuntimeError("a learned-motion kinematic clone is required")
        if start_timestamp_s is None:
            clone_index = 0
        else:
            clone_index = self._find_clone_index(
                start_timestamp_s,
                tolerance_s=clone_tolerance_s,
            )
        dt = float(self.timestamp_s - self._clone_timestamps_s[clone_index])
        if dt <= 0.0 or not np.isfinite(dt):
            raise ValueError("kinematic-residual window duration must be positive")

        H = np.zeros((3, self.P.shape[0]), dtype=np.float64)
        H[:, 6:9] = np.eye(3)
        H[:, self._clone_velocity_slice(clone_index)] = -dt * np.eye(3)
        H[:, self._clone_position_slice(clone_index)] = -np.eye(3)
        return H

    def predicted_kinematic_residual_body_end(
        self,
        *,
        start_timestamp_s: float | None = None,
        clone_tolerance_s: float = 1.0e-6,
    ) -> np.ndarray:
        """Return the kinematic residual expressed in the current body frame.

        h(x) = R_t^T * [(p_t - p_s) - v_s * dt]

        The learned measurement can therefore be predicted directly from
        body-frame gyro/thrust without using the estimator attitude as a
        network input. Attitude dependence remains explicit in h(x).
        """
        residual_w = self.predicted_kinematic_residual(
            start_timestamp_s=start_timestamp_s,
            clone_tolerance_s=clone_tolerance_s,
        )
        return self.R.T @ residual_w

    def kinematic_residual_body_end_jacobian(
        self,
        *,
        start_timestamp_s: float | None = None,
        clone_tolerance_s: float = 1.0e-6,
    ) -> np.ndarray:
        """Build H for R_t^T * [(p_t-p_s) - v_s*dt].

        The filter uses the right/local attitude error
            R_true = R_nominal Exp(dtheta).
        Therefore d(R^T y)/dtheta = [R^T y]_x.
        """
        if self.clone_count == 0:
            raise RuntimeError("a learned-motion kinematic clone is required")
        if start_timestamp_s is None:
            clone_index = 0
        else:
            clone_index = self._find_clone_index(
                start_timestamp_s,
                tolerance_s=clone_tolerance_s,
            )
        dt = float(self.timestamp_s - self._clone_timestamps_s[clone_index])
        if dt <= 0.0 or not np.isfinite(dt):
            raise ValueError("kinematic-residual window duration must be positive")

        residual_w = (
            self.p
            - self._clone_positions[clone_index]
            - self._clone_velocities[clone_index] * dt
        )
        predicted_b = self.R.T @ residual_w
        Rt = self.R.T

        H = np.zeros((3, self.P.shape[0]), dtype=np.float64)
        H[:, 0:3] = _skew(predicted_b)
        H[:, 6:9] = Rt
        H[:, self._clone_velocity_slice(clone_index)] = -dt * Rt
        H[:, self._clone_position_slice(clone_index)] = -Rt
        return H

    def predicted_kinematic_residual_body_end_gravity_compensated(
        self,
        *,
        start_timestamp_s: float | None = None,
        clone_tolerance_s: float = 1.0e-6,
    ) -> np.ndarray:
        """Return endpoint-body residual after removing known gravity motion.

        h(x) = R_t^T * [(p_t-p_s) - v_s*dt - 0.5*g*dt^2].
        """
        if self.clone_count == 0:
            raise RuntimeError("a learned-motion kinematic clone is required")
        if start_timestamp_s is None:
            clone_index = 0
        else:
            clone_index = self._find_clone_index(
                start_timestamp_s,
                tolerance_s=clone_tolerance_s,
            )
        dt = float(self.timestamp_s - self._clone_timestamps_s[clone_index])
        if dt <= 0.0 or not np.isfinite(dt):
            raise ValueError("kinematic-residual window duration must be positive")

        residual_w = (
            self.p
            - self._clone_positions[clone_index]
            - self._clone_velocities[clone_index] * dt
            - 0.5 * self.gravity_w * dt * dt
        )
        return self.R.T @ residual_w

    def kinematic_residual_body_end_gravity_compensated_jacobian(
        self,
        *,
        start_timestamp_s: float | None = None,
        clone_tolerance_s: float = 1.0e-6,
    ) -> np.ndarray:
        """Build H for the gravity-compensated endpoint-body residual."""
        if self.clone_count == 0:
            raise RuntimeError("a learned-motion kinematic clone is required")
        if start_timestamp_s is None:
            clone_index = 0
        else:
            clone_index = self._find_clone_index(
                start_timestamp_s,
                tolerance_s=clone_tolerance_s,
            )
        dt = float(self.timestamp_s - self._clone_timestamps_s[clone_index])
        if dt <= 0.0 or not np.isfinite(dt):
            raise ValueError("kinematic-residual window duration must be positive")

        residual_w = (
            self.p
            - self._clone_positions[clone_index]
            - self._clone_velocities[clone_index] * dt
            - 0.5 * self.gravity_w * dt * dt
        )
        predicted_b = self.R.T @ residual_w
        Rt = self.R.T

        H = np.zeros((3, self.P.shape[0]), dtype=np.float64)
        H[:, 0:3] = _skew(predicted_b)
        H[:, 6:9] = Rt
        H[:, self._clone_velocity_slice(clone_index)] = -dt * Rt
        H[:, self._clone_position_slice(clone_index)] = -Rt
        return H

    def propagate(self, *, gyro_b, accel_b, timestamp_s: float) -> None:
        t = float(timestamp_s)
        dt = t - self.timestamp_s
        if dt <= 0.0 or not np.isfinite(dt):
            raise ValueError("propagation timestamp must be finite and strictly increasing")

        gyro = np.asarray(gyro_b, dtype=np.float64).reshape(3)
        accel = np.asarray(accel_b, dtype=np.float64).reshape(3)
        omega = gyro - self.bg
        specific_force = accel - self.ba

        R_prev = self.R
        a_w = self.gravity_w + R_prev @ specific_force
        self.p = self.p + self.v * dt + 0.5 * a_w * dt * dt
        self.v = self.v + a_w * dt
        self.R = R_prev @ _exp_so3(omega * dt)
        self.timestamp_s = t

        F = np.eye(self._CURRENT_DIM, dtype=np.float64)
        F[0:3, 0:3] -= _skew(omega) * dt
        F[0:3, 12:15] = -np.eye(3) * dt
        F[3:6, 0:3] = -(R_prev @ _skew(specific_force)) * dt
        F[3:6, 9:12] = -R_prev * dt
        F[6:9, 3:6] = np.eye(3) * dt
        F[6:9, 0:3] = -0.5 * (R_prev @ _skew(specific_force)) * dt * dt
        F[6:9, 9:12] = -0.5 * R_prev * dt * dt

        G = np.zeros((self._CURRENT_DIM, 12), dtype=np.float64)
        G[0:3, 3:6] = -np.eye(3) * dt
        G[3:6, 0:3] = -R_prev * dt
        G[6:9, 0:3] = -0.5 * R_prev * dt * dt
        G[9:12, 6:9] = np.eye(3) * np.sqrt(dt)
        G[12:15, 9:12] = np.eye(3) * np.sqrt(dt)
        Qc = np.diag(
            [self.accel_noise_sigma**2] * 3
            + [self.gyro_noise_sigma**2] * 3
            + [self.accel_bias_rw_sigma**2] * 3
            + [self.gyro_bias_rw_sigma**2] * 3
        )

        total_dim = self.P.shape[0]
        F_aug = np.eye(total_dim, dtype=np.float64)
        F_aug[:self._CURRENT_DIM, :self._CURRENT_DIM] = F
        G_aug = np.zeros((total_dim, 12), dtype=np.float64)
        G_aug[:self._CURRENT_DIM, :] = G
        self.P = F_aug @ self.P @ F_aug.T + G_aug @ Qc @ G_aug.T
        self.P = 0.5 * (self.P + self.P.T)

    def _inject_error(self, dx: np.ndarray) -> None:
        dx = np.asarray(dx, dtype=np.float64).reshape(-1)
        expected_dim = self._CURRENT_DIM + self._CLONE_DIM * self.clone_count
        if dx.size != expected_dim:
            raise ValueError(f"error state has size {dx.size}, expected {expected_dim}")
        dtheta = dx[0:3]
        # Propagation linearization uses a right-multiplicative attitude error:
        #     R_true = R_nominal @ Exp(dtheta)
        # so injection must use the same convention.
        self.R = self.R @ _exp_so3(dtheta)
        self.v += dx[3:6]
        self.p += dx[6:9]
        self.ba += dx[9:12]
        self.bg += dx[12:15]
        for clone_index in range(self.clone_count):
            self._clone_velocities[clone_index] += dx[
                self._clone_velocity_slice(clone_index)
            ]
            self._clone_positions[clone_index] += dx[
                self._clone_position_slice(clone_index)
            ]

    def _kalman_update(self, residual: np.ndarray, H: np.ndarray, Rm: np.ndarray) -> None:
        residual = np.asarray(residual, dtype=np.float64).reshape(-1)
        Rm = np.asarray(Rm, dtype=np.float64)
        S = H @ self.P @ H.T + Rm
        PHt = self.P @ H.T
        K = np.linalg.solve(S.T, PHt.T).T
        dx = K @ residual
        self._inject_error(dx)
        I = np.eye(self.P.shape[0], dtype=np.float64)
        A = I - K @ H
        self.P = A @ self.P @ A.T + K @ Rm @ K.T

        # Error-state reset after attitude injection. For the local/right
        # multiplicative error used by F, the first-order reset Jacobian is
        # I - 0.5*[dtheta]_x on the attitude block.
        reset = np.eye(self.P.shape[0], dtype=np.float64)
        reset[0:3, 0:3] = np.eye(3) - 0.5 * _skew(dx[0:3])
        self.P = reset @ self.P @ reset.T
        self.P = 0.5 * (self.P + self.P.T)

    def update_learned_displacement(
        self,
        displacement_w,
        covariance_w,
        *,
        start_timestamp_s: float | None = None,
        clone_tolerance_s: float = 1.0e-6,
        marginalize_used_clone: bool = True,
    ) -> np.ndarray:
        """Fuse z = p_current - p_historical and return the innovation."""
        if self.clone_count == 0:
            raise RuntimeError("a learned-motion kinematic clone is required")
        if start_timestamp_s is None:
            clone_index = 0
        else:
            clone_index = self._find_clone_index(
                start_timestamp_s,
                tolerance_s=clone_tolerance_s,
            )

        z = np.asarray(displacement_w, dtype=np.float64).reshape(3)
        Rm = np.asarray(covariance_w, dtype=np.float64).reshape(3, 3)
        if not np.all(np.isfinite(Rm)):
            raise ValueError("learned displacement covariance must be finite")
        if np.linalg.eigvalsh(0.5 * (Rm + Rm.T))[0] <= 0.0:
            raise ValueError("learned displacement covariance must be positive definite")

        predicted = self.p - self._clone_positions[clone_index]
        residual = z - predicted
        H = self.relative_displacement_jacobian(
            start_timestamp_s=self._clone_timestamps_s[clone_index],
            clone_tolerance_s=clone_tolerance_s,
        )
        self._kalman_update(residual, H, Rm)

        if marginalize_used_clone:
            self.marginalize_clone(clone_index)
        return residual

    def update_learned_kinematic_residual(
        self,
        residual_displacement_w,
        covariance_w,
        *,
        start_timestamp_s: float | None = None,
        clone_tolerance_s: float = 1.0e-6,
        marginalize_used_clone: bool = True,
    ) -> np.ndarray:
        """Fuse z = (p_t - p_s) - v_s * dt and return the innovation."""
        if self.clone_count == 0:
            raise RuntimeError("a learned-motion kinematic clone is required")
        if start_timestamp_s is None:
            clone_index = 0
        else:
            clone_index = self._find_clone_index(
                start_timestamp_s,
                tolerance_s=clone_tolerance_s,
            )

        z = np.asarray(residual_displacement_w, dtype=np.float64).reshape(3)
        Rm = np.asarray(covariance_w, dtype=np.float64).reshape(3, 3)
        if not np.all(np.isfinite(Rm)):
            raise ValueError("learned residual covariance must be finite")
        if np.linalg.eigvalsh(0.5 * (Rm + Rm.T))[0] <= 0.0:
            raise ValueError("learned residual covariance must be positive definite")

        clone_timestamp = self._clone_timestamps_s[clone_index]
        predicted = self.predicted_kinematic_residual(
            start_timestamp_s=clone_timestamp,
            clone_tolerance_s=clone_tolerance_s,
        )
        innovation = z - predicted
        H = self.kinematic_residual_jacobian(
            start_timestamp_s=clone_timestamp,
            clone_tolerance_s=clone_tolerance_s,
        )
        self._kalman_update(innovation, H, Rm)

        if marginalize_used_clone:
            self.marginalize_clone(clone_index)
        return innovation

    def update_learned_kinematic_residual_body_end(
        self,
        residual_displacement_b_end,
        covariance_b_end,
        *,
        start_timestamp_s: float | None = None,
        clone_tolerance_s: float = 1.0e-6,
        marginalize_used_clone: bool = True,
    ) -> np.ndarray:
        """Fuse endpoint-body kinematic residual and return the innovation."""
        if self.clone_count == 0:
            raise RuntimeError("a learned-motion kinematic clone is required")
        if start_timestamp_s is None:
            clone_index = 0
        else:
            clone_index = self._find_clone_index(
                start_timestamp_s,
                tolerance_s=clone_tolerance_s,
            )

        z = np.asarray(residual_displacement_b_end, dtype=np.float64).reshape(3)
        Rm = np.asarray(covariance_b_end, dtype=np.float64).reshape(3, 3)
        if not np.all(np.isfinite(Rm)):
            raise ValueError("learned body residual covariance must be finite")
        if np.linalg.eigvalsh(0.5 * (Rm + Rm.T))[0] <= 0.0:
            raise ValueError(
                "learned body residual covariance must be positive definite"
            )

        clone_timestamp = self._clone_timestamps_s[clone_index]
        predicted = self.predicted_kinematic_residual_body_end(
            start_timestamp_s=clone_timestamp,
            clone_tolerance_s=clone_tolerance_s,
        )
        innovation = z - predicted
        H = self.kinematic_residual_body_end_jacobian(
            start_timestamp_s=clone_timestamp,
            clone_tolerance_s=clone_tolerance_s,
        )
        self._kalman_update(innovation, H, Rm)

        if marginalize_used_clone:
            self.marginalize_clone(clone_index)
        return innovation

    def update_learned_kinematic_residual_body_end_gravity_compensated(
        self,
        residual_displacement_b_end,
        covariance_b_end,
        *,
        start_timestamp_s: float | None = None,
        clone_tolerance_s: float = 1.0e-6,
        marginalize_used_clone: bool = True,
    ) -> np.ndarray:
        """Fuse gravity-compensated endpoint-body residual."""
        if self.clone_count == 0:
            raise RuntimeError("a learned-motion kinematic clone is required")
        if start_timestamp_s is None:
            clone_index = 0
        else:
            clone_index = self._find_clone_index(
                start_timestamp_s,
                tolerance_s=clone_tolerance_s,
            )

        z = np.asarray(residual_displacement_b_end, dtype=np.float64).reshape(3)
        Rm = np.asarray(covariance_b_end, dtype=np.float64).reshape(3, 3)
        if not np.all(np.isfinite(Rm)):
            raise ValueError("learned body residual covariance must be finite")
        if np.linalg.eigvalsh(0.5 * (Rm + Rm.T))[0] <= 0.0:
            raise ValueError(
                "learned body residual covariance must be positive definite"
            )

        clone_timestamp = self._clone_timestamps_s[clone_index]
        predicted = self.predicted_kinematic_residual_body_end_gravity_compensated(
            start_timestamp_s=clone_timestamp,
            clone_tolerance_s=clone_tolerance_s,
        )
        innovation = z - predicted
        H = self.kinematic_residual_body_end_gravity_compensated_jacobian(
            start_timestamp_s=clone_timestamp,
            clone_tolerance_s=clone_tolerance_s,
        )
        self._kalman_update(innovation, H, Rm)

        if marginalize_used_clone:
            self.marginalize_clone(clone_index)
        return innovation

    def update_absolute_position(self, position_w_b, covariance_w) -> np.ndarray:
        z = np.asarray(position_w_b, dtype=np.float64).reshape(3)
        Rm = np.asarray(covariance_w, dtype=np.float64).reshape(3, 3)
        residual = z - self.p
        H = np.zeros((3, self.P.shape[0]), dtype=np.float64)
        H[:, 6:9] = np.eye(3)
        self._kalman_update(residual, H, Rm)
        return residual

    def update_absolute_orientation(self, orientation_w_b_wxyz, covariance_rad2=None) -> np.ndarray:
        R_meas = quat_wxyz_to_rotmat(orientation_w_b_wxyz)
        # Right/local attitude residual, consistent with propagation and
        # injection: R_true = R_nominal @ Exp(dtheta).
        residual = _log_so3(self.R.T @ R_meas)
        if covariance_rad2 is None:
            covariance_rad2 = np.eye(3, dtype=np.float64) * np.deg2rad(3.0) ** 2
        Rm = np.asarray(covariance_rad2, dtype=np.float64).reshape(3, 3)
        H = np.zeros((3, self.P.shape[0]), dtype=np.float64)
        H[:, 0:3] = np.eye(3)
        self._kalman_update(residual, H, Rm)
        return residual

    def update_gate_pose(
        self,
        *,
        position_w_b,
        position_covariance_w,
        orientation_w_b_wxyz=None,
        orientation_covariance_rad2=None,
    ) -> None:
        self.update_absolute_position(position_w_b, position_covariance_w)
        if orientation_w_b_wxyz is not None:
            self.update_absolute_orientation(
                orientation_w_b_wxyz,
                orientation_covariance_rad2,
            )
