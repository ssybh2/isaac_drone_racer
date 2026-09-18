"""Standalone learned inertial odometry (IMO-style) estimator.

This module intentionally has no OpenVINS dependency.  It follows the structure
of Cioffi et al. (RAL 2023):

* IMU propagation estimates attitude, velocity, position and IMU biases.
* A learned 0.5 s relative displacement is injected as a Kalman measurement.
* Optional mapped-gate PnP measurements provide absolute pose anchors.

The learned displacement update keeps one cloned position state so the
measurement Jacobian is the paper-style relative-position form [-I, +I].
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
    """Error-state EKF with one cloned position for learned relative updates.

    Error-state order is [dtheta, dv, dp, dba, dbg] (15 states).  When a
    learned-displacement window is active, a 3-D clone of the window-start
    position is appended, making an 18-state covariance.
    """

    def __init__(
        self,
        *,
        gravity_w=(0.0, 0.0, -9.81),
        accel_noise_sigma=0.01,
        gyro_noise_sigma=0.001,
        accel_bias_rw_sigma=0.001,
        gyro_bias_rw_sigma=0.0001,
    ) -> None:
        self.gravity_w = np.asarray(gravity_w, dtype=np.float64).reshape(3)
        self.accel_noise_sigma = float(accel_noise_sigma)
        self.gyro_noise_sigma = float(gyro_noise_sigma)
        self.accel_bias_rw_sigma = float(accel_bias_rw_sigma)
        self.gyro_bias_rw_sigma = float(gyro_bias_rw_sigma)
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
            np.eye(15, dtype=np.float64) * 1.0e-3
            if initial_covariance is None
            else np.asarray(initial_covariance, dtype=np.float64).reshape(15, 15).copy()
        )
        self._clone_position: np.ndarray | None = None
        self._clone_timestamp_s: float | None = None

    @property
    def has_displacement_clone(self) -> bool:
        return self._clone_position is not None

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

    def begin_displacement_window(self) -> None:
        """Clone current position and augment covariance for a future relative update."""
        if self.has_displacement_clone:
            raise RuntimeError("a learned-displacement window is already active")
        self._clone_position = self.p.copy()
        self._clone_timestamp_s = self.timestamp_s

        P_aug = np.zeros((18, 18), dtype=np.float64)
        P_aug[:15, :15] = self.P
        # clone error equals current position error at augmentation time
        J = np.zeros((3, 15), dtype=np.float64)
        J[:, 6:9] = np.eye(3)
        P_aug[15:18, :15] = J @ self.P
        P_aug[:15, 15:18] = P_aug[15:18, :15].T
        P_aug[15:18, 15:18] = J @ self.P @ J.T
        self.P = P_aug

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

        F = np.eye(15, dtype=np.float64)
        F[0:3, 0:3] -= _skew(omega) * dt
        F[0:3, 12:15] = -np.eye(3) * dt
        F[3:6, 0:3] = -(R_prev @ _skew(specific_force)) * dt
        F[3:6, 9:12] = -R_prev * dt
        F[6:9, 3:6] = np.eye(3) * dt
        F[6:9, 0:3] = -0.5 * (R_prev @ _skew(specific_force)) * dt * dt
        F[6:9, 9:12] = -0.5 * R_prev * dt * dt

        G = np.zeros((15, 12), dtype=np.float64)
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

        if self.P.shape == (15, 15):
            self.P = F @ self.P @ F.T + G @ Qc @ G.T
        else:
            F_aug = np.eye(18, dtype=np.float64)
            F_aug[:15, :15] = F
            G_aug = np.zeros((18, 12), dtype=np.float64)
            G_aug[:15, :] = G
            self.P = F_aug @ self.P @ F_aug.T + G_aug @ Qc @ G_aug.T
        self.P = 0.5 * (self.P + self.P.T)

    def _inject_error(self, dx: np.ndarray) -> None:
        dx = np.asarray(dx, dtype=np.float64).reshape(-1)
        self.R = _exp_so3(dx[0:3]) @ self.R
        self.v += dx[3:6]
        self.p += dx[6:9]
        self.ba += dx[9:12]
        self.bg += dx[12:15]
        if dx.size == 18 and self._clone_position is not None:
            self._clone_position += dx[15:18]

    def _kalman_update(self, residual: np.ndarray, H: np.ndarray, Rm: np.ndarray) -> None:
        residual = np.asarray(residual, dtype=np.float64).reshape(-1)
        Rm = np.asarray(Rm, dtype=np.float64)
        S = H @ self.P @ H.T + Rm
        K = self.P @ H.T @ np.linalg.inv(S)
        dx = K @ residual
        self._inject_error(dx)
        I = np.eye(self.P.shape[0], dtype=np.float64)
        A = I - K @ H
        self.P = A @ self.P @ A.T + K @ Rm @ K.T
        self.P = 0.5 * (self.P + self.P.T)

    def update_learned_displacement(self, displacement_w, covariance_w) -> np.ndarray:
        """Fuse network-predicted relative displacement and return innovation.

        The innovation follows measurement-minus-prediction:
            r = dp_nn - (p_j - p_i)
        """
        if not self.has_displacement_clone or self.P.shape != (18, 18):
            raise RuntimeError("begin_displacement_window() must be called first")
        z = np.asarray(displacement_w, dtype=np.float64).reshape(3)
        Rm = np.asarray(covariance_w, dtype=np.float64).reshape(3, 3)
        predicted = self.p - self._clone_position
        residual = z - predicted

        H = np.zeros((3, 18), dtype=np.float64)
        H[:, 6:9] = np.eye(3)
        H[:, 15:18] = -np.eye(3)
        self._kalman_update(residual, H, Rm)

        # Marginalize the clone after the window update.
        self.P = self.P[:15, :15].copy()
        self._clone_position = None
        self._clone_timestamp_s = None
        return residual

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
        residual = _log_so3(R_meas @ self.R.T)
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
