"""Pure frame contract for diagnostic OpenVINS ground-truth initialization.

Isaac/ROS pose semantics use an active IMU-to-world quaternion ``q_WI``.
OpenVINS' internal IMU state stores ``q_GtoI`` (global/world to IMU), so the
orientation is inverted exactly once at this boundary.  This module has no ROS
or Isaac dependency so the frame convention can be regression-tested in pure
CI.
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class OpenVinsGtInitialization:
    """OpenVINS 17-state initialization fields before C++ handoff."""

    timestamp_s: float
    q_g_to_i_wxyz: np.ndarray
    position_g_i: np.ndarray
    velocity_g_i: np.ndarray
    gyro_bias_i: np.ndarray
    accel_bias_i: np.ndarray

    def as_openvins_state(self) -> np.ndarray:
        """Return ``[time, q_GtoI, p_IinG, v_IinG, bg, ba]``."""
        return np.concatenate(
            (
                np.array([self.timestamp_s], dtype=np.float64),
                self.q_g_to_i_wxyz,
                self.position_g_i,
                self.velocity_g_i,
                self.gyro_bias_i,
                self.accel_bias_i,
            )
        )


def _finite_vector(value, size: int, name: str) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float64).reshape(size)
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must be finite")
    return vector.copy()


def build_openvins_gt_initialization(
    timestamp_s: float,
    position_w_i,
    orientation_w_i_wxyz,
    linear_velocity_w_i,
    gyro_bias_i=None,
    accel_bias_i=None,
) -> OpenVinsGtInitialization:
    """Convert simulator world-frame truth to OpenVINS initialization fields.

    Args:
        timestamp_s: Sensor/simulator timestamp in seconds.
        position_w_i: IMU/body origin expressed in world/global frame.
        orientation_w_i_wxyz: Active IMU/body-to-world quaternion, scalar first.
        linear_velocity_w_i: IMU/body linear velocity expressed in world/global.
        gyro_bias_i: Optional gyro bias in IMU frame; defaults to zero.
        accel_bias_i: Optional accelerometer bias in IMU frame; defaults to zero.
    """
    timestamp = float(timestamp_s)
    if not np.isfinite(timestamp):
        raise ValueError("timestamp must be finite")

    q_w_i = _finite_vector(orientation_w_i_wxyz, 4, "orientation quaternion")
    norm = float(np.linalg.norm(q_w_i))
    if norm <= 0.0:
        raise ValueError("orientation quaternion must be non-zero")
    q_w_i /= norm

    # Inverse of a unit quaternion: q_GtoI = inverse(q_ItoG) = conjugate(q_WI).
    q_g_to_i = q_w_i.copy()
    q_g_to_i[1:] *= -1.0

    zero = np.zeros(3, dtype=np.float64)
    return OpenVinsGtInitialization(
        timestamp_s=timestamp,
        q_g_to_i_wxyz=q_g_to_i,
        position_g_i=_finite_vector(position_w_i, 3, "position"),
        velocity_g_i=_finite_vector(linear_velocity_w_i, 3, "linear velocity"),
        gyro_bias_i=_finite_vector(zero if gyro_bias_i is None else gyro_bias_i, 3, "gyro bias"),
        accel_bias_i=_finite_vector(zero if accel_bias_i is None else accel_bias_i, 3, "accel bias"),
    )
