"""OpenVINS integration boundary for the Swift-2023 reproduction.

OpenVINS stays an external ROS2 process.  This module provides:
- a transport-neutral odometry sample contract;
- one-time alignment from OpenVINS' local/global frame V into the known track
  world frame W;
- conversion to the repository's ``VioWorldEstimate`` consumed by the Swift
  VIO-drift Kalman filter;
- optional ROS2 publishers/subscriber used to bridge Isaac camera/IMU data to
  OpenVINS without importing ROS2 at module import time.

No OpenVINS GPL source code is vendored into this repository.
"""

from __future__ import annotations

from dataclasses import dataclass
from threading import Lock

import numpy as np

from .swift_vio_drift import VioWorldEstimate


def _normalize_quaternion_wxyz(q) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64).reshape(4)
    n = np.linalg.norm(q)
    if n <= 0.0 or not np.all(np.isfinite(q)):
        raise ValueError("Quaternion must be finite and non-zero")
    return q / n


def _quat_wxyz_to_rotmat(q) -> np.ndarray:
    w, x, y, z = _normalize_quaternion_wxyz(q)
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _rotmat_to_quat_wxyz(R) -> np.ndarray:
    R = np.asarray(R, dtype=np.float64).reshape(3, 3)
    tr = float(np.trace(R))
    if tr > 0.0:
        s = np.sqrt(tr + 1.0) * 2.0
        q = np.array(
            [0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s],
            dtype=np.float64,
        )
    else:
        i = int(np.argmax(np.diag(R)))
        if i == 0:
            s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
            q = np.array(
                [(R[2, 1] - R[1, 2]) / s, 0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s]
            )
        elif i == 1:
            s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
            q = np.array(
                [(R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s]
            )
        else:
            s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
            q = np.array(
                [(R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s]
            )
    q = _normalize_quaternion_wxyz(q)
    return -q if q[0] < 0.0 else q


@dataclass(frozen=True)
class OpenVinsOdomSample:
    """OpenVINS odometry expressed in its estimator frame V and IMU/body frame I."""

    timestamp_s: float
    position_v_i: np.ndarray
    orientation_v_i_wxyz: np.ndarray
    linear_velocity_v_i: np.ndarray
    pose_covariance: np.ndarray | None = None
    twist_covariance: np.ndarray | None = None

    def __post_init__(self) -> None:
        position = np.asarray(self.position_v_i, dtype=np.float64).reshape(3)
        velocity = np.asarray(self.linear_velocity_v_i, dtype=np.float64).reshape(3)
        orientation = _normalize_quaternion_wxyz(self.orientation_v_i_wxyz)
        if not np.all(np.isfinite(position)) or not np.all(np.isfinite(velocity)):
            raise ValueError("OpenVINS odometry contains non-finite values")
        object.__setattr__(self, "position_v_i", position)
        object.__setattr__(self, "linear_velocity_v_i", velocity)
        object.__setattr__(self, "orientation_v_i_wxyz", orientation)
        object.__setattr__(self, "timestamp_s", float(self.timestamp_s))
        if self.pose_covariance is not None:
            object.__setattr__(
                self, "pose_covariance", np.asarray(self.pose_covariance, dtype=np.float64).reshape(6, 6)
            )
        if self.twist_covariance is not None:
            object.__setattr__(
                self, "twist_covariance", np.asarray(self.twist_covariance, dtype=np.float64).reshape(6, 6)
            )


@dataclass(frozen=True)
class OpenVinsFrameAlignment:
    """Rigid alignment that maps OpenVINS estimator frame V into track world W."""

    R_wv: np.ndarray
    t_wv: np.ndarray

    def __post_init__(self) -> None:
        R = np.asarray(self.R_wv, dtype=np.float64).reshape(3, 3)
        t = np.asarray(self.t_wv, dtype=np.float64).reshape(3)
        if not np.allclose(R.T @ R, np.eye(3), atol=1.0e-6):
            raise ValueError("R_wv must be orthonormal")
        if np.linalg.det(R) < 0.0:
            raise ValueError("R_wv must be a proper rotation")
        object.__setattr__(self, "R_wv", R)
        object.__setattr__(self, "t_wv", t)

    @classmethod
    def from_reference_pose(
        cls,
        sample: OpenVinsOdomSample,
        *,
        reference_position_w_b,
        reference_orientation_w_b_wxyz,
    ) -> "OpenVinsFrameAlignment":
        """Align one initialized OpenVINS pose to a known start pose.

        In the current simulator the Isaac IMU is mounted on the body frame, so
        I == B.  On hardware, include the calibrated IMU/body transform before
        constructing this alignment if those frames differ.
        """
        R_vb = _quat_wxyz_to_rotmat(sample.orientation_v_i_wxyz)
        R_wb = _quat_wxyz_to_rotmat(reference_orientation_w_b_wxyz)
        R_wv = R_wb @ R_vb.T
        p_wb = np.asarray(reference_position_w_b, dtype=np.float64).reshape(3)
        t_wv = p_wb - R_wv @ sample.position_v_i
        return cls(R_wv=R_wv, t_wv=t_wv)

    def to_world(self, sample: OpenVinsOdomSample) -> VioWorldEstimate:
        R_vb = _quat_wxyz_to_rotmat(sample.orientation_v_i_wxyz)
        R_wb = self.R_wv @ R_vb
        return VioWorldEstimate(
            position_w_b=self.R_wv @ sample.position_v_i + self.t_wv,
            linear_velocity_w_b=self.R_wv @ sample.linear_velocity_v_i,
            orientation_w_b_wxyz=_rotmat_to_quat_wxyz(R_wb),
            timestamp_s=sample.timestamp_s,
        )


class OpenVinsSensorRateGate:
    """Deterministic source-rate gating for the Swift 200-Hz IMU / 30-Hz camera bridge."""

    def __init__(self, *, imu_hz: float = 200.0, camera_hz: float = 30.0) -> None:
        if imu_hz <= 0.0 or camera_hz <= 0.0:
            raise ValueError("OpenVINS source rates must be positive")
        self.imu_period_s = 1.0 / float(imu_hz)
        self.camera_period_s = 1.0 / float(camera_hz)
        self.reset()

    def reset(self) -> None:
        self._last_imu_s: float | None = None
        self._last_camera_s: float | None = None

    @staticmethod
    def _due(timestamp_s: float, last_s: float | None, period_s: float) -> bool:
        return last_s is None or float(timestamp_s) - last_s >= period_s - 1.0e-9

    def imu_due(self, timestamp_s: float) -> bool:
        if self._due(timestamp_s, self._last_imu_s, self.imu_period_s):
            self._last_imu_s = float(timestamp_s)
            return True
        return False

    def camera_due(self, timestamp_s: float) -> bool:
        if self._due(timestamp_s, self._last_camera_s, self.camera_period_s):
            self._last_camera_s = float(timestamp_s)
            return True
        return False


def _ros_stamp(msg, timestamp_s: float) -> None:
    sec = int(np.floor(float(timestamp_s)))
    nanosec = int(round((float(timestamp_s) - sec) * 1.0e9))
    if nanosec >= 1_000_000_000:
        sec += 1
        nanosec -= 1_000_000_000
    msg.header.stamp.sec = sec
    msg.header.stamp.nanosec = nanosec


class OpenVinsRos2Bridge:
    """Optional ROS2 I/O adapter for an external ``ov_msckf`` process.

    ROS2 imports happen lazily so pure Python estimator tests do not require a
    ROS installation. Call ``spin_once`` regularly from the single-vehicle
    diagnostic environment.
    """

    def __init__(
        self,
        *,
        image_topic: str = "/swift/camera/image_raw",
        imu_topic: str = "/swift/imu",
        odom_topic: str = "/ov_msckf/odomimu",
        node_name: str = "isaac_swift_openvins_bridge",
        camera_frame_id: str = "cam0",
        imu_frame_id: str = "imu",
    ) -> None:
        try:
            import rclpy
            from nav_msgs.msg import Odometry
            from sensor_msgs.msg import Image, Imu
        except ImportError as exc:  # pragma: no cover - depends on ROS2 runtime
            raise ImportError(
                "OpenVinsRos2Bridge requires ROS2 Python packages (rclpy, sensor_msgs, nav_msgs)."
            ) from exc

        if not rclpy.ok():
            rclpy.init(args=None)
        self._rclpy = rclpy
        self._Image = Image
        self._Imu = Imu
        self._node = rclpy.create_node(node_name)
        self._image_pub = self._node.create_publisher(Image, image_topic, 10)
        self._imu_pub = self._node.create_publisher(Imu, imu_topic, 200)
        self._odom_sub = self._node.create_subscription(Odometry, odom_topic, self._on_odom, 20)
        self.camera_frame_id = camera_frame_id
        self.imu_frame_id = imu_frame_id
        self._latest_lock = Lock()
        self._latest: OpenVinsOdomSample | None = None

    @property
    def node(self):
        return self._node

    @property
    def latest(self) -> OpenVinsOdomSample | None:
        with self._latest_lock:
            return self._latest

    def _on_odom(self, msg) -> None:
        stamp = msg.header.stamp
        timestamp_s = float(stamp.sec) + float(stamp.nanosec) * 1.0e-9
        q = msg.pose.pose.orientation
        p = msg.pose.pose.position
        v = msg.twist.twist.linear
        sample = OpenVinsOdomSample(
            timestamp_s=timestamp_s,
            position_v_i=np.array([p.x, p.y, p.z], dtype=np.float64),
            orientation_v_i_wxyz=np.array([q.w, q.x, q.y, q.z], dtype=np.float64),
            linear_velocity_v_i=np.array([v.x, v.y, v.z], dtype=np.float64),
            pose_covariance=np.asarray(msg.pose.covariance, dtype=np.float64).reshape(6, 6),
            twist_covariance=np.asarray(msg.twist.covariance, dtype=np.float64).reshape(6, 6),
        )
        with self._latest_lock:
            self._latest = sample

    def publish_rgb(self, rgb_image: np.ndarray, *, timestamp_s: float) -> None:
        image = np.asarray(rgb_image)
        if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("OpenVINS RGB input must be uint8 HxWx3")
        msg = self._Image()
        _ros_stamp(msg, timestamp_s)
        msg.header.frame_id = self.camera_frame_id
        msg.height = int(image.shape[0])
        msg.width = int(image.shape[1])
        msg.encoding = "rgb8"
        msg.is_bigendian = 0
        msg.step = int(image.shape[1] * 3)
        msg.data = np.ascontiguousarray(image).tobytes()
        self._image_pub.publish(msg)

    def publish_imu(
        self,
        *,
        angular_velocity_b,
        linear_acceleration_b,
        timestamp_s: float,
    ) -> None:
        omega = np.asarray(angular_velocity_b, dtype=np.float64).reshape(3)
        accel = np.asarray(linear_acceleration_b, dtype=np.float64).reshape(3)
        msg = self._Imu()
        _ros_stamp(msg, timestamp_s)
        msg.header.frame_id = self.imu_frame_id
        msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z = map(float, omega)
        msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z = map(float, accel)
        msg.orientation_covariance[0] = -1.0
        self._imu_pub.publish(msg)

    def spin_once(self, timeout_sec: float = 0.0) -> OpenVinsOdomSample | None:
        self._rclpy.spin_once(self._node, timeout_sec=float(timeout_sec))
        return self.latest

    def close(self) -> None:
        self._node.destroy_node()
