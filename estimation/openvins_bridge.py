"""OpenVINS integration boundary for the Swift-2023 reproduction.

OpenVINS stays an external ROS2 process. This module provides:
- a transport-neutral odometry sample contract;
- one-time alignment from OpenVINS' local/global frame V into the known track
  world frame W;
- conversion to the repository's ``VioWorldEstimate`` consumed by the Swift
  VIO-drift Kalman filter;
- deterministic source-rate gates for simulator -> ROS transport;
- optional ROS2 publishers/subscriber without importing ROS2 at module import.

OpenVINS' ROS2 ``odomimu`` message publishes pose in its global frame but
``twist.twist.linear`` in the local IMU frame. The conversion below therefore
rotates the reported velocity by the OpenVINS pose before applying V->W
alignment. This detail is easy to miss and is covered by regression tests.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import sys
from threading import Lock

import numpy as np

from .swift_vio_drift import VioWorldEstimate


def _restore_pythonpath_from_environment() -> tuple[str, ...]:
    """Restore ROS paths removed from ``sys.path`` by embedded Isaac Kit.

    Isaac Sim may rebuild ``sys.path`` while starting its embedded Kit runtime,
    even though the shell-sourced ROS ``PYTHONPATH`` remains in the process
    environment. Re-adding existing entries here keeps ROS optional and avoids
    hard-coding a distro or installation prefix.
    """
    candidates = [
        value
        for value in os.environ.get("PYTHONPATH", "").split(os.pathsep)
        if value and Path(value).is_dir()
    ]
    added = []
    for value in reversed(candidates):
        if value not in sys.path:
            sys.path.insert(0, value)
            added.append(value)
    return tuple(reversed(added))


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
    """One OpenVINS ``odomimu`` sample.

    ``position_v_i`` and ``orientation_v_i_wxyz`` are the IMU pose in the
    estimator/global frame V. For compatibility the velocity field keeps its
    historical name ``linear_velocity_v_i``, but the ROS2 publisher actually
    places *local-frame* velocity ``v_IinI`` in ``twist.twist.linear``.
    """

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
        timestamp = float(self.timestamp_s)
        if not np.isfinite(timestamp):
            raise ValueError("OpenVINS timestamp must be finite")
        if not np.all(np.isfinite(position)) or not np.all(np.isfinite(velocity)):
            raise ValueError("OpenVINS odometry contains non-finite values")
        object.__setattr__(self, "position_v_i", position)
        object.__setattr__(self, "linear_velocity_v_i", velocity)
        object.__setattr__(self, "orientation_v_i_wxyz", orientation)
        object.__setattr__(self, "timestamp_s", timestamp)
        if self.pose_covariance is not None:
            covariance = np.asarray(self.pose_covariance, dtype=np.float64).reshape(6, 6)
            if not np.all(np.isfinite(covariance)):
                raise ValueError("OpenVINS pose covariance contains non-finite values")
            object.__setattr__(self, "pose_covariance", covariance)
        if self.twist_covariance is not None:
            covariance = np.asarray(self.twist_covariance, dtype=np.float64).reshape(6, 6)
            if not np.all(np.isfinite(covariance)):
                raise ValueError("OpenVINS twist covariance contains non-finite values")
            object.__setattr__(self, "twist_covariance", covariance)

    @property
    def linear_velocity_i(self) -> np.ndarray:
        """Local IMU-frame linear velocity from OpenVINS ``odomimu``."""
        return self.linear_velocity_v_i


@dataclass(frozen=True)
class OpenVinsFrameAlignment:
    """Rigid alignment that maps OpenVINS estimator frame V into track world W."""

    R_wv: np.ndarray
    t_wv: np.ndarray

    def __post_init__(self) -> None:
        R = np.asarray(self.R_wv, dtype=np.float64).reshape(3, 3)
        t = np.asarray(self.t_wv, dtype=np.float64).reshape(3)
        if not np.all(np.isfinite(R)) or not np.all(np.isfinite(t)):
            raise ValueError("OpenVINS alignment must be finite")
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
        """Align one initialized OpenVINS pose to a known start/reference pose.

        In the current simulator the Isaac IMU is mounted on the body frame, so
        I == B. Hardware must account for a non-identity IMU/body transform
        before constructing this alignment.
        """
        R_vi = _quat_wxyz_to_rotmat(sample.orientation_v_i_wxyz)
        R_wb = _quat_wxyz_to_rotmat(reference_orientation_w_b_wxyz)
        R_wv = R_wb @ R_vi.T
        p_wb = np.asarray(reference_position_w_b, dtype=np.float64).reshape(3)
        t_wv = p_wb - R_wv @ sample.position_v_i
        return cls(R_wv=R_wv, t_wv=t_wv)

    def to_world(self, sample: OpenVinsOdomSample) -> VioWorldEstimate:
        """Map an OpenVINS pose/velocity sample into track world W.

        OpenVINS publishes ``v_IinI`` in the odometry twist. Convert it into V
        with the published IMU orientation first, then apply V->W alignment.
        """
        R_vi = _quat_wxyz_to_rotmat(sample.orientation_v_i_wxyz)
        R_wb = self.R_wv @ R_vi
        velocity_v = R_vi @ sample.linear_velocity_i
        return VioWorldEstimate(
            position_w_b=self.R_wv @ sample.position_v_i + self.t_wv,
            linear_velocity_w_b=self.R_wv @ velocity_v,
            orientation_w_b_wxyz=_rotmat_to_quat_wxyz(R_wb),
            timestamp_s=sample.timestamp_s,
        )


class OpenVinsSensorRateGate:
    """Phase-preserving source-rate scheduler for Swift's IMU/camera bridge.

    A naive ``now-last >= period`` gate drifts when the source clock cannot hit
    the requested period exactly (for example a 30 Hz camera sampled from a
    100 Hz render clock becomes 25 Hz). This scheduler keeps ideal deadlines
    and emits on the first source tick at or after each deadline, preserving the
    requested *average* rate without accumulating quantization error.
    """

    def __init__(self, *, imu_hz: float = 200.0, camera_hz: float = 30.0) -> None:
        if imu_hz <= 0.0 or camera_hz <= 0.0:
            raise ValueError("OpenVINS source rates must be positive")
        self.imu_period_s = 1.0 / float(imu_hz)
        self.camera_period_s = 1.0 / float(camera_hz)
        self.reset()

    def reset(self) -> None:
        self._next_imu_s: float | None = None
        self._next_camera_s: float | None = None
        self._last_imu_timestamp_s: float | None = None
        self._last_camera_timestamp_s: float | None = None

    @staticmethod
    def _check_monotonic(timestamp_s: float, last_s: float | None, stream: str) -> float:
        timestamp = float(timestamp_s)
        if not np.isfinite(timestamp):
            raise ValueError("Sensor timestamp must be finite")
        if last_s is not None and timestamp < last_s - 1.0e-9:
            raise ValueError(f"OpenVINS {stream} timestamps must be monotonic")
        return timestamp

    @staticmethod
    def _due(timestamp_s: float, next_s: float | None, period_s: float) -> tuple[bool, float]:
        if next_s is None:
            return True, timestamp_s + period_s
        if timestamp_s + 1.0e-9 < next_s:
            return False, next_s
        # Keep the ideal phase even if the source tick arrives late or several
        # deadlines were skipped.
        lateness_s = max(0.0, timestamp_s - next_s)
        periods_elapsed = int(np.floor(lateness_s / period_s)) + 1
        return True, next_s + periods_elapsed * period_s

    def imu_due(self, timestamp_s: float) -> bool:
        timestamp = self._check_monotonic(timestamp_s, self._last_imu_timestamp_s, "IMU")
        self._last_imu_timestamp_s = timestamp
        due, self._next_imu_s = self._due(timestamp, self._next_imu_s, self.imu_period_s)
        return due

    def camera_due(self, timestamp_s: float) -> bool:
        timestamp = self._check_monotonic(timestamp_s, self._last_camera_timestamp_s, "camera")
        self._last_camera_timestamp_s = timestamp
        due, self._next_camera_s = self._due(timestamp, self._next_camera_s, self.camera_period_s)
        return due


def _ros_stamp(msg, timestamp_s: float) -> None:
    sec = int(np.floor(float(timestamp_s)))
    nanosec = int(round((float(timestamp_s) - sec) * 1.0e9))
    if nanosec >= 1_000_000_000:
        sec += 1
        nanosec -= 1_000_000_000
    msg.header.stamp.sec = sec
    msg.header.stamp.nanosec = nanosec


class OpenVinsRos2Bridge:
    """Optional ROS2 I/O adapter for an external ``ov_msckf`` process."""

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
        _restore_pythonpath_from_environment()
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
        self._odom_callback_count = 0
        self.last_drain_count = 0

    @property
    def node(self):
        return self._node

    @property
    def latest(self) -> OpenVinsOdomSample | None:
        with self._latest_lock:
            return self._latest

    @property
    def odom_callback_count(self) -> int:
        with self._latest_lock:
            return int(self._odom_callback_count)

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
            # OpenVINS ROS2Visualizer publishes v_IinI here despite the generic
            # nav_msgs/Odometry field name.
            linear_velocity_v_i=np.array([v.x, v.y, v.z], dtype=np.float64),
            pose_covariance=np.asarray(msg.pose.covariance, dtype=np.float64).reshape(6, 6),
            twist_covariance=np.asarray(msg.twist.covariance, dtype=np.float64).reshape(6, 6),
        )
        with self._latest_lock:
            self._latest = sample
            self._odom_callback_count += 1

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
        if not np.all(np.isfinite(omega)) or not np.all(np.isfinite(accel)):
            raise ValueError("OpenVINS IMU input must be finite")
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

    def drain_latest(self, *, max_callbacks: int = 32) -> OpenVinsOdomSample | None:
        """Drain queued odometry callbacks and return the newest sample.

        OpenVINS publishes propagated ``odomimu`` at the IMU rate (200 Hz),
        while the Isaac control loop normally consumes state at 100 Hz. Calling
        ``spin_once`` only once per control cycle would therefore service at
        most half the odometry callbacks and can build an ever-growing stale
        queue. This bounded non-blocking drain collapses queued odometry to the
        newest sample before control/fusion consumes it.
        """
        if max_callbacks < 1:
            raise ValueError("max_callbacks must be at least 1")

        processed = 0
        for _ in range(int(max_callbacks)):
            before = self.odom_callback_count
            self._rclpy.spin_once(self._node, timeout_sec=0.0)
            after = self.odom_callback_count
            if after == before:
                break
            processed += after - before
        self.last_drain_count = processed
        return self.latest

    def close(self) -> None:
        self._node.destroy_node()
