from threading import Lock

import numpy as np
import pytest

from estimation.openvins_bridge import (
    OpenVinsFrameAlignment,
    OpenVinsOdomSample,
    OpenVinsRos2Bridge,
    OpenVinsSensorRateGate,
    _restore_pythonpath_from_environment,
)


def test_openvins_alignment_recovers_reference_pose_and_velocity():
    sample = OpenVinsOdomSample(
        timestamp_s=1.0,
        position_v_i=np.array([2.0, -1.0, 0.5]),
        orientation_v_i_wxyz=np.array([1.0, 0.0, 0.0, 0.0]),
        linear_velocity_v_i=np.array([1.0, 0.0, 0.0]),
    )
    q_wb = np.array([np.sqrt(0.5), 0.0, 0.0, np.sqrt(0.5)])
    p_wb = np.array([5.0, 3.0, 1.0])

    alignment = OpenVinsFrameAlignment.from_reference_pose(
        sample,
        reference_position_w_b=p_wb,
        reference_orientation_w_b_wxyz=q_wb,
    )
    world = alignment.to_world(sample)

    np.testing.assert_allclose(world.position_w_b, p_wb, atol=1e-9)
    np.testing.assert_allclose(world.orientation_w_b_wxyz, q_wb, atol=1e-9)
    np.testing.assert_allclose(world.linear_velocity_w_b, np.array([0.0, 1.0, 0.0]), atol=1e-9)


def test_openvins_local_twist_velocity_is_rotated_by_openvins_orientation():
    # Published odom orientation is I->V. A local +X velocity must therefore
    # become global +Y for a +90 deg yaw pose.
    q_90z = np.array([np.sqrt(0.5), 0.0, 0.0, np.sqrt(0.5)])
    sample = OpenVinsOdomSample(
        timestamp_s=2.0,
        position_v_i=np.zeros(3),
        orientation_v_i_wxyz=q_90z,
        linear_velocity_v_i=np.array([1.0, 0.0, 0.0]),
    )
    alignment = OpenVinsFrameAlignment(
        R_wv=np.eye(3),
        t_wv=np.zeros(3),
    )
    world = alignment.to_world(sample)
    np.testing.assert_allclose(world.linear_velocity_w_b, [0.0, 1.0, 0.0], atol=1e-9)


def test_rate_gate_preserves_30hz_average_on_100hz_render_clock():
    gate = OpenVinsSensorRateGate(imu_hz=200.0, camera_hz=30.0)
    times = np.arange(0.0, 10.0, 0.01)
    emitted = [t for t in times if gate.camera_due(float(t))]
    assert len(emitted) == 300
    intervals = np.diff(emitted)
    assert intervals.min() >= 0.03 - 1e-9
    assert intervals.max() <= 0.04 + 1e-9


def test_rate_gate_preserves_200hz_average_on_400hz_physics_clock():
    gate = OpenVinsSensorRateGate(imu_hz=200.0, camera_hz=30.0)
    times = np.arange(0.0, 1.0, 0.0025)
    emitted = [t for t in times if gate.imu_due(float(t))]
    assert len(emitted) == 200


def test_rate_gate_rejects_backward_timestamps_per_stream():
    gate = OpenVinsSensorRateGate()
    assert gate.imu_due(0.1)
    with pytest.raises(ValueError, match="IMU timestamps must be monotonic"):
        gate.imu_due(0.09)

    assert gate.camera_due(0.2)
    with pytest.raises(ValueError, match="camera timestamps must be monotonic"):
        gate.camera_due(0.19)


def test_ros_pythonpath_can_be_restored_after_embedded_runtime_rewrites_sys_path(
    monkeypatch, tmp_path
):
    ros_site = tmp_path / "ros_site_packages"
    ros_site.mkdir()
    original_sys_path = list(__import__("sys").path)
    monkeypatch.setenv("PYTHONPATH", str(ros_site))
    monkeypatch.setattr(__import__("sys"), "path", list(original_sys_path))

    added = _restore_pythonpath_from_environment()

    assert added == (str(ros_site),)
    assert __import__("sys").path[0] == str(ros_site)


class _FakeRclpyQueue:
    def __init__(self, bridge, samples):
        self.bridge = bridge
        self.samples = list(samples)

    def spin_once(self, node, timeout_sec=0.0):
        del node, timeout_sec
        if not self.samples:
            return
        sample = self.samples.pop(0)
        with self.bridge._latest_lock:
            self.bridge._latest = sample
            self.bridge._odom_callback_count += 1


def _sample(timestamp_s: float) -> OpenVinsOdomSample:
    return OpenVinsOdomSample(
        timestamp_s=timestamp_s,
        position_v_i=np.array([timestamp_s, 0.0, 0.0]),
        orientation_v_i_wxyz=np.array([1.0, 0.0, 0.0, 0.0]),
        linear_velocity_v_i=np.zeros(3),
    )


def _fake_ros_bridge(samples) -> OpenVinsRos2Bridge:
    # Avoid importing/initializing ROS2 in this pure-Python regression test.
    bridge = object.__new__(OpenVinsRos2Bridge)
    bridge._node = object()
    bridge._latest_lock = Lock()
    bridge._latest = None
    bridge._odom_callback_count = 0
    bridge.last_drain_count = 0
    bridge._rclpy = _FakeRclpyQueue(bridge, samples)
    return bridge


def test_drain_latest_collapses_200hz_queue_to_newest_sample():
    bridge = _fake_ros_bridge([_sample(0.005), _sample(0.010), _sample(0.015)])

    latest = bridge.drain_latest(max_callbacks=32)

    assert latest is not None
    assert latest.timestamp_s == pytest.approx(0.015)
    assert bridge.last_drain_count == 3
    assert bridge.odom_callback_count == 3


def test_drain_latest_is_bounded_and_can_resume_next_cycle():
    bridge = _fake_ros_bridge([_sample(0.005), _sample(0.010), _sample(0.015)])

    first = bridge.drain_latest(max_callbacks=2)
    assert first.timestamp_s == pytest.approx(0.010)
    assert bridge.last_drain_count == 2

    second = bridge.drain_latest(max_callbacks=2)
    assert second.timestamp_s == pytest.approx(0.015)
    assert bridge.last_drain_count == 1


def test_drain_latest_rejects_nonpositive_bound():
    bridge = _fake_ros_bridge([])
    with pytest.raises(ValueError, match="max_callbacks must be at least 1"):
        bridge.drain_latest(max_callbacks=0)
