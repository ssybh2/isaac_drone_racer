import numpy as np
import pytest

from estimation.openvins_bridge import (
    OpenVinsFrameAlignment,
    OpenVinsOdomSample,
    OpenVinsSensorRateGate,
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
