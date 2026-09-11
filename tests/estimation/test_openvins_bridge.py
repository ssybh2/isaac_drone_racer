import numpy as np

from estimation.openvins_bridge import (
    OpenVinsFrameAlignment,
    OpenVinsOdomSample,
    OpenVinsSensorRateGate,
)


def test_openvins_alignment_recovers_reference_pose():
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


def test_openvins_rate_gate_matches_swift_source_rates():
    gate = OpenVinsSensorRateGate(imu_hz=200.0, camera_hz=30.0)
    assert gate.imu_due(0.0)
    assert not gate.imu_due(0.0025)
    assert gate.imu_due(0.005)

    assert gate.camera_due(0.0)
    assert not gate.camera_due(0.02)
    assert gate.camera_due(1.0 / 30.0)
