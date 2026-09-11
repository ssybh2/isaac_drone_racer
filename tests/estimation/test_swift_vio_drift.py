import numpy as np

from estimation.swift_vio_drift import VioDriftKalmanFilter, VioWorldEstimate


def _vio(position_x: float, timestamp_s: float) -> VioWorldEstimate:
    return VioWorldEstimate(
        position_w_b=np.array([position_x, 0.0, 0.0]),
        linear_velocity_w_b=np.zeros(3),
        orientation_w_b_wxyz=np.array([1.0, 0.0, 0.0, 0.0]),
        timestamp_s=timestamp_s,
    )


def test_small_gate_covariance_strongly_corrects_vio_position_drift():
    filt = VioDriftKalmanFilter(sigma_pos=0.05, sigma_vel=0.1)
    filt.reset(0.0)
    vio = _vio(1.0, 0.01)

    fused = filt.step(
        vio,
        gate_positions_w_b=(np.zeros(3),),
        position_covariances_w=(np.eye(3) * 1.0e-3,),
    )

    assert abs(fused.position_w_b[0]) < 0.1
    assert fused.orientation_w_b_wxyz.tolist() == [1.0, 0.0, 0.0, 0.0]


def test_large_gate_covariance_preserves_vio_more_than_small_covariance():
    vio = _vio(1.0, 0.01)

    small = VioDriftKalmanFilter(sigma_pos=0.05, sigma_vel=0.1)
    small.reset(0.0)
    fused_small = small.step(
        vio,
        gate_positions_w_b=(np.zeros(3),),
        position_covariances_w=(np.eye(3) * 1.0e-3,),
    )

    large = VioDriftKalmanFilter(sigma_pos=0.05, sigma_vel=0.1)
    large.reset(0.0)
    fused_large = large.step(
        vio,
        gate_positions_w_b=(np.zeros(3),),
        position_covariances_w=(np.eye(3) * 10.0,),
    )

    assert abs(fused_small.position_w_b[0]) < abs(fused_large.position_w_b[0])
    assert fused_large.position_w_b[0] > 0.9


def test_prediction_only_keeps_orientation_from_vio():
    filt = VioDriftKalmanFilter()
    filt.reset(0.0)
    vio = VioWorldEstimate(
        position_w_b=np.array([0.4, -0.1, 1.0]),
        linear_velocity_w_b=np.array([1.0, 2.0, 3.0]),
        orientation_w_b_wxyz=np.array([0.9238795, 0.0, 0.0, 0.3826834]),
        timestamp_s=0.01,
    )

    fused = filt.step(vio)

    assert np.allclose(fused.orientation_w_b_wxyz, vio.orientation_w_b_wxyz)
