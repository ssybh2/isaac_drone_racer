import numpy as np
import pytest

from estimation.swift_vio_drift import VioDriftKalmanFilter, VioWorldEstimate


def _vio(position_x: float, timestamp_s: float) -> VioWorldEstimate:
    return VioWorldEstimate(
        position_w_b=np.array([position_x, 0.0, 0.0]),
        linear_velocity_w_b=np.zeros(3),
        orientation_w_b_wxyz=np.array([1.0, 0.0, 0.0, 0.0]),
        timestamp_s=timestamp_s,
    )


def test_small_gate_covariance_strongly_corrects_vio_position_drift_without_gate():
    filt = VioDriftKalmanFilter(sigma_pos=0.05, sigma_vel=0.1, innovation_gate_chi2=None)
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

    small = VioDriftKalmanFilter(sigma_pos=0.05, sigma_vel=0.1, innovation_gate_chi2=None)
    small.reset(0.0)
    fused_small = small.step(
        vio,
        gate_positions_w_b=(np.zeros(3),),
        position_covariances_w=(np.eye(3) * 1.0e-3,),
    )

    large = VioDriftKalmanFilter(sigma_pos=0.05, sigma_vel=0.1, innovation_gate_chi2=None)
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


def test_mahalanobis_gate_rejects_catastrophic_gate_pose():
    filt = VioDriftKalmanFilter(sigma_pos=0.05, sigma_vel=0.1)
    filt.reset(0.0)
    vio = _vio(0.0, 0.01)

    fused = filt.step(
        vio,
        gate_positions_w_b=(np.array([50.0, 0.0, 0.0]),),
        position_covariances_w=(np.eye(3) * 1.0e-3,),
    )

    diagnostics = filt.last_update_diagnostics
    assert diagnostics.attempted == 1
    assert diagnostics.accepted == 0
    assert diagnostics.rejected == 1
    np.testing.assert_allclose(fused.position_w_b, vio.position_w_b, atol=1e-12)


def test_joseph_update_keeps_covariance_symmetric_positive_semidefinite():
    filt = VioDriftKalmanFilter(
        sigma_pos=0.05,
        sigma_vel=0.1,
        innovation_gate_chi2=None,
    )
    filt.reset(0.0)
    vio = _vio(0.2, 0.01)
    filt.step(
        vio,
        gate_positions_w_b=(np.zeros(3),),
        position_covariances_w=(np.eye(3) * 1.0e-3,),
    )
    np.testing.assert_allclose(filt.P, filt.P.T, atol=1e-12)
    assert np.linalg.eigvalsh(filt.P).min() >= -1e-12


def test_same_timestamp_prediction_does_not_add_process_noise():
    filt = VioDriftKalmanFilter()
    filt.reset(1.0)

    filt.predict(1.0)
    np.testing.assert_allclose(filt.P, np.zeros((6, 6)), atol=0.0)


def test_nominal_100hz_step_matches_paper_process_covariance():
    filt = VioDriftKalmanFilter(sigma_pos=0.05, sigma_vel=0.1, nominal_rate_hz=100.0)
    filt.reset(0.0)

    filt.predict(0.01)

    np.testing.assert_allclose(np.diag(filt.P)[:3], np.full(3, 0.05), atol=1e-12)
    np.testing.assert_allclose(np.diag(filt.P)[3:], np.full(3, 0.1), atol=1e-12)


def test_split_timestamp_alignment_does_not_double_process_noise():
    single = VioDriftKalmanFilter(sigma_pos=0.05, sigma_vel=0.1, nominal_rate_hz=100.0)
    single.reset(0.0)
    single.predict(0.01)

    split = VioDriftKalmanFilter(sigma_pos=0.05, sigma_vel=0.1, nominal_rate_hz=100.0)
    split.reset(0.0)
    split.predict(0.005)
    split.predict(0.01)

    # Splitting a 10 ms interval around a camera update should contribute about
    # one nominal step of process uncertainty, not two full Q additions.
    assert np.trace(split.P) == pytest.approx(np.trace(single.P), rel=1.0e-3)


def test_invalid_nominal_rate_is_rejected():
    with pytest.raises(ValueError, match="nominal_rate_hz"):
        VioDriftKalmanFilter(nominal_rate_hz=0.0)
