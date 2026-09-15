import numpy as np
import pytest

from estimation.learned_vio_drift import LearnedDisplacementMeasurement, LearnedVioDriftFilter
from estimation.swift_vio_drift import VioWorldEstimate


def _vio(x: float, t: float, vx: float = 0.0) -> VioWorldEstimate:
    return VioWorldEstimate(
        position_w_b=np.array([x, 0.0, 0.0]),
        linear_velocity_w_b=np.array([vx, 0.0, 0.0]),
        orientation_w_b_wxyz=np.array([1.0, 0.0, 0.0, 0.0]),
        timestamp_s=t,
    )


def test_prediction_only_preserves_raw_vio_when_drift_state_is_zero():
    filt = LearnedVioDriftFilter()
    fused = filt.step(_vio(1.2, 0.1))
    np.testing.assert_allclose(fused.position_w_b, [1.2, 0.0, 0.0])
    np.testing.assert_allclose(fused.linear_velocity_w_b, [0.0, 0.0, 0.0])


def test_relative_displacement_measurement_removes_known_vio_drift():
    filt = LearnedVioDriftFilter(
        sigma_position=1.0e-4,
        sigma_velocity=1.0e-4,
        innovation_gate_chi2=None,
    )
    filt.reset(0.0, anchor_vio_position_w=np.zeros(3))
    filt.step(_vio(0.0, 0.0))
    raw = _vio(1.5, 0.5, vx=3.0)
    measurement = LearnedDisplacementMeasurement(
        displacement_w=np.array([1.0, 0.0, 0.0]),
        covariance_w=np.eye(3) * 1.0e-5,
        start_timestamp_s=0.0,
        end_timestamp_s=0.5,
    )
    corrected = filt.step(raw, measurement=measurement)
    assert corrected.position_w_b[0] == pytest.approx(1.0, abs=0.03)
    assert filt.last_update_diagnostics.accepted == 1


def test_catastrophic_learned_measurement_is_rejected_by_nis_gate():
    filt = LearnedVioDriftFilter(innovation_gate_chi2=16.26623619623813)
    filt.reset(0.0, anchor_vio_position_w=np.zeros(3))
    filt.step(_vio(0.0, 0.0))
    raw = _vio(0.1, 0.5)
    measurement = LearnedDisplacementMeasurement(
        displacement_w=np.array([100.0, 0.0, 0.0]),
        covariance_w=np.eye(3) * 1.0e-4,
        start_timestamp_s=0.0,
        end_timestamp_s=0.5,
    )
    corrected = filt.step(raw, measurement=measurement)
    assert filt.last_update_diagnostics.accepted == 0
    np.testing.assert_allclose(corrected.position_w_b, raw.position_w_b, atol=1e-9)


def test_joseph_update_keeps_covariance_symmetric_psd_and_orientation_unchanged():
    filt = LearnedVioDriftFilter(innovation_gate_chi2=None)
    filt.reset(0.0, anchor_vio_position_w=np.zeros(3))
    q = np.array([0.9238795, 0.0, 0.0, 0.3826834])
    raw = VioWorldEstimate(np.array([0.2, 0.0, 0.0]), np.zeros(3), q, 0.5)
    measurement = LearnedDisplacementMeasurement(
        displacement_w=np.array([0.0, 0.0, 0.0]),
        covariance_w=np.eye(3) * 0.01,
        start_timestamp_s=0.0,
        end_timestamp_s=0.5,
    )
    corrected = filt.step(raw, measurement=measurement)
    np.testing.assert_allclose(filt.P, filt.P.T, atol=1e-12)
    assert np.linalg.eigvalsh(filt.P).min() >= -1e-10
    np.testing.assert_allclose(corrected.orientation_w_b_wxyz, q / np.linalg.norm(q), atol=1e-12)


def test_reset_clears_drift_and_reanchors():
    filt = LearnedVioDriftFilter(innovation_gate_chi2=None)
    filt.reset(0.0, anchor_vio_position_w=np.zeros(3))
    measurement = LearnedDisplacementMeasurement(
        displacement_w=np.zeros(3),
        covariance_w=np.eye(3) * 1.0e-4,
        start_timestamp_s=0.0,
        end_timestamp_s=0.5,
    )
    filt.step(_vio(0.5, 0.5), measurement=measurement)
    assert np.linalg.norm(filt.current_position_drift_w) > 0.1
    filt.reset(1.0, anchor_vio_position_w=np.array([4.0, 0.0, 0.0]))
    np.testing.assert_allclose(filt.current_position_drift_w, np.zeros(3))
    np.testing.assert_allclose(filt.anchor_vio_position_w, [4.0, 0.0, 0.0])


def test_absolute_position_anchor_corrects_current_drift_without_changing_attitude():
    filt = LearnedVioDriftFilter(
        sigma_position=1.0e-4,
        sigma_velocity=1.0e-4,
        innovation_gate_chi2=None,
    )
    filt.reset(0.0, anchor_vio_position_w=np.zeros(3))
    raw = _vio(5.0, 1.0, vx=0.5)
    predicted = filt.step(raw)
    np.testing.assert_allclose(predicted.position_w_b, [5.0, 0.0, 0.0])

    corrected = filt.apply_absolute_position(
        raw,
        position_w_b=np.array([2.0, 0.0, 0.0]),
        covariance_w=np.eye(3) * 1.0e-6,
    )

    assert corrected.position_w_b[0] == pytest.approx(2.0, abs=0.02)
    assert filt.current_position_drift_w[0] == pytest.approx(3.0, abs=0.02)
    np.testing.assert_allclose(
        corrected.orientation_w_b_wxyz,
        raw.orientation_w_b_wxyz,
        atol=1.0e-12,
    )
    np.testing.assert_allclose(filt.P, filt.P.T, atol=1.0e-12)
    assert np.linalg.eigvalsh(filt.P).min() >= -1.0e-10
