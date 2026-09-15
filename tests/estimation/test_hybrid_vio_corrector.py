import numpy as np
import pytest

from estimation.hybrid_vio_corrector import HybridLearnedVioCorrector
from estimation.learned_motion import DisplacementPrediction
from estimation.learned_vio_drift import LearnedVioDriftFilter
from estimation.swift_vio_drift import VioWorldEstimate


class ConstantPredictor:
    def __init__(self, displacement):
        self.displacement = np.asarray(displacement, dtype=np.float64)
        self.calls = 0

    def predict(self, window):
        self.calls += 1
        return DisplacementPrediction(self.displacement, np.eye(3) * 1.0e-5)


def _vio(x, t, vx=0.0, q=None):
    if q is None:
        q = np.array([1.0, 0.0, 0.0, 0.0])
    return VioWorldEstimate(np.array([x, 0.0, 0.0]), np.array([vx, 0.0, 0.0]), q, t)


def _feed_motion(corrector, start=0.0, end=0.5):
    for t in np.arange(start, end + 1.0e-9, 0.01):
        corrector.ingest_motion_sample(
            float(t), gyro_w=np.zeros(3), thrust_w=np.array([0.0, 0.0, 6.0])
        )


def test_corrector_is_passthrough_until_full_learned_window_exists():
    predictor = ConstantPredictor([0.4, 0.0, 0.0])
    corrector = HybridLearnedVioCorrector(predictor, window_time_s=0.5, sample_rate_hz=100.0)
    _feed_motion(corrector, end=0.3)

    first = corrector.step(_vio(0.0, 0.0))
    mid = corrector.step(_vio(0.3, 0.3, vx=1.0))

    np.testing.assert_allclose(first.corrected.position_w_b, [0.0, 0.0, 0.0])
    np.testing.assert_allclose(mid.corrected.position_w_b, [0.3, 0.0, 0.0])
    assert predictor.calls == 0
    assert mid.learned_update_attempted is False


def test_corrector_uses_learned_displacement_to_remove_relative_vio_drift():
    predictor = ConstantPredictor([1.0, 0.0, 0.0])
    drift_filter = LearnedVioDriftFilter(
        sigma_position=1.0e-4,
        sigma_velocity=1.0e-4,
        innovation_gate_chi2=None,
    )
    corrector = HybridLearnedVioCorrector(
        predictor,
        window_time_s=0.5,
        sample_rate_hz=100.0,
        drift_filter=drift_filter,
    )
    _feed_motion(corrector)
    corrector.step(_vio(0.0, 0.0))

    result = corrector.step(_vio(1.5, 0.5, vx=3.0))

    assert predictor.calls == 1
    assert result.learned_update_attempted is True
    assert result.learned_update_accepted is True
    assert result.corrected.position_w_b[0] == pytest.approx(1.0, abs=0.03)
    np.testing.assert_allclose(result.raw.position_w_b, [1.5, 0.0, 0.0])


def test_corrector_preserves_openvins_attitude_exactly():
    predictor = ConstantPredictor([0.0, 0.0, 0.0])
    corrector = HybridLearnedVioCorrector(
        predictor,
        window_time_s=0.5,
        sample_rate_hz=100.0,
        innovation_gate_chi2=None,
    )
    _feed_motion(corrector)
    corrector.step(_vio(0.0, 0.0))
    q = np.array([0.9238795, 0.0, 0.0, 0.3826834])
    result = corrector.step(_vio(0.2, 0.5, q=q))
    np.testing.assert_allclose(result.corrected.orientation_w_b_wxyz, q / np.linalg.norm(q))


def test_rejected_learned_window_is_advanced_instead_of_replayed_forever():
    predictor = ConstantPredictor([100.0, 0.0, 0.0])
    corrector = HybridLearnedVioCorrector(
        predictor,
        window_time_s=0.5,
        sample_rate_hz=100.0,
        innovation_gate_chi2=16.26623619623813,
    )
    for t in np.arange(0.0, 1.01, 0.01):
        corrector.ingest_motion_sample(
            float(t), gyro_w=np.zeros(3), thrust_w=np.array([0.0, 0.0, 6.0])
        )
    corrector.step(_vio(0.0, 0.0))
    first = corrector.step(_vio(0.1, 0.5))
    second = corrector.step(_vio(0.2, 1.0))

    assert first.learned_update_attempted is True
    assert first.learned_update_accepted is False
    assert second.learned_update_attempted is True
    assert predictor.calls == 2
    assert corrector.window_start_timestamp_s == pytest.approx(1.0)
