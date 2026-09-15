import numpy as np
import pytest

from estimation.hybrid_vio_corrector import HybridLearnedVioCorrector, body_motion_to_world
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


class RecordingPredictor(ConstantPredictor):
    def __init__(self, displacement):
        super().__init__(displacement)
        self.last_window = None

    def predict(self, window):
        self.last_window = window
        return super().predict(window)


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


def test_body_motion_is_rotated_to_world_using_raw_vio_attitude():
    # 90 degree yaw: body +X maps to world +Y; body +Z is unchanged.
    q = np.array([np.sqrt(0.5), 0.0, 0.0, np.sqrt(0.5)])
    gyro_w, thrust_w = body_motion_to_world(
        _vio(0.0, 0.0, q=q),
        gyro_b=np.array([1.0, 0.0, 0.0]),
        thrust_b=np.array([0.0, 0.0, 6.0]),
    )
    np.testing.assert_allclose(gyro_w, [0.0, 1.0, 0.0], atol=1e-8)
    np.testing.assert_allclose(thrust_w, [0.0, 0.0, 6.0], atol=1e-8)


def test_body_motion_history_is_timestamp_aligned_and_rotated_before_prediction():
    predictor = RecordingPredictor([0.0, 0.0, 0.0])
    corrector = HybridLearnedVioCorrector(
        predictor,
        window_time_s=0.5,
        sample_rate_hz=100.0,
        innovation_gate_chi2=None,
    )
    q = np.array([np.sqrt(0.5), 0.0, 0.0, np.sqrt(0.5)])
    for t in np.arange(0.0, 0.51, 0.01):
        corrector.ingest_body_motion_sample(
            float(t),
            gyro_b=np.array([1.0, 0.0, 0.0]),
            thrust_b=np.array([0.0, 0.0, 6.0]),
        )
    corrector.step(_vio(0.0, 0.0, q=q))
    corrector.step(_vio(0.0, 0.5, q=q))

    assert predictor.last_window is not None
    np.testing.assert_allclose(predictor.last_window.features[0, :], 0.0, atol=1e-7)
    np.testing.assert_allclose(predictor.last_window.features[1, :], 1.0, atol=1e-7)
    np.testing.assert_allclose(predictor.last_window.features[5, :], 6.0, atol=1e-7)


def test_result_exposes_exact_window_and_raw_vio_displacement_for_gt_diagnostics():
    predictor = ConstantPredictor([1.0, 0.0, 0.0])
    corrector = HybridLearnedVioCorrector(
        predictor,
        window_time_s=0.5,
        sample_rate_hz=100.0,
        innovation_gate_chi2=None,
    )
    _feed_motion(corrector)
    corrector.step(_vio(0.0, 0.0))

    result = corrector.step(_vio(1.5, 0.5))

    assert result.prediction_start_timestamp_s == pytest.approx(0.0)
    assert result.prediction_end_timestamp_s == pytest.approx(0.5)
    np.testing.assert_allclose(result.raw_window_displacement_w, [1.5, 0.0, 0.0])
    np.testing.assert_allclose(result.prediction.displacement_w, [1.0, 0.0, 0.0])


def test_corrector_can_apply_sparse_absolute_position_anchor_after_relative_update():
    predictor = ConstantPredictor([2.0, 0.0, 0.0])
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
    raw = _vio(3.0, 0.5)
    relative = corrector.step(raw)
    assert relative.corrected.position_w_b[0] == pytest.approx(2.0, abs=0.03)

    anchored = corrector.apply_absolute_position(
        raw,
        position_w_b=np.array([1.0, 0.0, 0.0]),
        covariance_w=np.eye(3) * 1.0e-6,
    )

    assert anchored.absolute_position_update_applied is True
    assert abs(anchored.corrected.position_w_b[0] - 1.0) < abs(
        relative.corrected.position_w_b[0] - 1.0
    )
    assert anchored.corrected.position_w_b[0] == pytest.approx(1.0, abs=0.12)
    assert anchored.raw is relative.raw


def test_raw_vio_jump_isolation_keeps_corrected_position_continuous():
    predictor = ConstantPredictor([0.0, 0.0, 0.0])
    corrector = HybridLearnedVioCorrector(
        predictor,
        window_time_s=0.5,
        sample_rate_hz=100.0,
        raw_vio_jump_isolation=True,
        raw_vio_jump_threshold_m=0.5,
    )

    first = corrector.step(_vio(0.0, 0.0, vx=0.0))
    jumped = corrector.step(_vio(5.0, 0.01, vx=0.0))
    after = corrector.step(_vio(5.01, 0.02, vx=1.0))

    np.testing.assert_allclose(first.corrected.position_w_b, [0.0, 0.0, 0.0])
    np.testing.assert_allclose(jumped.raw.position_w_b, [5.0, 0.0, 0.0])
    assert jumped.raw_vio_jump_detected is True
    assert jumped.raw_vio_jump_residual_m == pytest.approx(5.0)
    np.testing.assert_allclose(jumped.raw_vio_jump_compensation_w, [5.0, 0.0, 0.0])
    np.testing.assert_allclose(jumped.corrected.position_w_b, [0.0, 0.0, 0.0], atol=1e-9)
    assert after.raw_vio_jump_detected is False
    assert after.corrected.position_w_b[0] == pytest.approx(0.01, abs=0.006)


def test_explicit_drift_velocity_from_displacement_propagates_correction_between_windows():
    predictor = ConstantPredictor([1.0, 0.0, 0.0])
    corrector = HybridLearnedVioCorrector(
        predictor,
        window_time_s=0.5,
        sample_rate_hz=100.0,
        relative_position_only=True,
        learned_drift_velocity_from_displacement=True,
        learned_drift_velocity_sigma_floor_mps=0.01,
        innovation_gate_chi2=None,
    )
    _feed_motion(corrector)
    corrector.step(_vio(0.0, 0.0, vx=3.0))

    boundary = corrector.step(_vio(1.5, 0.5, vx=3.0))
    mid = corrector.step(_vio(2.25, 0.75, vx=3.0))

    assert boundary.learned_update_accepted is True
    assert boundary.learned_velocity_update_applied is True
    np.testing.assert_allclose(
        boundary.learned_velocity_drift_measurement_w,
        [1.0, 0.0, 0.0],
        atol=1.0e-9,
    )
    assert boundary.corrected.position_w_b[0] == pytest.approx(1.0, abs=0.03)
    assert boundary.corrected.linear_velocity_w_b[0] == pytest.approx(2.0, abs=0.03)
    assert mid.corrected.position_w_b[0] == pytest.approx(1.5, abs=0.05)
    assert mid.corrected.linear_velocity_w_b[0] == pytest.approx(2.0, abs=0.03)


def test_position_residual_slew_defers_learned_position_jump_and_releases_at_rate_limit():
    predictor = ConstantPredictor([1.0, 0.0, 0.0])
    corrector = HybridLearnedVioCorrector(
        predictor,
        window_time_s=0.5,
        sample_rate_hz=100.0,
        relative_position_only=True,
        learned_drift_velocity_from_displacement=True,
        learned_drift_velocity_sigma_floor_mps=0.01,
        learned_position_residual_slew=True,
        learned_position_residual_max_rate_mps=4.0,
        innovation_gate_chi2=None,
    )
    _feed_motion(corrector)
    corrector.step(_vio(0.0, 0.0, vx=3.0))
    before = corrector.step(_vio(1.47, 0.49, vx=3.0))
    boundary = corrector.step(_vio(1.5, 0.5, vx=3.0))

    assert boundary.learned_update_accepted is True
    assert boundary.learned_position_injection_norm_m > 0.4
    assert boundary.learned_position_release_norm_m <= 0.040000001
    assert boundary.learned_position_pending_norm_m > 0.3
    assert abs(boundary.corrected.position_w_b[0] - before.corrected.position_w_b[0]) < 0.1

    pending_before = boundary.learned_position_pending_norm_m
    after = corrector.step(_vio(1.53, 0.51, vx=3.0))
    assert after.learned_position_release_norm_m == pytest.approx(0.04, abs=1.0e-8)
    assert after.learned_position_pending_norm_m < pending_before
    np.testing.assert_allclose(
        corrector.drift_filter.anchor_position_drift_w,
        corrector.drift_filter.current_position_drift_w,
        atol=0.02,
    )


def test_absolute_position_anchor_clears_pending_learned_position_residual():
    predictor = ConstantPredictor([1.0, 0.0, 0.0])
    corrector = HybridLearnedVioCorrector(
        predictor,
        window_time_s=0.5,
        sample_rate_hz=100.0,
        relative_position_only=True,
        learned_position_residual_slew=True,
        learned_position_residual_max_rate_mps=4.0,
        innovation_gate_chi2=None,
    )
    _feed_motion(corrector)
    corrector.step(_vio(0.0, 0.0, vx=3.0))
    corrector.step(_vio(1.47, 0.49, vx=3.0))
    raw = _vio(1.5, 0.5, vx=3.0)
    boundary = corrector.step(raw)
    assert boundary.learned_position_pending_norm_m > 0.3

    anchored = corrector.apply_absolute_position(
        raw,
        position_w_b=np.array([1.0, 0.0, 0.0]),
        covariance_w=np.eye(3) * 1.0e-6,
    )

    assert anchored.absolute_position_update_applied is True
    assert anchored.learned_position_pending_norm_m == pytest.approx(0.0, abs=1.0e-12)
    np.testing.assert_allclose(corrector.pending_position_drift_w, np.zeros(3), atol=1.0e-12)
