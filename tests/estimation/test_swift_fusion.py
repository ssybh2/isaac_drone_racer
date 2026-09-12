from dataclasses import dataclass

import numpy as np

from estimation.swift_fusion import SwiftPerceptionFusion
from estimation.swift_vio_drift import VioDriftKalmanFilter, VioWorldEstimate
from perception.corner_detection import CornerObservation


@dataclass
class _Measurement:
    position_w_b: np.ndarray
    position_covariance_w: np.ndarray


class _Builder:
    def __init__(self):
        self.reference_positions = []

    def build(self, observation, *, gate_index=None, reference_position_w_b=None):
        self.reference_positions.append(np.asarray(reference_position_w_b).copy())
        return _Measurement(
            position_w_b=np.asarray(reference_position_w_b).copy(),
            position_covariance_w=np.eye(3) * 1.0e-3,
        )


def _vio(t, x):
    return VioWorldEstimate(
        position_w_b=np.array([x, 0.0, 0.0]),
        linear_velocity_w_b=np.array([10.0, 0.0, 0.0]),
        orientation_w_b_wxyz=np.array([1.0, 0.0, 0.0, 0.0]),
        timestamp_s=t,
    )


def _obs(t):
    return CornerObservation(
        corners_uv=np.array([[10.0, 90.0], [90.0, 90.0], [90.0, 10.0], [10.0, 10.0]]),
        visible=np.ones(4, dtype=bool),
        confidence=np.ones(4),
        timestamp_s=t,
        source="test",
    )


def test_gate_association_uses_vio_interpolated_at_camera_timestamp():
    builder = _Builder()
    fusion = SwiftPerceptionFusion(
        builder,
        VioDriftKalmanFilter(innovation_gate_chi2=None),
    )
    fusion.reset(0.0)
    fusion.step(_vio(0.0, 0.0))
    result = fusion.step(_vio(0.1, 1.0), gate_observation=_obs(0.05), gate_index=0)

    assert result.measurement_accepted
    np.testing.assert_allclose(builder.reference_positions[-1], [0.5, 0.0, 0.0], atol=1e-9)
    assert result.fused_state.timestamp_s == 0.1


def test_stale_out_of_sequence_gate_is_rejected_instead_of_fused_at_wrong_time():
    builder = _Builder()
    fusion = SwiftPerceptionFusion(builder)
    fusion.reset(0.0)
    fusion.step(_vio(0.0, 0.0))
    fusion.step(_vio(0.1, 1.0))
    result = fusion.step(_vio(0.2, 2.0), gate_observation=_obs(0.05), gate_index=0)

    assert not result.measurement_accepted
    assert "older than the current drift-filter state" in result.rejection_reason
    assert not builder.reference_positions
