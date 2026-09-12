import numpy as np
import pytest

from perception.corner_detection import CornerObservation
from perception.hybrid_keypoint_detector import VisibilityGuardedGateCornerDetector


class _StubDetector:
    def __init__(self, observation: CornerObservation):
        self.observation = observation

    def detect(self, rgb_image, *, timestamp_s=0.0):
        return CornerObservation(
            corners_uv=self.observation.corners_uv,
            visible=self.observation.visible,
            confidence=self.observation.confidence,
            timestamp_s=timestamp_s,
            source=self.observation.source,
        )


def _observation(*, visible, confidence, source):
    return CornerObservation(
        corners_uv=np.array([[10.0, 90.0], [90.0, 90.0], [90.0, 10.0], [10.0, 10.0]]),
        visible=np.asarray(visible),
        confidence=np.asarray(confidence),
        source=source,
    )


def test_guard_keeps_coordinate_corners_but_can_reject_visibility():
    coordinate = _StubDetector(
        _observation(visible=[True] * 4, confidence=[0.99] * 4, source="coordinate")
    )
    guard = _StubDetector(
        _observation(visible=[True] * 4, confidence=[0.9, 0.8, 0.74, 0.95], source="guard")
    )
    detector = VisibilityGuardedGateCornerDetector(
        coordinate, guard, visibility_threshold=0.75
    )

    result = detector.detect(np.zeros((100, 100, 3), dtype=np.uint8), timestamp_s=1.25)

    np.testing.assert_allclose(result.corners_uv, coordinate.observation.corners_uv)
    assert result.visible.tolist() == [True, True, False, True]
    np.testing.assert_allclose(result.confidence, [0.9, 0.8, 0.74, 0.95])
    assert result.timestamp_s == 1.25
    assert result.source == "coordinate+visibility_guard:guard"


def test_guard_cannot_reenable_coordinate_detector_rejection():
    coordinate = _StubDetector(
        _observation(
            visible=[True, False, True, True], confidence=[0.99] * 4, source="coordinate"
        )
    )
    guard = _StubDetector(
        _observation(visible=[True] * 4, confidence=[1.0] * 4, source="guard")
    )
    detector = VisibilityGuardedGateCornerDetector(coordinate, guard)

    result = detector.detect(np.zeros((100, 100, 3), dtype=np.uint8))

    assert result.visible.tolist() == [True, False, True, True]


def test_visibility_threshold_must_be_a_probability():
    stub = _StubDetector(
        _observation(visible=[True] * 4, confidence=[1.0] * 4, source="stub")
    )
    with pytest.raises(ValueError, match="visibility_threshold"):
        VisibilityGuardedGateCornerDetector(stub, stub, visibility_threshold=1.1)
