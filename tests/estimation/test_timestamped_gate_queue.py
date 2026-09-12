import numpy as np
import pytest

from estimation.timestamped_gate_queue import PendingGateFrame, TimestampedGateFrameQueue
from perception.corner_detection import CornerObservation


def _frame(timestamp_s: float) -> PendingGateFrame:
    observation = CornerObservation(
        corners_uv=np.zeros((4, 2)),
        visible=np.ones(4, dtype=bool),
        confidence=np.ones(4),
        timestamp_s=timestamp_s,
        source="test",
    )
    return PendingGateFrame(observation, None, np.zeros((2, 2, 3), dtype=np.uint8))


def test_delayed_vio_pops_camera_frames_in_timestamp_order():
    queue = TimestampedGateFrameQueue(max_age_s=2.0)
    queue.append(_frame(1.00))
    queue.append(_frame(1.03))

    assert queue.pop_ready(0.99) is None
    assert queue.pop_ready(1.01).observation.timestamp_s == pytest.approx(1.00)
    assert queue.pop_ready(1.04).observation.timestamp_s == pytest.approx(1.03)


def test_queue_discards_frames_older_than_first_vio_sample():
    queue = TimestampedGateFrameQueue(max_age_s=2.0)
    for timestamp in (1.00, 1.03, 1.06):
        queue.append(_frame(timestamp))

    queue.discard_before(1.05)

    assert queue.dropped_count == 2
    assert queue.pop_ready(1.06).observation.timestamp_s == pytest.approx(1.06)


def test_queue_is_age_bounded_while_vio_is_unavailable():
    queue = TimestampedGateFrameQueue(max_age_s=0.05)
    queue.append(_frame(1.00))
    queue.append(_frame(1.03))
    queue.append(_frame(1.06))

    assert queue.dropped_count == 1
    assert len(queue) == 2


def test_queue_rejects_out_of_order_camera_timestamps():
    queue = TimestampedGateFrameQueue()
    queue.append(_frame(2.0))

    with pytest.raises(ValueError, match="monotonic"):
        queue.append(_frame(1.0))
