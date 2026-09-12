"""Bounded timestamp queue for camera observations awaiting delayed VIO."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

from perception.corner_detection import CornerObservation


@dataclass(frozen=True)
class PendingGateFrame:
    observation: CornerObservation
    gate_index: int | None
    rgb: np.ndarray


class TimestampedGateFrameQueue:
    """Retain camera frames until VIO reaches their source timestamp."""

    def __init__(self, *, max_age_s: float = 2.0) -> None:
        if max_age_s <= 0.0:
            raise ValueError("max_age_s must be positive")
        self.max_age_s = float(max_age_s)
        self._frames: deque[PendingGateFrame] = deque()
        self.dropped_count = 0

    def __len__(self) -> int:
        return len(self._frames)

    def clear(self) -> None:
        self._frames.clear()

    def append(self, frame: PendingGateFrame) -> None:
        timestamp = float(frame.observation.timestamp_s)
        if self._frames and timestamp < self._frames[-1].observation.timestamp_s - 1.0e-9:
            raise ValueError("Gate frame timestamps must be monotonic")
        self._frames.append(frame)
        while self._frames and timestamp - self._frames[0].observation.timestamp_s > self.max_age_s:
            self._frames.popleft()
            self.dropped_count += 1

    def discard_before(self, timestamp_s: float) -> None:
        """Discard frames older than the first VIO state retained for interpolation."""
        timestamp = float(timestamp_s)
        while self._frames and self._frames[0].observation.timestamp_s < timestamp - 1.0e-9:
            self._frames.popleft()
            self.dropped_count += 1

    def pop_ready(self, timestamp_s: float) -> PendingGateFrame | None:
        """Pop the oldest frame only after VIO has reached its timestamp."""
        if not self._frames:
            return None
        if self._frames[0].observation.timestamp_s > float(timestamp_s) + 1.0e-9:
            return None
        return self._frames.popleft()
