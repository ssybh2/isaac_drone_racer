"""Short fixed-lag history for timestamp-aligned VIO/gate fusion."""

from __future__ import annotations

from collections import deque

import numpy as np

from .swift_vio_drift import VioWorldEstimate


def _slerp_wxyz(q0, q1, alpha: float) -> np.ndarray:
    q0 = np.asarray(q0, dtype=np.float64).reshape(4)
    q1 = np.asarray(q1, dtype=np.float64).reshape(4)
    q0 = q0 / np.linalg.norm(q0)
    q1 = q1 / np.linalg.norm(q1)
    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    dot = float(np.clip(dot, -1.0, 1.0))
    if dot > 0.9995:
        q = q0 + float(alpha) * (q1 - q0)
        return q / np.linalg.norm(q)
    theta = np.arccos(dot)
    sin_theta = np.sin(theta)
    w0 = np.sin((1.0 - float(alpha)) * theta) / sin_theta
    w1 = np.sin(float(alpha) * theta) / sin_theta
    q = w0 * q0 + w1 * q1
    return q / np.linalg.norm(q)


class VioWorldEstimateBuffer:
    """Monotonic VIO state history with interpolation at camera timestamps."""

    def __init__(self, *, max_age_s: float = 2.0, max_samples: int = 4096) -> None:
        if max_age_s <= 0.0:
            raise ValueError("max_age_s must be positive")
        if max_samples < 2:
            raise ValueError("max_samples must be at least 2")
        self.max_age_s = float(max_age_s)
        self.max_samples = int(max_samples)
        self._samples: deque[VioWorldEstimate] = deque()

    def clear(self) -> None:
        self._samples.clear()

    def __len__(self) -> int:
        return len(self._samples)

    @property
    def oldest(self) -> VioWorldEstimate | None:
        return None if not self._samples else self._samples[0]

    @property
    def latest(self) -> VioWorldEstimate | None:
        return None if not self._samples else self._samples[-1]

    def push(self, sample: VioWorldEstimate) -> None:
        timestamp = float(sample.timestamp_s)
        if self._samples and timestamp < self._samples[-1].timestamp_s - 1.0e-9:
            raise ValueError("VIO buffer timestamps must be monotonic")
        if self._samples and abs(timestamp - self._samples[-1].timestamp_s) <= 1.0e-9:
            self._samples[-1] = sample
        else:
            self._samples.append(sample)
        newest = self._samples[-1].timestamp_s
        while len(self._samples) > self.max_samples:
            self._samples.popleft()
        while len(self._samples) > 1 and newest - self._samples[0].timestamp_s > self.max_age_s:
            self._samples.popleft()

    def interpolate(self, timestamp_s: float) -> VioWorldEstimate:
        """Interpolate a state inside the retained time interval.

        No extrapolation is performed: a gate measurement outside the retained
        VIO interval is safer to reject than to fuse at the wrong time.
        """
        if not self._samples:
            raise ValueError("VIO buffer is empty")
        timestamp = float(timestamp_s)
        if not np.isfinite(timestamp):
            raise ValueError("Requested VIO timestamp must be finite")
        first = self._samples[0]
        last = self._samples[-1]
        if timestamp < first.timestamp_s - 1.0e-9 or timestamp > last.timestamp_s + 1.0e-9:
            raise ValueError(
                f"Requested timestamp {timestamp:.6f}s is outside VIO history "
                f"[{first.timestamp_s:.6f}, {last.timestamp_s:.6f}]s"
            )
        if abs(timestamp - first.timestamp_s) <= 1.0e-9:
            return first
        if abs(timestamp - last.timestamp_s) <= 1.0e-9:
            return last

        samples = list(self._samples)
        for left, right in zip(samples[:-1], samples[1:]):
            if left.timestamp_s - 1.0e-9 <= timestamp <= right.timestamp_s + 1.0e-9:
                dt = right.timestamp_s - left.timestamp_s
                if dt <= 1.0e-12:
                    return right
                alpha = float(np.clip((timestamp - left.timestamp_s) / dt, 0.0, 1.0))
                return VioWorldEstimate(
                    position_w_b=(1.0 - alpha) * left.position_w_b + alpha * right.position_w_b,
                    linear_velocity_w_b=(1.0 - alpha) * left.linear_velocity_w_b
                    + alpha * right.linear_velocity_w_b,
                    orientation_w_b_wxyz=_slerp_wxyz(
                        left.orientation_w_b_wxyz,
                        right.orientation_w_b_wxyz,
                        alpha,
                    ),
                    timestamp_s=timestamp,
                )
        raise RuntimeError("Unable to bracket requested VIO timestamp")
