"""Dataset utilities for supervised learned-motion training from diagnostic CSV traces."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


_REQUIRED_COLUMNS = (
    "t_s",
    "truth_px", "truth_py", "truth_pz",
    "truth_qw", "truth_qx", "truth_qy", "truth_qz",
    "imu_gx", "imu_gy", "imu_gz",
    "thrust_b_x", "thrust_b_y", "thrust_b_z",
)


@dataclass(frozen=True)
class TraceWindows:
    features: np.ndarray
    targets: np.ndarray
    start_timestamps_s: np.ndarray
    end_timestamps_s: np.ndarray
    source_path: Path


def _quat_wxyz_to_rotmat(q) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64).reshape(4)
    norm = float(np.linalg.norm(q))
    if norm <= 0.0 or not np.isfinite(norm):
        raise ValueError("truth quaternion must be finite and non-zero")
    w, x, y, z = q / norm
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _read_trace(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        columns = set(reader.fieldnames or ())
        missing = [name for name in _REQUIRED_COLUMNS if name not in columns]
        if missing:
            raise ValueError(f"trace {path} missing required columns: {', '.join(missing)}")
        rows = list(reader)
    if len(rows) < 2:
        raise ValueError(f"trace {path} must contain at least two rows")

    def col(name: str) -> np.ndarray:
        try:
            values = np.asarray([float(row[name]) for row in rows], dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"trace {path} contains invalid numeric values in {name}") from exc
        if not np.all(np.isfinite(values)):
            raise ValueError(f"trace {path} contains non-finite values in {name}")
        return values

    timestamps = col("t_s")
    if np.any(np.diff(timestamps) <= 0.0):
        raise ValueError(f"trace {path} timestamps must be strictly increasing")
    positions = np.column_stack((col("truth_px"), col("truth_py"), col("truth_pz")))
    quaternions = np.column_stack(
        (col("truth_qw"), col("truth_qx"), col("truth_qy"), col("truth_qz"))
    )
    gyro_b = np.column_stack((col("imu_gx"), col("imu_gy"), col("imu_gz")))
    thrust_b = np.column_stack((col("thrust_b_x"), col("thrust_b_y"), col("thrust_b_z")))

    features_w = np.empty((len(rows), 6), dtype=np.float64)
    for index, (q, gyro, thrust) in enumerate(zip(quaternions, gyro_b, thrust_b)):
        R_wb = _quat_wxyz_to_rotmat(q)
        features_w[index, 0:3] = R_wb @ gyro
        features_w[index, 3:6] = R_wb @ thrust
    return timestamps, positions, features_w


def load_trace_windows(
    path: str | Path,
    *,
    window_time_s: float = 0.5,
    sample_rate_hz: float = 100.0,
    stride_time_s: float = 0.01,
) -> TraceWindows:
    """Convert one complete trace into fixed-rate motion windows and truth dp labels."""
    path = Path(path)
    if window_time_s <= 0.0 or sample_rate_hz <= 0.0 or stride_time_s <= 0.0:
        raise ValueError("window_time_s, sample_rate_hz and stride_time_s must be positive")
    sample_count = int(round(float(window_time_s) * float(sample_rate_hz)))
    if sample_count < 2:
        raise ValueError("learned motion window must contain at least two samples")

    timestamps, positions, source_features = _read_trace(path)
    first = float(timestamps[0])
    last_start = float(timestamps[-1] - window_time_s)
    if last_start < first - 1.0e-9:
        raise ValueError(f"trace {path} is shorter than window_time_s={window_time_s}")
    num_windows = int(np.floor((last_start - first) / stride_time_s + 1.0e-9)) + 1
    starts = first + np.arange(num_windows, dtype=np.float64) * float(stride_time_s)
    ends = starts + float(window_time_s)

    features = np.empty((num_windows, 6, sample_count), dtype=np.float32)
    targets = np.empty((num_windows, 3), dtype=np.float32)
    sample_offsets = np.arange(sample_count, dtype=np.float64) / float(sample_rate_hz)
    for window_index, (start, end) in enumerate(zip(starts, ends)):
        sample_times = start + sample_offsets
        for channel in range(6):
            features[window_index, channel, :] = np.interp(
                sample_times, timestamps, source_features[:, channel]
            ).astype(np.float32)
        p_start = np.array(
            [np.interp(start, timestamps, positions[:, axis]) for axis in range(3)],
            dtype=np.float64,
        )
        p_end = np.array(
            [np.interp(end, timestamps, positions[:, axis]) for axis in range(3)],
            dtype=np.float64,
        )
        targets[window_index, :] = (p_end - p_start).astype(np.float32)

    return TraceWindows(
        features=features,
        targets=targets,
        start_timestamps_s=starts,
        end_timestamps_s=ends,
        source_path=path,
    )


def split_trace_paths(
    paths: Iterable[str | Path],
    *,
    val_fraction: float = 0.15,
    test_fraction: float = 0.15,
    seed: int = 0,
) -> tuple[list[Path], list[Path], list[Path]]:
    """Split at the whole-trajectory/file level to prevent temporal leakage."""
    normalized = [Path(path) for path in paths]
    if not normalized:
        raise ValueError("at least one trace path is required")
    if not 0.0 <= val_fraction < 1.0 or not 0.0 <= test_fraction < 1.0:
        raise ValueError("validation/test fractions must lie in [0, 1)")
    if val_fraction + test_fraction >= 1.0:
        raise ValueError("validation and test fractions must leave a non-empty training fraction")
    rng = np.random.default_rng(int(seed))
    order = rng.permutation(len(normalized))
    shuffled = [normalized[int(i)] for i in order]
    n_val = int(round(len(shuffled) * float(val_fraction)))
    n_test = int(round(len(shuffled) * float(test_fraction)))
    if n_val + n_test >= len(shuffled):
        excess = n_val + n_test - (len(shuffled) - 1)
        while excess > 0 and (n_test > 0 or n_val > 0):
            if n_test >= n_val and n_test > 0:
                n_test -= 1
            elif n_val > 0:
                n_val -= 1
            excess -= 1
    val = shuffled[:n_val]
    test = shuffled[n_val : n_val + n_test]
    train = shuffled[n_val + n_test :]
    return train, val, test
