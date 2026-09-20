"""Dataset utilities for supervised learned-motion training from diagnostic CSV traces."""

from __future__ import annotations

import csv
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Iterable

import numpy as np

from .learned_motion import endpoint_body_gyro_aligned_features


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
    target_mode: str = "displacement"


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


def _read_trace(
    path: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
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

    velocity_columns = ("truth_vx", "truth_vy", "truth_vz")
    velocities = None
    if all(name in columns for name in velocity_columns):
        velocities = np.column_stack(tuple(col(name) for name in velocity_columns))

    features_b = np.column_stack((gyro_b, thrust_b))
    features_w = np.empty((len(rows), 6), dtype=np.float64)
    for index, (q, gyro, thrust) in enumerate(zip(quaternions, gyro_b, thrust_b)):
        R_wb = _quat_wxyz_to_rotmat(q)
        features_w[index, 0:3] = R_wb @ gyro
        features_w[index, 3:6] = R_wb @ thrust
    return timestamps, positions, quaternions, features_w, features_b, velocities


def load_trace_windows(
    path: str | Path,
    *,
    window_time_s: float = 0.5,
    sample_rate_hz: float = 100.0,
    stride_time_s: float = 0.01,
    target_mode: str = "displacement",
) -> TraceWindows:
    """Convert one complete trace into fixed-rate learned-motion windows.

    target_mode="displacement" produces dp = p1 - p0.

    target_mode="kinematic_residual" produces
        dp_residual_w = (p1 - p0) - v0 * window_time_s
    in world coordinates.

    target_mode="kinematic_residual_body_end" uses body-frame gyro/thrust
    features and produces
        dp_residual_b1 = R_wb(t1)^T * dp_residual_w.
    This removes estimator attitude from the network input contract while
    making the output-frame dependence explicit in the EKF measurement model.

    target_mode="kinematic_residual_body_end_gyro_aligned" uses the same
    endpoint-body residual target, but first uses gyro-only relative attitude
    integration to rotate every body gyro/thrust sample into the window-end
    body frame. This keeps a common feature frame without any global attitude.

    target_mode="kinematic_residual_body_end_gravity_compensated" additionally
    subtracts the known 0.5*g*dt^2 term before expressing the target in the
    endpoint body frame. This removes absolute gravity direction from what the
    network must infer from gyro+thrust alone.

    target_mode="displacement_body_end_gyro_aligned" produces the full
        dp_b1 = R_wb(t1)^T * (p1 - p0)
    displacement while using the same gyro-only endpoint-body aligned features
    as V6.1. Unlike the kinematic-residual targets, it does not subtract
    v_start*dt or gravity motion.

    target_mode="delta_velocity_gravity_compensated" produces
        dv_specific_w = (v1 - v0) - g * window_time_s
    in world coordinates.

    target_mode="delta_velocity_body_end_gyro_aligned" produces
        dv_specific_b1 = R_wb(t1)^T * [(v1-v0) - g*window_time_s]
    and uses gyro-only endpoint-body-aligned body gyro/thrust features. It is
    independent of global yaw and does not require GT/EKF global attitude as a
    network input.
    """
    path = Path(path)
    if window_time_s <= 0.0 or sample_rate_hz <= 0.0 or stride_time_s <= 0.0:
        raise ValueError("window_time_s, sample_rate_hz and stride_time_s must be positive")
    sample_count = int(round(float(window_time_s) * float(sample_rate_hz)))
    if sample_count < 2:
        raise ValueError("learned motion window must contain at least two samples")

    valid_target_modes = (
        "displacement",
        "kinematic_residual",
        "kinematic_residual_body_end",
        "kinematic_residual_body_end_gyro_aligned",
        "kinematic_residual_body_end_gravity_compensated",
        "displacement_body_end_gyro_aligned",
        "delta_velocity_gravity_compensated",
        "delta_velocity_body_end_gyro_aligned",
    )
    if target_mode not in valid_target_modes:
        raise ValueError(
            "target_mode must be one of " + ", ".join(repr(v) for v in valid_target_modes)
        )
    (
        timestamps,
        positions,
        quaternions,
        features_w,
        features_b,
        velocities,
    ) = _read_trace(path)
    source_features = (
        features_b
        if target_mode in (
            "kinematic_residual_body_end",
            "kinematic_residual_body_end_gyro_aligned",
            "kinematic_residual_body_end_gravity_compensated",
            "displacement_body_end_gyro_aligned",
            "delta_velocity_body_end_gyro_aligned",
        )
        else features_w
    )
    if target_mode in (
        "kinematic_residual",
        "kinematic_residual_body_end",
        "kinematic_residual_body_end_gyro_aligned",
        "kinematic_residual_body_end_gravity_compensated",
        "delta_velocity_gravity_compensated",
        "delta_velocity_body_end_gyro_aligned",
    ) and velocities is None:
        raise ValueError(
            f"trace {path} needs truth_vx/truth_vy/truth_vz for "
            "velocity-dependent targets"
        )
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
    rotation_samples = (
        np.stack([_quat_wxyz_to_rotmat(q) for q in quaternions], axis=0)
        if target_mode in (
            "kinematic_residual_body_end",
            "kinematic_residual_body_end_gyro_aligned",
            "kinematic_residual_body_end_gravity_compensated",
            "displacement_body_end_gyro_aligned",
            "delta_velocity_body_end_gyro_aligned",
        )
        else None
    )
    for window_index, (start, end) in enumerate(zip(starts, ends)):
        sample_times = start + sample_offsets
        for channel in range(6):
            features[window_index, channel, :] = np.interp(
                sample_times, timestamps, source_features[:, channel]
            ).astype(np.float32)
        if target_mode in (
            "kinematic_residual_body_end_gyro_aligned",
            "kinematic_residual_body_end_gravity_compensated",
            "displacement_body_end_gyro_aligned",
            "delta_velocity_body_end_gyro_aligned",
        ):
            features[window_index, :, :] = endpoint_body_gyro_aligned_features(
                features[window_index, :, :],
                sample_times,
                end_timestamp_s=end,
            )
        p_start = np.array(
            [np.interp(start, timestamps, positions[:, axis]) for axis in range(3)],
            dtype=np.float64,
        )
        p_end = np.array(
            [np.interp(end, timestamps, positions[:, axis]) for axis in range(3)],
            dtype=np.float64,
        )
        target = p_end - p_start
        if target_mode in (
            "kinematic_residual",
            "kinematic_residual_body_end",
            "kinematic_residual_body_end_gyro_aligned",
            "kinematic_residual_body_end_gravity_compensated",
        ):
            v_start = np.array(
                [
                    np.interp(start, timestamps, velocities[:, axis])
                    for axis in range(3)
                ],
                dtype=np.float64,
            )
            target = target - v_start * float(window_time_s)

        if target_mode in (
            "delta_velocity_gravity_compensated",
            "delta_velocity_body_end_gyro_aligned",
        ):
            v_start = np.array(
                [
                    np.interp(start, timestamps, velocities[:, axis])
                    for axis in range(3)
                ],
                dtype=np.float64,
            )
            v_end = np.array(
                [
                    np.interp(end, timestamps, velocities[:, axis])
                    for axis in range(3)
                ],
                dtype=np.float64,
            )
            gravity_w = np.array([0.0, 0.0, -9.81], dtype=np.float64)
            target = v_end - v_start - gravity_w * float(window_time_s)

        if target_mode == "kinematic_residual_body_end_gravity_compensated":
            gravity_w = np.array([0.0, 0.0, -9.81], dtype=np.float64)
            target = target - 0.5 * gravity_w * float(window_time_s) ** 2

        if target_mode in (
            "kinematic_residual_body_end",
            "kinematic_residual_body_end_gyro_aligned",
            "kinematic_residual_body_end_gravity_compensated",
            "displacement_body_end_gyro_aligned",
            "delta_velocity_body_end_gyro_aligned",
        ):
            # Interpolate the endpoint attitude and project the matrix back to
            # SO(3). The target is expressed in the endpoint body frame.
            R_interp = np.empty((3, 3), dtype=np.float64)
            for row in range(3):
                for col in range(3):
                    R_interp[row, col] = np.interp(
                        end,
                        timestamps,
                        rotation_samples[:, row, col],
                    )
            U, _, Vt = np.linalg.svd(R_interp)
            R_end = U @ Vt
            if np.linalg.det(R_end) < 0.0:
                U[:, -1] *= -1.0
                R_end = U @ Vt
            target = R_end.T @ target

        targets[window_index, :] = target.astype(np.float32)

    return TraceWindows(
        features=features,
        targets=targets,
        start_timestamps_s=starts,
        end_timestamps_s=ends,
        source_path=path,
        target_mode=target_mode,
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


def load_trace_split_manifest(
    manifest_path: str | Path,
) -> tuple[list[Path], list[Path], list[Path]]:
    """Load explicit train/val/test trace paths from a dataset manifest.

    Manifest paths are resolved relative to manifest.parent / path_base.
    This keeps experiment splits deterministic and prevents random splitting
    from placing an entire motion family only in the training set.
    """
    manifest_path = Path(manifest_path).expanduser().resolve()
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    if data.get("schema") != "isaac_drone_racer.imo_dataset_manifest.v2":
        raise ValueError(
            "unsupported IMO dataset manifest schema: "
            f"{data.get('schema')!r}"
        )

    path_base = Path(data.get("path_base", "."))
    if path_base.is_absolute():
        base_dir = path_base
    else:
        base_dir = (manifest_path.parent / path_base).resolve()

    traces = data.get("traces")
    if not isinstance(traces, list) or not traces:
        raise ValueError("manifest must contain a non-empty traces list")

    splits = {"train": [], "val": [], "test": []}
    seen_paths: set[Path] = set()

    for index, entry in enumerate(traces):
        if not isinstance(entry, dict):
            raise ValueError(f"manifest trace entry {index} must be an object")
        split = entry.get("split")
        if split not in splits:
            raise ValueError(
                f"manifest trace entry {index} has invalid split {split!r}"
            )
        raw_path = entry.get("path")
        if not raw_path:
            raise ValueError(f"manifest trace entry {index} is missing path")
        path = Path(raw_path)
        if not path.is_absolute():
            path = (base_dir / path).resolve()
        else:
            path = path.resolve()
        if path in seen_paths:
            raise ValueError(f"manifest contains duplicate trace path: {path}")
        seen_paths.add(path)
        splits[split].append(path)

    if not splits["train"]:
        raise ValueError("manifest must contain at least one training trace")
    return splits["train"], splits["val"], splits["test"]
