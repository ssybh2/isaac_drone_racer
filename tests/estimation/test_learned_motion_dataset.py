import csv
import json
from pathlib import Path

import numpy as np
import pytest

from estimation.learned_motion_dataset import (
    load_trace_split_manifest,
    load_trace_windows,
    split_trace_paths,
)


FIELDS = [
    "t_s",
    "truth_px", "truth_py", "truth_pz",
    "truth_qw", "truth_qx", "truth_qy", "truth_qz",
    "imu_gx", "imu_gy", "imu_gz",
    "thrust_b_x", "thrust_b_y", "thrust_b_z",
]


def _write_trace(path: Path, duration_s: float = 1.0, yaw_90: bool = False) -> None:
    q = [np.sqrt(0.5), 0.0, 0.0, np.sqrt(0.5)] if yaw_90 else [1.0, 0.0, 0.0, 0.0]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        for t in np.arange(0.0, duration_s + 1.0e-9, 0.01):
            writer.writerow(
                {
                    "t_s": t,
                    "truth_px": 2.0 * t,
                    "truth_py": 0.0,
                    "truth_pz": 1.0,
                    "truth_qw": q[0],
                    "truth_qx": q[1],
                    "truth_qy": q[2],
                    "truth_qz": q[3],
                    "imu_gx": 1.0,
                    "imu_gy": 0.0,
                    "imu_gz": 0.0,
                    "thrust_b_x": 0.0,
                    "thrust_b_y": 0.0,
                    "thrust_b_z": 6.0,
                }
            )


def test_trace_windows_build_world_frame_features_and_truth_displacement(tmp_path):
    trace = tmp_path / "trace.csv"
    _write_trace(trace, yaw_90=True)

    windows = load_trace_windows(
        trace,
        window_time_s=0.5,
        sample_rate_hz=100.0,
        stride_time_s=0.5,
    )

    assert windows.features.shape == (2, 6, 50)
    assert windows.targets.shape == (2, 3)
    np.testing.assert_allclose(windows.features[:, 0, :], 0.0, atol=1e-7)
    np.testing.assert_allclose(windows.features[:, 1, :], 1.0, atol=1e-7)
    np.testing.assert_allclose(windows.features[:, 5, :], 6.0, atol=1e-7)
    np.testing.assert_allclose(windows.targets, [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]], atol=1e-8)


def test_missing_thrust_columns_are_rejected(tmp_path):
    trace = tmp_path / "bad.csv"
    with trace.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["t_s", "truth_px"])
        writer.writeheader()
        writer.writerow({"t_s": 0.0, "truth_px": 0.0})
    with pytest.raises(ValueError, match="missing required columns"):
        load_trace_windows(trace)


def test_trace_split_is_by_whole_file_and_is_deterministic():
    paths = [Path(f"trace_{i}.csv") for i in range(10)]
    a = split_trace_paths(paths, val_fraction=0.2, test_fraction=0.2, seed=7)
    b = split_trace_paths(paths, val_fraction=0.2, test_fraction=0.2, seed=7)
    assert a == b
    train, val, test = a
    assert len(train) == 6 and len(val) == 2 and len(test) == 2
    assert set(train).isdisjoint(val)
    assert set(train).isdisjoint(test)
    assert set(val).isdisjoint(test)



def test_explicit_manifest_preserves_declared_whole_trace_splits(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    manifest_dir = tmp_path / "config"
    manifest_dir.mkdir()

    manifest = {
        "schema": "isaac_drone_racer.imo_dataset_manifest.v2",
        "path_base": "..",
        "traces": [
            {"path": "data/train_a.csv", "split": "train"},
            {"path": "data/train_b.csv", "split": "train"},
            {"path": "data/val_a.csv", "split": "val"},
            {"path": "data/test_a.csv", "split": "test"},
        ],
    }
    path = manifest_dir / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")

    train, val, test = load_trace_split_manifest(path)

    assert train == [
        (data_dir / "train_a.csv").resolve(),
        (data_dir / "train_b.csv").resolve(),
    ]
    assert val == [(data_dir / "val_a.csv").resolve()]
    assert test == [(data_dir / "test_a.csv").resolve()]


def test_explicit_manifest_rejects_duplicate_paths(tmp_path):
    manifest = {
        "schema": "isaac_drone_racer.imo_dataset_manifest.v2",
        "traces": [
            {"path": "same.csv", "split": "train"},
            {"path": "same.csv", "split": "test"},
        ],
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate trace path"):
        load_trace_split_manifest(path)
