"""Audit whether 0.5 s displacement is observable without start velocity.

The current learned-motion TCN receives only world-frame gyro and
mass-normalized thrust. In a low-drag rigid-body model, two windows can have
nearly identical thrust/gyro histories but different initial translational
velocities, so their displacements differ by roughly v_start * dt.

This audit decomposes each GT displacement into:

    dp = v_start * dt + residual

and reports how much of the displacement magnitude/error budget remains after
the constant-velocity term. A large reduction, especially on Lissajous and
racing-like traces, is evidence that start velocity should be represented
explicitly rather than forcing the TCN to infer an unobservable quantity.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from estimation.learned_motion_dataset import load_trace_split_manifest


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("manifest", type=Path)
parser.add_argument("--window_time_s", type=float, default=0.5)
parser.add_argument("--stride_time_s", type=float, default=0.01)
parser.add_argument(
    "--split",
    choices=("train", "val", "test", "all"),
    default="test",
)
parser.add_argument("--output", type=Path, default=None)
args = parser.parse_args()


_REQUIRED = (
    "t_s",
    "truth_px", "truth_py", "truth_pz",
    "truth_vx", "truth_vy", "truth_vz",
)


def _read_trace(path: Path):
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        columns = set(reader.fieldnames or ())
        missing = [name for name in _REQUIRED if name not in columns]
        if missing:
            raise ValueError(
                f"trace {path} missing required columns: {', '.join(missing)}"
            )
        rows = list(reader)

    def col(name: str) -> np.ndarray:
        values = np.asarray([float(row[name]) for row in rows], dtype=np.float64)
        if not np.all(np.isfinite(values)):
            raise ValueError(f"trace {path} contains non-finite {name}")
        return values

    t = col("t_s")
    if np.any(np.diff(t) <= 0.0):
        raise ValueError(f"trace {path} timestamps must be strictly increasing")
    p = np.column_stack((col("truth_px"), col("truth_py"), col("truth_pz")))
    v = np.column_stack((col("truth_vx"), col("truth_vy"), col("truth_vz")))
    return t, p, v


def _windows(path: Path) -> dict:
    t, p, v = _read_trace(path)
    dt = float(args.window_time_s)
    stride = float(args.stride_time_s)
    if dt <= 0.0 or stride <= 0.0:
        raise ValueError("window_time_s and stride_time_s must be positive")

    first = float(t[0])
    last_start = float(t[-1] - dt)
    if last_start < first - 1.0e-9:
        raise ValueError(f"trace {path} shorter than requested window")
    count = int(np.floor((last_start - first) / stride + 1.0e-9)) + 1
    starts = first + np.arange(count, dtype=np.float64) * stride
    ends = starts + dt

    dp = np.empty((count, 3), dtype=np.float64)
    v0 = np.empty((count, 3), dtype=np.float64)
    for i, (start, end) in enumerate(zip(starts, ends)):
        p0 = np.array(
            [np.interp(start, t, p[:, axis]) for axis in range(3)],
            dtype=np.float64,
        )
        p1 = np.array(
            [np.interp(end, t, p[:, axis]) for axis in range(3)],
            dtype=np.float64,
        )
        v0[i] = np.array(
            [np.interp(start, t, v[:, axis]) for axis in range(3)],
            dtype=np.float64,
        )
        dp[i] = p1 - p0

    constant_velocity = v0 * dt
    residual = dp - constant_velocity

    dp_norm2 = np.sum(dp * dp, axis=1)
    residual_norm2 = np.sum(residual * residual, axis=1)
    baseline_error_norm = np.sqrt(residual_norm2)

    return {
        "path": str(path),
        "samples": int(count),
        "dp_norm_rms_m": float(np.sqrt(np.mean(dp_norm2))),
        "constant_velocity_baseline_rmse_m": float(
            np.sqrt(np.mean(residual_norm2))
        ),
        "residual_to_displacement_rms_ratio": float(
            np.sqrt(np.mean(residual_norm2))
            / max(np.sqrt(np.mean(dp_norm2)), 1.0e-12)
        ),
        "axis_dp_rmse_m": np.sqrt(np.mean(dp * dp, axis=0)).tolist(),
        "axis_residual_rmse_m": np.sqrt(
            np.mean(residual * residual, axis=0)
        ).tolist(),
        "axis_v0_std_mps": np.std(v0, axis=0).tolist(),
        "v0_speed_mean_mps": float(np.mean(np.linalg.norm(v0, axis=1))),
        "v0_speed_p95_mps": float(
            np.percentile(np.linalg.norm(v0, axis=1), 95)
        ),
        "baseline_error_p95_m": float(np.percentile(baseline_error_norm, 95)),
    }


def _aggregate(items: list[dict]) -> dict:
    if not items:
        return {}
    total = sum(item["samples"] for item in items)
    dp_axis_mse = np.zeros(3, dtype=np.float64)
    residual_axis_mse = np.zeros(3, dtype=np.float64)
    for item in items:
        w = item["samples"] / total
        dp_axis_mse += w * np.asarray(item["axis_dp_rmse_m"]) ** 2
        residual_axis_mse += (
            w * np.asarray(item["axis_residual_rmse_m"]) ** 2
        )
    dp_norm_rms = float(np.sqrt(np.sum(dp_axis_mse)))
    residual_norm_rms = float(np.sqrt(np.sum(residual_axis_mse)))
    return {
        "traces": len(items),
        "samples": int(total),
        "dp_norm_rms_m": dp_norm_rms,
        "constant_velocity_baseline_rmse_m": residual_norm_rms,
        "residual_to_displacement_rms_ratio": (
            residual_norm_rms / max(dp_norm_rms, 1.0e-12)
        ),
        "axis_dp_rmse_m": np.sqrt(dp_axis_mse).tolist(),
        "axis_residual_rmse_m": np.sqrt(residual_axis_mse).tolist(),
    }


def main() -> None:
    train, val, test = load_trace_split_manifest(args.manifest)
    split_paths = {"train": train, "val": val, "test": test}
    selected = (
        ("train", "val", "test")
        if args.split == "all"
        else (args.split,)
    )

    report = {
        "schema": "isaac_drone_racer.learned_motion_observability_audit.v1",
        "manifest": str(args.manifest.expanduser().resolve()),
        "window_time_s": float(args.window_time_s),
        "stride_time_s": float(args.stride_time_s),
        "splits": {},
    }

    for split in selected:
        items = [_windows(path) for path in split_paths[split]]
        aggregate = _aggregate(items)
        report["splits"][split] = {
            "aggregate": aggregate,
            "traces": items,
        }
        print(
            f"[observability] {split} "
            f"dp_rms={aggregate['dp_norm_rms_m']:.4f}m "
            f"v0_baseline_rmse={aggregate['constant_velocity_baseline_rmse_m']:.4f}m "
            f"ratio={aggregate['residual_to_displacement_rms_ratio']:.3f}",
            flush=True,
        )
        for item in items:
            print(
                f"  {Path(item['path']).name:42s} "
                f"dp_rms={item['dp_norm_rms_m']:.4f}m "
                f"v0_baseline_rmse={item['constant_velocity_baseline_rmse_m']:.4f}m "
                f"ratio={item['residual_to_displacement_rms_ratio']:.3f} "
                f"v0_p95={item['v0_speed_p95_mps']:.3f}m/s",
                flush=True,
            )

    if args.output is not None:
        output = args.output.expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"[observability] wrote {output}", flush=True)


if __name__ == "__main__":
    main()
