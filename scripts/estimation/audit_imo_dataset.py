"""Audit 0.5 s displacement coverage for an explicit IMO dataset manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from estimation.learned_motion_dataset import (
    load_trace_split_manifest,
    load_trace_windows,
)


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("manifest", type=Path)
parser.add_argument("--window_time_s", type=float, default=0.5)
parser.add_argument("--sample_rate_hz", type=float, default=100.0)
parser.add_argument("--stride_time_s", type=float, default=0.05)
parser.add_argument(
    "--target_mode",
    choices=(
        "displacement",
        "kinematic_residual",
        "kinematic_residual_body_end",
        "kinematic_residual_body_end_gyro_aligned",
        "kinematic_residual_body_end_gravity_compensated",
    ),
    default="displacement",
    help="Audit the same target representation that will be used for training.",
)
parser.add_argument("--output", type=Path, default=None)
parser.add_argument(
    "--heldout_support_ratio",
    type=float,
    default=1.25,
    help=(
        "Warn when a val/test axis p95 displacement exceeds this multiple of "
        "the training axis max displacement."
    ),
)
args = parser.parse_args()


def _metrics(path: Path) -> dict:
    windows = load_trace_windows(
        path,
        window_time_s=args.window_time_s,
        sample_rate_hz=args.sample_rate_hz,
        stride_time_s=args.stride_time_s,
        target_mode=args.target_mode,
    )
    target = windows.targets.astype(np.float64)
    norm = np.linalg.norm(target, axis=1)
    return {
        "path": str(path),
        "windows": int(len(target)),
        "axis_mean_m": np.mean(target, axis=0).tolist(),
        "axis_std_m": np.std(target, axis=0).tolist(),
        "axis_abs_p95_m": np.percentile(np.abs(target), 95, axis=0).tolist(),
        "axis_abs_max_m": np.max(np.abs(target), axis=0).tolist(),
        "norm_mean_m": float(np.mean(norm)),
        "norm_p95_m": float(np.percentile(norm, 95)),
        "norm_max_m": float(np.max(norm)),
    }


def _aggregate(items: list[dict], paths: list[Path]) -> dict:
    targets = []
    for path in paths:
        windows = load_trace_windows(
            path,
            window_time_s=args.window_time_s,
            sample_rate_hz=args.sample_rate_hz,
            stride_time_s=args.stride_time_s,
            target_mode=args.target_mode,
        )
        targets.append(windows.targets.astype(np.float64))
    if not targets:
        return {"traces": 0, "windows": 0}
    target = np.concatenate(targets, axis=0)
    norm = np.linalg.norm(target, axis=1)
    return {
        "traces": len(paths),
        "windows": int(len(target)),
        "axis_mean_m": np.mean(target, axis=0).tolist(),
        "axis_std_m": np.std(target, axis=0).tolist(),
        "axis_abs_p95_m": np.percentile(np.abs(target), 95, axis=0).tolist(),
        "axis_abs_max_m": np.max(np.abs(target), axis=0).tolist(),
        "norm_mean_m": float(np.mean(norm)),
        "norm_p95_m": float(np.percentile(norm, 95)),
        "norm_max_m": float(np.max(norm)),
    }


def main() -> None:
    train, val, test = load_trace_split_manifest(args.manifest)
    split_paths = {"train": train, "val": val, "test": test}

    missing = [
        str(path)
        for paths in split_paths.values()
        for path in paths
        if not path.exists()
    ]
    if missing:
        raise FileNotFoundError(
            "Dataset manifest is not fully collected. Missing traces:\n"
            + "\n".join(missing)
        )

    report = {
        "schema": "isaac_drone_racer.imo_dataset_audit.v1",
        "manifest": str(args.manifest.expanduser().resolve()),
        "window_time_s": float(args.window_time_s),
        "sample_rate_hz": float(args.sample_rate_hz),
        "stride_time_s": float(args.stride_time_s),
        "target_mode": str(args.target_mode),
        "splits": {},
    }

    for split, paths in split_paths.items():
        per_trace = [_metrics(path) for path in paths]
        report["splits"][split] = {
            "aggregate": _aggregate(per_trace, paths),
            "traces": per_trace,
        }

    for split in ("train", "val", "test"):
        agg = report["splits"][split]["aggregate"]
        print(
            f"[IMO audit] {split:5s} traces={agg['traces']:2d} "
            f"windows={agg['windows']:6d} "
            f"abs_p95_xyz={agg.get('axis_abs_p95_m')} "
            f"abs_max_xyz={agg.get('axis_abs_max_m')} "
            f"norm_p95={agg.get('norm_p95_m')}",
            flush=True,
        )

    train_axis_max = np.asarray(
        report["splits"]["train"]["aggregate"]["axis_abs_max_m"],
        dtype=np.float64,
    )
    support_ratio = float(args.heldout_support_ratio)
    if support_ratio <= 0.0 or not np.isfinite(support_ratio):
        raise ValueError("--heldout_support_ratio must be positive and finite")

    warnings = []
    for split in ("val", "test"):
        print(f"\n[IMO audit] {split} per-trace:")
        for item in report["splits"][split]["traces"]:
            p95 = np.asarray(item["axis_abs_p95_m"], dtype=np.float64)
            ratio = np.divide(
                p95,
                np.maximum(train_axis_max, 1.0e-12),
            )
            out_axes = [
                "XYZ"[axis]
                for axis, value in enumerate(ratio)
                if value > support_ratio
            ]
            status = ""
            if out_axes:
                warning = {
                    "split": split,
                    "path": item["path"],
                    "axes": out_axes,
                    "p95_to_train_max_ratio": ratio.tolist(),
                }
                warnings.append(warning)
                status = f"  OUT_OF_SUPPORT={','.join(out_axes)}"
            print(
                f"  {Path(item['path']).name:38s} "
                f"p95_xyz={np.round(item['axis_abs_p95_m'], 4).tolist()} "
                f"max_norm={item['norm_max_m']:.4f}{status}"
            )

    report["heldout_support_ratio"] = support_ratio
    report["support_warnings"] = warnings
    if warnings:
        print(
            f"\n[IMO audit] WARNING: {len(warnings)} held-out traces exceed "
            f"{support_ratio:.2f}x training-axis max support.",
            flush=True,
        )
    else:
        print(
            "\n[IMO audit] held-out p95 displacement is within configured "
            "training support on all axes.",
            flush=True,
        )

    if args.output is not None:
        output = args.output.expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"\n[IMO audit] wrote {output}", flush=True)


if __name__ == "__main__":
    main()
