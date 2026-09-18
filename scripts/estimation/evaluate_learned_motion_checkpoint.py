"""Evaluate a learned-motion checkpoint including covariance calibration.

This is intentionally offline: it loads the explicit V2 train/val/test manifest
and reports displacement-mean accuracy plus uncertainty calibration. The EKF
uses the network covariance as a measurement covariance, so a checkpoint with
acceptable RMSE but badly overconfident log-variance is not safe to fuse.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from estimation.learned_motion import build_tcn
from estimation.learned_motion_dataset import (
    load_trace_split_manifest,
    load_trace_windows,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--split",
        choices=("train", "val", "test", "all"),
        default="test",
    )
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def _trace_metrics(torch, model, path: Path, metadata: dict, device, batch_size: int) -> dict:
    windows = load_trace_windows(
        path,
        window_time_s=float(metadata.get("window_time_s", 0.5)),
        sample_rate_hz=float(metadata.get("sample_rate_hz", 100.0)),
        stride_time_s=float(metadata.get("stride_time_s", 0.01)),
    )
    x = torch.from_numpy(windows.features).float()
    y = torch.from_numpy(windows.targets).float()
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(x, y),
        batch_size=batch_size,
        shuffle=False,
    )

    residual_parts = []
    variance_parts = []
    target_parts = []
    prediction_parts = []
    nll_parts = []

    model.eval()
    with torch.no_grad():
        for features, targets in loader:
            features = features.to(device)
            targets = targets.to(device)
            output = model(features)
            prediction = output[:, :3]
            log_var = torch.clamp(output[:, 3:], min=-12.0, max=6.0)
            variance = torch.exp(log_var) + 1.0e-6
            residual = prediction - targets
            nll = 0.5 * (log_var + residual.pow(2) / variance)

            residual_parts.append(residual.cpu().numpy())
            variance_parts.append(variance.cpu().numpy())
            target_parts.append(targets.cpu().numpy())
            prediction_parts.append(prediction.cpu().numpy())
            nll_parts.append(nll.cpu().numpy())

    residual = np.concatenate(residual_parts, axis=0).astype(np.float64)
    variance = np.concatenate(variance_parts, axis=0).astype(np.float64)
    target = np.concatenate(target_parts, axis=0).astype(np.float64)
    prediction = np.concatenate(prediction_parts, axis=0).astype(np.float64)
    nll = np.concatenate(nll_parts, axis=0).astype(np.float64)

    sigma = np.sqrt(variance)
    squared = residual**2
    norm_squared = np.sum(squared, axis=1)
    normalized_squared = squared / variance
    normalized_squared_norm = np.sum(normalized_squared, axis=1)

    return {
        "path": str(path),
        "samples": int(len(residual)),
        # Historical training metric: RMS over all scalar XYZ coordinates.
        "coordinate_rmse_m": float(np.sqrt(np.mean(squared))),
        # Directly comparable to estimator diagnostics: RMS Euclidean dp error.
        "norm_rmse_m": float(np.sqrt(np.mean(norm_squared))),
        "axis_rmse_m": np.sqrt(np.mean(squared, axis=0)).tolist(),
        "axis_bias_m": np.mean(residual, axis=0).tolist(),
        "target_axis_std_m": np.std(target, axis=0).tolist(),
        "prediction_axis_std_m": np.std(prediction, axis=0).tolist(),
        "nll": float(np.mean(nll)),
        "predicted_sigma_mean_m": np.mean(sigma, axis=0).tolist(),
        "predicted_sigma_median_m": np.median(sigma, axis=0).tolist(),
        "normalized_squared_error_axis_mean": np.mean(
            normalized_squared, axis=0
        ).tolist(),
        "normalized_squared_error_norm_mean": float(
            np.mean(normalized_squared_norm)
        ),
        "one_sigma_coverage_axis": np.mean(
            np.abs(residual) <= sigma, axis=0
        ).tolist(),
        "two_sigma_coverage_axis": np.mean(
            np.abs(residual) <= 2.0 * sigma, axis=0
        ).tolist(),
    }


def _aggregate(items: list[dict]) -> dict:
    if not items:
        return {}
    total = sum(item["samples"] for item in items)
    # Weight per-trace scalar diagnostics by sample count. Axis RMSE is
    # recomputed from weighted MSE, not averaged as RMSE.
    axis_mse = np.zeros(3, dtype=np.float64)
    axis_bias = np.zeros(3, dtype=np.float64)
    nse_axis = np.zeros(3, dtype=np.float64)
    sigma_mean = np.zeros(3, dtype=np.float64)
    one_sigma = np.zeros(3, dtype=np.float64)
    two_sigma = np.zeros(3, dtype=np.float64)
    nll = 0.0

    for item in items:
        w = item["samples"] / total
        axis_rmse = np.asarray(item["axis_rmse_m"], dtype=np.float64)
        axis_mse += w * axis_rmse**2
        axis_bias += w * np.asarray(item["axis_bias_m"], dtype=np.float64)
        nse_axis += w * np.asarray(
            item["normalized_squared_error_axis_mean"], dtype=np.float64
        )
        sigma_mean += w * np.asarray(
            item["predicted_sigma_mean_m"], dtype=np.float64
        )
        one_sigma += w * np.asarray(
            item["one_sigma_coverage_axis"], dtype=np.float64
        )
        two_sigma += w * np.asarray(
            item["two_sigma_coverage_axis"], dtype=np.float64
        )
        nll += w * float(item["nll"])

    axis_rmse = np.sqrt(axis_mse)
    return {
        "traces": len(items),
        "samples": int(total),
        "coordinate_rmse_m": float(np.sqrt(np.mean(axis_mse))),
        "norm_rmse_m": float(np.sqrt(np.sum(axis_mse))),
        "axis_rmse_m": axis_rmse.tolist(),
        "axis_bias_m": axis_bias.tolist(),
        "nll": float(nll),
        "predicted_sigma_mean_m": sigma_mean.tolist(),
        "normalized_squared_error_axis_mean": nse_axis.tolist(),
        "normalized_squared_error_norm_mean": float(np.sum(nse_axis)),
        "one_sigma_coverage_axis": one_sigma.tolist(),
        "two_sigma_coverage_axis": two_sigma.tolist(),
        "calibration_reference": {
            "normalized_squared_error_norm_mean_expected": 3.0,
            "one_sigma_coverage_expected": 0.6827,
            "two_sigma_coverage_expected": 0.9545,
        },
    }


def main() -> None:
    args = _parse_args()
    try:
        import torch
    except ImportError as exc:
        raise ImportError("PyTorch is required to evaluate the checkpoint") from exc

    checkpoint_path = args.checkpoint.expanduser().resolve()
    manifest_path = args.manifest.expanduser().resolve()
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    metadata = dict(checkpoint.get("metadata", {}))
    input_dim = int(metadata.get("input_dim", 6))
    output_dim = int(metadata.get("output_dim", 6))
    if input_dim != 6 or output_dim != 6:
        raise ValueError(
            "current evaluator expects the 6-input/6-output displacement TCN"
        )

    requested_device = str(args.device)
    if requested_device.startswith("cuda") and not torch.cuda.is_available():
        requested_device = "cpu"
    device = torch.device(requested_device)

    model = build_tcn(input_dim=input_dim, output_dim=output_dim).to(device)
    model.load_state_dict(checkpoint.get("model_state_dict", checkpoint))
    model.eval()

    train, val, test = load_trace_split_manifest(manifest_path)
    split_paths = {"train": train, "val": val, "test": test}
    selected = (
        ("train", "val", "test")
        if args.split == "all"
        else (args.split,)
    )

    report = {
        "schema": "isaac_drone_racer.learned_motion_checkpoint_eval.v1",
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "checkpoint_selection_metric": checkpoint.get("selection_metric"),
        "checkpoint_selection_value": checkpoint.get("selection_value"),
        "manifest": str(manifest_path),
        "splits": {},
    }

    for split in selected:
        paths = split_paths[split]
        items = [
            _trace_metrics(
                torch,
                model,
                path,
                metadata,
                device,
                int(args.batch_size),
            )
            for path in paths
        ]
        report["splits"][split] = {
            "aggregate": _aggregate(items),
            "traces": items,
        }

        agg = report["splits"][split]["aggregate"]
        print(
            f"[learned-motion-eval] {split} "
            f"coord_rmse={agg['coordinate_rmse_m']:.4f}m "
            f"norm_rmse={agg['norm_rmse_m']:.4f}m "
            f"axis_rmse={np.round(agg['axis_rmse_m'], 4).tolist()} "
            f"nll={agg['nll']:.4f}",
            flush=True,
        )
        print(
            f"[learned-motion-eval] {split} "
            f"NSE_norm={agg['normalized_squared_error_norm_mean']:.3f} "
            f"(ideal~3) "
            f"1sigma={np.round(agg['one_sigma_coverage_axis'], 3).tolist()} "
            f"2sigma={np.round(agg['two_sigma_coverage_axis'], 3).tolist()}",
            flush=True,
        )

        for item in items:
            print(
                f"  {Path(item['path']).name:42s} "
                f"norm_rmse={item['norm_rmse_m']:.4f}m "
                f"bias={np.round(item['axis_bias_m'], 4).tolist()} "
                f"NSE={item['normalized_squared_error_norm_mean']:.2f}",
                flush=True,
            )

    output = (
        args.output.expanduser().resolve()
        if args.output is not None
        else checkpoint_path.with_suffix(checkpoint_path.suffix + ".eval.json")
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"[learned-motion-eval] wrote {output}", flush=True)


if __name__ == "__main__":
    main()
