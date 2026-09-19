"""Audit whether learned-motion bias is constant or feature-conditioned.

This tool is diagnostic only. It performs leave-one-trace-out cross-validation
within an offline manifest split and fits residual corrections using only
deployment-available quantities:
  - network predicted mean (3)
  - network predicted sigma (3)
  - 0.5 s TCN input-window channel mean/std/endpoint (6 each)

For every held-out trace it compares:
  1) raw network prediction
  2) constant residual-bias correction estimated from the other traces
  3) ridge affine residual correction estimated from the other traces

No evaluation replay is used to fit the correction.
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
        choices=("train", "val", "test"),
        default="val",
    )
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--ridge-alpha",
        type=float,
        default=1.0e-2,
        help="Ridge coefficient after train-fold feature standardization.",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _predict_trace(torch, model, path: Path, metadata: dict, device, batch_size: int):
    windows = load_trace_windows(
        path,
        window_time_s=float(metadata.get("window_time_s", 0.5)),
        sample_rate_hz=float(metadata.get("sample_rate_hz", 100.0)),
        stride_time_s=float(metadata.get("stride_time_s", 0.01)),
        target_mode=str(metadata.get("target_mode", "displacement")),
    )

    x_np = windows.features.astype(np.float64)
    y_np = windows.targets.astype(np.float64)
    x = torch.from_numpy(windows.features).float()
    loader = torch.utils.data.DataLoader(
        x,
        batch_size=int(batch_size),
        shuffle=False,
    )

    outputs = []
    model.eval()
    with torch.no_grad():
        for features in loader:
            output = model(features.to(device))
            outputs.append(output.detach().cpu().numpy())
    output_np = np.concatenate(outputs, axis=0).astype(np.float64)

    prediction = output_np[:, :3]
    log_var = np.clip(output_np[:, 3:], -12.0, 6.0)
    sigma = np.sqrt(np.exp(log_var) + 1.0e-6)
    residual = prediction - y_np

    feature_mean = np.mean(x_np, axis=2)
    feature_std = np.std(x_np, axis=2)
    feature_end = x_np[:, :, -1]
    conditioning = np.concatenate(
        (
            prediction,
            sigma,
            feature_mean,
            feature_std,
            feature_end,
        ),
        axis=1,
    )

    return {
        "path": str(path),
        "conditioning": conditioning,
        "residual": residual,
        "prediction": prediction,
        "target": y_np,
        "samples": int(len(residual)),
    }


def _fit_affine_ridge(
    x_train: np.ndarray,
    residual_train: np.ndarray,
    alpha: float,
):
    x_train = np.asarray(x_train, dtype=np.float64)
    residual_train = np.asarray(residual_train, dtype=np.float64)

    mean = np.mean(x_train, axis=0)
    scale = np.std(x_train, axis=0)
    scale = np.where(scale < 1.0e-9, 1.0, scale)
    x_std = (x_train - mean) / scale
    design = np.concatenate(
        (np.ones((len(x_std), 1), dtype=np.float64), x_std),
        axis=1,
    )

    reg = np.eye(design.shape[1], dtype=np.float64) * float(alpha)
    reg[0, 0] = 0.0
    lhs = design.T @ design + reg
    rhs = design.T @ residual_train
    beta = np.linalg.solve(lhs, rhs)
    return mean, scale, beta


def _predict_affine(
    x: np.ndarray,
    mean: np.ndarray,
    scale: np.ndarray,
    beta: np.ndarray,
) -> np.ndarray:
    x_std = (np.asarray(x, dtype=np.float64) - mean) / scale
    design = np.concatenate(
        (np.ones((len(x_std), 1), dtype=np.float64), x_std),
        axis=1,
    )
    return design @ beta


def _metrics(errors: np.ndarray) -> dict:
    errors = np.asarray(errors, dtype=np.float64).reshape(-1, 3)
    norm = np.linalg.norm(errors, axis=1)
    return {
        "samples": int(len(errors)),
        "axis_bias_m": np.mean(errors, axis=0).tolist(),
        "axis_rmse_m": np.sqrt(np.mean(errors**2, axis=0)).tolist(),
        "norm_rmse_m": float(np.sqrt(np.mean(norm**2))),
        "norm_mean_m": float(np.mean(norm)),
        "norm_max_m": float(np.max(norm)),
    }


def _aggregate(items: list[dict], key: str) -> dict:
    errors = np.concatenate([item[key] for item in items], axis=0)
    return _metrics(errors)


def main() -> None:
    args = _parse_args()
    if args.ridge_alpha < 0.0:
        raise ValueError("--ridge-alpha must be non-negative")

    try:
        import torch
    except ImportError as exc:
        raise ImportError("PyTorch is required for this audit") from exc

    checkpoint_path = args.checkpoint.expanduser().resolve()
    manifest_path = args.manifest.expanduser().resolve()
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    metadata = dict(checkpoint.get("metadata", {}))

    requested_device = str(args.device)
    if requested_device.startswith("cuda") and not torch.cuda.is_available():
        requested_device = "cpu"
    device = torch.device(requested_device)

    model = build_tcn(
        input_dim=int(metadata.get("input_dim", 6)),
        output_dim=int(metadata.get("output_dim", 6)),
    ).to(device)
    model.load_state_dict(checkpoint.get("model_state_dict", checkpoint))
    model.eval()

    train_paths, val_paths, test_paths = load_trace_split_manifest(manifest_path)
    split_paths = {
        "train": list(train_paths),
        "val": list(val_paths),
        "test": list(test_paths),
    }
    paths = split_paths[str(args.split)]
    if len(paths) < 2:
        raise ValueError(
            f"split {args.split!r} needs at least two traces for leave-one-trace-out"
        )

    traces = [
        _predict_trace(
            torch,
            model,
            path,
            metadata,
            device,
            int(args.batch_size),
        )
        for path in paths
    ]

    fold_reports = []
    fold_arrays = []

    for held_index, held in enumerate(traces):
        train = [item for i, item in enumerate(traces) if i != held_index]
        x_train = np.concatenate([item["conditioning"] for item in train], axis=0)
        e_train = np.concatenate([item["residual"] for item in train], axis=0)

        constant_bias = np.mean(e_train, axis=0)
        mean, scale, beta = _fit_affine_ridge(
            x_train,
            e_train,
            float(args.ridge_alpha),
        )

        raw_error = held["residual"].copy()
        constant_error = raw_error - constant_bias
        affine_hat = _predict_affine(
            held["conditioning"],
            mean,
            scale,
            beta,
        )
        affine_error = raw_error - affine_hat

        fold_reports.append(
            {
                "held_out_trace": held["path"],
                "train_samples": int(len(e_train)),
                "held_out_samples": int(len(raw_error)),
                "constant_bias_fit_m": constant_bias.tolist(),
                "raw": _metrics(raw_error),
                "constant_debias": _metrics(constant_error),
                "affine_conditioned_debias": _metrics(affine_error),
            }
        )
        fold_arrays.append(
            {
                "raw": raw_error,
                "constant": constant_error,
                "affine": affine_error,
            }
        )

    aggregate_items = [
        {
            "raw": item["raw"],
            "constant": item["constant"],
            "affine": item["affine"],
        }
        for item in fold_arrays
    ]
    report = {
        "schema": "isaac_drone_racer.learned_motion_bias_conditioning_audit.v1",
        "checkpoint": str(checkpoint_path),
        "manifest": str(manifest_path),
        "split": str(args.split),
        "target_mode": str(metadata.get("target_mode", "displacement")),
        "ridge_alpha": float(args.ridge_alpha),
        "conditioning_features": [
            "prediction_xyz",
            "predicted_sigma_xyz",
            "window_channel_mean_6",
            "window_channel_std_6",
            "window_channel_endpoint_6",
        ],
        "folds": fold_reports,
        "aggregate": {
            "raw": _aggregate(aggregate_items, "raw"),
            "constant_debias": _aggregate(aggregate_items, "constant"),
            "affine_conditioned_debias": _aggregate(aggregate_items, "affine"),
        },
    }

    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    agg = report["aggregate"]
    print("[bias-conditioning] leave-one-trace-out")
    for name in ("raw", "constant_debias", "affine_conditioned_debias"):
        item = agg[name]
        print(
            f"  {name:28s} "
            f"norm_rmse={1000.0 * item['norm_rmse_m']:.4f} mm "
            f"bias_mm={np.round(1000.0 * np.asarray(item['axis_bias_m']), 4).tolist()}"
        )

    raw_rmse = float(agg["raw"]["norm_rmse_m"])
    for name in ("constant_debias", "affine_conditioned_debias"):
        rmse = float(agg[name]["norm_rmse_m"])
        improvement = 100.0 * (raw_rmse - rmse) / max(raw_rmse, 1.0e-12)
        print(f"  improvement {name:21s}: {improvement:+.2f}%")

    print("\n[bias-conditioning] held-out traces")
    for fold in fold_reports:
        raw = 1000.0 * float(fold["raw"]["norm_rmse_m"])
        const = 1000.0 * float(fold["constant_debias"]["norm_rmse_m"])
        affine = 1000.0 * float(fold["affine_conditioned_debias"]["norm_rmse_m"])
        print(
            f"  {Path(fold['held_out_trace']).name:42s} "
            f"raw={raw:7.3f} const={const:7.3f} affine={affine:7.3f} mm"
        )

    print(f"[bias-conditioning] wrote {output}")


if __name__ == "__main__":
    main()
