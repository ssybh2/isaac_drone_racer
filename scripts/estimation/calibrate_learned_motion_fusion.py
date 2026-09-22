"""Calibrate learned delta-velocity fusion on the validation split only.

The learned-motion network predicts endpoint-body gravity-compensated delta
velocity. This script evaluates the checkpoint at the *runtime fusion cadence*
(default 2 Hz for a 0.5 s window) and writes a small calibration artifact with
the mean prediction residual E[prediction - truth]. Runtime subtracts that bias
before converting the measurement to the configured fusion frame.

No test traces are read or used.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from estimation.learned_motion import build_tcn
from estimation.learned_motion_dataset import load_trace_split_manifest, load_trace_windows


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--fusion-rate-hz", type=float, default=2.0)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.fusion_rate_hz <= 0.0:
        raise ValueError("--fusion-rate-hz must be positive")
    try:
        import torch
    except ImportError as exc:
        raise ImportError("PyTorch is required for learned-motion calibration") from exc

    checkpoint_path = args.checkpoint.expanduser().resolve()
    manifest_path = args.manifest.expanduser().resolve()
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    metadata = dict(checkpoint.get("metadata", {}))
    target_mode = str(metadata.get("target_mode", ""))
    if target_mode != "delta_velocity_body_end_gyro_aligned":
        raise ValueError(
            "fusion calibration currently requires "
            "target_mode='delta_velocity_body_end_gyro_aligned'"
        )

    _, val_paths, _ = load_trace_split_manifest(manifest_path)
    if not val_paths:
        raise ValueError("manifest has no validation traces")

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

    stride_s = 1.0 / float(args.fusion_rate_hz)
    residual_parts: list[np.ndarray] = []
    variance_parts: list[np.ndarray] = []

    with torch.no_grad():
        for path in val_paths:
            windows = load_trace_windows(
                path,
                window_time_s=float(metadata.get("window_time_s", 0.5)),
                sample_rate_hz=float(metadata.get("sample_rate_hz", 100.0)),
                stride_time_s=stride_s,
                target_mode=target_mode,
            )
            x = torch.from_numpy(windows.features).float()
            y = torch.from_numpy(windows.targets).float()
            loader = torch.utils.data.DataLoader(
                torch.utils.data.TensorDataset(x, y),
                batch_size=int(args.batch_size),
                shuffle=False,
            )
            for features, targets in loader:
                output = model(features.to(device))
                prediction = output[:, :3]
                log_var = torch.clamp(output[:, 3:], min=-12.0, max=6.0)
                variance = torch.exp(log_var) + 1.0e-6
                residual_parts.append(
                    (prediction - targets.to(device)).cpu().numpy()
                )
                variance_parts.append(variance.cpu().numpy())

    residual = np.concatenate(residual_parts, axis=0).astype(np.float64)
    variance = np.concatenate(variance_parts, axis=0).astype(np.float64)
    squared = residual**2
    sigma = np.sqrt(variance)
    nse_axis = np.mean(squared / variance, axis=0)
    report = {
        "schema": "isaac_drone_racer.learned_delta_velocity_fusion_calibration.v1",
        "checkpoint": str(checkpoint_path),
        "manifest": str(manifest_path),
        "split": "val",
        "target_mode": target_mode,
        "target_frame": "body_end",
        "target_unit": "m/s",
        "window_time_s": float(metadata.get("window_time_s", 0.5)),
        "sample_rate_hz": float(metadata.get("sample_rate_hz", 100.0)),
        "fusion_rate_hz": float(args.fusion_rate_hz),
        "fusion_stride_time_s": float(stride_s),
        "samples": int(residual.shape[0]),
        "bias_body_end_mps": np.mean(residual, axis=0).tolist(),
        "axis_rmse_mps": np.sqrt(np.mean(squared, axis=0)).tolist(),
        "norm_rmse_mps": float(np.sqrt(np.mean(np.sum(squared, axis=1)))),
        "predicted_sigma_mean_mps": np.mean(sigma, axis=0).tolist(),
        "nse_axis_mean": nse_axis.tolist(),
        "nse_norm_mean": float(np.sum(nse_axis)),
        "one_sigma_coverage_axis": np.mean(np.abs(residual) <= sigma, axis=0).tolist(),
        "two_sigma_coverage_axis": np.mean(np.abs(residual) <= 2.0 * sigma, axis=0).tolist(),
        "contract": (
            "Validation split only. bias_body_end_mps is E[prediction-truth] "
            "at the runtime fusion cadence and must be subtracted from the "
            "network prediction before fusion."
        ),
    }

    output = (
        args.output.expanduser().resolve()
        if args.output is not None
        else Path(str(checkpoint_path) + ".fusion_calibration.json")
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print(
        "[learned-motion-calibration] "
        f"samples={report['samples']} "
        f"bias={np.round(report['bias_body_end_mps'], 6).tolist()}m/s "
        f"norm_rmse={report['norm_rmse_mps']:.6f}m/s "
        f"NSE_norm={report['nse_norm_mean']:.3f}",
        flush=True,
    )
    print(f"[learned-motion-calibration] wrote {output}", flush=True)


if __name__ == "__main__":
    main()
