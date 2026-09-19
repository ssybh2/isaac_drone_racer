"""Train the hybrid VIO learned relative-displacement TCN from complete CSV traces."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from estimation.learned_motion import build_tcn
from estimation.learned_motion_dataset import (
    load_trace_split_manifest,
    load_trace_windows,
    split_trace_paths,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "traces",
        nargs="*",
        type=Path,
        help="Complete trajectory CSV traces. Omit when --manifest is used.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Explicit V2 train/val/test manifest. Mutually exclusive with positional traces.",
    )
    parser.add_argument("--output", type=Path, default=Path("artifacts/learned_motion/model.pt"))
    parser.add_argument("--window_time_s", type=float, default=0.5)
    parser.add_argument("--sample_rate_hz", type=float, default=100.0)
    parser.add_argument("--stride_time_s", type=float, default=0.01)
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
        help=(
            "Train direct displacement, world-frame residual displacement, "
            "or endpoint-body residual displacement after subtracting "
            "v_start * window_time_s."
        ),
    )
    parser.add_argument("--val_fraction", type=float, default=0.15)
    parser.add_argument("--test_fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1.0e-4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument(
        "--train_sampling",
        choices=("window_uniform", "trace_balanced"),
        default="window_uniform",
        help=(
            "window_uniform samples every window equally (legacy behavior). "
            "trace_balanced uses inverse per-trace window-count weights so every "
            "training trajectory contributes equal expected mass per epoch."
        ),
    )
    parser.add_argument(
        "--selection_metric",
        choices=("nll", "rmse"),
        default="nll",
        help="Metric used to select the best validation checkpoint.",
    )
    return parser.parse_args()


def _stack_paths(paths, args, *, return_counts: bool = False):
    feature_parts = []
    target_parts = []
    counts = []
    for path in paths:
        windows = load_trace_windows(
            path,
            window_time_s=args.window_time_s,
            sample_rate_hz=args.sample_rate_hz,
            stride_time_s=args.stride_time_s,
            target_mode=args.target_mode,
        )
        feature_parts.append(windows.features)
        target_parts.append(windows.targets)
        counts.append(int(windows.features.shape[0]))
    if not feature_parts:
        features = np.empty(
            (0, 6, int(round(args.window_time_s * args.sample_rate_hz))),
            np.float32,
        )
        targets = np.empty((0, 3), np.float32)
    else:
        features = np.concatenate(feature_parts, axis=0)
        targets = np.concatenate(target_parts, axis=0)
    if return_counts:
        return features, targets, counts
    return features, targets


def trace_balanced_sample_weights(trace_counts) -> np.ndarray:
    """Return per-window weights with equal total probability mass per trace."""
    counts = np.asarray(trace_counts, dtype=np.int64).reshape(-1)
    if counts.size == 0 or np.any(counts <= 0):
        raise ValueError("trace_counts must contain positive window counts")
    return np.concatenate(
        [
            np.full(int(count), 1.0 / float(count), dtype=np.float64)
            for count in counts
        ]
    )


def _gaussian_displacement_nll(torch, output, target):
    mean = output[:, :3]
    log_var = torch.clamp(output[:, 3:], min=-12.0, max=6.0)
    inv_var = torch.exp(-log_var)
    error2 = (target - mean).pow(2)
    loss = 0.5 * (log_var + error2 * inv_var)
    return loss.mean(), mean


def _evaluate(torch, model, loader, device):
    if loader is None:
        return None
    model.eval()
    losses = []
    residual_parts = []
    sigma_parts = []
    with torch.no_grad():
        for features, targets in loader:
            features = features.to(device)
            targets = targets.to(device)
            output = model(features)
            loss, mean = _gaussian_displacement_nll(torch, output, targets)
            log_var = torch.clamp(output[:, 3:], min=-12.0, max=6.0)
            sigma = torch.sqrt(torch.exp(log_var) + 1.0e-6)
            losses.append(float(loss.detach().cpu()))
            residual_parts.append((mean - targets).detach().cpu().numpy())
            sigma_parts.append(sigma.detach().cpu().numpy())
    if not losses:
        return None
    residual = np.concatenate(residual_parts, axis=0).astype(np.float64)
    sigma = np.concatenate(sigma_parts, axis=0).astype(np.float64)
    squared = residual**2
    norm_squared = np.sum(squared, axis=1)
    return {
        "nll": float(np.mean(losses)),
        # Historical scalar-coordinate RMSE, kept for selection compatibility.
        "rmse_m": float(np.sqrt(np.mean(squared))),
        "norm_rmse_m": float(np.sqrt(np.mean(norm_squared))),
        "axis_rmse_m": np.sqrt(np.mean(squared, axis=0)).tolist(),
        "axis_bias_m": np.mean(residual, axis=0).tolist(),
        "predicted_sigma_mean_m": np.mean(sigma, axis=0).tolist(),
        "samples": int(squared.shape[0]),
    }


def _evaluate_path(torch, model, path, args, device):
    features, targets = _stack_paths([path], args)
    if features.shape[0] == 0:
        return None
    dataset = torch.utils.data.TensorDataset(
        torch.from_numpy(features).float(),
        torch.from_numpy(targets).float(),
    )
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    return _evaluate(torch, model, loader, device)


def _per_trace_metrics(torch, model, paths, args, device):
    return {
        str(path): _evaluate_path(torch, model, path, args, device)
        for path in paths
    }


def main() -> None:
    args = _parse_args()
    if args.epochs < 1 or args.batch_size < 1:
        raise ValueError("epochs and batch_size must be positive")

    try:
        import torch
    except ImportError as exc:
        raise ImportError("PyTorch is required to train the learned motion model") from exc

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    if args.manifest is not None and args.traces:
        raise ValueError("Use either --manifest or positional traces, not both")
    if args.manifest is None and not args.traces:
        raise ValueError("Provide positional traces or --manifest")

    split_source = "random_whole_trace_split"
    manifest_path = None
    if args.manifest is not None:
        manifest_path = args.manifest.expanduser().resolve()
        train_paths, val_paths, test_paths = load_trace_split_manifest(manifest_path)
        split_source = "explicit_manifest"
    else:
        trace_paths = [path.expanduser().resolve() for path in args.traces]
        train_paths, val_paths, test_paths = split_trace_paths(
            trace_paths,
            val_fraction=args.val_fraction,
            test_fraction=args.test_fraction,
            seed=args.seed,
        )

    all_paths = list(train_paths) + list(val_paths) + list(test_paths)
    missing = [str(path) for path in all_paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing trace files: {missing}")
    train_x, train_y, train_trace_counts = _stack_paths(
        train_paths,
        args,
        return_counts=True,
    )
    val_x, val_y = _stack_paths(val_paths, args)
    test_x, test_y = _stack_paths(test_paths, args)
    if train_x.shape[0] == 0:
        raise ValueError("No training windows were produced")

    requested_device = str(args.device)
    if requested_device.startswith("cuda") and not torch.cuda.is_available():
        print("[learned-motion] CUDA requested but unavailable; falling back to CPU")
        requested_device = "cpu"
    device = torch.device(requested_device)

    def loader(
        features,
        targets,
        *,
        shuffle,
        trace_counts=None,
        sampling="window_uniform",
    ):
        if features.shape[0] == 0:
            return None
        dataset = torch.utils.data.TensorDataset(
            torch.from_numpy(features).float(),
            torch.from_numpy(targets).float(),
        )
        if sampling == "trace_balanced":
            if not trace_counts or sum(trace_counts) != len(dataset):
                raise ValueError(
                    "trace-balanced sampling requires per-trace counts matching "
                    "the concatenated training dataset"
                )
            sample_weights = trace_balanced_sample_weights(trace_counts)
            sampler = torch.utils.data.WeightedRandomSampler(
                weights=torch.from_numpy(sample_weights).double(),
                num_samples=len(dataset),
                replacement=True,
            )
            return torch.utils.data.DataLoader(
                dataset,
                batch_size=args.batch_size,
                sampler=sampler,
                shuffle=False,
                num_workers=args.num_workers,
            )
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=shuffle,
            num_workers=args.num_workers,
        )

    train_loader = loader(
        train_x,
        train_y,
        shuffle=True,
        trace_counts=train_trace_counts,
        sampling=args.train_sampling,
    )
    val_loader = loader(val_x, val_y, shuffle=False)
    test_loader = loader(test_x, test_y, shuffle=False)

    model = build_tcn(input_dim=6, output_dim=6).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "schema": "isaac_drone_racer.learned_motion_tcn.v1",
        "input_dim": 6,
        "output_dim": 6,
        "window_time_s": float(args.window_time_s),
        "sample_rate_hz": float(args.sample_rate_hz),
        "stride_time_s": float(args.stride_time_s),
        "target_mode": str(args.target_mode),
        "gravity_compensated_target": bool(
            args.target_mode == "kinematic_residual_body_end_gravity_compensated"
        ),
        "feature_frame": (
            "body_endpoint_gyro_aligned"
            if args.target_mode in (
                "kinematic_residual_body_end_gyro_aligned",
                "kinematic_residual_body_end_gravity_compensated",
            )
            else (
                "body"
                if args.target_mode == "kinematic_residual_body_end"
                else "world"
            )
        ),
        "features": (
            ["gyro_end_x", "gyro_end_y", "gyro_end_z", "thrust_end_x", "thrust_end_y", "thrust_end_z"]
            if args.target_mode in (
                "kinematic_residual_body_end_gyro_aligned",
                "kinematic_residual_body_end_gravity_compensated",
            )
            else (
                ["gyro_b_x", "gyro_b_y", "gyro_b_z", "thrust_b_x", "thrust_b_y", "thrust_b_z"]
                if args.target_mode in (
                    "kinematic_residual_body_end",
                    "kinematic_residual_body_end_gyro_aligned",
                    "kinematic_residual_body_end_gravity_compensated",
                )
                else ["gyro_w_x", "gyro_w_y", "gyro_w_z", "thrust_w_x", "thrust_w_y", "thrust_w_z"]
            )
        ),
        "outputs": (
            ["dp_w_x", "dp_w_y", "dp_w_z", "log_var_x", "log_var_y", "log_var_z"]
            if args.target_mode == "displacement"
            else (
                [
                    "dp_residual_b_end_x",
                    "dp_residual_b_end_y",
                    "dp_residual_b_end_z",
                    "log_var_x",
                    "log_var_y",
                    "log_var_z",
                ]
                if args.target_mode == "kinematic_residual_body_end"
                else [
                    "dp_residual_w_x",
                    "dp_residual_w_y",
                    "dp_residual_w_z",
                    "log_var_x",
                    "log_var_y",
                    "log_var_z",
                ]
            )
        ),
        "train_traces": [str(path) for path in train_paths],
        "val_traces": [str(path) for path in val_paths],
        "test_traces": [str(path) for path in test_paths],
        "split_source": split_source,
        "manifest": None if manifest_path is None else str(manifest_path),
        "selection_metric": str(args.selection_metric),
        "train_sampling": str(args.train_sampling),
        "train_trace_window_counts": {
            str(path): int(count)
            for path, count in zip(train_paths, train_trace_counts)
        },
        "seed": int(args.seed),
    }

    best_metric = float("inf")
    best_epoch = -1
    for epoch in range(args.epochs):
        model.train()
        epoch_losses = []
        for features, targets in train_loader:
            features = features.to(device)
            targets = targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            output = model(features)
            loss, _ = _gaussian_displacement_nll(torch, output, targets)
            loss.backward()
            optimizer.step()
            epoch_losses.append(float(loss.detach().cpu()))

        train_nll = float(np.mean(epoch_losses))
        val_metrics = _evaluate(torch, model, val_loader, device)
        if val_metrics is None:
            selection_value = train_nll
        elif args.selection_metric == "rmse":
            selection_value = val_metrics["rmse_m"]
        else:
            selection_value = val_metrics["nll"]
        if selection_value < best_metric:
            best_metric = selection_value
            best_epoch = epoch
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "metadata": metadata,
                    "epoch": int(epoch),
                    "selection_metric": str(args.selection_metric),
                    "selection_value": float(selection_value),
                },
                args.output,
            )
        if epoch == 0 or (epoch + 1) % 10 == 0 or epoch == args.epochs - 1:
            val_text = "n/a" if val_metrics is None else f"{val_metrics['nll']:.6f}"
            print(
                f"[learned-motion] epoch={epoch + 1}/{args.epochs} "
                f"train_nll={train_nll:.6f} val_nll={val_text}"
            )

    checkpoint = torch.load(args.output, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    val_metrics = _evaluate(torch, model, val_loader, device)
    test_metrics = _evaluate(torch, model, test_loader, device)
    report = {
        "checkpoint": str(args.output.resolve()),
        "best_epoch": int(best_epoch),
        "best_selection_metric": str(args.selection_metric),
        "best_selection_value": float(best_metric),
        "train_windows": int(train_x.shape[0]),
        "val_windows": int(val_x.shape[0]),
        "test_windows": int(test_x.shape[0]),
        "validation": val_metrics,
        "test": test_metrics,
        "train_by_trace": _per_trace_metrics(
            torch, model, train_paths, args, device
        ),
        "validation_by_trace": _per_trace_metrics(
            torch, model, val_paths, args, device
        ),
        "test_by_trace": _per_trace_metrics(
            torch, model, test_paths, args, device
        ),
        "metadata": metadata,
    }
    report_path = args.output.with_suffix(args.output.suffix + ".json")
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"[learned-motion] saved checkpoint: {args.output}")
    print(f"[learned-motion] saved report: {report_path}")


if __name__ == "__main__":
    main()
