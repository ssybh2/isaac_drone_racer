"""Train the hybrid VIO learned relative-displacement TCN from complete CSV traces."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from estimation.learned_motion import build_tcn
from estimation.learned_motion_dataset import load_trace_windows, split_trace_paths


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("traces", nargs="+", type=Path, help="Complete trajectory CSV traces.")
    parser.add_argument("--output", type=Path, default=Path("artifacts/learned_motion/model.pt"))
    parser.add_argument("--window_time_s", type=float, default=0.5)
    parser.add_argument("--sample_rate_hz", type=float, default=100.0)
    parser.add_argument("--stride_time_s", type=float, default=0.01)
    parser.add_argument("--val_fraction", type=float, default=0.15)
    parser.add_argument("--test_fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1.0e-4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num_workers", type=int, default=0)
    return parser.parse_args()


def _stack_paths(paths, args):
    feature_parts = []
    target_parts = []
    for path in paths:
        windows = load_trace_windows(
            path,
            window_time_s=args.window_time_s,
            sample_rate_hz=args.sample_rate_hz,
            stride_time_s=args.stride_time_s,
        )
        feature_parts.append(windows.features)
        target_parts.append(windows.targets)
    if not feature_parts:
        return np.empty((0, 6, int(round(args.window_time_s * args.sample_rate_hz))), np.float32), np.empty((0, 3), np.float32)
    return np.concatenate(feature_parts, axis=0), np.concatenate(target_parts, axis=0)


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
    squared_errors = []
    with torch.no_grad():
        for features, targets in loader:
            features = features.to(device)
            targets = targets.to(device)
            output = model(features)
            loss, mean = _gaussian_displacement_nll(torch, output, targets)
            losses.append(float(loss.detach().cpu()))
            squared_errors.append((mean - targets).pow(2).detach().cpu().numpy())
    if not losses:
        return None
    squared = np.concatenate(squared_errors, axis=0)
    return {
        "nll": float(np.mean(losses)),
        "rmse_m": float(np.sqrt(np.mean(squared))),
        "axis_rmse_m": np.sqrt(np.mean(squared, axis=0)).tolist(),
        "samples": int(squared.shape[0]),
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

    trace_paths = [path.expanduser().resolve() for path in args.traces]
    missing = [str(path) for path in trace_paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing trace files: {missing}")
    train_paths, val_paths, test_paths = split_trace_paths(
        trace_paths,
        val_fraction=args.val_fraction,
        test_fraction=args.test_fraction,
        seed=args.seed,
    )
    train_x, train_y = _stack_paths(train_paths, args)
    val_x, val_y = _stack_paths(val_paths, args)
    test_x, test_y = _stack_paths(test_paths, args)
    if train_x.shape[0] == 0:
        raise ValueError("No training windows were produced")

    requested_device = str(args.device)
    if requested_device.startswith("cuda") and not torch.cuda.is_available():
        print("[learned-motion] CUDA requested but unavailable; falling back to CPU")
        requested_device = "cpu"
    device = torch.device(requested_device)

    def loader(features, targets, *, shuffle):
        if features.shape[0] == 0:
            return None
        dataset = torch.utils.data.TensorDataset(
            torch.from_numpy(features).float(),
            torch.from_numpy(targets).float(),
        )
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=shuffle,
            num_workers=args.num_workers,
        )

    train_loader = loader(train_x, train_y, shuffle=True)
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
        "features": ["gyro_w_x", "gyro_w_y", "gyro_w_z", "thrust_w_x", "thrust_w_y", "thrust_w_z"],
        "outputs": ["dp_w_x", "dp_w_y", "dp_w_z", "log_var_x", "log_var_y", "log_var_z"],
        "train_traces": [str(path) for path in train_paths],
        "val_traces": [str(path) for path in val_paths],
        "test_traces": [str(path) for path in test_paths],
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
        selection_metric = train_nll if val_metrics is None else val_metrics["nll"]
        if selection_metric < best_metric:
            best_metric = selection_metric
            best_epoch = epoch
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "metadata": metadata,
                    "epoch": int(epoch),
                    "selection_nll": float(selection_metric),
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
        "best_selection_nll": float(best_metric),
        "train_windows": int(train_x.shape[0]),
        "val_windows": int(val_x.shape[0]),
        "test_windows": int(test_x.shape[0]),
        "validation": val_metrics,
        "test": test_metrics,
        "metadata": metadata,
    }
    report_path = args.output.with_suffix(args.output.suffix + ".json")
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"[learned-motion] saved checkpoint: {args.output}")
    print(f"[learned-motion] saved report: {report_path}")


if __name__ == "__main__":
    main()
