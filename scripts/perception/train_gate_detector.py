"""Train and evaluate the Stage2B ordered gate-corner detector."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import sys
from time import perf_counter

sys.path.insert(0, str(Path(__file__).parents[2]))

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from perception.keypoint_detector import GateKeypointNet, Stage2KeypointDataset


def split_ids(root: Path, validation_fraction: float, seed: int) -> tuple[list[str], list[str]]:
    ids = sorted(path.stem for path in (root / "labels").glob("*.json"))
    if len(ids) < 2:
        raise ValueError("Stage2B training requires at least two labeled samples")
    random.Random(seed).shuffle(ids)
    validation_count = max(1, int(round(len(ids) * validation_fraction)))
    return ids[validation_count:], ids[:validation_count]


def losses(predicted, logits, target, visible):
    mask = visible[..., None]
    denominator = mask.sum().clamp_min(1.0) * 2.0
    corner_loss = (((predicted - target) ** 2) * mask).sum() / denominator
    visibility_loss = F.binary_cross_entropy_with_logits(logits, visible)
    return corner_loss + 0.1 * visibility_loss, corner_loss, visibility_loss


@torch.inference_mode()
def evaluate(model, loader, device):
    model.eval()
    squared_error_px = 0.0
    visible_corner_count = 0.0
    visibility_correct = 0.0
    visibility_count = 0
    for images, target, visible, _ in loader:
        images, target, visible = images.to(device), target.to(device), visible.to(device)
        predicted, logits = model(images)
        scale = torch.tensor([loader.dataset.input_size, loader.dataset.input_size], device=device)
        squared_error_px += ((((predicted - target) * scale) ** 2) * visible[..., None]).sum().item()
        visible_corner_count += visible.sum().item()
        visibility_correct += ((torch.sigmoid(logits) >= 0.5) == visible.bool()).sum().item()
        visibility_count += visible.numel()
    rmse_px = (squared_error_px / max(visible_corner_count, 1.0)) ** 0.5
    return {
        "corner_rmse_px_at_input_resolution": rmse_px,
        "visibility_accuracy": visibility_correct / visibility_count,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument(
        "--validation_dataset",
        type=Path,
        default=None,
        help="Independent validation dataset. When set, all --dataset samples are used for training.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--input_size", type=int, default=128)
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=1.0e-3)
    parser.add_argument("--validation_fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.validation_dataset is None:
        train_ids, validation_ids = split_ids(args.dataset, args.validation_fraction, args.seed)
        validation_root = args.dataset
        validation_mode = "shuffled_split"
    else:
        train_ids = sorted(path.stem for path in (args.dataset / "labels").glob("*.json"))
        validation_ids = sorted(
            path.stem for path in (args.validation_dataset / "labels").glob("*.json")
        )
        if not train_ids or not validation_ids:
            raise ValueError("Training and independent validation datasets must both be non-empty")
        validation_root = args.validation_dataset
        validation_mode = "independent_dataset"
    train_dataset = Stage2KeypointDataset(args.dataset, train_ids, input_size=args.input_size)
    validation_dataset = Stage2KeypointDataset(
        validation_root, validation_ids, input_size=args.input_size
    )
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
        num_workers=4,
    )
    validation_loader = DataLoader(validation_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    device = torch.device(args.device)
    model = GateKeypointNet(args.input_size, args.width).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1.0e-4)
    best_rmse = float("inf")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    start = perf_counter()

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        for images, target, visible, _ in train_loader:
            images, target, visible = images.to(device), target.to(device), visible.to(device)
            predicted, logits = model(images)
            loss, _, _ = losses(predicted, logits, target, visible)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            running += loss.item() * images.shape[0]

        metrics = evaluate(model, validation_loader, device)
        print(
            f"epoch={epoch:03d} train_loss={running / len(train_dataset):.6f} "
            f"val_corner_rmse_px={metrics['corner_rmse_px_at_input_resolution']:.3f} "
            f"val_visibility_acc={metrics['visibility_accuracy']:.4f}",
            flush=True,
        )
        if metrics["corner_rmse_px_at_input_resolution"] < best_rmse:
            best_rmse = metrics["corner_rmse_px_at_input_resolution"]
            metadata = {
                "schema": "isaac_drone_racer.stage2b_detector.v1",
                "architecture": "spatial_heatmap_softargmax",
                "dataset": str(args.dataset.resolve()),
                "validation_dataset": str(validation_root.resolve()),
                "validation_mode": validation_mode,
                "train_samples": len(train_dataset),
                "validation_samples": len(validation_dataset),
                "epoch": epoch,
                "seed": args.seed,
                **metrics,
            }
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "input_size": args.input_size,
                    "width": args.width,
                    "metadata": metadata,
                },
                args.output,
            )
            args.output.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    print(
        f"TRAINING_COMPLETE best_rmse_px={best_rmse:.3f} "
        f"elapsed_s={perf_counter() - start:.1f} checkpoint={args.output}"
    )


if __name__ == "__main__":
    main()
