"""Train a torchvision Keypoint R-CNN Stage2B comparison baseline."""

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
from torch.utils.data import DataLoader

from perception.torchvision_keypoint_detector import (
    TorchvisionStage2KeypointDataset,
    build_keypoint_rcnn,
)


def collate(batch):
    images, targets, sample_ids = zip(*batch)
    return list(images), list(targets), sample_ids


@torch.inference_mode()
def validation_corner_rmse(model, loader, device) -> float:
    model.eval()
    squared_error = 0.0
    corner_count = 0
    for images, targets, _ in loader:
        outputs = model([image.to(device) for image in images])
        for output, target in zip(outputs, targets):
            visible = target["keypoints"][0, :, 2].bool()
            if not len(output["scores"]) or not visible.any():
                continue
            predicted = output["keypoints"][0, :, :2].cpu()
            truth = target["keypoints"][0, :, :2]
            squared_error += ((predicted[visible] - truth[visible]) ** 2).sum().item()
            corner_count += int(visible.sum())
    return float(np.sqrt(squared_error / max(corner_count, 1)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--validation_dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--image_size", type=int, default=128)
    parser.add_argument("--learning_rate", type=float, default=0.005)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    train_dataset = TorchvisionStage2KeypointDataset(args.dataset)
    validation_dataset = TorchvisionStage2KeypointDataset(args.validation_dataset)
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
        num_workers=4,
        collate_fn=collate,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=collate,
    )

    device = torch.device(args.device)
    model = build_keypoint_rcnn(image_size=args.image_size).to(device)
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.SGD(parameters, lr=args.learning_rate, momentum=0.9, weight_decay=5.0e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.1)
    best_rmse = float("inf")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    started = perf_counter()

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        samples = 0
        for images, targets, _ in train_loader:
            images = [image.to(device) for image in images]
            targets = [{key: value.to(device) for key, value in target.items()} for target in targets]
            losses = model(images, targets)
            loss = sum(losses.values())
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            running_loss += float(loss) * len(images)
            samples += len(images)
        scheduler.step()
        rmse = validation_corner_rmse(model, validation_loader, device)
        print(
            f"epoch={epoch:03d} train_loss={running_loss / max(samples, 1):.6f} "
            f"val_corner_rmse_px={rmse:.3f}",
            flush=True,
        )
        if rmse < best_rmse:
            best_rmse = rmse
            metadata = {
                "schema": "isaac_drone_racer.stage2b_torchvision_detector.v1",
                "architecture": "torchvision_keypointrcnn_resnet50_fpn",
                "dataset": str(args.dataset.resolve()),
                "validation_dataset": str(args.validation_dataset.resolve()),
                "train_samples": len(train_dataset),
                "validation_samples": len(validation_dataset),
                "epoch": epoch,
                "seed": args.seed,
                "corner_rmse_px_at_original_resolution": rmse,
            }
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "image_size": args.image_size,
                    "metadata": metadata,
                },
                args.output,
            )
            args.output.with_suffix(".json").write_text(
                json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
            )

    print(
        f"TRAINING_COMPLETE best_rmse_px={best_rmse:.3f} "
        f"elapsed_s={perf_counter() - started:.1f} checkpoint={args.output}"
    )


if __name__ == "__main__":
    main()
