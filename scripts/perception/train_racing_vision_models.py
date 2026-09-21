"""Fine-tune the Stage2 racing vision stack on real GT-policy racing frames.

This script trains the two models used by the current production hybrid:
1. torchvision Keypoint R-CNN for gate instance + corner coordinates;
2. GateKeypointNet for ordered-corner visibility/confidence guarding.

The racing dataset is episode-split by the collector. Training can optionally
mix the legacy Stage2 dataset, while validation is intentionally performed on
held-out racing episodes so the primary metric reflects deployment conditions.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import ConcatDataset, DataLoader, Dataset

from perception.keypoint_detector import GateKeypointNet, Stage2KeypointDataset
from perception.torchvision_keypoint_detector import (
    TorchvisionStage2KeypointDataset,
    _decode_keypoint_visibility,
    build_keypoint_rcnn,
)


def _sample_ids(root: Path, *, min_visible_corners: int = 0) -> list[str]:
    ids: list[str] = []
    for label_path in sorted((root / "labels").glob("*.json")):
        payload = json.loads(label_path.read_text(encoding="utf-8"))
        visible_count = int(sum(bool(v) for v in payload["visible"]))
        if visible_count >= int(min_visible_corners):
            ids.append(label_path.stem)
    if not ids:
        raise ValueError(
            f"No samples with >= {min_visible_corners} visible corners under {root}"
        )
    return ids


def _motion_blur(image: np.ndarray, kernel_size: int, angle_deg: float) -> np.ndarray:
    kernel_size = max(3, int(kernel_size) | 1)
    kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
    kernel[kernel_size // 2, :] = 1.0
    matrix = cv2.getRotationMatrix2D(
        (0.5 * (kernel_size - 1), 0.5 * (kernel_size - 1)),
        float(angle_deg),
        1.0,
    )
    kernel = cv2.warpAffine(kernel, matrix, (kernel_size, kernel_size))
    kernel_sum = float(kernel.sum())
    if kernel_sum <= 1.0e-8:
        return image
    kernel /= kernel_sum
    return cv2.filter2D(image, -1, kernel, borderType=cv2.BORDER_REFLECT101)


def _augment_image(image: torch.Tensor) -> torch.Tensor:
    array = image.permute(1, 2, 0).cpu().numpy().astype(np.float32)

    gain = random.uniform(0.55, 1.45)
    contrast = random.uniform(0.70, 1.30)
    mean = array.mean(axis=(0, 1), keepdims=True)
    array = (array - mean) * contrast + mean
    array *= gain

    if random.random() < 0.55:
        kernel = random.choice((3, 5, 7, 9, 11, 13, 15))
        angle = random.uniform(0.0, 180.0)
        array = _motion_blur(array, kernel, angle)
    elif random.random() < 0.20:
        sigma = random.uniform(0.4, 1.8)
        array = cv2.GaussianBlur(array, (5, 5), sigmaX=sigma)

    noise_std = random.uniform(0.0, 0.045)
    if noise_std > 0.0:
        array += np.random.normal(0.0, noise_std, array.shape).astype(np.float32)

    array = np.clip(array, 0.0, 1.0)
    return torch.from_numpy(array.copy()).permute(2, 0, 1)


class _AugmentedDataset(Dataset):
    def __init__(self, base: Dataset):
        self.base = base

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index):
        item = self.base[index]
        image = _augment_image(item[0])
        return (image, *item[1:])


def _repeat_dataset(dataset: Dataset, repeats: int) -> Dataset:
    repeats = max(1, int(repeats))
    if repeats == 1:
        return dataset
    return ConcatDataset([dataset for _ in range(repeats)])


def _worker_init(worker_id: int) -> None:
    seed = int(torch.initial_seed() % (2**32))
    random.seed(seed + worker_id)
    np.random.seed(seed + worker_id)


def _rcnn_collate(batch):
    images, targets, sample_ids = zip(*batch)
    return list(images), list(targets), sample_ids


def _build_rcnn_training_dataset(
    racing_root: Path,
    legacy_root: Path | None,
    *,
    racing_repeat: int,
    min_visible_corners: int,
) -> Dataset:
    racing_ids = _sample_ids(
        racing_root,
        min_visible_corners=min_visible_corners,
    )
    racing = _AugmentedDataset(
        TorchvisionStage2KeypointDataset(racing_root, racing_ids)
    )
    pieces: list[Dataset] = [_repeat_dataset(racing, racing_repeat)]
    if legacy_root is not None:
        legacy_ids = _sample_ids(
            legacy_root,
            min_visible_corners=min_visible_corners,
        )
        pieces.append(
            _AugmentedDataset(
                TorchvisionStage2KeypointDataset(legacy_root, legacy_ids)
            )
        )
    return pieces[0] if len(pieces) == 1 else ConcatDataset(pieces)


def _build_guard_training_dataset(
    racing_root: Path,
    legacy_root: Path | None,
    *,
    racing_repeat: int,
    input_size: int,
) -> Dataset:
    racing_ids = _sample_ids(racing_root, min_visible_corners=0)
    racing = _AugmentedDataset(
        Stage2KeypointDataset(
            racing_root,
            racing_ids,
            input_size=input_size,
        )
    )
    pieces: list[Dataset] = [_repeat_dataset(racing, racing_repeat)]
    if legacy_root is not None:
        legacy_ids = _sample_ids(legacy_root, min_visible_corners=0)
        pieces.append(
            _AugmentedDataset(
                Stage2KeypointDataset(
                    legacy_root,
                    legacy_ids,
                    input_size=input_size,
                )
            )
        )
    return pieces[0] if len(pieces) == 1 else ConcatDataset(pieces)


@torch.inference_mode()
def _evaluate_rcnn(
    model,
    loader,
    device,
    *,
    detection_threshold: float,
) -> dict:
    model.eval()
    eligible = 0
    detected = 0
    usable = 0
    squared_error = 0.0
    visible_corner_count = 0

    for images, targets, _ in loader:
        outputs = model([image.to(device) for image in images])
        for image, output, target in zip(images, outputs, targets):
            gt_visible = target["keypoints"][0, :, 2].bool()
            if int(gt_visible.sum().item()) < 2:
                continue
            eligible += 1

            if not len(output["scores"]):
                continue
            instance_score = float(output["scores"][0].detach().cpu())
            if instance_score < float(detection_threshold):
                continue
            detected += 1

            predicted = output["keypoints"][0, :, :2].detach().cpu()
            truth = target["keypoints"][0, :, :2]
            squared_error += ((predicted[gt_visible] - truth[gt_visible]) ** 2).sum().item()
            visible_corner_count += int(gt_visible.sum().item())

            raw_scores = output.get("keypoints_scores")
            logits = None
            if raw_scores is not None and len(raw_scores):
                logits = raw_scores[0].detach().cpu().numpy()
            predicted_visible, _ = _decode_keypoint_visibility(
                predicted.numpy(),
                image_width=int(image.shape[2]),
                image_height=int(image.shape[1]),
                keypoint_logits=logits,
                instance_score=instance_score,
                confidence_threshold=0.0,
                min_quad_area_px2=16.0,
            )
            if int(predicted_visible.sum()) >= 2:
                usable += 1

    return {
        "eligible_frames_gt_ge2": int(eligible),
        "instance_detection_rate": float(detected / max(eligible, 1)),
        "coordinate_usable_rate": float(usable / max(eligible, 1)),
        "corner_rmse_px": float(
            np.sqrt(squared_error / max(visible_corner_count, 1))
        ),
    }


def _guard_losses(predicted, logits, target, visible):
    mask = visible[..., None]
    denominator = mask.sum().clamp_min(1.0) * 2.0
    corner_loss = (((predicted - target) ** 2) * mask).sum() / denominator
    visibility_loss = F.binary_cross_entropy_with_logits(logits, visible)
    return corner_loss + 0.1 * visibility_loss


@torch.inference_mode()
def _evaluate_guard(
    model,
    loader,
    device,
    *,
    input_size: int,
    visibility_threshold: float,
) -> dict:
    model.eval()
    squared_error = 0.0
    visible_corner_count = 0.0
    eligible_frames = 0
    usable_frames = 0
    ineligible_frames = 0
    false_usable_frames = 0
    tp = fp = fn = tn = 0

    for images, target, visible, _ in loader:
        images = images.to(device)
        target = target.to(device)
        visible = visible.to(device)
        predicted, logits = model(images)
        confidence = torch.sigmoid(logits)
        predicted_visible = confidence >= float(visibility_threshold)
        truth_visible = visible.bool()

        scale = torch.tensor(
            [input_size, input_size],
            device=device,
            dtype=predicted.dtype,
        )
        squared_error += (
            (((predicted - target) * scale) ** 2) * visible[..., None]
        ).sum().item()
        visible_corner_count += visible.sum().item()

        tp += int((predicted_visible & truth_visible).sum().item())
        fp += int((predicted_visible & ~truth_visible).sum().item())
        fn += int((~predicted_visible & truth_visible).sum().item())
        tn += int((~predicted_visible & ~truth_visible).sum().item())

        truth_count = truth_visible.sum(dim=1)
        predicted_count = predicted_visible.sum(dim=1)
        eligible_mask = truth_count >= 2
        ineligible_mask = ~eligible_mask
        eligible_frames += int(eligible_mask.sum().item())
        usable_frames += int(
            (eligible_mask & (predicted_count >= 2)).sum().item()
        )
        ineligible_frames += int(ineligible_mask.sum().item())
        false_usable_frames += int(
            (ineligible_mask & (predicted_count >= 2)).sum().item()
        )

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2.0 * precision * recall / max(precision + recall, 1.0e-12)
    return {
        "corner_rmse_px_at_input_resolution": float(
            np.sqrt(squared_error / max(visible_corner_count, 1.0))
        ),
        "visibility_precision": float(precision),
        "visibility_recall": float(recall),
        "visibility_f1": float(f1),
        "usable_ge2_rate_given_gt_ge2": float(
            usable_frames / max(eligible_frames, 1)
        ),
        "false_usable_ge2_rate_given_gt_lt2": float(
            false_usable_frames / max(ineligible_frames, 1)
        ),
        "eligible_frames_gt_ge2": int(eligible_frames),
        "ineligible_frames_gt_lt2": int(ineligible_frames),
        "corner_confusion": {
            "tp": int(tp),
            "fp": int(fp),
            "fn": int(fn),
            "tn": int(tn),
        },
    }


def _train_rcnn(args, device: torch.device) -> dict:
    train_dataset = _build_rcnn_training_dataset(
        args.racing_train,
        args.legacy_train,
        racing_repeat=args.racing_repeat,
        min_visible_corners=args.rcnn_min_visible_corners,
    )
    validation_ids = _sample_ids(
        args.racing_val,
        min_visible_corners=args.rcnn_min_visible_corners,
    )
    validation_dataset = TorchvisionStage2KeypointDataset(
        args.racing_val,
        validation_ids,
    )

    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.rcnn_batch_size,
        shuffle=True,
        generator=generator,
        num_workers=args.num_workers,
        collate_fn=_rcnn_collate,
        worker_init_fn=_worker_init,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=args.rcnn_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=_rcnn_collate,
        worker_init_fn=_worker_init,
    )

    model = build_keypoint_rcnn(image_size=args.rcnn_image_size).to(device)
    init_checkpoint = Path(args.rcnn_init).expanduser()
    if init_checkpoint.exists():
        payload = torch.load(
            init_checkpoint,
            map_location=device,
            weights_only=False,
        )
        model.load_state_dict(payload["model_state_dict"])
        print(f"[racing-vision][rcnn] warm start: {init_checkpoint}", flush=True)
    else:
        print(
            f"[racing-vision][rcnn] init checkpoint not found; training from scratch: "
            f"{init_checkpoint}",
            flush=True,
        )

    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        parameters,
        lr=args.rcnn_learning_rate,
        momentum=0.9,
        weight_decay=5.0e-4,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(args.rcnn_epochs, 1),
    )

    output = args.output_dir / "torchvision_keypointrcnn_racing_best.pt"
    best_score = -float("inf")
    best_metrics: dict = {}
    started = perf_counter()

    for epoch in range(1, args.rcnn_epochs + 1):
        model.train()
        running_loss = 0.0
        samples = 0
        for images, targets, _ in train_loader:
            images = [image.to(device) for image in images]
            targets = [
                {key: value.to(device) for key, value in target.items()}
                for target in targets
            ]
            losses = model(images, targets)
            loss = sum(losses.values())
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, 5.0)
            optimizer.step()
            running_loss += float(loss.item()) * len(images)
            samples += len(images)
        scheduler.step()

        metrics = _evaluate_rcnn(
            model,
            validation_loader,
            device,
            detection_threshold=args.rcnn_detection_threshold,
        )
        score = (
            metrics["coordinate_usable_rate"]
            + metrics["instance_detection_rate"]
            - 0.002 * metrics["corner_rmse_px"]
        )
        print(
            "[racing-vision][rcnn] "
            f"epoch={epoch:03d} "
            f"loss={running_loss / max(samples, 1):.6f} "
            f"detect={metrics['instance_detection_rate']:.4f} "
            f"usable={metrics['coordinate_usable_rate']:.4f} "
            f"rmse={metrics['corner_rmse_px']:.3f}px",
            flush=True,
        )
        if score > best_score:
            best_score = score
            best_metrics = dict(metrics)
            metadata = {
                "schema": "isaac_drone_racer.stage2b_torchvision_detector.racing_v1",
                "architecture": "torchvision_keypointrcnn_resnet50_fpn",
                "racing_train": str(args.racing_train.resolve()),
                "racing_val": str(args.racing_val.resolve()),
                "legacy_train": (
                    None
                    if args.legacy_train is None
                    else str(args.legacy_train.resolve())
                ),
                "racing_repeat": int(args.racing_repeat),
                "train_samples_effective": len(train_dataset),
                "validation_samples": len(validation_dataset),
                "epoch": int(epoch),
                "seed": int(args.seed),
                "image_size": int(args.rcnn_image_size),
                **metrics,
            }
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "image_size": int(args.rcnn_image_size),
                    "metadata": metadata,
                },
                output,
            )
            output.with_suffix(".json").write_text(
                json.dumps(metadata, indent=2) + "\n",
                encoding="utf-8",
            )

    return {
        "checkpoint": str(output),
        "elapsed_s": float(perf_counter() - started),
        "best_metrics": best_metrics,
    }


def _train_guard(args, device: torch.device) -> dict:
    train_dataset = _build_guard_training_dataset(
        args.racing_train,
        args.legacy_train,
        racing_repeat=args.racing_repeat,
        input_size=args.guard_input_size,
    )
    validation_ids = _sample_ids(
        args.racing_val,
        min_visible_corners=0,
    )
    validation_dataset = Stage2KeypointDataset(
        args.racing_val,
        validation_ids,
        input_size=args.guard_input_size,
    )

    generator = torch.Generator().manual_seed(args.seed + 17)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.guard_batch_size,
        shuffle=True,
        generator=generator,
        num_workers=args.num_workers,
        worker_init_fn=_worker_init,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=args.guard_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        worker_init_fn=_worker_init,
    )

    model = GateKeypointNet(
        args.guard_input_size,
        args.guard_width,
    ).to(device)
    init_checkpoint = Path(args.guard_init).expanduser()
    if init_checkpoint.exists():
        payload = torch.load(
            init_checkpoint,
            map_location=device,
            weights_only=False,
        )
        model.load_state_dict(payload["model_state_dict"])
        print(f"[racing-vision][guard] warm start: {init_checkpoint}", flush=True)
    else:
        print(
            f"[racing-vision][guard] init checkpoint not found; training from scratch: "
            f"{init_checkpoint}",
            flush=True,
        )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.guard_learning_rate,
        weight_decay=1.0e-4,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(args.guard_epochs, 1),
    )

    output = args.output_dir / "gate_keypoint_net_racing_best.pt"
    best_score = -float("inf")
    best_metrics: dict = {}
    started = perf_counter()

    for epoch in range(1, args.guard_epochs + 1):
        model.train()
        running_loss = 0.0
        samples = 0
        for images, target, visible, _ in train_loader:
            images = images.to(device)
            target = target.to(device)
            visible = visible.to(device)
            predicted, logits = model(images)
            loss = _guard_losses(predicted, logits, target, visible)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            running_loss += float(loss.item()) * int(images.shape[0])
            samples += int(images.shape[0])
        scheduler.step()

        metrics = _evaluate_guard(
            model,
            validation_loader,
            device,
            input_size=args.guard_input_size,
            visibility_threshold=args.guard_visibility_threshold,
        )
        score = (
            metrics["usable_ge2_rate_given_gt_ge2"]
            + 0.25 * metrics["visibility_f1"]
            - metrics["false_usable_ge2_rate_given_gt_lt2"]
            - 0.002 * metrics["corner_rmse_px_at_input_resolution"]
        )
        print(
            "[racing-vision][guard] "
            f"epoch={epoch:03d} "
            f"loss={running_loss / max(samples, 1):.6f} "
            f"usable={metrics['usable_ge2_rate_given_gt_ge2']:.4f} "
            f"false_usable={metrics['false_usable_ge2_rate_given_gt_lt2']:.4f} "
            f"f1={metrics['visibility_f1']:.4f} "
            f"rmse={metrics['corner_rmse_px_at_input_resolution']:.3f}px",
            flush=True,
        )
        if score > best_score:
            best_score = score
            best_metrics = dict(metrics)
            metadata = {
                "schema": "isaac_drone_racer.stage2b_detector.racing_v1",
                "architecture": "spatial_heatmap_softargmax",
                "racing_train": str(args.racing_train.resolve()),
                "racing_val": str(args.racing_val.resolve()),
                "legacy_train": (
                    None
                    if args.legacy_train is None
                    else str(args.legacy_train.resolve())
                ),
                "racing_repeat": int(args.racing_repeat),
                "train_samples_effective": len(train_dataset),
                "validation_samples": len(validation_dataset),
                "epoch": int(epoch),
                "seed": int(args.seed),
                "input_size": int(args.guard_input_size),
                "width": int(args.guard_width),
                "visibility_threshold": float(args.guard_visibility_threshold),
                **metrics,
            }
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "input_size": int(args.guard_input_size),
                    "width": int(args.guard_width),
                    "metadata": metadata,
                },
                output,
            )
            output.with_suffix(".json").write_text(
                json.dumps(metadata, indent=2) + "\n",
                encoding="utf-8",
            )

    return {
        "checkpoint": str(output),
        "elapsed_s": float(perf_counter() - started),
        "best_metrics": best_metrics,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--racing-train", type=Path, required=True)
    parser.add_argument("--racing-val", type=Path, required=True)
    parser.add_argument("--legacy-train", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--model",
        choices=("both", "rcnn", "guard"),
        default="both",
    )
    parser.add_argument("--racing-repeat", type=int, default=3)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )

    parser.add_argument(
        "--rcnn-init",
        default=(
            "artifacts/stage2_next_steps_20260911/checkpoints/"
            "torchvision_keypointrcnn_best.pt"
        ),
    )
    parser.add_argument("--rcnn-epochs", type=int, default=12)
    parser.add_argument("--rcnn-batch-size", type=int, default=8)
    parser.add_argument("--rcnn-image-size", type=int, default=256)
    parser.add_argument("--rcnn-learning-rate", type=float, default=2.0e-3)
    parser.add_argument("--rcnn-detection-threshold", type=float, default=0.5)
    parser.add_argument("--rcnn-min-visible-corners", type=int, default=2)

    parser.add_argument(
        "--guard-init",
        default=(
            "artifacts/stage2_next_steps_20260911/checkpoints/"
            "gate_keypoint_net_best.pt"
        ),
    )
    parser.add_argument("--guard-epochs", type=int, default=30)
    parser.add_argument("--guard-batch-size", type=int, default=64)
    parser.add_argument("--guard-input-size", type=int, default=256)
    parser.add_argument("--guard-width", type=int, default=32)
    parser.add_argument("--guard-learning-rate", type=float, default=5.0e-4)
    parser.add_argument("--guard-visibility-threshold", type=float, default=0.75)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    for path in (args.racing_train, args.racing_val):
        if not (path / "labels").exists() or not (path / "images").exists():
            raise FileNotFoundError(f"Invalid Stage2 dataset root: {path}")
    if args.legacy_train is not None:
        if not (args.legacy_train / "labels").exists():
            raise FileNotFoundError(
                f"Invalid legacy Stage2 dataset root: {args.legacy_train}"
            )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    summary = {
        "schema": "isaac_drone_racer.racing_vision_training.v1",
        "device": str(device),
        "model": args.model,
        "racing_repeat": int(args.racing_repeat),
    }

    if args.model in ("both", "rcnn"):
        summary["rcnn"] = _train_rcnn(args, device)
    if args.model in ("both", "guard"):
        summary["guard"] = _train_guard(args, device)

    (args.output_dir / "training_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    print("=" * 96)
    print("RACING VISION TRAINING COMPLETE")
    print("=" * 96)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
