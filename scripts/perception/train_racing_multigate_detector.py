"""Train a multi-instance gate Keypoint R-CNN on Circular-12 racing data.

The collector labels every mapped gate in every frame. This trainer consumes
those map-wide labels directly, keeps negative/no-usable-gate frames, selects
checkpoints on held-out racing episodes, and evaluates the final model on the
untouched test split.

The model has 12 foreground classes, one per physical Circular-12 gate. The
high-contrast gate texture therefore supervises color -> global Gate ID while
the keypoint head simultaneously learns the four semantic gate corners. Runtime
association uses the predicted identity to select the known-map landmark and
still applies pixel reprojection gating before any EKF update.
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
from torch.utils.data import DataLoader, Dataset

from perception.racing_multigate_dataset import (
    RacingMultiGateKeypointDataset,
    collate,
)
from perception.torchvision_keypoint_detector import (
    _decode_keypoint_visibility,
    build_keypoint_rcnn,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("artifacts/racing_estimator/circular12_pitch40_v1"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/racing_vision/circular12_pitch40_multigate"),
    )
    parser.add_argument(
        "--warm-start",
        type=Path,
        default=Path(
            "artifacts/stage2_next_steps_20260911/checkpoints/"
            "torchvision_keypointrcnn_best.pt"
        ),
    )
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=0.0025)
    parser.add_argument("--weight-decay", type=float, default=5.0e-4)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--min-visible-corners", type=int, default=2)
    parser.add_argument("--box-padding-px", type=float, default=10.0)
    parser.add_argument("--detection-threshold", type=float, default=0.35)
    parser.add_argument("--keypoint-confidence-threshold", type=float, default=0.35)
    parser.add_argument("--match-rmse-px", type=float, default=35.0)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def _motion_blur(image: np.ndarray, kernel_size: int, angle_deg: float) -> np.ndarray:
    kernel_size = max(3, int(kernel_size) | 1)
    kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
    kernel[kernel_size // 2, :] = 1.0
    center = (0.5 * (kernel_size - 1), 0.5 * (kernel_size - 1))
    matrix = cv2.getRotationMatrix2D(center, float(angle_deg), 1.0)
    kernel = cv2.warpAffine(kernel, matrix, (kernel_size, kernel_size))
    total = float(kernel.sum())
    if total <= 1.0e-8:
        return image
    kernel /= total
    return cv2.filter2D(image, -1, kernel, borderType=cv2.BORDER_REFLECT101)


def _augment_image(image: torch.Tensor) -> torch.Tensor:
    array = image.permute(1, 2, 0).cpu().numpy().astype(np.float32)
    gain = random.uniform(0.55, 1.45)
    contrast = random.uniform(0.70, 1.30)
    mean = array.mean(axis=(0, 1), keepdims=True)
    array = (array - mean) * contrast + mean
    array *= gain

    draw = random.random()
    if draw < 0.60:
        array = _motion_blur(
            array,
            random.choice((3, 5, 7, 9, 11, 13, 15)),
            random.uniform(0.0, 180.0),
        )
    elif draw < 0.75:
        array = cv2.GaussianBlur(
            array,
            (5, 5),
            sigmaX=random.uniform(0.4, 1.8),
        )

    noise_std = random.uniform(0.0, 0.045)
    if noise_std > 0.0:
        array += np.random.normal(0.0, noise_std, array.shape).astype(np.float32)
    array = np.clip(array, 0.0, 1.0)
    return torch.from_numpy(array.copy()).permute(2, 0, 1)


class _AugmentedDataset(Dataset):
    def __init__(self, base: Dataset) -> None:
        self.base = base

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index):
        image, target, info = self.base[index]
        return _augment_image(image), target, info


def _worker_init(worker_id: int) -> None:
    seed = int(torch.initial_seed() % (2**32))
    random.seed(seed + worker_id)
    np.random.seed(seed + worker_id)


def _prediction_instances(
    output: dict,
    *,
    image_width: int,
    image_height: int,
    detection_threshold: float,
    keypoint_confidence_threshold: float,
) -> list[tuple[np.ndarray, np.ndarray, float, int | None]]:
    scores = output.get("scores")
    if scores is None:
        return []
    raw_keypoint_scores = output.get("keypoints_scores")
    raw_labels = output.get("labels")
    result: list[tuple[np.ndarray, np.ndarray, float, int | None]] = []
    for index in range(int(len(scores))):
        instance_score = float(scores[index].detach().cpu())
        if instance_score < float(detection_threshold):
            break
        corners = output["keypoints"][index, :, :2].detach().cpu().numpy()
        logits = None
        if raw_keypoint_scores is not None and len(raw_keypoint_scores) > index:
            logits = raw_keypoint_scores[index].detach().cpu().numpy()
        visible, _ = _decode_keypoint_visibility(
            corners,
            image_width=image_width,
            image_height=image_height,
            keypoint_logits=logits,
            instance_score=instance_score,
            confidence_threshold=keypoint_confidence_threshold,
            min_quad_area_px2=16.0,
        )
        if int(visible.sum()) >= 2:
            gate_id = None
            if raw_labels is not None and len(raw_labels) > index:
                label = int(raw_labels[index].detach().cpu())
                if 1 <= label <= 12:
                    gate_id = label
            result.append(
                (corners.astype(np.float64), visible, instance_score, gate_id)
            )
    return result


def _greedy_match(
    predictions: list[tuple[np.ndarray, np.ndarray, float, int | None]],
    gt_corners: np.ndarray,
    gt_visible: np.ndarray,
    *,
    max_rmse_px: float,
) -> list[tuple[int, int, float, np.ndarray]]:
    candidates: list[tuple[float, int, int, np.ndarray]] = []
    for pred_index, (pred_corners, pred_visible, _) in enumerate(predictions):
        for gt_index in range(len(gt_corners)):
            common = np.asarray(gt_visible[gt_index], dtype=bool) & np.asarray(
                pred_visible, dtype=bool
            )
            if int(common.sum()) < 2:
                continue
            residual = pred_corners[common] - gt_corners[gt_index, common]
            rmse = float(np.sqrt(np.mean(np.sum(residual * residual, axis=1))))
            if np.isfinite(rmse) and rmse <= float(max_rmse_px):
                candidates.append((rmse, pred_index, gt_index, common))

    candidates.sort(key=lambda item: item[0])
    used_pred: set[int] = set()
    used_gt: set[int] = set()
    matches: list[tuple[int, int, float, np.ndarray]] = []
    for rmse, pred_index, gt_index, common in candidates:
        if pred_index in used_pred or gt_index in used_gt:
            continue
        used_pred.add(pred_index)
        used_gt.add(gt_index)
        matches.append((pred_index, gt_index, rmse, common))
    return matches


@torch.inference_mode()
def _evaluate(
    model,
    loader,
    device: torch.device,
    *,
    detection_threshold: float,
    keypoint_confidence_threshold: float,
    match_rmse_px: float,
) -> dict:
    model.eval()
    gt_instances = 0
    pred_instances = 0
    matched_instances = 0
    positive_frames = 0
    positive_frames_hit = 0
    negative_frames = 0
    negative_frames_false_positive = 0
    coordinate_sq_sum = 0.0
    matched_corner_count = 0
    matched_gate_id_correct = 0
    matched_gate_id_count = 0

    for images, _, infos in loader:
        outputs = model([image.to(device) for image in images])
        for image, output, info in zip(images, outputs, infos):
            predictions = _prediction_instances(
                output,
                image_width=int(image.shape[2]),
                image_height=int(image.shape[1]),
                detection_threshold=detection_threshold,
                keypoint_confidence_threshold=keypoint_confidence_threshold,
            )
            gt_count = int(len(info.gate_indices))
            gt_instances += gt_count
            pred_instances += len(predictions)

            if gt_count > 0:
                positive_frames += 1
            else:
                negative_frames += 1
                if predictions:
                    negative_frames_false_positive += 1

            matches = _greedy_match(
                predictions,
                info.corners_uv,
                info.visible_masks,
                max_rmse_px=match_rmse_px,
            )
            matched_instances += len(matches)
            if gt_count > 0 and matches:
                positive_frames_hit += 1

            for pred_index, gt_index, _, common in matches:
                pred_corners = predictions[pred_index][0]
                residual = pred_corners[common] - info.corners_uv[gt_index, common]
                coordinate_sq_sum += float(np.sum(residual * residual))
                matched_corner_count += int(common.sum())

                predicted_gate_id = predictions[pred_index][3]
                truth_gate_id = int(info.gate_indices[gt_index]) + 1
                if predicted_gate_id is not None:
                    matched_gate_id_count += 1
                    matched_gate_id_correct += int(
                        int(predicted_gate_id) == truth_gate_id
                    )

    precision = matched_instances / max(pred_instances, 1)
    recall = matched_instances / max(gt_instances, 1)
    coordinate_rmse = float(
        np.sqrt(coordinate_sq_sum / max(2 * matched_corner_count, 1))
    )
    radial_rmse = float(
        np.sqrt(coordinate_sq_sum / max(matched_corner_count, 1))
    )
    negative_fp_rate = negative_frames_false_positive / max(negative_frames, 1)
    positive_hit_rate = positive_frames_hit / max(positive_frames, 1)
    gate_id_accuracy = matched_gate_id_correct / max(matched_gate_id_count, 1)
    return {
        "gt_instances": int(gt_instances),
        "predicted_usable_instances": int(pred_instances),
        "matched_instances": int(matched_instances),
        "instance_precision": float(precision),
        "instance_recall": float(recall),
        "positive_frame_hit_rate": float(positive_hit_rate),
        "negative_frame_false_positive_rate": float(negative_fp_rate),
        "matched_corner_coordinate_rmse_px": coordinate_rmse,
        "matched_corner_radial_rmse_px": radial_rmse,
        "gate_id_evaluated_matches": int(matched_gate_id_count),
        "gate_id_correct_matches": int(matched_gate_id_correct),
        "gate_id_accuracy": float(gate_id_accuracy),
        "recommended_pixel_sigma_px": float(max(1.0, coordinate_rmse)),
    }


def _build_loader(
    root: Path,
    *,
    batch_size: int,
    num_workers: int,
    min_visible_corners: int,
    box_padding_px: float,
    augment: bool,
    seed: int,
    shuffle: bool,
):
    dataset: Dataset = RacingMultiGateKeypointDataset(
        root,
        min_visible_corners=min_visible_corners,
        box_padding_px=box_padding_px,
    )
    if augment:
        dataset = _AugmentedDataset(dataset)
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
        num_workers=num_workers,
        collate_fn=collate,
        worker_init_fn=_worker_init,
    ), dataset


def main() -> None:
    args = _parse_args()
    if args.epochs < 1 or args.batch_size < 1:
        raise ValueError("epochs and batch-size must be positive")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    requested_device = str(args.device)
    if requested_device.startswith("cuda") and not torch.cuda.is_available():
        print("[multigate-vision] CUDA unavailable; using CPU", flush=True)
        requested_device = "cpu"
    device = torch.device(requested_device)

    root = args.dataset_root.expanduser().resolve()
    train_root = root / "vision" / "train"
    val_root = root / "vision" / "val"
    test_root = root / "vision" / "test"
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    train_loader, train_dataset = _build_loader(
        train_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        min_visible_corners=args.min_visible_corners,
        box_padding_px=args.box_padding_px,
        augment=True,
        seed=args.seed,
        shuffle=True,
    )
    val_loader, val_dataset = _build_loader(
        val_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        min_visible_corners=args.min_visible_corners,
        box_padding_px=args.box_padding_px,
        augment=False,
        seed=args.seed + 1,
        shuffle=False,
    )
    test_loader, test_dataset = _build_loader(
        test_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        min_visible_corners=args.min_visible_corners,
        box_padding_px=args.box_padding_px,
        augment=False,
        seed=args.seed + 2,
        shuffle=False,
    )

    num_gate_ids = 12
    num_classes = num_gate_ids + 1  # background + Gate IDs 1..12
    model = build_keypoint_rcnn(
        image_size=args.image_size,
        num_classes=num_classes,
    ).to(device)
    warm_start = args.warm_start.expanduser().resolve()
    if warm_start.exists():
        payload = torch.load(warm_start, map_location=device, weights_only=False)
        source_state = payload["model_state_dict"]
        target_state = model.state_dict()
        compatible = {
            key: value
            for key, value in source_state.items()
            if key in target_state and target_state[key].shape == value.shape
        }
        missing, unexpected = model.load_state_dict(compatible, strict=False)
        print(
            "[multigate-vision] warm start: "
            f"{warm_start} compatible_tensors={len(compatible)}/"
            f"{len(target_state)} skipped_or_missing={len(missing)} "
            f"unexpected={len(unexpected)}",
            flush=True,
        )
    else:
        print(
            f"[multigate-vision] warm start not found; training from scratch: {warm_start}",
            flush=True,
        )

    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.SGD(
        parameters,
        lr=float(args.learning_rate),
        momentum=0.9,
        weight_decay=float(args.weight_decay),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(int(args.epochs), 1),
    )

    checkpoint_path = output_dir / "torchvision_keypointrcnn_multigate_best.pt"
    best_score = -float("inf")
    best_epoch = -1
    best_val: dict = {}
    started = perf_counter()

    for epoch in range(1, int(args.epochs) + 1):
        model.train()
        running_loss = 0.0
        batches = 0
        for images, targets, _ in train_loader:
            images_device = [image.to(device) for image in images]
            targets_device = [
                {key: value.to(device) for key, value in target.items()}
                for target in targets
            ]
            losses = model(images_device, targets_device)
            loss = sum(losses.values())
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, 5.0)
            optimizer.step()
            running_loss += float(loss.detach().cpu())
            batches += 1
        scheduler.step()

        metrics = _evaluate(
            model,
            val_loader,
            device,
            detection_threshold=args.detection_threshold,
            keypoint_confidence_threshold=args.keypoint_confidence_threshold,
            match_rmse_px=args.match_rmse_px,
        )
        score = (
            metrics["instance_recall"]
            + 0.25 * metrics["instance_precision"]
            + 0.25 * metrics["positive_frame_hit_rate"]
            - 0.25 * metrics["negative_frame_false_positive_rate"]
            - 0.003 * metrics["matched_corner_coordinate_rmse_px"]
            + 0.50 * metrics["gate_id_accuracy"]
        )
        print(
            "[multigate-vision] "
            f"epoch={epoch:03d} "
            f"loss={running_loss / max(batches, 1):.5f} "
            f"precision={metrics['instance_precision']:.4f} "
            f"recall={metrics['instance_recall']:.4f} "
            f"frame_hit={metrics['positive_frame_hit_rate']:.4f} "
            f"neg_fp={metrics['negative_frame_false_positive_rate']:.4f} "
            f"coord_rmse={metrics['matched_corner_coordinate_rmse_px']:.3f}px "
            f"gate_id_acc={metrics['gate_id_accuracy']:.4f}",
            flush=True,
        )

        if score > best_score:
            best_score = float(score)
            best_epoch = int(epoch)
            best_val = dict(metrics)
            metadata = {
                "schema": "isaac_drone_racer.circular12_multigate_keypointrcnn.v2",
                "architecture": "torchvision_keypointrcnn_resnet50_fpn",
                "num_classes": int(num_classes),
                "gate_identity_mode": "gate_id_class",
                "gate_id_count": int(num_gate_ids),
                "gate_identity_source": (
                    "high_contrast_texture+supervised_global_gate_id"
                ),
                "dataset_root": str(root),
                "train_root": str(train_root),
                "val_root": str(val_root),
                "test_root": str(test_root),
                "train_frames": int(len(train_dataset)),
                "val_frames": int(len(val_dataset)),
                "test_frames": int(len(test_dataset)),
                "min_visible_corners": int(args.min_visible_corners),
                "box_padding_px": float(args.box_padding_px),
                "detection_threshold": float(args.detection_threshold),
                "keypoint_confidence_threshold": float(
                    args.keypoint_confidence_threshold
                ),
                "match_rmse_px": float(args.match_rmse_px),
                "image_size": int(args.image_size),
                "epoch": int(epoch),
                "seed": int(args.seed),
                "validation": metrics,
            }
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "image_size": int(args.image_size),
                    "num_classes": int(num_classes),
                    "metadata": metadata,
                },
                checkpoint_path,
            )
            checkpoint_path.with_suffix(".json").write_text(
                json.dumps(metadata, indent=2) + "\n",
                encoding="utf-8",
            )

    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(payload["model_state_dict"])
    test_metrics = _evaluate(
        model,
        test_loader,
        device,
        detection_threshold=args.detection_threshold,
        keypoint_confidence_threshold=args.keypoint_confidence_threshold,
        match_rmse_px=args.match_rmse_px,
    )
    report = {
        "schema": "isaac_drone_racer.circular12_multigate_training_report.v2",
        "checkpoint": str(checkpoint_path),
        "best_epoch": best_epoch,
        "best_score": best_score,
        "elapsed_s": float(perf_counter() - started),
        "validation": best_val,
        "test": test_metrics,
        "runtime_contract": {
            "detector_api": "TorchvisionGateCornerDetector.detect_all",
            "map_identity": (
                "detector Gate ID class -> known-map gate index; "
                "pixel reprojection remains the consistency gate"
            ),
            "gate_measurement_model": "direct_reprojection",
            "visibility_guard": "not used; per-instance Keypoint R-CNN confidence only",
            "recommended_pixel_sigma_px": float(
                best_val.get("recommended_pixel_sigma_px", 3.0)
            ),
        },
    }
    report_path = output_dir / "training_report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("=" * 96)
    print("CIRCULAR-12 MULTI-GATE VISION TRAINING COMPLETE")
    print("=" * 96)
    print(json.dumps(report, indent=2))
    print(f"[multigate-vision] checkpoint: {checkpoint_path}")
    print(f"[multigate-vision] report: {report_path}")


if __name__ == "__main__":
    main()
