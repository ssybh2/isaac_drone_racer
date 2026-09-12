"""Calibrate Swift Keypoint R-CNN visibility gating and ordinary corner noise."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parents[2]))

import numpy as np
import torch
from torch.utils.data import DataLoader

from perception.keypoint_detector import TorchGateCornerDetector
from perception.torchvision_keypoint_detector import (
    TorchvisionGateCornerDetector,
    TorchvisionStage2KeypointDataset,
    _decode_keypoint_visibility,
)


def _collate(batch):
    images, targets, sample_ids = zip(*batch)
    return list(images), list(targets), list(sample_ids)


def _percentile(values: list[float], q: float) -> float | None:
    return None if not values else float(np.percentile(values, q))


def _threshold_metrics(records: list[dict], threshold: float, min_quad_area_px2: float) -> dict:
    complete_truth = 0
    partial_truth = 0
    accepted_complete_truth = 0
    accepted_partial_truth = 0
    visible_true_positive = 0
    visible_false_positive = 0
    visible_false_negative = 0
    radial_errors: list[float] = []
    signed_errors: list[np.ndarray] = []

    for record in records:
        truth_visible = record["truth_visible"]
        truth_complete = bool(np.all(truth_visible))
        complete_truth += truth_complete
        partial_truth += not truth_complete
        if record["instance_score"] < record["detection_threshold"]:
            predicted_visible = np.zeros(4, dtype=bool)
        else:
            keypoint_logits = record["keypoint_logits"]
            instance_score = record["instance_score"]
            if record["auxiliary_visibility_confidence"] is not None:
                probabilities = np.clip(
                    record["auxiliary_visibility_confidence"], 1.0e-6, 1.0 - 1.0e-6
                )
                keypoint_logits = np.log(probabilities / (1.0 - probabilities))
                instance_score = 1.0
            predicted_visible, _ = _decode_keypoint_visibility(
                record["corners_uv"],
                image_width=record["image_width"],
                image_height=record["image_height"],
                keypoint_logits=keypoint_logits,
                instance_score=instance_score,
                confidence_threshold=threshold,
                min_quad_area_px2=min_quad_area_px2,
            )
        predicted_complete = bool(np.all(predicted_visible))
        accepted_complete_truth += truth_complete and predicted_complete
        accepted_partial_truth += (not truth_complete) and predicted_complete
        visible_true_positive += int(np.sum(predicted_visible & truth_visible))
        visible_false_positive += int(np.sum(predicted_visible & ~truth_visible))
        visible_false_negative += int(np.sum(~predicted_visible & truth_visible))
        if truth_complete and predicted_complete:
            residual = record["corners_uv"] - record["truth_corners_uv"]
            radial_errors.extend(np.linalg.norm(residual, axis=1).tolist())
            signed_errors.append(residual)

    precision_denominator = visible_true_positive + visible_false_positive
    recall_denominator = visible_true_positive + visible_false_negative
    visibility_precision = visible_true_positive / max(precision_denominator, 1)
    visibility_recall = visible_true_positive / max(recall_denominator, 1)
    return {
        "keypoint_confidence_threshold": threshold,
        "complete_truth_samples": complete_truth,
        "partial_truth_samples": partial_truth,
        "complete_recall": accepted_complete_truth / max(complete_truth, 1),
        "partial_false_accept_rate": accepted_partial_truth / max(partial_truth, 1),
        "accepted_complete_truth_samples": accepted_complete_truth,
        "accepted_partial_truth_samples": accepted_partial_truth,
        "visibility_precision": visibility_precision,
        "visibility_recall": visibility_recall,
        "visibility_f1": 2.0
        * visibility_precision
        * visibility_recall
        / max(visibility_precision + visibility_recall, 1.0e-12),
        "accepted_corner_error_px_mean": None if not radial_errors else float(np.mean(radial_errors)),
        "accepted_corner_error_px_rmse": (
            None if not radial_errors else float(np.sqrt(np.mean(np.square(radial_errors))))
        ),
        "accepted_corner_error_px_p95": _percentile(radial_errors, 95),
        "signed_errors": signed_errors,
    }


def _corner_noise_calibration(signed_errors: list[np.ndarray]) -> dict:
    residuals = np.asarray(signed_errors, dtype=np.float64).reshape(-1, 4, 2)
    bias = np.median(residuals, axis=0)
    centered = (residuals - bias).reshape(-1)
    absolute = np.abs(centered)
    robust_sigma = float(1.4826 * np.median(absolute))
    cutoff = float(np.percentile(absolute, 95))
    trimmed = centered[absolute <= cutoff]
    return {
        "samples": int(residuals.shape[0]),
        "median_signed_bias_px_by_corner_xy": bias.tolist(),
        "isotropic_robust_sigma_px": robust_sigma,
        "central_95pct_component_std_px": float(np.std(trimmed, ddof=1)),
        "absolute_component_p95_px": cutoff,
        "recommended_corner_sigma_px": robust_sigma,
        "method": "1.4826 * MAD after removing per-corner x/y median bias",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--visibility_checkpoint",
        type=Path,
        help="Optional compact detector checkpoint used only for learned visibility gating.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--detection_threshold", type=float, default=0.5)
    parser.add_argument("--min_quad_area_px2", type=float, default=16.0)
    parser.add_argument("--max_partial_false_accept_rate", type=float, default=0.01)
    args = parser.parse_args()

    dataset = TorchvisionStage2KeypointDataset(args.dataset)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=_collate,
    )
    detector = TorchvisionGateCornerDetector(args.checkpoint, device=args.device)
    visibility_detector = (
        None
        if args.visibility_checkpoint is None
        else TorchGateCornerDetector(
            args.visibility_checkpoint,
            device=args.device,
            visibility_threshold=0.0,
        )
    )
    records = []
    detector.model.eval()
    with torch.inference_mode():
        for images, targets, sample_ids in loader:
            outputs = detector.model([image.to(detector.device) for image in images])
            for image, target, sample_id, output in zip(images, targets, sample_ids, outputs):
                auxiliary_visibility_confidence = None
                if visibility_detector is not None:
                    rgb = image.mul(255.0).byte().permute(1, 2, 0).numpy()
                    auxiliary_visibility_confidence = visibility_detector.detect(rgb).confidence
                if len(output["scores"]):
                    instance_score = float(output["scores"][0].detach().cpu())
                    corners_uv = output["keypoints"][0, :, :2].detach().cpu().numpy()
                    scores = output.get("keypoints_scores")
                    logits = None if scores is None else scores[0].detach().cpu().numpy()
                else:
                    instance_score = 0.0
                    corners_uv = np.zeros((4, 2), dtype=np.float64)
                    logits = None
                records.append(
                    {
                        "sample_id": sample_id,
                        "instance_score": instance_score,
                        "detection_threshold": args.detection_threshold,
                        "corners_uv": corners_uv,
                        "keypoint_logits": logits,
                        "auxiliary_visibility_confidence": auxiliary_visibility_confidence,
                        "truth_corners_uv": target["keypoints"][0, :, :2].numpy(),
                        "truth_visible": target["keypoints"][0, :, 2].numpy() > 0.0,
                        "image_width": int(image.shape[2]),
                        "image_height": int(image.shape[1]),
                    }
                )

    thresholds = [float(value) for value in np.linspace(0.05, 0.95, 19)]
    metrics = [_threshold_metrics(records, value, args.min_quad_area_px2) for value in thresholds]
    eligible = [
        item
        for item in metrics
        if item["partial_false_accept_rate"] <= args.max_partial_false_accept_rate
        and item["accepted_complete_truth_samples"] > 0
    ]
    candidates = eligible or [item for item in metrics if item["accepted_complete_truth_samples"] > 0]
    if not candidates:
        raise RuntimeError("No confidence threshold accepted any complete validation sample")
    selected = max(
        candidates,
        key=lambda item: (
            -item["partial_false_accept_rate"] if not eligible else item["complete_recall"],
            item["complete_recall"],
            -float(item["accepted_corner_error_px_p95"]),
        ),
    )
    noise = _corner_noise_calibration(selected["signed_errors"])
    selected_report = {key: value for key, value in selected.items() if key != "signed_errors"}
    metrics_report = [
        {key: value for key, value in item.items() if key != "signed_errors"}
        for item in metrics
    ]

    report = {
        "schema": "isaac_drone_racer.swift_detector_calibration.v1",
        "dataset": str(args.dataset),
        "checkpoint": str(args.checkpoint),
        "visibility_checkpoint": (
            None
            if args.visibility_checkpoint is None
            else str(args.visibility_checkpoint)
        ),
        "detection_threshold": args.detection_threshold,
        "min_quad_area_px2": args.min_quad_area_px2,
        "selection_constraint": {
            "max_partial_false_accept_rate": args.max_partial_false_accept_rate,
            "constraint_satisfied": bool(eligible),
        },
        "selected": selected_report,
        "corner_noise_calibration": noise,
        "threshold_metrics": metrics_report,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "selected": selected_report, "corner_noise_calibration": noise}, indent=2))


if __name__ == "__main__":
    main()
