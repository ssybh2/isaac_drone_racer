"""Evaluate Circular-12 Gate-ID confusion independently of corner geometry.

The detector is matched to GT instances using only semantic-corner geometry.
Gate identity is evaluated *after* that geometry match, so the resulting 12x12
matrix answers the intended question: once the network found the correct
physical gate, which Gate ID / texture did it call it?

Outputs:
  * confusion_counts.csv
  * confusion_row_normalized.csv
  * confusion_matrix.png
  * confusion_report.json
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np

from perception.gate_identity import identity_for_gate_id
from perception.racing_multigate_dataset import RacingMultiGateKeypointDataset
from perception.torchvision_keypoint_detector import TorchvisionGateCornerDetector


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path(
            "artifacts/racing_vision/"
            "circular12_color20_h207_diversified_v1"
        ),
    )
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(
            "artifacts/racing_vision/"
            "circular12_color20_h207_gateid_kprcnn_v1/"
            "torchvision_keypointrcnn_multigate_best.pt"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "artifacts/racing_vision/"
            "circular12_color20_h207_gateid_kprcnn_v1/"
            "gate_id_diagnostics/test"
        ),
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--detection-threshold", type=float, default=0.35)
    parser.add_argument("--keypoint-confidence-threshold", type=float, default=0.35)
    parser.add_argument("--match-rmse-px", type=float, default=35.0)
    parser.add_argument("--min-visible-corners", type=int, default=2)
    parser.add_argument("--max-instances", type=int, default=10)
    return parser.parse_args()


def _match_predictions(
    predictions,
    gt_corners: np.ndarray,
    gt_visible: np.ndarray,
    *,
    max_rmse_px: float,
) -> list[tuple[int, int, float, np.ndarray]]:
    candidates: list[tuple[float, int, int, np.ndarray]] = []
    for pred_index, observation in enumerate(predictions):
        pred_visible = np.asarray(observation.visible, dtype=bool).reshape(4)
        pred_corners = np.asarray(observation.corners_uv, dtype=np.float64).reshape(4, 2)
        if int(pred_visible.sum()) < 2:
            continue
        for gt_index in range(len(gt_corners)):
            common = pred_visible & np.asarray(gt_visible[gt_index], dtype=bool)
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


def _write_matrix_csv(path: Path, matrix: np.ndarray, *, normalized: bool) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["truth\\pred"] + [f"G{i:02d}" for i in range(1, 13)])
        for truth_id in range(1, 13):
            row = matrix[truth_id - 1]
            if normalized:
                values = [f"{float(value):.6f}" for value in row]
            else:
                values = [int(value) for value in row]
            writer.writerow([f"G{truth_id:02d}"] + values)


def _render_confusion_png(
    path: Path,
    counts: np.ndarray,
    row_normalized: np.ndarray,
) -> None:
    cell = 74
    left = 150
    top = 120
    width = left + 12 * cell + 30
    height = top + 12 * cell + 90
    canvas = np.full((height, width, 3), 245, dtype=np.uint8)

    cv2.putText(
        canvas,
        "Circular-12 Gate-ID Confusion (row = truth, column = prediction)",
        (24, 42),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        (20, 20, 20),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        "cell text: count / row percentage",
        (24, 76),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (70, 70, 70),
        1,
        cv2.LINE_AA,
    )

    for gate_id in range(1, 13):
        identity = identity_for_gate_id(gate_id)
        label = f"G{gate_id:02d}"
        x = left + (gate_id - 1) * cell
        cv2.putText(
            canvas,
            label,
            (x + 12, top - 46),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (20, 20, 20),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            identity.color_name[:9],
            (x + 2, top - 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.32,
            (70, 70, 70),
            1,
            cv2.LINE_AA,
        )

        y = top + (gate_id - 1) * cell
        cv2.putText(
            canvas,
            label,
            (20, y + 27),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (20, 20, 20),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            identity.color_name[:12],
            (62, y + 27),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.32,
            (70, 70, 70),
            1,
            cv2.LINE_AA,
        )

    for truth in range(12):
        for pred in range(12):
            fraction = float(row_normalized[truth, pred])
            intensity = int(round(255.0 * (1.0 - min(fraction, 1.0))))
            if truth == pred:
                fill = (intensity, 255, intensity)
            else:
                fill = (255, intensity, intensity)
            x1 = left + pred * cell
            y1 = top + truth * cell
            x2 = x1 + cell - 2
            y2 = y1 + cell - 2
            cv2.rectangle(canvas, (x1, y1), (x2, y2), fill, -1)
            cv2.rectangle(canvas, (x1, y1), (x2, y2), (190, 190, 190), 1)

            count = int(counts[truth, pred])
            pct = 100.0 * fraction
            cv2.putText(
                canvas,
                str(count),
                (x1 + 8, y1 + 29),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.46,
                (15, 15, 15),
                1,
                cv2.LINE_AA,
            )
            cv2.putText(
                canvas,
                f"{pct:.0f}%",
                (x1 + 8, y1 + 54),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.39,
                (40, 40, 40),
                1,
                cv2.LINE_AA,
            )

    if not cv2.imwrite(str(path), canvas):
        raise RuntimeError(f"failed to write confusion image: {path}")


def main() -> None:
    args = _parse_args()
    split_root = args.dataset_root.expanduser().resolve() / "vision" / args.split
    checkpoint = args.checkpoint.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = RacingMultiGateKeypointDataset(
        split_root,
        min_visible_corners=int(args.min_visible_corners),
    )
    detector = TorchvisionGateCornerDetector(
        checkpoint,
        device=args.device,
        detection_threshold=float(args.detection_threshold),
        keypoint_confidence_threshold=float(args.keypoint_confidence_threshold),
        max_instances=int(args.max_instances),
    )
    if detector.gate_identity_mode != "gate_id_class":
        raise RuntimeError(
            "checkpoint is not a Gate-ID classifier: "
            f"gate_identity_mode={detector.gate_identity_mode!r}"
        )

    counts = np.zeros((12, 12), dtype=np.int64)
    matched_geometry = 0
    matched_with_id = 0
    correct_id = 0
    total_gt = 0
    total_predictions = 0
    corner_sq_sum = 0.0
    corner_count = 0

    for index in range(len(dataset)):
        image, _, info = dataset[index]
        rgb = (
            image.mul(255.0)
            .clamp(0.0, 255.0)
            .byte()
            .permute(1, 2, 0)
            .cpu()
            .numpy()
        )
        predictions = detector.detect_all(rgb, timestamp_s=float(index))
        usable_predictions = [
            obs for obs in predictions
            if int(np.asarray(obs.visible, dtype=bool).sum())
            >= int(args.min_visible_corners)
        ]
        matches = _match_predictions(
            usable_predictions,
            info.corners_uv,
            info.visible_masks,
            max_rmse_px=float(args.match_rmse_px),
        )

        total_gt += len(info.gate_indices)
        total_predictions += len(usable_predictions)
        matched_geometry += len(matches)

        for pred_index, gt_index, _, common in matches:
            observation = usable_predictions[pred_index]
            truth_gate_id = int(info.gate_indices[gt_index]) + 1
            predicted_gate_id = observation.gate_id

            residual = (
                np.asarray(observation.corners_uv, dtype=np.float64)[common]
                - info.corners_uv[gt_index, common]
            )
            corner_sq_sum += float(np.sum(residual * residual))
            corner_count += int(common.sum())

            if predicted_gate_id is None:
                continue
            predicted_gate_id = int(predicted_gate_id)
            if not 1 <= predicted_gate_id <= 12:
                continue
            counts[truth_gate_id - 1, predicted_gate_id - 1] += 1
            matched_with_id += 1
            correct_id += int(predicted_gate_id == truth_gate_id)

        if (index + 1) % 100 == 0 or index + 1 == len(dataset):
            print(
                f"[gate-id-confusion] {index + 1}/{len(dataset)} "
                f"geom_matches={matched_geometry} id_matches={matched_with_id}",
                flush=True,
            )

    row_totals = counts.sum(axis=1, keepdims=True)
    row_normalized = np.divide(
        counts,
        np.maximum(row_totals, 1),
        dtype=np.float64,
    )
    per_gate = []
    for gate_id in range(1, 13):
        identity = identity_for_gate_id(gate_id)
        row = counts[gate_id - 1]
        total = int(row.sum())
        correct = int(row[gate_id - 1])
        wrong = row.copy()
        wrong[gate_id - 1] = 0
        confused_with = int(np.argmax(wrong)) + 1 if int(wrong.sum()) else None
        per_gate.append(
            {
                "gate_id": gate_id,
                "color_name": identity.color_name,
                "matched_instances": total,
                "correct": correct,
                "accuracy": float(correct / max(total, 1)),
                "most_confused_with_gate_id": confused_with,
                "most_confused_with_count": (
                    0 if confused_with is None else int(row[confused_with - 1])
                ),
            }
        )

    report = {
        "schema": "isaac_drone_racer.gate_id_confusion.v1",
        "dataset_root": str(args.dataset_root.expanduser().resolve()),
        "split": args.split,
        "checkpoint": str(checkpoint),
        "thresholds": {
            "detection": float(args.detection_threshold),
            "keypoint_confidence": float(args.keypoint_confidence_threshold),
            "match_rmse_px": float(args.match_rmse_px),
            "min_visible_corners": int(args.min_visible_corners),
        },
        "gt_instances": int(total_gt),
        "predicted_usable_instances": int(total_predictions),
        "geometry_matched_instances": int(matched_geometry),
        "geometry_recall": float(matched_geometry / max(total_gt, 1)),
        "geometry_precision": float(matched_geometry / max(total_predictions, 1)),
        "gate_id_evaluated_matches": int(matched_with_id),
        "gate_id_correct_matches": int(correct_id),
        "gate_id_accuracy": float(correct_id / max(matched_with_id, 1)),
        "corner_coordinate_rmse_px": float(
            np.sqrt(corner_sq_sum / max(2 * corner_count, 1))
        ),
        "per_gate": per_gate,
        "confusion_counts": counts.tolist(),
        "confusion_row_normalized": row_normalized.tolist(),
        "note": (
            "GT/prediction instance matching uses corner geometry only; "
            "predicted Gate ID is scored after the geometry match."
        ),
    }

    _write_matrix_csv(
        output_dir / "confusion_counts.csv",
        counts,
        normalized=False,
    )
    _write_matrix_csv(
        output_dir / "confusion_row_normalized.csv",
        row_normalized,
        normalized=True,
    )
    _render_confusion_png(
        output_dir / "confusion_matrix.png",
        counts,
        row_normalized,
    )
    (output_dir / "confusion_report.json").write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )

    print("=" * 96)
    print("CIRCULAR-12 GATE-ID CONFUSION COMPLETE")
    print("=" * 96)
    print(
        f"gate_id_accuracy={report['gate_id_accuracy']:.4f} "
        f"geometry_recall={report['geometry_recall']:.4f} "
        f"geometry_precision={report['geometry_precision']:.4f} "
        f"corner_rmse={report['corner_coordinate_rmse_px']:.3f}px"
    )
    for item in per_gate:
        confusion = (
            "none"
            if item["most_confused_with_gate_id"] is None
            else (
                f"G{item['most_confused_with_gate_id']:02d}"
                f" x{item['most_confused_with_count']}"
            )
        )
        print(
            f"G{item['gate_id']:02d} {item['color_name']:<13s} "
            f"acc={item['accuracy']:.4f} n={item['matched_instances']:4d} "
            f"top_confusion={confusion}",
            flush=True,
        )
    print(f"[gate-id-confusion] outputs: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
