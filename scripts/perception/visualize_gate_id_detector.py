"""Render a detector audit video with Gate ID, score, corners and GT match.

The video is intentionally diagnostic rather than a training input. Predictions
are drawn with their predicted Gate-ID palette color. Geometry-matched GT
instances are annotated so identity mistakes are immediately visible.
"""

from __future__ import annotations

import argparse
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
        "--output",
        type=Path,
        default=Path(
            "artifacts/racing_vision/"
            "circular12_color20_h207_gateid_kprcnn_v1/"
            "gate_id_diagnostics/test/detector_audit.mp4"
        ),
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--detection-threshold", type=float, default=0.35)
    parser.add_argument("--keypoint-confidence-threshold", type=float, default=0.35)
    parser.add_argument("--match-rmse-px", type=float, default=35.0)
    parser.add_argument("--min-visible-corners", type=int, default=2)
    parser.add_argument("--max-instances", type=int, default=10)
    parser.add_argument("--fps", type=float, default=25.0)
    parser.add_argument(
        "--run-index",
        type=int,
        default=None,
        help="Render only one dataset run index. Recommended for temporal review.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="0 means all selected frames.",
    )
    return parser.parse_args()


def _score_from_source(source: str) -> float | None:
    marker = "score="
    if marker not in source:
        return None
    try:
        token = source.split(marker, 1)[1].split(":", 1)[0]
        return float(token)
    except ValueError:
        return None


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
    matches = []
    for rmse, pred_index, gt_index, common in candidates:
        if pred_index in used_pred or gt_index in used_gt:
            continue
        used_pred.add(pred_index)
        used_gt.add(gt_index)
        matches.append((pred_index, gt_index, rmse, common))
    return matches


def _gate_bgr(gate_id: int | None) -> tuple[int, int, int]:
    if gate_id is None or not 1 <= int(gate_id) <= 12:
        return (220, 220, 220)
    rgb = identity_for_gate_id(int(gate_id)).rgb
    return (int(rgb[2]), int(rgb[1]), int(rgb[0]))


def _draw_poly(
    frame: np.ndarray,
    corners: np.ndarray,
    visible: np.ndarray,
    color: tuple[int, int, int],
    *,
    thickness: int,
) -> None:
    corners = np.asarray(corners, dtype=np.float64).reshape(4, 2)
    visible = np.asarray(visible, dtype=bool).reshape(4)
    for idx in range(4):
        if visible[idx]:
            point = tuple(int(round(v)) for v in corners[idx])
            cv2.circle(frame, point, 4, color, -1, lineType=cv2.LINE_AA)
    if bool(np.all(visible)):
        contour = np.round(corners).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(
            frame,
            [contour],
            True,
            color,
            thickness,
            lineType=cv2.LINE_AA,
        )


def _select_sample_ids(split_root: Path, run_index: int | None) -> list[str]:
    label_paths = sorted((split_root / "labels").glob("*.json"))
    selected = []
    for path in label_paths:
        if run_index is None:
            selected.append(path.stem)
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        if int(payload.get("run_index", -1)) == int(run_index):
            selected.append(path.stem)
    if not selected:
        raise ValueError(
            f"no frames selected for split={split_root.name!r}, "
            f"run_index={run_index!r}"
        )
    return selected


def main() -> None:
    args = _parse_args()
    split_root = args.dataset_root.expanduser().resolve() / "vision" / args.split
    sample_ids = _select_sample_ids(split_root, args.run_index)
    if args.max_frames > 0:
        sample_ids = sample_ids[: int(args.max_frames)]

    dataset = RacingMultiGateKeypointDataset(
        split_root,
        sample_ids_override=sample_ids,
        min_visible_corners=int(args.min_visible_corners),
    )
    detector = TorchvisionGateCornerDetector(
        args.checkpoint.expanduser().resolve(),
        device=args.device,
        detection_threshold=float(args.detection_threshold),
        keypoint_confidence_threshold=float(args.keypoint_confidence_threshold),
        max_instances=int(args.max_instances),
    )

    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(args.fps),
        (256, 256),
    )
    if not writer.isOpened():
        raise RuntimeError(f"failed to open video writer: {output}")

    total_matches = 0
    correct_ids = 0
    wrong_ids = 0

    try:
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
            frame = cv2.cvtColor(np.ascontiguousarray(rgb), cv2.COLOR_RGB2BGR)
            predictions = detector.detect_all(rgb, timestamp_s=float(index))
            usable = [
                obs for obs in predictions
                if int(np.asarray(obs.visible, dtype=bool).sum())
                >= int(args.min_visible_corners)
            ]
            matches = _match_predictions(
                usable,
                info.corners_uv,
                info.visible_masks,
                max_rmse_px=float(args.match_rmse_px),
            )
            match_by_pred = {
                pred_idx: (gt_idx, rmse)
                for pred_idx, gt_idx, rmse, _ in matches
            }

            # Thin GT overlays first.
            for gt_index, gate_index in enumerate(info.gate_indices):
                truth_gate_id = int(gate_index) + 1
                _draw_poly(
                    frame,
                    info.corners_uv[gt_index],
                    info.visible_masks[gt_index],
                    _gate_bgr(truth_gate_id),
                    thickness=1,
                )

            for pred_index, observation in enumerate(usable):
                predicted_gate_id = observation.gate_id
                color = _gate_bgr(predicted_gate_id)
                _draw_poly(
                    frame,
                    observation.corners_uv,
                    observation.visible,
                    color,
                    thickness=2,
                )

                visible_points = np.asarray(observation.corners_uv)[
                    np.asarray(observation.visible, dtype=bool)
                ]
                if len(visible_points):
                    anchor = visible_points.mean(axis=0)
                else:
                    anchor = np.array([6.0, 18.0 + 16.0 * pred_index])
                x = int(np.clip(round(anchor[0]) + 4, 0, 180))
                y = int(np.clip(round(anchor[1]) - 5, 12, 246))
                score = _score_from_source(observation.source)
                pred_text = (
                    "G??"
                    if predicted_gate_id is None
                    else f"G{int(predicted_gate_id):02d}"
                )
                score_text = "" if score is None else f" s={score:.2f}"

                if pred_index in match_by_pred:
                    gt_index, rmse = match_by_pred[pred_index]
                    truth_gate_id = int(info.gate_indices[gt_index]) + 1
                    correct = (
                        predicted_gate_id is not None
                        and int(predicted_gate_id) == truth_gate_id
                    )
                    total_matches += 1
                    correct_ids += int(correct)
                    wrong_ids += int(not correct)
                    status = "OK" if correct else "ID!"
                    text = (
                        f"P:{pred_text}{score_text} "
                        f"T:G{truth_gate_id:02d} {status} {rmse:.1f}px"
                    )
                    text_color = (40, 230, 40) if correct else (40, 40, 255)
                else:
                    text = f"P:{pred_text}{score_text} unmatched"
                    text_color = (0, 165, 255)

                cv2.putText(
                    frame,
                    text,
                    (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.38,
                    text_color,
                    1,
                    cv2.LINE_AA,
                )

            header = (
                f"{args.split} {info.sample_id} "
                f"pred={len(usable)} gt={len(info.gate_indices)}"
            )
            cv2.rectangle(frame, (0, 0), (255, 20), (0, 0, 0), -1)
            cv2.putText(
                frame,
                header[:52],
                (4, 14),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.36,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
            writer.write(frame)

            if (index + 1) % 100 == 0 or index + 1 == len(dataset):
                print(
                    f"[gate-id-video] {index + 1}/{len(dataset)} "
                    f"matches={total_matches} correct_id={correct_ids} "
                    f"wrong_id={wrong_ids}",
                    flush=True,
                )
    finally:
        writer.release()

    accuracy = correct_ids / max(total_matches, 1)
    print("=" * 88)
    print("GATE-ID DETECTOR AUDIT VIDEO COMPLETE")
    print("=" * 88)
    print(
        f"frames={len(dataset)} matches={total_matches} "
        f"id_accuracy={accuracy:.4f} output={output}",
        flush=True,
    )


if __name__ == "__main__":
    main()
