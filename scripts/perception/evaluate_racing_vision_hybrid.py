"""Evaluate production-style hybrid gate-corner availability on racing data."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from perception.hybrid_keypoint_detector import VisibilityGuardedGateCornerDetector
from perception.keypoint_detector import TorchGateCornerDetector
from perception.torchvision_keypoint_detector import TorchvisionGateCornerDetector


def _bucket(value: float, edges: tuple[float, ...]) -> str:
    lower = 0.0
    for upper in edges:
        if value < upper:
            return f"[{lower:g},{upper:g})"
        lower = upper
    return f"[{lower:g},inf)"


def _new_counter() -> dict:
    return {
        "frames": 0,
        "gt_ge2": 0,
        "coordinate_ge2": 0,
        "guard_ge2": 0,
        "hybrid_ge2": 0,
        "hybrid_ge2_given_gt_ge2": 0,
    }


def _update_counter(
    counter: dict,
    *,
    gt_count: int,
    coordinate_count: int,
    guard_count: int,
    hybrid_count: int,
) -> None:
    counter["frames"] += 1
    eligible = gt_count >= 2
    counter["gt_ge2"] += int(eligible)
    counter["coordinate_ge2"] += int(coordinate_count >= 2)
    counter["guard_ge2"] += int(guard_count >= 2)
    counter["hybrid_ge2"] += int(hybrid_count >= 2)
    counter["hybrid_ge2_given_gt_ge2"] += int(eligible and hybrid_count >= 2)


def _finalize(counter: dict) -> dict:
    result = dict(counter)
    result["hybrid_availability_given_gt_ge2"] = (
        counter["hybrid_ge2_given_gt_ge2"] / max(counter["gt_ge2"], 1)
    )
    result["coordinate_ge2_rate"] = (
        counter["coordinate_ge2"] / max(counter["frames"], 1)
    )
    result["guard_ge2_rate"] = counter["guard_ge2"] / max(counter["frames"], 1)
    result["hybrid_ge2_rate"] = counter["hybrid_ge2"] / max(counter["frames"], 1)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--coordinate-checkpoint", type=Path, required=True)
    parser.add_argument("--visibility-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--detection-threshold", type=float, default=0.5)
    parser.add_argument("--visibility-threshold", type=float, default=0.75)
    args = parser.parse_args()

    labels = sorted((args.dataset / "labels").glob("*.json"))
    if not labels:
        raise ValueError(f"No labels found under {args.dataset}")

    coordinate = TorchvisionGateCornerDetector(
        args.coordinate_checkpoint,
        device=args.device,
        detection_threshold=args.detection_threshold,
        # Production hybrid delegates corner visibility to the guard.
        keypoint_confidence_threshold=0.0,
    )
    guard = TorchGateCornerDetector(
        args.visibility_checkpoint,
        device=args.device,
        visibility_threshold=0.0,
    )
    hybrid = VisibilityGuardedGateCornerDetector(
        coordinate,
        guard,
        visibility_threshold=args.visibility_threshold,
    )

    overall = _new_counter()
    speed_buckets = defaultdict(_new_counter)
    rate_buckets = defaultdict(_new_counter)
    squared_error = 0.0
    corner_count = 0
    failures: list[dict] = []

    speed_edges = (3.0, 6.0, 9.0, 12.0, 15.0)
    rate_edges = (2.0, 4.0, 6.0, 8.0, 10.0)

    for index, label_path in enumerate(labels):
        payload = json.loads(label_path.read_text(encoding="utf-8"))
        image_path = args.dataset / "images" / f"{label_path.stem}.png"
        bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise RuntimeError(f"Unable to read {image_path}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        coordinate_obs = coordinate.detect(rgb, timestamp_s=float(index))
        guard_obs = guard.detect(rgb, timestamp_s=float(index))
        hybrid_obs = hybrid.detect(rgb, timestamp_s=float(index))

        gt_visible = np.asarray(payload["visible"], dtype=bool)
        gt_corners = np.asarray(payload["corners_uv"], dtype=np.float64)
        coordinate_count = int(np.asarray(coordinate_obs.visible, dtype=bool).sum())
        guard_count = int(
            (
                np.asarray(guard_obs.confidence, dtype=np.float64)
                >= float(args.visibility_threshold)
            ).sum()
        )
        hybrid_count = int(np.asarray(hybrid_obs.visible, dtype=bool).sum())
        gt_count = int(gt_visible.sum())

        extra = payload.get("extra") or {}
        speed = float(extra.get("speed_mps", 0.0))
        body_rate = float(extra.get("body_rate_norm_radps", 0.0))

        fields = {
            "gt_count": gt_count,
            "coordinate_count": coordinate_count,
            "guard_count": guard_count,
            "hybrid_count": hybrid_count,
        }
        _update_counter(overall, **fields)
        _update_counter(speed_buckets[_bucket(speed, speed_edges)], **fields)
        _update_counter(rate_buckets[_bucket(body_rate, rate_edges)], **fields)

        if gt_count > 0 and coordinate_count > 0:
            predicted = np.asarray(coordinate_obs.corners_uv, dtype=np.float64)
            squared_error += float(
                np.square(predicted[gt_visible] - gt_corners[gt_visible]).sum()
            )
            corner_count += gt_count

        if gt_count >= 2 and hybrid_count < 2:
            failures.append(
                {
                    "sample_id": label_path.stem,
                    "speed_mps": speed,
                    "body_rate_norm_radps": body_rate,
                    "gt_visible_corners": gt_count,
                    "coordinate_visible_corners": coordinate_count,
                    "guard_visible_corners": guard_count,
                    "hybrid_visible_corners": hybrid_count,
                    "guard_confidence": np.asarray(
                        guard_obs.confidence,
                        dtype=np.float64,
                    ).tolist(),
                }
            )

    summary = {
        "schema": "isaac_drone_racer.racing_vision_hybrid_eval.v1",
        "dataset": str(args.dataset.resolve()),
        "coordinate_checkpoint": str(args.coordinate_checkpoint.resolve()),
        "visibility_checkpoint": str(args.visibility_checkpoint.resolve()),
        "detection_threshold": float(args.detection_threshold),
        "visibility_threshold": float(args.visibility_threshold),
        "overall": _finalize(overall),
        "corner_rmse_px_on_gt_visible": float(
            np.sqrt(squared_error / max(corner_count, 1))
        ),
        "speed_buckets_mps": {
            key: _finalize(value) for key, value in sorted(speed_buckets.items())
        },
        "body_rate_buckets_radps": {
            key: _finalize(value) for key, value in sorted(rate_buckets.items())
        },
        "eligible_failures": failures,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    print("=" * 96)
    print("RACING VISION HYBRID EVALUATION")
    print("=" * 96)
    print(json.dumps({k: v for k, v in summary.items() if k != "eligible_failures"}, indent=2))
    print(f"[racing-vision] eligible failures: {len(failures)}")
    print(f"[racing-vision] wrote: {args.output}")


if __name__ == "__main__":
    main()
