"""Evaluate Stage2B corner and downstream PnP errors on a dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import platform
import random
import sys
import time

sys.path.insert(0, str(Path(__file__).parents[2]))

import cv2
import numpy as np
import torch

from perception.camera_model import CameraCalibration
from perception.corner_detection import CornerObservation
from perception.keypoint_detector import TorchGateCornerDetector
from perception.pose_recovery import GatePoseRecovery
from perception.rigid_transform import RigidTransform, rotation_error_rad, translation_error_m
from perception.stage2_calibration import load_stage2_gate_geometry, stage2_camera_to_body
from perception.torchvision_keypoint_detector import TorchvisionGateCornerDetector


DISTANCE_BINS_M = ((2.5, 4.0), (4.0, 5.5), (5.5, 7.0), (7.0, 8.0), (8.0, math.inf))
PROJECTED_WIDTH_BINS_PX = ((0.0, 64.0), (64.0, 96.0), (96.0, 128.0), (128.0, math.inf))


def transform_from_json(payload: dict) -> RigidTransform:
    matrix = np.asarray(payload["matrix"], dtype=np.float64)
    return RigidTransform(
        matrix[:3, :3],
        matrix[:3, 3],
        to_frame=payload["to_frame"],
        from_frame=payload["from_frame"],
    )


def percentile(values: list[float], q: float) -> float | None:
    return None if not values else float(np.percentile(values, q))


def mean(values: list[float]) -> float | None:
    return None if not values else float(np.mean(values))


def _metric_summary(records: list[dict]) -> dict:
    complete_records = [record for record in records if record["ground_truth_complete"]]
    pnp_records = [record for record in complete_records if record["pnp_success"]]
    corner_errors = [error for record in complete_records for error in record["corner_errors_px"]]
    gate_translation = [record["gate_translation_error_m"] for record in pnp_records]
    gate_rotation = [record["gate_rotation_error_deg"] for record in pnp_records]
    body_translation = [record["body_translation_error_m"] for record in pnp_records]
    body_rotation = [record["body_rotation_error_deg"] for record in pnp_records]
    predicted_complete = sum(record["predicted_complete"] for record in complete_records)

    return {
        "samples": len(records),
        "ground_truth_complete_samples": len(complete_records),
        "predicted_complete_samples": predicted_complete,
        "pnp_successful_samples": len(pnp_records),
        "complete_detection_rate": predicted_complete / max(len(complete_records), 1),
        "pnp_success_rate": len(pnp_records) / max(len(complete_records), 1),
        "corner_error_px_mean": mean(corner_errors),
        "corner_error_px_rmse": (
            None if not corner_errors else float(np.sqrt(np.mean(np.square(corner_errors))))
        ),
        "corner_error_px_p95": percentile(corner_errors, 95),
        "gate_translation_error_m_mean": mean(gate_translation),
        "gate_translation_error_m_p95": percentile(gate_translation, 95),
        "gate_rotation_error_deg_mean": mean(gate_rotation),
        "gate_rotation_error_deg_p95": percentile(gate_rotation, 95),
        "body_translation_error_m_mean": mean(body_translation),
        "body_translation_error_m_p95": percentile(body_translation, 95),
        "body_rotation_error_deg_mean": mean(body_rotation),
        "body_rotation_error_deg_p95": percentile(body_rotation, 95),
    }


def _binned_metrics(records: list[dict], field: str, bins: tuple[tuple[float, float], ...]) -> list[dict]:
    output = []
    for lower, upper in bins:
        selected = [
            record
            for record in records
            if record[field] >= lower and (record[field] < upper or math.isinf(upper))
        ]
        metrics = _metric_summary(selected)
        metrics["range"] = [lower, None if math.isinf(upper) else upper]
        metrics["range_semantics"] = "lower_inclusive_upper_exclusive"
        output.append(metrics)
    return output


def _projected_gate_width(corners_uv: np.ndarray) -> float:
    bottom = np.linalg.norm(corners_uv[1] - corners_uv[0])
    top = np.linalg.norm(corners_uv[2] - corners_uv[3])
    return float(0.5 * (bottom + top))


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _load_rgb(dataset: Path, sample_id: str) -> np.ndarray:
    bgr = cv2.imread(str(dataset / "images" / f"{sample_id}.png"), cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError(f"Unable to read image {sample_id}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--detector",
        choices=("gate-keypoint-net", "torchvision-keypoint-rcnn"),
        default="gate-keypoint-net",
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--subset",
        choices=("all", "shuffled-validation"),
        default="shuffled-validation",
        help="Use every sample for an independent dataset, or reproduce the development split.",
    )
    parser.add_argument("--validation_fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--warmup_runs", type=int, default=10)
    args = parser.parse_args()

    all_ids = sorted(path.stem for path in (args.dataset / "labels").glob("*.json"))
    if not all_ids:
        raise ValueError(f"No labels found under {args.dataset}")
    ids = list(all_ids)
    if args.subset == "shuffled-validation":
        if not 0.0 < args.validation_fraction <= 1.0:
            raise ValueError("validation_fraction must be in (0, 1]")
        random.Random(args.seed).shuffle(ids)
        validation_count = max(1, int(round(len(ids) * args.validation_fraction)))
        ids = ids[:validation_count]

    if args.detector == "gate-keypoint-net":
        detector = TorchGateCornerDetector(args.checkpoint, device=args.device)
    else:
        detector = TorchvisionGateCornerDetector(args.checkpoint, device=args.device)
    device = detector.device
    geometry = load_stage2_gate_geometry()
    T_bc = stage2_camera_to_body()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    warmup_rgb = _load_rgb(args.dataset, ids[0])
    for _ in range(max(0, args.warmup_runs)):
        detector.detect(warmup_rgb)
    _synchronize(device)

    records = []
    inference_times_ms = []
    for sample_id in ids:
        label = json.loads((args.dataset / "labels" / f"{sample_id}.json").read_text(encoding="utf-8"))
        rgb = _load_rgb(args.dataset, sample_id)
        _synchronize(device)
        started = time.perf_counter()
        detected = detector.detect(rgb)
        _synchronize(device)
        inference_times_ms.append((time.perf_counter() - started) * 1000.0)

        oracle = CornerObservation(label["corners_uv"], visible=label["visible"], source="dataset_oracle")
        camera_payload = label["camera"]
        camera = CameraCalibration(
            K=camera_payload["K"],
            image_width=camera_payload["image_width"],
            image_height=camera_payload["image_height"],
            distortion=camera_payload["distortion"],
            model=camera_payload["model"],
        )
        T_wg = transform_from_json(label["truth"]["T_wg"])
        T_wc = transform_from_json(label["truth"]["T_wc"])
        T_wb = transform_from_json(label["truth"]["T_wb"])
        T_cg_truth = T_wc.inverse() @ T_wg
        opening_center_c = T_cg_truth.transform_points(geometry.center_g)
        record = {
            "sample_id": sample_id,
            "distance_m": float(np.linalg.norm(opening_center_c)),
            "projected_gate_width_px": _projected_gate_width(oracle.corners_uv),
            "ground_truth_complete": oracle.complete,
            "predicted_complete": bool(detected.complete),
            "pnp_success": False,
            "corner_errors_px": [],
        }
        if not oracle.complete:
            records.append(record)
            continue

        record["corner_errors_px"] = np.linalg.norm(
            detected.corners_uv - oracle.corners_uv, axis=1
        ).tolist()
        if not detected.complete:
            records.append(record)
            continue

        try:
            solution = GatePoseRecovery(geometry, camera, T_bc).recover(detected, T_wg=T_wg)
        except (RuntimeError, ValueError):
            records.append(record)
            continue
        record.update(
            {
                "pnp_success": True,
                "gate_translation_error_m": translation_error_m(solution.T_cg, T_cg_truth),
                "gate_rotation_error_deg": float(
                    np.degrees(rotation_error_rad(solution.T_cg, T_cg_truth))
                ),
                "body_translation_error_m": translation_error_m(solution.T_wb_est, T_wb),
                "body_rotation_error_deg": float(
                    np.degrees(rotation_error_rad(solution.T_wb_est, T_wb))
                ),
            }
        )
        records.append(record)

    checkpoint_hash = hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
    manifest_path = args.dataset / "manifest.json"
    manifest_hash = hashlib.sha256(manifest_path.read_bytes()).hexdigest() if manifest_path.exists() else None
    aggregate = _metric_summary(records)
    summary = {
        "schema": "isaac_drone_racer.stage2b_evaluation.v2",
        "dataset": str(args.dataset.resolve()),
        "dataset_manifest": str(manifest_path.resolve()) if manifest_path.exists() else None,
        "dataset_manifest_sha256": manifest_hash,
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": checkpoint_hash,
        "detector": args.detector,
        "subset": args.subset,
        "split_seed": args.seed if args.subset == "shuffled-validation" else None,
        "validation_fraction": args.validation_fraction if args.subset == "shuffled-validation" else None,
        "evaluated_samples": len(ids),
        **aggregate,
        "distance_bins_m": _binned_metrics(records, "distance_m", DISTANCE_BINS_M),
        "projected_gate_width_bins_px": _binned_metrics(
            records, "projected_gate_width_px", PROJECTED_WIDTH_BINS_PX
        ),
        "inference": {
            "latency_ms_mean": mean(inference_times_ms),
            "latency_ms_p50": percentile(inference_times_ms, 50),
            "latency_ms_p95": percentile(inference_times_ms, 95),
            "warmup_runs": max(0, args.warmup_runs),
            "device": str(device),
            "hardware": torch.cuda.get_device_name(device) if device.type == "cuda" else platform.processor(),
            "peak_cuda_memory_mib": (
                float(torch.cuda.max_memory_allocated(device) / (1024**2)) if device.type == "cuda" else None
            ),
            "checkpoint_size_mib": float(args.checkpoint.stat().st_size / (1024**2)),
            "model_parameters": sum(parameter.numel() for parameter in detector.model.parameters()),
        },
    }
    text = json.dumps(summary, indent=2) + "\n"
    print(text, end="")
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
