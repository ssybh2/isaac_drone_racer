"""Evaluate Stage2B corner and downstream PnP errors on a held-out split."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
import sys

sys.path.insert(0, str(Path(__file__).parents[2]))

import cv2
import numpy as np

from perception.camera_model import CameraCalibration
from perception.corner_detection import CornerObservation
from perception.gate_usd_config import load_gate_keypoint_calibration
from perception.keypoint_detector import TorchGateCornerDetector
from perception.pose_recovery import GatePoseRecovery
from perception.rigid_transform import RigidTransform, rotation_error_rad, translation_error_m


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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--validation_fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    ids = sorted(path.stem for path in (args.dataset / "labels").glob("*.json"))
    random.Random(args.seed).shuffle(ids)
    validation_count = max(1, int(round(len(ids) * args.validation_fraction)))
    ids = ids[:validation_count]
    detector = TorchGateCornerDetector(args.checkpoint, device=args.device)
    geometry = load_gate_keypoint_calibration("assets/gate/gate_keypoints.json")
    T_bc = RigidTransform(
        np.array([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]]),
        np.array([0.14, 0.0, 0.05]),
        to_frame="B",
        from_frame="C",
    )

    corner_errors = []
    gate_translation_errors = []
    gate_rotation_errors = []
    body_translation_errors = []
    body_rotation_errors = []
    gt_complete = 0
    predicted_complete = 0
    pnp_success = 0

    for sample_id in ids:
        label = json.loads((args.dataset / "labels" / f"{sample_id}.json").read_text(encoding="utf-8"))
        bgr = cv2.imread(str(args.dataset / "images" / f"{sample_id}.png"), cv2.IMREAD_COLOR)
        if bgr is None:
            raise RuntimeError(f"Unable to read image {sample_id}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        detected = detector.detect(rgb)
        oracle = CornerObservation(label["corners_uv"], visible=label["visible"], source="dataset_oracle")
        if not oracle.complete:
            continue
        gt_complete += 1
        corner_errors.extend(np.linalg.norm(detected.corners_uv - oracle.corners_uv, axis=1).tolist())
        if not detected.complete:
            continue
        predicted_complete += 1

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
        try:
            solution = GatePoseRecovery(geometry, camera, T_bc).recover(detected, T_wg=T_wg)
        except (RuntimeError, ValueError):
            continue
        pnp_success += 1
        gate_translation_errors.append(translation_error_m(solution.T_cg, T_cg_truth))
        gate_rotation_errors.append(rotation_error_rad(solution.T_cg, T_cg_truth))
        body_translation_errors.append(translation_error_m(solution.T_wb_est, T_wb))
        body_rotation_errors.append(rotation_error_rad(solution.T_wb_est, T_wb))

    checkpoint_hash = hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
    summary = {
        "schema": "isaac_drone_racer.stage2b_evaluation.v1",
        "dataset": str(args.dataset.resolve()),
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": checkpoint_hash,
        "validation_samples": len(ids),
        "ground_truth_complete_samples": gt_complete,
        "predicted_complete_samples": predicted_complete,
        "pnp_successful_samples": pnp_success,
        "complete_detection_rate": predicted_complete / max(gt_complete, 1),
        "pnp_success_rate": pnp_success / max(gt_complete, 1),
        "corner_error_px_mean": float(np.mean(corner_errors)) if corner_errors else None,
        "corner_error_px_rmse": float(np.sqrt(np.mean(np.square(corner_errors)))) if corner_errors else None,
        "corner_error_px_p95": percentile(corner_errors, 95),
        "gate_translation_error_m_mean": float(np.mean(gate_translation_errors)) if gate_translation_errors else None,
        "gate_translation_error_m_p95": percentile(gate_translation_errors, 95),
        "gate_rotation_error_deg_mean": (
            float(np.degrees(np.mean(gate_rotation_errors))) if gate_rotation_errors else None
        ),
        "gate_rotation_error_deg_p95": (
            None
            if not gate_rotation_errors
            else float(np.degrees(np.percentile(gate_rotation_errors, 95)))
        ),
        "body_translation_error_m_mean": float(np.mean(body_translation_errors)) if body_translation_errors else None,
        "body_translation_error_m_p95": percentile(body_translation_errors, 95),
        "body_rotation_error_deg_mean": (
            float(np.degrees(np.mean(body_rotation_errors))) if body_rotation_errors else None
        ),
        "body_rotation_error_deg_p95": (
            None
            if not body_rotation_errors
            else float(np.degrees(np.percentile(body_rotation_errors, 95)))
        ),
    }
    text = json.dumps(summary, indent=2) + "\n"
    print(text, end="")
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
