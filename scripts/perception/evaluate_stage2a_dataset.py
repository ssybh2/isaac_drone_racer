"""Verify oracle projection -> PnP equivalence over a Stage2 dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parents[2]))

import numpy as np

from perception.camera_model import CameraCalibration
from perception.corner_detection import CornerObservation
from perception.pose_recovery import GatePoseRecovery
from perception.rigid_transform import RigidTransform, rotation_error_rad, translation_error_m
from perception.stage2_calibration import load_stage2_gate_geometry, stage2_camera_to_body


def transform_from_json(payload: dict) -> RigidTransform:
    matrix = np.asarray(payload["matrix"], dtype=np.float64)
    return RigidTransform(
        matrix[:3, :3], matrix[:3, 3], to_frame=payload["to_frame"], from_frame=payload["from_frame"]
    )


def stats(values: list[float], scale: float = 1.0) -> dict[str, float]:
    array = np.asarray(values) * scale
    return {"mean": float(array.mean()), "p95": float(np.percentile(array, 95)), "max": float(array.max())}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    geometry = load_stage2_gate_geometry()
    T_bc = stage2_camera_to_body()
    reprojection, gate_t, gate_r, body_t, body_r = [], [], [], [], []
    total = 0

    for label_path in sorted((args.dataset / "labels").glob("*.json")):
        total += 1
        label = json.loads(label_path.read_text(encoding="utf-8"))
        observation = CornerObservation(label["corners_uv"], visible=label["visible"])
        if not observation.complete:
            continue
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
        result = GatePoseRecovery(geometry, camera, T_bc).recover(observation, T_wg=T_wg)
        T_cg_truth = T_wc.inverse() @ T_wg
        reprojection.append(result.pnp.reprojection_rmse_px)
        gate_t.append(translation_error_m(result.T_cg, T_cg_truth))
        gate_r.append(rotation_error_rad(result.T_cg, T_cg_truth))
        body_t.append(translation_error_m(result.T_wb_est, T_wb))
        body_r.append(rotation_error_rad(result.T_wb_est, T_wb))

    summary = {
        "schema": "isaac_drone_racer.stage2a_dataset_equivalence.v1",
        "dataset": str(args.dataset.resolve()),
        "total_samples": total,
        "complete_samples": len(reprojection),
        "reprojection_error_px": stats(reprojection),
        "gate_translation_error_m": stats(gate_t),
        "gate_rotation_error_deg": stats(gate_r, 180.0 / np.pi),
        "body_translation_error_m": stats(body_t),
        "body_rotation_error_deg": stats(body_r, 180.0 / np.pi),
    }
    text = json.dumps(summary, indent=2) + "\n"
    print(text, end="")
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
