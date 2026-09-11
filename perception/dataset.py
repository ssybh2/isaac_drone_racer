"""Dataset writer for Stage2B RGB + oracle keypoint supervision."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np

from .camera_model import CameraCalibration
from .corner_detection import CornerObservation
from .rigid_transform import RigidTransform


def _transform_json(T: RigidTransform | None):
    if T is None:
        return None
    return {
        "to_frame": T.to_frame,
        "from_frame": T.from_frame,
        "matrix": T.as_matrix().tolist(),
    }


@dataclass
class Stage2DatasetWriter:
    """Write deterministic labels that can train and audit a Stage2B detector."""

    root: Path | str

    def __post_init__(self) -> None:
        self.root = Path(self.root)
        (self.root / "images").mkdir(parents=True, exist_ok=True)
        (self.root / "labels").mkdir(parents=True, exist_ok=True)

    def write_sample(
        self,
        sample_id: str,
        rgb: np.ndarray,
        corners: CornerObservation,
        camera: CameraCalibration,
        *,
        gate_index: int,
        T_wg: RigidTransform,
        T_wc: RigidTransform,
        T_wb: RigidTransform | None = None,
        extra: dict | None = None,
    ) -> tuple[Path, Path]:
        try:
            import cv2
        except ImportError as exc:  # pragma: no cover
            raise ImportError("OpenCV is required to write Stage2B RGB datasets") from exc

        image = np.asarray(rgb)
        if image.ndim != 3 or image.shape[-1] != 3:
            raise ValueError("rgb must have shape (H, W, 3)")
        image_path = self.root / "images" / f"{sample_id}.png"
        label_path = self.root / "labels" / f"{sample_id}.json"

        # Isaac returns RGB; OpenCV writers expect BGR.
        if not cv2.imwrite(str(image_path), image[..., ::-1]):
            raise RuntimeError(f"Failed to write {image_path}")

        payload = {
            "schema": "isaac_drone_racer.stage2_keypoints.v1",
            "sample_id": sample_id,
            "gate_index": int(gate_index),
            "corner_names": ["left_bottom", "right_bottom", "right_top", "left_top"],
            "corners_uv": corners.corners_uv.tolist(),
            "visible": corners.visible.tolist(),
            "confidence": corners.confidence.tolist(),
            "timestamp_s": corners.timestamp_s,
            "source": corners.source,
            "camera": {
                "K": camera.K.tolist(),
                "image_width": camera.image_width,
                "image_height": camera.image_height,
                "distortion": None if camera.dist_coeffs is None else camera.dist_coeffs.tolist(),
                "model": camera.model,
            },
            "truth": {
                "T_wg": _transform_json(T_wg),
                "T_wc": _transform_json(T_wc),
                "T_wb": _transform_json(T_wb),
            },
            "extra": extra or {},
        }
        label_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return image_path, label_path
