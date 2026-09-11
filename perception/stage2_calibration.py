"""Authoritative Stage 2 gate and reference-camera calibration."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .gate_geometry import CORNER_NAMES, GateGeometry
from .gate_usd_config import load_gate_keypoint_calibration
from .rigid_transform import RigidTransform


PROJECT_ROOT = Path(__file__).resolve().parents[1]
GATE_KEYPOINT_CALIBRATION_PATH = PROJECT_ROOT / "assets" / "gate" / "gate_keypoints.json"

CAMERA_MODEL = "pinhole"
CAMERA_OPTICAL_CONVENTION = "opencv_ros_x_right_y_down_z_forward"

# Isaac CameraCfg mount values. ``convention='world'`` applies to the authored
# camera axes; Isaac exposes the corresponding OpenCV/ROS optical pose through
# ``quat_w_ros``.
CAMERA_OFFSET_POS_B = (0.14, 0.0, 0.05)
CAMERA_OFFSET_ROT_WXYZ = (1.0, 0.0, 0.0, 0.0)
CAMERA_OFFSET_CONVENTION = "world"

# Validated transform from OpenCV/ROS optical frame C to drone body frame B.
CAMERA_TO_BODY_ROTATION = (
    (0.0, 0.0, 1.0),
    (-1.0, 0.0, 0.0),
    (0.0, -1.0, 0.0),
)


def stage2_camera_to_body() -> RigidTransform:
    """Return the single configured Stage2 transform ``T_bc``."""
    return RigidTransform(
        np.asarray(CAMERA_TO_BODY_ROTATION, dtype=np.float64),
        np.asarray(CAMERA_OFFSET_POS_B, dtype=np.float64),
        to_frame="B",
        from_frame="C",
    )


def load_stage2_gate_geometry() -> GateGeometry:
    """Load the authoritative gate-actor opening keypoints."""
    geometry = load_gate_keypoint_calibration(GATE_KEYPOINT_CALIBRATION_PATH)
    if geometry.corner_names != CORNER_NAMES:
        raise ValueError(f"Unexpected Stage2 corner order {geometry.corner_names}")
    return geometry
