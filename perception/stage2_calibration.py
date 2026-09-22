"""Authoritative Stage 2 gate and reference-camera calibration."""

from __future__ import annotations

from pathlib import Path
import math

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

# New high-speed Circular-12 racing camera mount. Positive pitch-up means the
# optical forward axis acquires a positive body-Z component. The legacy Stage2
# camera remains 0 deg unless a caller explicitly requests this mount.
RACING_CAMERA_PITCH_UP_DEG = 40.0

# Validated transform from OpenCV/ROS optical frame C to drone body frame B.
CAMERA_TO_BODY_ROTATION = (
    (0.0, 0.0, 1.0),
    (-1.0, 0.0, 0.0),
    (0.0, -1.0, 0.0),
)

# The rendered 256x256 Stage2/OpenVINS camera contract validated by the RGB
# overlay artifact. Keep these values synchronized with
# config/openvins/swift_sim/kalibr_imucam_chain.yaml. Runtime code verifies the
# Isaac camera matrix before publishing frames to OpenVINS so a future camera
# or IsaacLab configuration change fails closed instead of silently corrupting
# VIO calibration.
OPENVINS_CAMERA_RESOLUTION = (256, 256)  # width, height
OPENVINS_CAMERA_INTRINSICS = (
    293.19970703125,  # fx
    293.19970703125,  # fy
    128.0,  # cx
    128.0,  # cy
)


def openvins_camera_matrix() -> np.ndarray:
    """Return the validated 256x256 OpenVINS pinhole intrinsic matrix."""
    fx, fy, cx, cy = OPENVINS_CAMERA_INTRINSICS
    return np.array(
        [
            [fx, 0.0, cx],
            [0.0, fy, cy],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def camera_mount_quaternion_wxyz(
    pitch_up_deg: float = 0.0,
) -> tuple[float, float, float, float]:
    """Return Isaac mount quaternion for a positive optical pitch-up angle.

    With the legacy 0-deg mount the OpenCV optical +Z axis maps to body +X.
    A positive pitch-up rotates that optical forward axis toward body +Z.
    In the body XYZ convention this is a negative rotation about body +Y.
    """
    angle = math.radians(float(pitch_up_deg))
    half = 0.5 * angle
    return (
        float(math.cos(half)),
        0.0,
        float(-math.sin(half)),
        0.0,
    )


def camera_to_body_rotation(
    pitch_up_deg: float = 0.0,
) -> np.ndarray:
    """Return optical-frame C -> body-frame B rotation for the mount angle."""
    angle = math.radians(float(pitch_up_deg))
    ca = math.cos(angle)
    sa = math.sin(angle)
    # R_y(-angle): body-fixed mount rotation that tips optical +Z upward.
    R_pitch_up = np.array(
        [
            [ca, 0.0, -sa],
            [0.0, 1.0, 0.0],
            [sa, 0.0, ca],
        ],
        dtype=np.float64,
    )
    return R_pitch_up @ np.asarray(
        CAMERA_TO_BODY_ROTATION,
        dtype=np.float64,
    )


def stage2_camera_to_body(
    pitch_up_deg: float = 0.0,
) -> RigidTransform:
    """Return the Stage2/racing camera transform ``T_bc``."""
    return RigidTransform(
        camera_to_body_rotation(pitch_up_deg),
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
