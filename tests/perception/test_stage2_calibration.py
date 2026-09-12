import re
from pathlib import Path

import numpy as np

from perception.gate_geometry import CORNER_NAMES
from perception.stage2_calibration import (
    CAMERA_MODEL,
    CAMERA_OPTICAL_CONVENTION,
    GATE_KEYPOINT_CALIBRATION_PATH,
    OPENVINS_CAMERA_INTRINSICS,
    OPENVINS_CAMERA_RESOLUTION,
    load_stage2_gate_geometry,
    openvins_camera_matrix,
    stage2_camera_to_body,
)


def test_authoritative_stage2_calibration_is_complete():
    assert CAMERA_MODEL == "pinhole"
    assert CAMERA_OPTICAL_CONVENTION == "opencv_ros_x_right_y_down_z_forward"
    assert isinstance(GATE_KEYPOINT_CALIBRATION_PATH, Path)
    assert GATE_KEYPOINT_CALIBRATION_PATH.is_file()

    geometry = load_stage2_gate_geometry()
    assert geometry.corner_names == CORNER_NAMES
    np.testing.assert_allclose(geometry.center_g, [-0.0661411, 0.0, 1.0668], atol=1e-6)

    T_bc = stage2_camera_to_body()
    np.testing.assert_allclose(
        T_bc.R,
        [[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]],
    )
    np.testing.assert_allclose(T_bc.t, [0.14, 0.0, 0.05])
    assert (T_bc.to_frame, T_bc.from_frame) == ("B", "C")


def test_openvins_256px_camera_contract_matches_validated_overlay():
    assert OPENVINS_CAMERA_RESOLUTION == (256, 256)
    np.testing.assert_allclose(
        OPENVINS_CAMERA_INTRINSICS,
        [293.19970703125, 293.19970703125, 128.0, 128.0],
        atol=0.0,
    )
    np.testing.assert_allclose(
        openvins_camera_matrix(),
        [
            [293.19970703125, 0.0, 128.0],
            [0.0, 293.19970703125, 128.0],
            [0.0, 0.0, 1.0],
        ],
        atol=0.0,
    )


def _openvins_camera_yaml_text() -> str:
    project_root = GATE_KEYPOINT_CALIBRATION_PATH.parents[2]
    yaml_path = project_root / "config" / "openvins" / "swift_sim" / "kalibr_imucam_chain.yaml"
    return yaml_path.read_text(encoding="utf-8")


def test_openvins_yaml_matches_authoritative_camera_contract():
    text = _openvins_camera_yaml_text()

    intrinsics_match = re.search(r"intrinsics:\s*\[([^\]]+)\]", text)
    resolution_match = re.search(r"resolution:\s*\[([^\]]+)\]", text)
    assert intrinsics_match is not None
    assert resolution_match is not None

    yaml_intrinsics = tuple(float(value.strip()) for value in intrinsics_match.group(1).split(","))
    yaml_resolution = tuple(int(value.strip()) for value in resolution_match.group(1).split(","))

    np.testing.assert_allclose(yaml_intrinsics, OPENVINS_CAMERA_INTRINSICS, atol=0.0)
    assert yaml_resolution == OPENVINS_CAMERA_RESOLUTION


def test_openvins_yaml_extrinsic_is_inverse_of_authoritative_camera_to_body():
    text = _openvins_camera_yaml_text()
    matrix_block = text.split("T_cam_imu:", 1)[1].split("cam_overlaps:", 1)[0]
    rows = re.findall(r"-\s*\[([^\]]+)\]", matrix_block)
    assert len(rows) == 4
    T_cam_imu_yaml = np.asarray(
        [[float(value.strip()) for value in row.split(",")] for row in rows],
        dtype=np.float64,
    )

    # Simulation uses I == B. Stage2 stores T_bc (camera -> body), whereas
    # OpenVINS expects T_cam_imu (IMU -> optical camera), so the YAML must hold
    # exactly inverse(T_bc).
    T_cb = stage2_camera_to_body().inverse()
    T_cam_imu_expected = np.eye(4, dtype=np.float64)
    T_cam_imu_expected[:3, :3] = T_cb.R
    T_cam_imu_expected[:3, 3] = T_cb.t

    np.testing.assert_allclose(T_cam_imu_yaml, T_cam_imu_expected, atol=1e-12)
