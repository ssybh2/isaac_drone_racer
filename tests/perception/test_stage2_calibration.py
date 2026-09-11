from pathlib import Path

import numpy as np

from perception.gate_geometry import CORNER_NAMES
from perception.stage2_calibration import (
    CAMERA_MODEL,
    CAMERA_OPTICAL_CONVENTION,
    GATE_KEYPOINT_CALIBRATION_PATH,
    load_stage2_gate_geometry,
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
