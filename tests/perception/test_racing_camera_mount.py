import numpy as np

from perception.stage2_calibration import (
    RACING_CAMERA_PITCH_UP_DEG,
    camera_mount_quaternion_wxyz,
    camera_to_body_rotation,
    stage2_camera_to_body,
)


def test_racing_camera_40deg_pitch_up_points_optical_axis_upward():
    angle_deg = float(RACING_CAMERA_PITCH_UP_DEG)
    assert angle_deg == 40.0

    R_bc = camera_to_body_rotation(angle_deg)
    optical_forward_b = R_bc @ np.array([0.0, 0.0, 1.0])
    expected = np.array(
        [
            np.cos(np.deg2rad(angle_deg)),
            0.0,
            np.sin(np.deg2rad(angle_deg)),
        ]
    )
    np.testing.assert_allclose(optical_forward_b, expected, atol=1.0e-12)

    T_bc = stage2_camera_to_body(angle_deg)
    np.testing.assert_allclose(T_bc.R, R_bc, atol=1.0e-12)


def test_racing_camera_mount_quaternion_is_negative_body_y_rotation():
    q = np.asarray(camera_mount_quaternion_wxyz(40.0), dtype=np.float64)
    expected = np.array(
        [
            np.cos(np.deg2rad(20.0)),
            0.0,
            -np.sin(np.deg2rad(20.0)),
            0.0,
        ]
    )
    np.testing.assert_allclose(q, expected, atol=1.0e-12)
