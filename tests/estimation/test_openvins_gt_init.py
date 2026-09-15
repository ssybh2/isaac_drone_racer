import importlib.util
from pathlib import Path

import numpy as np
import pytest


MODULE_PATH = Path(__file__).parents[2] / "estimation" / "openvins_gt_init.py"
SPEC = importlib.util.spec_from_file_location("openvins_gt_init", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)
build_openvins_gt_initialization = MODULE.build_openvins_gt_initialization


def test_identity_preserves_translation_velocity_and_zero_biases():
    state = build_openvins_gt_initialization(
        1.25,
        [1.0, 2.0, 3.0],
        [1.0, 0.0, 0.0, 0.0],
        [4.0, 5.0, 6.0],
    )

    np.testing.assert_allclose(state.q_g_to_i_wxyz, [1.0, 0.0, 0.0, 0.0])
    np.testing.assert_allclose(state.position_g_i, [1.0, 2.0, 3.0])
    np.testing.assert_allclose(state.velocity_g_i, [4.0, 5.0, 6.0])
    np.testing.assert_allclose(state.gyro_bias_i, [0.0, 0.0, 0.0])
    np.testing.assert_allclose(state.accel_bias_i, [0.0, 0.0, 0.0])
    np.testing.assert_allclose(
        state.as_openvins_state(),
        [1.25, 1.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    )


def test_world_from_imu_yaw_is_inverted_for_openvins_q_g_to_i():
    c = np.sqrt(0.5)
    state = build_openvins_gt_initialization(
        2.0,
        [0.0, 0.0, 0.0],
        [c, 0.0, 0.0, c],
        [0.0, 0.0, 0.0],
    )
    np.testing.assert_allclose(state.q_g_to_i_wxyz, [c, 0.0, 0.0, -c], atol=1.0e-12)


def test_quaternion_is_normalized_and_invalid_zero_quaternion_rejected():
    state = build_openvins_gt_initialization(
        0.0,
        [0.0, 0.0, 0.0],
        [2.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
    )
    np.testing.assert_allclose(state.q_g_to_i_wxyz, [1.0, 0.0, 0.0, 0.0])

    with pytest.raises(ValueError, match="quaternion"):
        build_openvins_gt_initialization(
            0.0,
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        )
