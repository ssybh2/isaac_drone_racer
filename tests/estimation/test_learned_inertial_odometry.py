import importlib.util
from pathlib import Path

import numpy as np


MODULE = Path(__file__).resolve().parents[2] / "estimation" / "learned_inertial_odometry.py"
spec = importlib.util.spec_from_file_location("learned_inertial_odometry", MODULE)
lio = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(lio)


def test_static_specific_force_keeps_pose_nearly_fixed():
    est = lio.LearnedInertialOdometry()
    est.reset(position_w_b=(1.0, 2.0, 3.0))
    for k in range(1, 101):
        est.propagate(
            gyro_b=(0.0, 0.0, 0.0),
            accel_b=(0.0, 0.0, 9.81),
            timestamp_s=k * 0.01,
        )
    state = est.state()
    assert np.allclose(state.position_w_b, (1.0, 2.0, 3.0), atol=1e-8)
    assert np.allclose(state.linear_velocity_w_b, 0.0, atol=1e-8)


def test_gyro_propagates_yaw():
    est = lio.LearnedInertialOdometry()
    est.reset()
    est.propagate(
        gyro_b=(0.0, 0.0, np.pi / 2.0),
        accel_b=(0.0, 0.0, 9.81),
        timestamp_s=1.0,
    )
    R = lio.quat_wxyz_to_rotmat(est.state().orientation_w_b_wxyz)
    x_axis_world = R @ np.array([1.0, 0.0, 0.0])
    assert np.allclose(x_axis_world[:2], (0.0, 1.0), atol=1e-6)


def test_learned_relative_displacement_pulls_position_toward_measurement():
    est = lio.LearnedInertialOdometry()
    est.reset(initial_covariance=np.eye(15) * 0.1)
    est.begin_displacement_window()

    # Propagate with an intentionally biased acceleration so the EKF drifts.
    for k in range(1, 51):
        est.propagate(
            gyro_b=(0.0, 0.0, 0.0),
            accel_b=(0.4, 0.0, 9.81),
            timestamp_s=k * 0.01,
        )
    before = est.state().position_w_b[0]
    residual = est.update_learned_displacement(
        displacement_w=(0.0, 0.0, 0.0),
        covariance_w=np.eye(3) * 1e-4,
    )
    after = est.state().position_w_b[0]
    assert residual[0] < 0.0
    assert abs(after) < abs(before)
    assert not est.has_displacement_clone
    assert est.state().covariance.shape == (15, 15)


def test_gate_position_update_moves_state_toward_absolute_anchor():
    est = lio.LearnedInertialOdometry()
    est.reset(position_w_b=(5.0, 0.0, 0.0), initial_covariance=np.eye(15))
    est.update_absolute_position(
        position_w_b=(1.0, 0.0, 0.0),
        covariance_w=np.eye(3) * 1e-4,
    )
    assert est.state().position_w_b[0] < 1.01
