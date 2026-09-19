import copy

import numpy as np

import estimation.learned_inertial_odometry as lio


def _propagate_static(est, start_step, end_step, dt=0.01):
    for k in range(start_step, end_step + 1):
        est.propagate(
            gyro_b=(0.0, 0.0, 0.0),
            accel_b=(0.0, 0.0, 9.81),
            timestamp_s=k * dt,
        )


def test_static_specific_force_keeps_pose_nearly_fixed():
    est = lio.LearnedInertialOdometry()
    est.reset(position_w_b=(1.0, 2.0, 3.0))
    _propagate_static(est, 1, 100)
    state = est.state()
    assert np.allclose(state.position_w_b, (1.0, 2.0, 3.0), atol=1e-8)
    assert np.allclose(state.linear_velocity_w_b, 0.0, atol=1e-8)


def test_attitude_error_injection_is_right_multiplicative():
    est = lio.LearnedInertialOdometry()
    initial_q = np.array(
        [np.cos(np.deg2rad(30.0) / 2.0), 0.0, 0.0, np.sin(np.deg2rad(30.0) / 2.0)]
    )
    est.reset(orientation_w_b_wxyz=initial_q)

    R_before = est.R.copy()
    dx = np.zeros(15)
    dx[0:3] = np.array([0.02, -0.01, 0.03])
    expected = R_before @ lio._exp_so3(dx[0:3])

    est._inject_error(dx)

    assert np.allclose(est.R, expected, atol=1e-12)


def test_absolute_orientation_update_uses_local_right_error():
    est = lio.LearnedInertialOdometry()
    initial_q = np.array(
        [np.cos(np.deg2rad(45.0) / 2.0), 0.0, 0.0, np.sin(np.deg2rad(45.0) / 2.0)]
    )
    est.reset(
        orientation_w_b_wxyz=initial_q,
        initial_covariance=np.eye(15) * 0.1,
    )

    R_target = est.R @ lio._exp_so3(np.array([0.04, -0.02, 0.01]))
    q_target = lio.rotmat_to_quat_wxyz(R_target)

    before = lio._log_so3(est.R.T @ R_target)
    est.update_absolute_orientation(
        q_target,
        covariance_rad2=np.eye(3) * 1.0e-8,
    )
    after = lio._log_so3(est.R.T @ R_target)

    assert np.linalg.norm(after) < np.linalg.norm(before) * 1.0e-3
    assert np.linalg.eigvalsh(est.state().covariance).min() > -1.0e-10


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


def test_multi_clone_augmentation_preserves_cross_covariance():
    est = lio.LearnedInertialOdometry(max_position_clones=11)
    est.reset(initial_covariance=np.eye(15) * 0.1)

    est.clone_current_position()
    assert est.clone_count == 1
    assert est.state().covariance.shape == (21, 21)

    for k in range(1, 6):
        est.propagate(
            gyro_b=(0.0, 0.0, 0.0),
            accel_b=(0.0, 0.0, 9.81),
            timestamp_s=k * 0.01,
        )
    est.clone_current_position()

    P = est.state().covariance
    assert est.clone_count == 2
    assert P.shape == (27, 27)
    assert np.linalg.norm(P[15:21, 21:27]) > 1.0e-6
    assert np.allclose(P, P.T, atol=1e-12)


def test_20hz_clone_cadence_retains_half_second_history():
    est = lio.LearnedInertialOdometry(max_position_clones=11)
    est.reset()
    est.clone_current_position()

    for k in range(1, 11):
        est.propagate(
            gyro_b=(0.0, 0.0, 0.0),
            accel_b=(0.0, 0.0, 9.81),
            timestamp_s=k * 0.05,
        )
        est.clone_current_position()

    timestamps = np.asarray(est.clone_timestamps_s)
    assert est.clone_count == 11
    assert est.state().covariance.shape == (81, 81)
    assert np.allclose(np.diff(timestamps), 0.05, atol=1e-12)
    assert np.isclose(timestamps[-1] - timestamps[0], 0.5)

    H = est.relative_displacement_jacobian(start_timestamp_s=0.0)
    assert np.allclose(H[:, 6:9], np.eye(3))
    assert np.allclose(H[:, 18:21], -np.eye(3))


def test_relative_displacement_jacobian_selects_historical_clone():
    est = lio.LearnedInertialOdometry(max_position_clones=11)
    est.reset()
    est.clone_current_position()
    for k in range(1, 4):
        est.propagate(
            gyro_b=(0.0, 0.0, 0.0),
            accel_b=(0.0, 0.0, 9.81),
            timestamp_s=k * 0.05,
        )
        est.clone_current_position()

    H = est.relative_displacement_jacobian(start_timestamp_s=0.10)
    clone_start = 15 + 6 * 2
    assert np.allclose(H[:, 6:9], np.eye(3))
    assert np.allclose(H[:, clone_start + 3 : clone_start + 6], -np.eye(3))
    assert np.count_nonzero(H) == 6


def test_clone_marginalization_updates_covariance_dimension_and_order():
    est = lio.LearnedInertialOdometry(max_position_clones=11)
    est.reset()
    est.clone_current_position()
    for k in range(1, 5):
        est.propagate(
            gyro_b=(0.0, 0.0, 0.0),
            accel_b=(0.0, 0.0, 9.81),
            timestamp_s=k * 0.05,
        )
        est.clone_current_position()

    removed = est.marginalize_clones_before(0.10)
    assert removed == 2
    assert np.allclose(est.clone_timestamps_s, (0.10, 0.15, 0.20))
    assert est.state().covariance.shape == (33, 33)
    assert np.allclose(est.state().covariance, est.state().covariance.T, atol=1e-12)


def test_kinematic_residual_removes_known_start_velocity_term():
    est = lio.LearnedInertialOdometry(max_position_clones=11)
    est.reset(
        linear_velocity_w_b=(2.0, -1.0, 0.5),
        initial_covariance=np.eye(15) * 0.05,
    )
    est.clone_current_position()

    for k in range(1, 51):
        est.propagate(
            gyro_b=(0.0, 0.0, 0.0),
            accel_b=(0.0, 0.0, 9.81),
            timestamp_s=k * 0.01,
        )

    predicted = est.predicted_kinematic_residual(
        start_timestamp_s=0.0,
        clone_tolerance_s=1e-9,
    )
    assert np.allclose(predicted, 0.0, atol=1e-9)

    H = est.kinematic_residual_jacobian(
        start_timestamp_s=0.0,
        clone_tolerance_s=1e-9,
    )
    assert np.allclose(H[:, 6:9], np.eye(3))
    assert np.allclose(H[:, 15:18], -0.5 * np.eye(3))
    assert np.allclose(H[:, 18:21], -np.eye(3))

    innovation = est.update_learned_kinematic_residual(
        residual_displacement_w=(0.0, 0.0, 0.0),
        covariance_w=np.eye(3) * 0.01,
        start_timestamp_s=0.0,
        clone_tolerance_s=1e-9,
    )
    assert np.allclose(innovation, 0.0, atol=1e-9)
    assert est.clone_count == 0
    assert est.state().covariance.shape == (15, 15)


def test_body_end_kinematic_residual_jacobian_matches_finite_difference():
    est = lio.LearnedInertialOdometry(max_position_clones=11)
    q0 = lio.rotmat_to_quat_wxyz(
        lio._exp_so3(np.array([0.15, -0.10, 0.25]))
    )
    est.reset(
        position_w_b=(0.3, -0.2, 1.1),
        linear_velocity_w_b=(1.2, -0.4, 0.2),
        orientation_w_b_wxyz=q0,
        initial_covariance=np.eye(15) * 0.05,
    )
    est.clone_current_position()

    for k in range(1, 51):
        est.propagate(
            gyro_b=(0.03, -0.02, 0.08),
            accel_b=(0.4, -0.2, 9.9),
            timestamp_s=k * 0.01,
        )

    H = est.kinematic_residual_body_end_jacobian(
        start_timestamp_s=0.0,
        clone_tolerance_s=1e-9,
    )
    base = est.predicted_kinematic_residual_body_end(
        start_timestamp_s=0.0,
        clone_tolerance_s=1e-9,
    )
    eps = 1.0e-7

    # Check current attitude, current position, cloned velocity and cloned
    # position columns. These are all terms in the endpoint-body measurement.
    columns = list(range(0, 3)) + list(range(6, 9))
    columns += list(range(15, 18)) + list(range(18, 21))
    for col in columns:
        perturbed = copy.deepcopy(est)
        dx = np.zeros(perturbed.P.shape[0], dtype=np.float64)
        dx[col] = eps
        perturbed._inject_error(dx)
        value = perturbed.predicted_kinematic_residual_body_end(
            start_timestamp_s=0.0,
            clone_tolerance_s=1e-9,
        )
        numerical = (value - base) / eps
        np.testing.assert_allclose(
            numerical,
            H[:, col],
            rtol=2e-5,
            atol=2e-6,
        )


def test_body_end_kinematic_residual_update_accepts_exact_measurement():
    est = lio.LearnedInertialOdometry(max_position_clones=11)
    est.reset(
        linear_velocity_w_b=(1.0, 0.2, -0.1),
        initial_covariance=np.eye(15) * 0.05,
    )
    est.clone_current_position()
    for k in range(1, 51):
        est.propagate(
            gyro_b=(0.0, 0.0, 0.2),
            accel_b=(0.2, 0.1, 9.81),
            timestamp_s=k * 0.01,
        )

    z = est.predicted_kinematic_residual_body_end(
        start_timestamp_s=0.0,
        clone_tolerance_s=1e-9,
    ).copy()
    innovation = est.update_learned_kinematic_residual_body_end(
        z,
        np.eye(3) * 0.01,
        start_timestamp_s=0.0,
        clone_tolerance_s=1e-9,
    )
    np.testing.assert_allclose(innovation, 0.0, atol=1e-12)
    assert est.clone_count == 0
    assert np.linalg.eigvalsh(est.state().covariance).min() > -1e-10


def test_gravity_compensated_body_residual_jacobian_matches_finite_difference():
    est = lio.LearnedInertialOdometry(max_position_clones=11)
    q0 = lio.rotmat_to_quat_wxyz(
        lio._exp_so3(np.array([0.12, -0.08, 0.20]))
    )
    est.reset(
        position_w_b=(0.2, -0.1, 1.0),
        linear_velocity_w_b=(0.8, -0.3, 0.1),
        orientation_w_b_wxyz=q0,
        initial_covariance=np.eye(15) * 0.05,
    )
    est.clone_current_position()
    for k in range(1, 51):
        est.propagate(
            gyro_b=(0.02, -0.01, 0.07),
            accel_b=(0.3, -0.1, 9.7),
            timestamp_s=k * 0.01,
        )

    H = est.kinematic_residual_body_end_gravity_compensated_jacobian(
        start_timestamp_s=0.0,
        clone_tolerance_s=1e-9,
    )
    base = est.predicted_kinematic_residual_body_end_gravity_compensated(
        start_timestamp_s=0.0,
        clone_tolerance_s=1e-9,
    )
    eps = 1.0e-7
    columns = list(range(0, 3)) + list(range(6, 9))
    columns += list(range(15, 18)) + list(range(18, 21))

    for col in columns:
        perturbed = copy.deepcopy(est)
        dx = np.zeros(perturbed.P.shape[0], dtype=np.float64)
        dx[col] = eps
        perturbed._inject_error(dx)
        value = (
            perturbed.predicted_kinematic_residual_body_end_gravity_compensated(
                start_timestamp_s=0.0,
                clone_tolerance_s=1e-9,
            )
        )
        numerical = (value - base) / eps
        np.testing.assert_allclose(
            numerical,
            H[:, col],
            rtol=2e-5,
            atol=2e-6,
        )


def test_gravity_compensated_body_residual_is_zero_for_ballistic_gravity():
    est = lio.LearnedInertialOdometry(max_position_clones=11)
    est.reset(
        position_w_b=(0.0, 0.0, 0.0),
        linear_velocity_w_b=(0.0, 0.0, 0.0),
        initial_covariance=np.eye(15) * 0.05,
    )
    est.clone_current_position()

    # Free fall has zero specific force; gravity is handled explicitly by the
    # nominal dynamics and should disappear from the compensated measurement.
    for k in range(1, 51):
        est.propagate(
            gyro_b=(0.0, 0.0, 0.0),
            accel_b=(0.0, 0.0, 0.0),
            timestamp_s=k * 0.01,
        )

    z = est.predicted_kinematic_residual_body_end_gravity_compensated(
        start_timestamp_s=0.0,
        clone_tolerance_s=1e-9,
    )
    np.testing.assert_allclose(z, 0.0, atol=1e-9)


def test_learned_relative_displacement_pulls_position_toward_measurement():
    est = lio.LearnedInertialOdometry(max_position_clones=11)
    est.reset(initial_covariance=np.eye(15) * 0.1)
    est.begin_displacement_window()

    for k in range(1, 51):
        est.propagate(
            gyro_b=(0.0, 0.0, 0.0),
            accel_b=(0.4, 0.0, 9.81),
            timestamp_s=k * 0.01,
        )
        if k % 5 == 0:
            est.clone_current_position()

    before = est.state().position_w_b[0]
    residual = est.update_learned_displacement(
        displacement_w=(0.0, 0.0, 0.0),
        covariance_w=np.eye(3) * 1e-4,
        start_timestamp_s=0.0,
    )
    after = est.state().position_w_b[0]

    assert residual[0] < 0.0
    assert abs(after) < abs(before)
    assert est.clone_count == 10
    assert est.state().covariance.shape == (75, 75)


def test_overlapping_half_second_updates_can_run_continuously():
    est = lio.LearnedInertialOdometry(max_position_clones=11)
    est.reset(initial_covariance=np.eye(15) * 0.05)
    est.clone_current_position()

    update_count = 0
    for k in range(1, 21):
        t = k * 0.05
        est.propagate(
            gyro_b=(0.0, 0.0, 0.0),
            accel_b=(0.0, 0.0, 9.81),
            timestamp_s=t,
        )
        est.clone_current_position()
        if t >= 0.5 - 1e-12:
            start = round(t - 0.5, 12)
            residual = est.update_learned_displacement(
                displacement_w=(0.0, 0.0, 0.0),
                covariance_w=np.eye(3) * 0.01,
                start_timestamp_s=start,
                clone_tolerance_s=1e-9,
            )
            assert np.allclose(residual, 0.0, atol=1e-10)
            update_count += 1

    assert update_count == 11
    assert est.clone_count == 10
    assert np.allclose(est.state().position_w_b, 0.0, atol=1e-9)
    assert np.allclose(est.state().covariance, est.state().covariance.T, atol=1e-12)
    assert np.linalg.eigvalsh(est.state().covariance).min() > -1e-10


def test_max_clone_count_bounds_fixed_lag_state():
    est = lio.LearnedInertialOdometry(max_position_clones=3)
    est.reset()
    est.clone_current_position()
    for k in range(1, 6):
        est.propagate(
            gyro_b=(0.0, 0.0, 0.0),
            accel_b=(0.0, 0.0, 9.81),
            timestamp_s=k * 0.05,
        )
        est.clone_current_position()
    assert est.clone_count == 3
    assert np.allclose(est.clone_timestamps_s, (0.15, 0.20, 0.25))
    assert est.state().covariance.shape == (33, 33)


def test_learned_covariance_protection_blocks_millimetre_horizontal_sigma():
    raw = np.diag(np.square([0.0027, 0.20, 0.003]))
    protected = lio.protected_displacement_covariance(
        raw,
        sigma_floor_xyz_m=(0.10, 0.10, 0.01),
        covariance_scale=1.25,
    )
    sigma = np.sqrt(np.diag(protected))
    assert np.allclose(sigma, (0.125, 0.25, 0.0125), atol=1e-12)


def test_gate_position_update_moves_state_toward_absolute_anchor_with_clones():
    est = lio.LearnedInertialOdometry()
    est.reset(position_w_b=(5.0, 0.0, 0.0), initial_covariance=np.eye(15))
    est.clone_current_position()
    est.update_absolute_position(
        position_w_b=(1.0, 0.0, 0.0),
        covariance_w=np.eye(3) * 1e-4,
    )
    assert est.state().position_w_b[0] < 1.01
    assert est.clone_count == 1
    assert est.state().covariance.shape == (21, 21)


def test_learned_kalman_gain_modes_zero_expected_rows():
    est = lio.LearnedInertialOdometry(max_position_clones=3)
    est.reset(initial_covariance=np.eye(15) * 0.1)
    est.clone_current_position()
    est.propagate(
        gyro_b=(0.0, 0.0, 0.0),
        accel_b=(0.0, 0.0, 9.81),
        timestamp_s=0.05,
    )
    est.clone_current_position()

    K = np.ones((est.P.shape[0], 3), dtype=np.float64)

    full = est._constrain_kalman_gain(K, "full")
    np.testing.assert_allclose(full, K)

    freeze_attitude_bias = est._constrain_kalman_gain(
        K,
        "freeze_attitude_bias",
    )
    np.testing.assert_allclose(freeze_attitude_bias[0:3], 0.0)
    np.testing.assert_allclose(freeze_attitude_bias[9:15], 0.0)
    np.testing.assert_allclose(freeze_attitude_bias[3:9], 1.0)
    for clone_index in range(est.clone_count):
        np.testing.assert_allclose(
            freeze_attitude_bias[est._clone_slice(clone_index)],
            1.0,
        )

    freeze_position = est._constrain_kalman_gain(K, "freeze_position")
    np.testing.assert_allclose(freeze_position[6:9], 0.0)
    np.testing.assert_allclose(freeze_position[0:6], 1.0)
    np.testing.assert_allclose(freeze_position[9:15], 1.0)
    for clone_index in range(est.clone_count):
        np.testing.assert_allclose(
            freeze_position[est._clone_velocity_slice(clone_index)],
            1.0,
        )
        np.testing.assert_allclose(
            freeze_position[est._clone_position_slice(clone_index)],
            0.0,
        )

    combined = est._constrain_kalman_gain(
        K,
        "freeze_position_attitude_bias",
    )
    np.testing.assert_allclose(combined[0:3], 0.0)
    np.testing.assert_allclose(combined[3:6], 1.0)
    np.testing.assert_allclose(combined[6:15], 0.0)
    for clone_index in range(est.clone_count):
        np.testing.assert_allclose(
            combined[est._clone_velocity_slice(clone_index)],
            1.0,
        )
        np.testing.assert_allclose(
            combined[est._clone_position_slice(clone_index)],
            0.0,
        )


def test_constrained_kalman_update_uses_effective_gain_for_joseph_update():
    est = lio.LearnedInertialOdometry(
        max_position_clones=2,
        learned_kalman_gain_mode="freeze_position_attitude_bias",
    )
    est.reset(initial_covariance=np.eye(15) * 0.1)
    est.clone_current_position()

    rng = np.random.default_rng(7)
    A0 = rng.normal(size=(est.P.shape[0], est.P.shape[0]))
    P0 = 0.002 * (A0 @ A0.T) + np.eye(est.P.shape[0]) * 0.05
    est.P = P0.copy()

    H = np.zeros((3, est.P.shape[0]), dtype=np.float64)
    H[:, 3:6] = np.eye(3)
    H[:, 6:9] = 0.4 * np.eye(3)
    H[:, est._clone_velocity_slice(0)] = -0.3 * np.eye(3)
    H[:, est._clone_position_slice(0)] = -0.2 * np.eye(3)
    Rm = np.eye(3) * 0.03
    residual = np.array([0.08, -0.04, 0.02], dtype=np.float64)

    S = H @ P0 @ H.T + Rm
    K_raw = np.linalg.solve(S.T, (P0 @ H.T).T).T
    K_eff = est._constrain_kalman_gain(
        K_raw,
        "freeze_position_attitude_bias",
    )
    dx_expected = K_eff @ residual
    I = np.eye(P0.shape[0], dtype=np.float64)
    joseph = I - K_eff @ H
    P_expected = joseph @ P0 @ joseph.T + K_eff @ Rm @ K_eff.T
    P_expected = 0.5 * (P_expected + P_expected.T)

    p_before = est.p.copy()
    R_before = est.R.copy()
    ba_before = est.ba.copy()
    bg_before = est.bg.copy()
    clone_p_before = est._clone_positions[0].copy()
    v_before = est.v.copy()
    clone_v_before = est._clone_velocities[0].copy()

    est._kalman_update(
        residual,
        H,
        Rm,
        gain_mode="freeze_position_attitude_bias",
    )

    np.testing.assert_allclose(est.p, p_before, atol=1e-12)
    np.testing.assert_allclose(est.R, R_before, atol=1e-12)
    np.testing.assert_allclose(est.ba, ba_before, atol=1e-12)
    np.testing.assert_allclose(est.bg, bg_before, atol=1e-12)
    np.testing.assert_allclose(est._clone_positions[0], clone_p_before, atol=1e-12)
    np.testing.assert_allclose(est.v, v_before + dx_expected[3:6], atol=1e-12)
    np.testing.assert_allclose(
        est._clone_velocities[0],
        clone_v_before + dx_expected[est._clone_velocity_slice(0)],
        atol=1e-12,
    )
    np.testing.assert_allclose(est.P, P_expected, rtol=1e-11, atol=1e-11)
    assert est.last_update_diagnostics["kalman_gain_mode"] == (
        "freeze_position_attitude_bias"
    )
    assert est.last_update_diagnostics["kalman_gain_position_norm"] == 0.0
    assert est.last_update_diagnostics["kalman_gain_clone_position_norm"] == 0.0


def test_learned_update_uses_configured_gain_mode_but_absolute_update_stays_full():
    est = lio.LearnedInertialOdometry(
        max_position_clones=2,
        learned_kalman_gain_mode="freeze_position",
    )
    est.reset(initial_covariance=np.eye(15) * 0.1)
    est.clone_current_position()
    est.propagate(
        gyro_b=(0.0, 0.0, 0.0),
        accel_b=(0.3, 0.0, 9.81),
        timestamp_s=0.5,
    )

    est.update_learned_kinematic_residual(
        residual_displacement_w=(0.0, 0.0, 0.0),
        covariance_w=np.eye(3) * 0.01,
        start_timestamp_s=0.0,
        clone_tolerance_s=1e-9,
    )
    assert est.last_update_diagnostics["kalman_gain_mode"] == "freeze_position"
    assert est.last_update_diagnostics["kalman_gain_position_norm"] == 0.0
    assert est.last_update_diagnostics["kalman_gain_clone_position_norm"] == 0.0

    est.update_absolute_position(
        position_w_b=(0.0, 0.0, 0.0),
        covariance_w=np.eye(3) * 0.1,
    )
    assert est.last_update_diagnostics["kalman_gain_mode"] == "full"
    assert est.last_update_diagnostics["kalman_gain_position_norm"] > 0.0
