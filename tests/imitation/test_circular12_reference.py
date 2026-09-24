import math

import numpy as np

from imitation.circular12_reference import Circular12Reference


def test_reference_preserves_radius_height_speed_and_rotation():
    reference = Circular12Reference()
    sample = reference.sample_phase(0.73)

    center = np.array(
        [reference.cfg.center_x_m, reference.cfg.center_y_m]
    )
    assert math.isclose(
        np.linalg.norm(sample.position_w[:2] - center),
        reference.cfg.radius_m,
        rel_tol=0.0,
        abs_tol=1.0e-9,
    )
    assert math.isclose(
        sample.position_w[2],
        reference.cfg.height_m,
        abs_tol=1.0e-12,
    )
    assert math.isclose(
        np.linalg.norm(sample.velocity_w),
        reference.cfg.speed_mps,
        rel_tol=0.0,
        abs_tol=1.0e-9,
    )
    np.testing.assert_allclose(
        sample.rotation_wb.T @ sample.rotation_wb,
        np.eye(3),
        atol=1.0e-10,
    )
    assert np.linalg.det(sample.rotation_wb) > 0.999999


def test_reference_bank_matches_steady_turn_physics():
    reference = Circular12Reference()
    expected = -math.atan2(
        reference.cfg.speed_mps**2 / reference.cfg.radius_m,
        reference.cfg.gravity_mps2,
    )
    assert math.isclose(reference.bank_rad, expected, abs_tol=1.0e-12)
    assert 65.0 < abs(reference.bank_deg) < 75.0
