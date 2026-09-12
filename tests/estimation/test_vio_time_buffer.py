import numpy as np
import pytest

from estimation.swift_vio_drift import VioWorldEstimate
from estimation.vio_time_buffer import VioWorldEstimateBuffer


def _sample(t, x, q):
    return VioWorldEstimate(
        position_w_b=np.array([x, 0.0, 0.0]),
        linear_velocity_w_b=np.array([2.0 * x, 0.0, 0.0]),
        orientation_w_b_wxyz=np.asarray(q, dtype=float),
        timestamp_s=t,
    )


def test_buffer_interpolates_position_velocity_and_orientation():
    buf = VioWorldEstimateBuffer(max_age_s=2.0)
    q0 = [1.0, 0.0, 0.0, 0.0]
    q90 = [np.sqrt(0.5), 0.0, 0.0, np.sqrt(0.5)]
    buf.push(_sample(0.0, 0.0, q0))
    buf.push(_sample(0.1, 1.0, q90))

    mid = buf.interpolate(0.05)
    np.testing.assert_allclose(mid.position_w_b, [0.5, 0.0, 0.0], atol=1e-9)
    np.testing.assert_allclose(mid.linear_velocity_w_b, [1.0, 0.0, 0.0], atol=1e-9)
    q45 = np.array([np.cos(np.pi / 8.0), 0.0, 0.0, np.sin(np.pi / 8.0)])
    np.testing.assert_allclose(mid.orientation_w_b_wxyz, q45, atol=1e-8)


def test_buffer_refuses_extrapolation_and_nonmonotonic_inserts():
    buf = VioWorldEstimateBuffer(max_age_s=2.0)
    buf.push(_sample(1.0, 0.0, [1, 0, 0, 0]))
    buf.push(_sample(1.1, 1.0, [1, 0, 0, 0]))
    with pytest.raises(ValueError, match="outside VIO history"):
        buf.interpolate(0.9)
    with pytest.raises(ValueError, match="monotonic"):
        buf.push(_sample(1.05, 0.5, [1, 0, 0, 0]))
