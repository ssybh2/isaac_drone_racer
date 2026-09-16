import builtins

import numpy as np
import pytest

from estimation.learned_motion import LearnedMotionBuffer, MotionWindow, TorchTcnDisplacementPredictor


def _fill(buffer: LearnedMotionBuffer, end_s: float = 0.6, dt: float = 0.01) -> None:
    for t in np.arange(0.0, end_s + 0.5 * dt, dt):
        buffer.append(
            float(t),
            gyro_w=np.array([t, 2.0 * t, 3.0 * t]),
            thrust_w=np.array([1.0, 2.0, 3.0]),
        )


def test_buffer_resamples_exact_half_second_window_to_100hz_six_channel_tensor():
    buffer = LearnedMotionBuffer(window_time_s=0.5, sample_rate_hz=100.0)
    _fill(buffer)

    window = buffer.window(0.0, 0.5)

    assert isinstance(window, MotionWindow)
    assert window.features.shape == (6, 50)
    assert window.timestamps_s.shape == (50,)
    assert window.start_timestamp_s == pytest.approx(0.0)
    assert window.end_timestamp_s == pytest.approx(0.5)
    assert window.timestamps_s[0] == pytest.approx(0.0)
    assert window.timestamps_s[-1] == pytest.approx(0.49)
    expected_thrust = np.repeat(np.array([[1.0], [2.0], [3.0]]), 50, axis=1)
    np.testing.assert_allclose(window.features[3:, :], expected_thrust)


def test_buffer_rejects_non_monotonic_motion_timestamps():
    buffer = LearnedMotionBuffer()
    buffer.append(1.0, gyro_w=np.zeros(3), thrust_w=np.zeros(3))
    with pytest.raises(ValueError, match="monotonic"):
        buffer.append(0.9, gyro_w=np.zeros(3), thrust_w=np.zeros(3))


def test_window_requires_configured_duration_and_coverage():
    buffer = LearnedMotionBuffer(window_time_s=0.5, sample_rate_hz=100.0)
    _fill(buffer, end_s=0.3)
    with pytest.raises(ValueError, match="duration"):
        buffer.window(0.0, 0.4)
    with pytest.raises(ValueError, match="cover"):
        buffer.window(0.0, 0.5)


def test_reset_discards_buffered_motion_history():
    buffer = LearnedMotionBuffer()
    _fill(buffer)
    assert len(buffer) > 0
    buffer.reset()
    assert len(buffer) == 0


def test_torch_predictor_import_is_lazy(monkeypatch, tmp_path):
    original_import = builtins.__import__

    def rejecting_import(name, *args, **kwargs):
        if name == "torch" or name.startswith("torch."):
            raise ImportError("torch intentionally unavailable")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", rejecting_import)
    with pytest.raises(ImportError, match="PyTorch"):
        TorchTcnDisplacementPredictor(tmp_path / "missing.pt")
