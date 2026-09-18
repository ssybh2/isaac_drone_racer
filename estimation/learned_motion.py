"""Learned short-horizon motion constraints for hybrid VIO.

NumPy-only buffering and contracts live at module scope. PyTorch is imported
only when constructing a trainable network or loading an inference checkpoint,
so estimator-only deployments and pure unit tests do not depend on torch.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np


@dataclass(frozen=True)
class MotionWindow:
    """Uniformly sampled gyro+thrust history for one displacement prediction."""

    features: np.ndarray
    timestamps_s: np.ndarray
    start_timestamp_s: float
    end_timestamp_s: float

    def __post_init__(self) -> None:
        features = np.asarray(self.features, dtype=np.float32)
        timestamps = np.asarray(self.timestamps_s, dtype=np.float64).reshape(-1)
        if features.ndim != 2 or features.shape[0] != 6:
            raise ValueError("MotionWindow features must have shape (6, N)")
        if features.shape[1] != timestamps.size:
            raise ValueError("MotionWindow feature/timestamp sample counts must match")
        if not np.all(np.isfinite(features)) or not np.all(np.isfinite(timestamps)):
            raise ValueError("MotionWindow values must be finite")
        object.__setattr__(self, "features", features)
        object.__setattr__(self, "timestamps_s", timestamps)
        object.__setattr__(self, "start_timestamp_s", float(self.start_timestamp_s))
        object.__setattr__(self, "end_timestamp_s", float(self.end_timestamp_s))


@dataclass(frozen=True)
class DisplacementPrediction:
    displacement_w: np.ndarray
    covariance_w: np.ndarray

    def __post_init__(self) -> None:
        displacement = np.asarray(self.displacement_w, dtype=np.float64).reshape(3)
        covariance = np.asarray(self.covariance_w, dtype=np.float64).reshape(3, 3)
        covariance = 0.5 * (covariance + covariance.T)
        if not np.all(np.isfinite(displacement)) or not np.all(np.isfinite(covariance)):
            raise ValueError("Displacement prediction must be finite")
        if np.linalg.eigvalsh(covariance)[0] <= 0.0:
            raise ValueError("Displacement prediction covariance must be positive definite")
        object.__setattr__(self, "displacement_w", displacement)
        object.__setattr__(self, "covariance_w", covariance)


class DisplacementPredictor(Protocol):
    def predict(self, window: MotionWindow) -> DisplacementPrediction:
        """Predict displacement and uncertainty over ``window``."""


class LearnedMotionBuffer:
    """Timestamped world-frame gyro/thrust buffer with fixed-rate resampling."""

    def __init__(
        self,
        *,
        window_time_s: float = 0.5,
        sample_rate_hz: float = 100.0,
        coverage_tolerance_s: float = 2.0e-3,
    ) -> None:
        if window_time_s <= 0.0 or not np.isfinite(window_time_s):
            raise ValueError("window_time_s must be positive and finite")
        if sample_rate_hz <= 0.0 or not np.isfinite(sample_rate_hz):
            raise ValueError("sample_rate_hz must be positive and finite")
        sample_count = int(round(float(window_time_s) * float(sample_rate_hz)))
        if sample_count < 2:
            raise ValueError("learned motion window must contain at least two samples")
        self.window_time_s = float(window_time_s)
        self.sample_rate_hz = float(sample_rate_hz)
        self.sample_period_s = 1.0 / self.sample_rate_hz
        self.sample_count = sample_count
        self.coverage_tolerance_s = float(coverage_tolerance_s)
        self._timestamps: list[float] = []
        self._features: list[np.ndarray] = []

    def __len__(self) -> int:
        return len(self._timestamps)

    @property
    def first_timestamp_s(self) -> float | None:
        return None if not self._timestamps else self._timestamps[0]

    @property
    def last_timestamp_s(self) -> float | None:
        return None if not self._timestamps else self._timestamps[-1]

    def reset(self) -> None:
        self._timestamps.clear()
        self._features.clear()

    def append(self, timestamp_s: float, *, gyro_w, thrust_w) -> None:
        timestamp = float(timestamp_s)
        gyro = np.asarray(gyro_w, dtype=np.float64).reshape(3)
        thrust = np.asarray(thrust_w, dtype=np.float64).reshape(3)
        if not np.isfinite(timestamp) or not np.all(np.isfinite(gyro)) or not np.all(np.isfinite(thrust)):
            raise ValueError("motion samples must be finite")
        if self._timestamps and timestamp <= self._timestamps[-1] + 1.0e-12:
            raise ValueError("motion sample timestamps must be strictly monotonic")
        self._timestamps.append(timestamp)
        self._features.append(np.concatenate((gyro, thrust)))

    def discard_before(self, timestamp_s: float) -> None:
        """Drop samples older than ``timestamp_s``, retaining one interpolation predecessor."""
        if len(self._timestamps) <= 1:
            return
        timestamp = float(timestamp_s)
        idx = int(np.searchsorted(np.asarray(self._timestamps), timestamp, side="left"))
        keep_from = max(0, idx - 1)
        if keep_from:
            del self._timestamps[:keep_from]
            del self._features[:keep_from]

    def window(self, start_timestamp_s: float, end_timestamp_s: float) -> MotionWindow:
        start = float(start_timestamp_s)
        end = float(end_timestamp_s)
        duration = end - start
        if abs(duration - self.window_time_s) > max(1.0e-9, 0.25 * self.sample_period_s):
            raise ValueError(
                f"requested learned-motion duration {duration:.6f}s does not match configured "
                f"{self.window_time_s:.6f}s"
            )
        if not self._timestamps:
            raise ValueError("motion buffer does not cover requested window")

        sample_times = start + np.arange(self.sample_count, dtype=np.float64) * self.sample_period_s
        required_first = float(sample_times[0])
        required_last = float(sample_times[-1])
        if (
            self._timestamps[0] > required_first + self.coverage_tolerance_s
            or self._timestamps[-1] < required_last - self.coverage_tolerance_s
        ):
            raise ValueError("motion buffer does not cover requested window")

        source_t = np.asarray(self._timestamps, dtype=np.float64)
        source_f = np.asarray(self._features, dtype=np.float64)
        sampled = np.empty((self.sample_count, 6), dtype=np.float64)
        for channel in range(6):
            sampled[:, channel] = np.interp(sample_times, source_t, source_f[:, channel])
        return MotionWindow(
            features=sampled.T.astype(np.float32),
            timestamps_s=sample_times,
            start_timestamp_s=start,
            end_timestamp_s=end,
        )


def _import_torch():
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - depends on deployment environment
        raise ImportError(
            "PyTorch is required to construct or run the learned motion TCN"
        ) from exc
    return torch


def build_tcn(*, input_dim: int = 6, output_dim: int = 6):
    """Build a causal dilated residual TCN inspired by the UZH IMO architecture."""
    torch = _import_torch()
    nn = torch.nn

    class CausalResidualBlock(nn.Module):
        def __init__(self, in_channels: int, out_channels: int, dilation: int, dropout: float):
            super().__init__()
            padding = dilation
            self.net = nn.Sequential(
                nn.ConstantPad1d((padding, 0), 0.0),
                nn.Conv1d(in_channels, out_channels, kernel_size=2, dilation=dilation),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.ConstantPad1d((padding, 0), 0.0),
                nn.Conv1d(out_channels, out_channels, kernel_size=2, dilation=dilation),
                nn.GELU(),
                nn.Dropout(dropout),
            )
            self.skip = (
                nn.Identity() if in_channels == out_channels else nn.Conv1d(in_channels, out_channels, 1)
            )
            self.activation = nn.GELU()

        def forward(self, x):
            return self.activation(self.net(x) + self.skip(x))

    class LearnedMotionTcn(nn.Module):
        def __init__(self):
            super().__init__()
            channels = (64, 64, 64, 64, 128, 128, 128)
            blocks = []
            in_channels = input_dim
            for level, out_channels in enumerate(channels):
                blocks.append(CausalResidualBlock(in_channels, out_channels, 2**level, 0.2))
                in_channels = out_channels
            self.temporal = nn.Sequential(*blocks)
            self.head = nn.Linear(channels[-1], output_dim)

        def forward(self, features):
            encoded = self.temporal(features)
            return self.head(encoded[:, :, -1])

    return LearnedMotionTcn()


class TorchTcnDisplacementPredictor:
    """Checkpoint-backed displacement predictor with lazy PyTorch dependency."""

    def __init__(self, checkpoint_path, *, device: str = "cpu", variance_floor: float = 1.0e-6):
        torch = _import_torch()
        self._torch = torch
        self.device = torch.device(device)
        self.variance_floor = float(variance_floor)
        if self.variance_floor <= 0.0:
            raise ValueError("variance_floor must be positive")
        path = Path(checkpoint_path).expanduser()
        checkpoint = torch.load(path, map_location=self.device)
        metadata = dict(checkpoint.get("metadata", {}))
        if int(metadata.get("input_dim", 6)) != 6 or int(metadata.get("output_dim", 6)) != 6:
            raise ValueError("learned motion checkpoint must use input_dim=6 and output_dim=6")
        self.window_time_s = float(metadata.get("window_time_s", 0.5))
        self.sample_rate_hz = float(metadata.get("sample_rate_hz", 100.0))
        self.target_mode = str(metadata.get("target_mode", "displacement"))
        if self.target_mode not in (
            "displacement",
            "kinematic_residual",
            "kinematic_residual_body_end",
        ):
            raise ValueError(
                f"unsupported learned-motion target_mode: {self.target_mode!r}"
            )
        self.feature_frame = str(metadata.get("feature_frame", "world"))
        if self.feature_frame not in ("world", "body"):
            raise ValueError(
                f"unsupported learned-motion feature_frame: {self.feature_frame!r}"
            )
        if (
            self.target_mode == "kinematic_residual_body_end"
            and self.feature_frame != "body"
        ):
            raise ValueError(
                "kinematic_residual_body_end checkpoints must use body-frame features"
            )
        self.model = build_tcn(input_dim=6, output_dim=6).to(self.device)
        state = checkpoint.get("model_state_dict", checkpoint)
        self.model.load_state_dict(state)
        self.model.eval()

    def predict(self, window: MotionWindow) -> DisplacementPrediction:
        if abs((window.end_timestamp_s - window.start_timestamp_s) - self.window_time_s) > 1.0e-6:
            raise ValueError("motion window duration does not match checkpoint metadata")
        expected_samples = int(round(self.window_time_s * self.sample_rate_hz))
        if window.features.shape != (6, expected_samples):
            raise ValueError("motion window sample count does not match checkpoint metadata")
        tensor = self._torch.from_numpy(window.features[None, ...]).to(
            device=self.device, dtype=self._torch.float32
        )
        with self._torch.no_grad():
            output = self.model(tensor)[0].detach().cpu().numpy().astype(np.float64)
        displacement = output[:3]
        log_variance = np.clip(output[3:], -12.0, 6.0)
        variance = np.exp(log_variance) + self.variance_floor
        return DisplacementPrediction(displacement, np.diag(variance))
