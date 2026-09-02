"""Configuration objects for phenomenological Stage 1 sensor errors."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class UniformRange:
    """Closed numeric range sampled uniformly per environment and episode."""

    low: float = 0.0
    high: float = 0.0

    def __post_init__(self) -> None:
        if self.low > self.high:
            raise ValueError(f"range lower bound {self.low} exceeds upper bound {self.high}")


def _zero_range() -> UniformRange:
    return UniformRange(0.0, 0.0)


@dataclass
class FakeVioCfg:
    """Fake VIO source timing and episode-randomized error ranges."""

    update_period_s: float = 0.01
    position_noise_std_m: UniformRange = field(default_factory=_zero_range)
    orientation_noise_std_rad: UniformRange = field(default_factory=_zero_range)
    velocity_noise_std_mps: UniformRange = field(default_factory=_zero_range)
    position_bias_m: UniformRange = field(default_factory=_zero_range)
    orientation_bias_rad: UniformRange = field(default_factory=_zero_range)
    velocity_bias_mps: UniformRange = field(default_factory=_zero_range)
    position_drift_std_m_per_sqrt_s: UniformRange = field(default_factory=_zero_range)
    roll_pitch_drift_std_rad_per_sqrt_s: UniformRange = field(default_factory=_zero_range)
    yaw_drift_std_rad_per_sqrt_s: UniformRange = field(default_factory=_zero_range)
    velocity_drift_std_mps_per_sqrt_s: UniformRange = field(default_factory=_zero_range)
    latency_s: UniformRange = field(default_factory=_zero_range)
    dropout_probability: UniformRange = field(default_factory=_zero_range)
    burst_dropout_probability: UniformRange = field(default_factory=_zero_range)
    burst_duration_s: UniformRange = field(default_factory=_zero_range)

    def __post_init__(self) -> None:
        if self.update_period_s <= 0.0:
            raise ValueError("update_period_s must be positive")
        for name in (
            "position_noise_std_m",
            "orientation_noise_std_rad",
            "velocity_noise_std_mps",
            "position_drift_std_m_per_sqrt_s",
            "roll_pitch_drift_std_rad_per_sqrt_s",
            "yaw_drift_std_rad_per_sqrt_s",
            "velocity_drift_std_mps_per_sqrt_s",
            "latency_s",
            "dropout_probability",
            "burst_dropout_probability",
            "burst_duration_s",
        ):
            value = getattr(self, name)
            if value.low < 0.0:
                raise ValueError(f"{name} must be non-negative")
        if self.dropout_probability.high > 1.0 or self.burst_dropout_probability.high > 1.0:
            raise ValueError("dropout probability must not exceed one")

    @classmethod
    def clean(cls, update_period_s: float = 0.01) -> FakeVioCfg:
        return cls(update_period_s=update_period_s)


@dataclass
class FakeImuCfg:
    """Fake gyroscope source timing and episode-randomized error ranges."""

    update_period_s: float = 0.0025
    noise_std_radps: UniformRange = field(default_factory=_zero_range)
    bias_radps: UniformRange = field(default_factory=_zero_range)
    bias_random_walk_std_radps_per_sqrt_s: UniformRange = field(default_factory=_zero_range)
    latency_s: UniformRange = field(default_factory=_zero_range)
    dropout_probability: UniformRange = field(default_factory=_zero_range)
    burst_dropout_probability: UniformRange = field(default_factory=_zero_range)
    burst_duration_s: UniformRange = field(default_factory=_zero_range)

    def __post_init__(self) -> None:
        if self.update_period_s <= 0.0:
            raise ValueError("update_period_s must be positive")
        for name in (
            "noise_std_radps",
            "bias_random_walk_std_radps_per_sqrt_s",
            "latency_s",
            "dropout_probability",
            "burst_dropout_probability",
            "burst_duration_s",
        ):
            value = getattr(self, name)
            if value.low < 0.0:
                raise ValueError(f"{name} must be non-negative")
        if self.dropout_probability.high > 1.0 or self.burst_dropout_probability.high > 1.0:
            raise ValueError("dropout probability must not exceed one")

    @classmethod
    def clean(cls, update_period_s: float = 0.0025) -> FakeImuCfg:
        return cls(update_period_s=update_period_s)
