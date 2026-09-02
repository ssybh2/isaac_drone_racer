"""Configuration objects for phenomenological Stage 1 sensor errors."""

from __future__ import annotations

import math
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


@dataclass
class FakeSensorPipelineCfg:
    """Named error profile plus its concrete, overrideable source configs."""

    profile: str = "clean"
    vio: FakeVioCfg = field(default_factory=FakeVioCfg.clean)
    imu: FakeImuCfg = field(default_factory=FakeImuCfg.clean)
    seed_offset: int = 10_000

    @classmethod
    def from_profile(cls, profile: str) -> FakeSensorPipelineCfg:
        vio, imu = fake_sensor_profile(profile)
        return cls(profile=profile, vio=vio, imu=imu)


def fake_sensor_profile(profile: str) -> tuple[FakeVioCfg, FakeImuCfg]:
    """Build clean, curriculum, nominal, or held-out stress priors."""
    degree = math.pi / 180.0
    if profile == "clean":
        return FakeVioCfg.clean(update_period_s=0.01), FakeImuCfg.clean(update_period_s=0.0025)
    if profile == "mild":
        return (
            FakeVioCfg(
                update_period_s=0.01,
                position_noise_std_m=UniformRange(0.002, 0.01),
                orientation_noise_std_rad=UniformRange(0.05 * degree, 0.3 * degree),
                velocity_noise_std_mps=UniformRange(0.005, 0.03),
                position_bias_m=UniformRange(-0.01, 0.01),
                orientation_bias_rad=UniformRange(-0.2 * degree, 0.2 * degree),
                velocity_bias_mps=UniformRange(-0.02, 0.02),
                position_drift_std_m_per_sqrt_s=UniformRange(0.0, 0.004),
                yaw_drift_std_rad_per_sqrt_s=UniformRange(0.0, 0.05 * degree),
                velocity_drift_std_mps_per_sqrt_s=UniformRange(0.0, 0.006),
                latency_s=UniformRange(0.0, 0.01),
                dropout_probability=UniformRange(0.0, 0.005),
            ),
            FakeImuCfg(
                update_period_s=0.0025,
                noise_std_radps=UniformRange(0.0005, 0.003),
                bias_radps=UniformRange(-0.002, 0.002),
                bias_random_walk_std_radps_per_sqrt_s=UniformRange(0.0, 0.0005),
                latency_s=UniformRange(0.0, 0.0025),
                dropout_probability=UniformRange(0.0, 0.001),
            ),
        )
    if profile == "nominal":
        return (
            FakeVioCfg(
                update_period_s=0.02,
                position_noise_std_m=UniformRange(0.005, 0.03),
                orientation_noise_std_rad=UniformRange(0.1 * degree, 1.0 * degree),
                velocity_noise_std_mps=UniformRange(0.01, 0.10),
                position_bias_m=UniformRange(-0.03, 0.03),
                orientation_bias_rad=UniformRange(-0.5 * degree, 0.5 * degree),
                velocity_bias_mps=UniformRange(-0.05, 0.05),
                position_drift_std_m_per_sqrt_s=UniformRange(0.001, 0.02),
                roll_pitch_drift_std_rad_per_sqrt_s=UniformRange(0.0, 0.05 * degree),
                yaw_drift_std_rad_per_sqrt_s=UniformRange(0.01 * degree, 0.30 * degree),
                velocity_drift_std_mps_per_sqrt_s=UniformRange(0.002, 0.03),
                latency_s=UniformRange(0.0, 0.04),
                dropout_probability=UniformRange(0.0, 0.03),
                burst_dropout_probability=UniformRange(0.0, 0.002),
                burst_duration_s=UniformRange(0.0, 0.10),
            ),
            FakeImuCfg(
                update_period_s=0.0025,
                noise_std_radps=UniformRange(0.001, 0.01),
                bias_radps=UniformRange(-0.01, 0.01),
                bias_random_walk_std_radps_per_sqrt_s=UniformRange(0.0001, 0.002),
                latency_s=UniformRange(0.0, 0.005),
                dropout_probability=UniformRange(0.0, 0.005),
            ),
        )
    if profile == "stress":
        return (
            FakeVioCfg(
                update_period_s=0.04,
                position_noise_std_m=UniformRange(0.03, 0.10),
                orientation_noise_std_rad=UniformRange(1.0 * degree, 3.0 * degree),
                velocity_noise_std_mps=UniformRange(0.10, 0.30),
                position_bias_m=UniformRange(-0.10, 0.10),
                orientation_bias_rad=UniformRange(-2.0 * degree, 2.0 * degree),
                velocity_bias_mps=UniformRange(-0.15, 0.15),
                position_drift_std_m_per_sqrt_s=UniformRange(0.02, 0.06),
                roll_pitch_drift_std_rad_per_sqrt_s=UniformRange(0.05 * degree, 0.2 * degree),
                yaw_drift_std_rad_per_sqrt_s=UniformRange(0.30 * degree, 1.0 * degree),
                velocity_drift_std_mps_per_sqrt_s=UniformRange(0.03, 0.10),
                latency_s=UniformRange(0.04, 0.10),
                dropout_probability=UniformRange(0.03, 0.10),
                burst_dropout_probability=UniformRange(0.002, 0.01),
                burst_duration_s=UniformRange(0.10, 0.30),
            ),
            FakeImuCfg(
                update_period_s=0.005,
                noise_std_radps=UniformRange(0.01, 0.04),
                bias_radps=UniformRange(-0.03, 0.03),
                bias_random_walk_std_radps_per_sqrt_s=UniformRange(0.002, 0.008),
                latency_s=UniformRange(0.005, 0.015),
                dropout_probability=UniformRange(0.005, 0.02),
                burst_dropout_probability=UniformRange(0.0, 0.002),
                burst_duration_s=UniformRange(0.0, 0.03),
            ),
        )
    raise ValueError(f"unknown Fake Sensor profile: {profile!r}")
