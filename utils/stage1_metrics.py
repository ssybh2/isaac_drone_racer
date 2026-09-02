"""Episode-level Stage 1 evaluation aggregation."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from statistics import fmean


@dataclass(frozen=True)
class EpisodeRecord:
    completed: bool
    gates_passed: int
    collision: bool
    flyaway: bool
    episode_return: float
    duration_s: float
    position_rmse_m: float
    attitude_rmse_rad: float
    velocity_rmse_mps: float
    vio_age_s: tuple[float, ...] = ()
    imu_age_s: tuple[float, ...] = ()
    vio_valid: tuple[bool, ...] = ()
    imu_valid: tuple[bool, ...] = ()
    vio_dropped: tuple[bool, ...] = ()
    imu_dropped: tuple[bool, ...] = ()

    def __post_init__(self) -> None:
        if len(self.vio_age_s) != len(self.vio_valid):
            raise ValueError("VIO age and validity sample counts must match")
        if len(self.imu_age_s) != len(self.imu_valid):
            raise ValueError("IMU age and validity sample counts must match")
        if self.vio_dropped and len(self.vio_age_s) != len(self.vio_dropped):
            raise ValueError("VIO age and dropout sample counts must match")
        if self.imu_dropped and len(self.imu_age_s) != len(self.imu_dropped):
            raise ValueError("IMU age and dropout sample counts must match")


def _mean(values: list[float]) -> float:
    return fmean(values) if values else 0.0


def _quantile(values: list[float], probability: float) -> float:
    """Linear quantile matching the common ``(n - 1) * q`` definition."""
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


class Stage1EpisodeAccumulator:
    """Collect immutable episode records and emit JSON/CSV-ready summaries."""

    def __init__(self) -> None:
        self.records: list[EpisodeRecord] = []

    def add(self, record: EpisodeRecord) -> None:
        self.records.append(record)

    def rows(self) -> list[dict]:
        return [asdict(record) for record in self.records]

    def summary(self) -> dict[str, float | int]:
        records = self.records
        count = len(records)
        vio_ages = [sample for record in records for sample in record.vio_age_s]
        imu_ages = [sample for record in records for sample in record.imu_age_s]
        vio_valid = [sample for record in records for sample in record.vio_valid]
        imu_valid = [sample for record in records for sample in record.imu_valid]
        vio_dropped = [sample for record in records for sample in record.vio_dropped]
        imu_dropped = [sample for record in records for sample in record.imu_dropped]
        return {
            "episodes": count,
            "completion_rate": _mean([float(record.completed) for record in records]),
            "collision_rate": _mean([float(record.collision) for record in records]),
            "flyaway_rate": _mean([float(record.flyaway) for record in records]),
            "mean_gates_passed": _mean([float(record.gates_passed) for record in records]),
            "mean_return": _mean([record.episode_return for record in records]),
            "mean_duration_s": _mean([record.duration_s for record in records]),
            "mean_position_rmse_m": _mean([record.position_rmse_m for record in records]),
            "mean_attitude_rmse_rad": _mean([record.attitude_rmse_rad for record in records]),
            "mean_velocity_rmse_mps": _mean([record.velocity_rmse_mps for record in records]),
            "vio_age_mean_s": _mean(vio_ages),
            "vio_age_p95_s": _quantile(vio_ages, 0.95),
            "imu_age_mean_s": _mean(imu_ages),
            "imu_age_p95_s": _quantile(imu_ages, 0.95),
            "vio_fresh_fraction": _mean([float(value) for value in vio_valid]),
            "imu_fresh_fraction": _mean([float(value) for value in imu_valid]),
            "vio_dropout_fraction": _mean([float(value) for value in vio_dropped])
            if vio_dropped
            else 0.0,
            "imu_dropout_fraction": _mean([float(value) for value in imu_dropped])
            if imu_dropped
            else 0.0,
        }
