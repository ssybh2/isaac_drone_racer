"""Tensorized phenomenological gyroscope provider."""

from __future__ import annotations

import math

import torch

from .fake_sensor_cfg import FakeImuCfg, UniformRange
from .state_estimate import ImuEstimate, SourceStatus


class FakeImu:
    """Generate delayed and imperfect body-frame angular velocity samples."""

    def __init__(self, num_envs: int, device: str | torch.device, cfg: FakeImuCfg, seed: int) -> None:
        self.num_envs = num_envs
        self.device = torch.device(device)
        self.cfg = cfg
        self.generator = torch.Generator(device=self.device)
        self.generator.manual_seed(seed)

        history_length = math.ceil(cfg.latency_s.high / cfg.update_period_s) + 3
        self._history_length = max(3, history_length)
        self._write_index = torch.zeros(num_envs, dtype=torch.long, device=self.device)
        self._history_timestamp_s = torch.full(
            (self._history_length, num_envs), -torch.inf, device=self.device
        )
        self._history_angular_velocity = torch.zeros(
            self._history_length, num_envs, 3, device=self.device
        )

        self._last_angular_velocity = torch.zeros(num_envs, 3, device=self.device)
        self._last_timestamp_s = torch.zeros(num_envs, device=self.device)
        self._last_call_timestamp_s = torch.zeros(num_envs, device=self.device)
        self._last_source_timestamp_s = torch.zeros(num_envs, device=self.device)
        self._next_update_timestamp_s = torch.zeros(num_envs, device=self.device)
        self._valid = torch.zeros(num_envs, dtype=torch.bool, device=self.device)

        self._bias = torch.zeros(num_envs, 3, device=self.device)
        self._bias_random_walk = torch.zeros(num_envs, 3, device=self.device)
        self._noise_std = torch.zeros(num_envs, 1, device=self.device)
        self._bias_random_walk_std = torch.zeros(num_envs, 1, device=self.device)
        self._latency_s = torch.zeros(num_envs, device=self.device)
        self._dropout_probability = torch.zeros(num_envs, device=self.device)
        self._burst_probability = torch.zeros(num_envs, device=self.device)
        self._burst_duration_s = torch.zeros(num_envs, device=self.device)
        self._burst_time_left_s = torch.zeros(num_envs, device=self.device)

    def _sample(self, value_range: UniformRange, shape: tuple[int, ...]) -> torch.Tensor:
        if value_range.low == value_range.high:
            return torch.full(shape, value_range.low, device=self.device)
        return torch.rand(shape, generator=self.generator, device=self.device) * (
            value_range.high - value_range.low
        ) + value_range.low

    def _normal(self, shape: tuple[int, ...]) -> torch.Tensor:
        return torch.randn(shape, generator=self.generator, device=self.device)

    def _measurement(self, angular_velocity_b: torch.Tensor, env_ids: torch.Tensor) -> torch.Tensor:
        count = len(env_ids)
        return (
            angular_velocity_b[env_ids]
            + self._bias[env_ids]
            + self._bias_random_walk[env_ids]
            + self._normal((count, 3)) * self._noise_std[env_ids]
        )

    def reset(
        self, env_ids: torch.Tensor, angular_velocity_b: torch.Tensor, timestamp_s: float | torch.Tensor
    ) -> ImuEstimate:
        env_ids = env_ids.to(device=self.device, dtype=torch.long)
        timestamp = self._timestamp_tensor(timestamp_s)
        count = len(env_ids)

        self._bias[env_ids] = self._sample(self.cfg.bias_radps, (count, 3))
        self._bias_random_walk[env_ids] = 0.0
        self._noise_std[env_ids] = self._sample(self.cfg.noise_std_radps, (count, 1))
        self._bias_random_walk_std[env_ids] = self._sample(
            self.cfg.bias_random_walk_std_radps_per_sqrt_s, (count, 1)
        )
        self._latency_s[env_ids] = self._sample(self.cfg.latency_s, (count,))
        self._dropout_probability[env_ids] = self._sample(self.cfg.dropout_probability, (count,))
        self._burst_probability[env_ids] = self._sample(
            self.cfg.burst_dropout_probability, (count,)
        )
        self._burst_duration_s[env_ids] = self._sample(self.cfg.burst_duration_s, (count,))
        self._burst_time_left_s[env_ids] = 0.0

        measurement = self._measurement(angular_velocity_b, env_ids)
        self._history_timestamp_s[:, env_ids] = timestamp[env_ids].unsqueeze(0)
        self._history_angular_velocity[:, env_ids] = measurement.unsqueeze(0)
        self._write_index[env_ids] = 0
        self._last_angular_velocity[env_ids] = measurement
        self._last_timestamp_s[env_ids] = timestamp[env_ids]
        self._last_call_timestamp_s[env_ids] = timestamp[env_ids]
        self._last_source_timestamp_s[env_ids] = timestamp[env_ids]
        self._next_update_timestamp_s[env_ids] = timestamp[env_ids] + self.cfg.update_period_s
        self._valid[env_ids] = True
        return self.estimate(timestamp)

    def update(
        self, angular_velocity_b: torch.Tensor, timestamp_s: float | torch.Tensor
    ) -> ImuEstimate:
        timestamp = self._timestamp_tensor(timestamp_s)
        dt = (timestamp - self._last_call_timestamp_s).clamp_min(0.0)
        self._last_call_timestamp_s = timestamp
        self._burst_time_left_s = (self._burst_time_left_s - dt).clamp_min(0.0)

        due = timestamp >= self._next_update_timestamp_s - 1.0e-7
        self._valid[:] = False
        env_ids = torch.arange(self.num_envs, device=self.device)
        source_dt = (timestamp - self._last_source_timestamp_s).clamp_min(0.0)
        next_bias_random_walk = self._bias_random_walk + (
            self._normal((self.num_envs, 3))
            * self._bias_random_walk_std
            * source_dt.sqrt().unsqueeze(-1)
        )
        self._bias_random_walk = torch.where(
            due.unsqueeze(-1), next_bias_random_walk, self._bias_random_walk
        )
        self._last_source_timestamp_s = torch.where(
            due, timestamp, self._last_source_timestamp_s
        )

        measurement = self._measurement(angular_velocity_b, env_ids)
        slots = self._write_index
        self._history_timestamp_s[slots, env_ids] = torch.where(
            due, timestamp, self._history_timestamp_s[slots, env_ids]
        )
        self._history_angular_velocity[slots, env_ids] = torch.where(
            due.unsqueeze(-1), measurement, self._history_angular_velocity[slots, env_ids]
        )
        self._write_index = torch.where(due, (slots + 1) % self._history_length, slots)
        periods = torch.floor(
            (timestamp - self._next_update_timestamp_s) / self.cfg.update_period_s
        ).clamp_min(0.0) + 1.0
        self._next_update_timestamp_s = torch.where(
            due,
            self._next_update_timestamp_s + periods * self.cfg.update_period_s,
            self._next_update_timestamp_s,
        )

        target_timestamp = timestamp - self._latency_s
        eligible = self._history_timestamp_s <= target_timestamp.unsqueeze(0) + 1.0e-7
        eligible_times = torch.where(eligible, self._history_timestamp_s, -torch.inf)
        candidate_slots = eligible_times.argmax(dim=0)
        candidate_timestamp = self._history_timestamp_s[candidate_slots, env_ids]
        fresh = candidate_timestamp > self._last_timestamp_s + 1.0e-7

        independent_dropout = torch.rand(
            (self.num_envs,), generator=self.generator, device=self.device
        ) < self._dropout_probability
        starts_burst = fresh & (
            torch.rand((self.num_envs,), generator=self.generator, device=self.device)
            < self._burst_probability
        )
        self._burst_time_left_s = torch.where(
            starts_burst, self._burst_duration_s, self._burst_time_left_s
        )
        deliver = fresh & ~independent_dropout & (self._burst_time_left_s <= 0.0)

        self._last_angular_velocity = torch.where(
            deliver.unsqueeze(-1),
            self._history_angular_velocity[candidate_slots, env_ids],
            self._last_angular_velocity,
        )
        self._last_timestamp_s = torch.where(deliver, candidate_timestamp, self._last_timestamp_s)
        self._valid = deliver
        return self.estimate(timestamp)

    def _timestamp_tensor(self, timestamp_s: float | torch.Tensor) -> torch.Tensor:
        if isinstance(timestamp_s, torch.Tensor):
            value = timestamp_s.to(device=self.device, dtype=torch.float32)
            if value.ndim == 0:
                return value.expand(self.num_envs).clone()
            if tuple(value.shape) != (self.num_envs,):
                raise ValueError(f"timestamp must have shape ({self.num_envs},)")
            return value
        return torch.full((self.num_envs,), float(timestamp_s), device=self.device)

    def estimate(self, publish_timestamp_s: float | torch.Tensor) -> ImuEstimate:
        publish_timestamp = self._timestamp_tensor(publish_timestamp_s)
        return ImuEstimate(
            angular_velocity_b=self._last_angular_velocity.clone(),
            status=SourceStatus(
                timestamp_s=self._last_timestamp_s.clone(),
                age_s=(publish_timestamp - self._last_timestamp_s).clamp_min(0.0),
                valid=self._valid.clone(),
            ),
        )
