"""Tensorized phenomenological VIO error provider."""

from __future__ import annotations

import math

import torch

from .fake_sensor_cfg import FakeVioCfg, UniformRange
from .frame_math import quaternion_from_rotation_vector, quaternion_multiply
from .state_estimate import GroundTruthState, SourceStatus, VioEstimate


class FakeVio:
    """Generate delayed and imperfect VIO states for many environments."""

    def __init__(self, num_envs: int, device: str | torch.device, cfg: FakeVioCfg, seed: int) -> None:
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
        self._history_position = torch.zeros(self._history_length, num_envs, 3, device=self.device)
        self._history_orientation = torch.zeros(self._history_length, num_envs, 4, device=self.device)
        self._history_velocity = torch.zeros(self._history_length, num_envs, 3, device=self.device)

        self._last_position = torch.zeros(num_envs, 3, device=self.device)
        self._last_orientation = torch.zeros(num_envs, 4, device=self.device)
        self._last_orientation[:, 0] = 1.0
        self._last_velocity = torch.zeros(num_envs, 3, device=self.device)
        self._last_timestamp_s = torch.zeros(num_envs, device=self.device)
        self._last_call_timestamp_s = torch.zeros(num_envs, device=self.device)
        self._next_update_timestamp_s = torch.zeros(num_envs, device=self.device)
        self._valid = torch.zeros(num_envs, dtype=torch.bool, device=self.device)

        self._position_bias = torch.zeros(num_envs, 3, device=self.device)
        self._orientation_bias = torch.zeros(num_envs, 3, device=self.device)
        self._velocity_bias = torch.zeros(num_envs, 3, device=self.device)
        self._position_drift = torch.zeros(num_envs, 3, device=self.device)
        self._orientation_drift = torch.zeros(num_envs, 3, device=self.device)
        self._velocity_drift = torch.zeros(num_envs, 3, device=self.device)
        self._position_noise_std = torch.zeros(num_envs, 1, device=self.device)
        self._orientation_noise_std = torch.zeros(num_envs, 1, device=self.device)
        self._velocity_noise_std = torch.zeros(num_envs, 1, device=self.device)
        self._position_drift_std = torch.zeros(num_envs, 1, device=self.device)
        self._roll_pitch_drift_std = torch.zeros(num_envs, 1, device=self.device)
        self._yaw_drift_std = torch.zeros(num_envs, 1, device=self.device)
        self._velocity_drift_std = torch.zeros(num_envs, 1, device=self.device)
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

    def _measurement(self, ground_truth: GroundTruthState, env_ids: torch.Tensor) -> tuple[torch.Tensor, ...]:
        count = len(env_ids)
        position = (
            ground_truth.position_w_b[env_ids]
            + self._position_bias[env_ids]
            + self._position_drift[env_ids]
            + self._normal((count, 3)) * self._position_noise_std[env_ids]
        )
        rotation_error = (
            self._orientation_bias[env_ids]
            + self._orientation_drift[env_ids]
            + self._normal((count, 3)) * self._orientation_noise_std[env_ids]
        )
        orientation = quaternion_multiply(
            ground_truth.orientation_w_b[env_ids], quaternion_from_rotation_vector(rotation_error)
        )
        velocity = (
            ground_truth.linear_velocity_w_b[env_ids]
            + self._velocity_bias[env_ids]
            + self._velocity_drift[env_ids]
            + self._normal((count, 3)) * self._velocity_noise_std[env_ids]
        )
        return position, orientation, velocity

    def _write_history(
        self,
        env_ids: torch.Tensor,
        timestamp_s: torch.Tensor,
        position: torch.Tensor,
        orientation: torch.Tensor,
        velocity: torch.Tensor,
    ) -> None:
        slots = self._write_index[env_ids]
        self._history_timestamp_s[slots, env_ids] = timestamp_s[env_ids]
        self._history_position[slots, env_ids] = position
        self._history_orientation[slots, env_ids] = orientation
        self._history_velocity[slots, env_ids] = velocity
        self._write_index[env_ids] = (slots + 1) % self._history_length

    def reset(
        self, env_ids: torch.Tensor, ground_truth: GroundTruthState, timestamp_s: float | torch.Tensor
    ) -> VioEstimate:
        env_ids = env_ids.to(device=self.device, dtype=torch.long)
        timestamp = self._timestamp_tensor(timestamp_s)
        count = len(env_ids)

        self._position_bias[env_ids] = self._sample(self.cfg.position_bias_m, (count, 3))
        self._orientation_bias[env_ids] = self._sample(self.cfg.orientation_bias_rad, (count, 3))
        self._velocity_bias[env_ids] = self._sample(self.cfg.velocity_bias_mps, (count, 3))
        self._position_drift[env_ids] = 0.0
        self._orientation_drift[env_ids] = 0.0
        self._velocity_drift[env_ids] = 0.0
        self._position_noise_std[env_ids] = self._sample(self.cfg.position_noise_std_m, (count, 1))
        self._orientation_noise_std[env_ids] = self._sample(self.cfg.orientation_noise_std_rad, (count, 1))
        self._velocity_noise_std[env_ids] = self._sample(self.cfg.velocity_noise_std_mps, (count, 1))
        self._position_drift_std[env_ids] = self._sample(
            self.cfg.position_drift_std_m_per_sqrt_s, (count, 1)
        )
        self._roll_pitch_drift_std[env_ids] = self._sample(
            self.cfg.roll_pitch_drift_std_rad_per_sqrt_s, (count, 1)
        )
        self._yaw_drift_std[env_ids] = self._sample(self.cfg.yaw_drift_std_rad_per_sqrt_s, (count, 1))
        self._velocity_drift_std[env_ids] = self._sample(
            self.cfg.velocity_drift_std_mps_per_sqrt_s, (count, 1)
        )
        self._latency_s[env_ids] = self._sample(self.cfg.latency_s, (count,))
        self._dropout_probability[env_ids] = self._sample(self.cfg.dropout_probability, (count,))
        self._burst_probability[env_ids] = self._sample(self.cfg.burst_dropout_probability, (count,))
        self._burst_duration_s[env_ids] = self._sample(self.cfg.burst_duration_s, (count,))
        self._burst_time_left_s[env_ids] = 0.0

        position, orientation, velocity = self._measurement(ground_truth, env_ids)
        self._history_timestamp_s[:, env_ids] = timestamp[env_ids].unsqueeze(0)
        self._history_position[:, env_ids] = position.unsqueeze(0)
        self._history_orientation[:, env_ids] = orientation.unsqueeze(0)
        self._history_velocity[:, env_ids] = velocity.unsqueeze(0)
        self._write_index[env_ids] = 0
        self._last_position[env_ids] = position
        self._last_orientation[env_ids] = orientation
        self._last_velocity[env_ids] = velocity
        self._last_timestamp_s[env_ids] = timestamp[env_ids]
        self._last_call_timestamp_s[env_ids] = timestamp[env_ids]
        self._next_update_timestamp_s[env_ids] = timestamp[env_ids] + self.cfg.update_period_s
        self._valid[:] = False
        self._valid[env_ids] = True
        return self.estimate(timestamp)

    def update(self, ground_truth: GroundTruthState, timestamp_s: float | torch.Tensor) -> VioEstimate:
        timestamp = self._timestamp_tensor(timestamp_s)
        dt = (timestamp - self._last_call_timestamp_s).clamp_min(0.0)
        self._last_call_timestamp_s = timestamp
        self._burst_time_left_s = (self._burst_time_left_s - dt).clamp_min(0.0)

        due = timestamp >= self._next_update_timestamp_s - 1.0e-7
        self._valid[:] = False
        env_ids = torch.arange(self.num_envs, device=self.device)
        due_dt = dt.sqrt().unsqueeze(-1)
        position_drift = self._position_drift + (
            self._normal((self.num_envs, 3)) * self._position_drift_std * due_dt
        )
        orientation_scale = self._roll_pitch_drift_std.expand(-1, 3).clone()
        orientation_scale[:, 2] = self._yaw_drift_std[:, 0]
        orientation_drift = self._orientation_drift + (
            self._normal((self.num_envs, 3)) * orientation_scale * due_dt
        )
        velocity_drift = self._velocity_drift + (
            self._normal((self.num_envs, 3)) * self._velocity_drift_std * due_dt
        )
        self._position_drift = torch.where(due.unsqueeze(-1), position_drift, self._position_drift)
        self._orientation_drift = torch.where(due.unsqueeze(-1), orientation_drift, self._orientation_drift)
        self._velocity_drift = torch.where(due.unsqueeze(-1), velocity_drift, self._velocity_drift)

        position, orientation, velocity = self._measurement(ground_truth, env_ids)
        slots = self._write_index
        self._history_timestamp_s[slots, env_ids] = torch.where(
            due, timestamp, self._history_timestamp_s[slots, env_ids]
        )
        self._history_position[slots, env_ids] = torch.where(
            due.unsqueeze(-1), position, self._history_position[slots, env_ids]
        )
        self._history_orientation[slots, env_ids] = torch.where(
            due.unsqueeze(-1), orientation, self._history_orientation[slots, env_ids]
        )
        self._history_velocity[slots, env_ids] = torch.where(
            due.unsqueeze(-1), velocity, self._history_velocity[slots, env_ids]
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
            torch.rand((self.num_envs,), generator=self.generator, device=self.device) < self._burst_probability
        )
        self._burst_time_left_s = torch.where(starts_burst, self._burst_duration_s, self._burst_time_left_s)
        deliver = fresh & ~independent_dropout & (self._burst_time_left_s <= 0.0)

        self._last_position = torch.where(
            deliver.unsqueeze(-1), self._history_position[candidate_slots, env_ids], self._last_position
        )
        self._last_orientation = torch.where(
            deliver.unsqueeze(-1), self._history_orientation[candidate_slots, env_ids], self._last_orientation
        )
        self._last_velocity = torch.where(
            deliver.unsqueeze(-1), self._history_velocity[candidate_slots, env_ids], self._last_velocity
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

    def estimate(self, publish_timestamp_s: float | torch.Tensor) -> VioEstimate:
        publish_timestamp = self._timestamp_tensor(publish_timestamp_s)
        return VioEstimate(
            position_v_b=self._last_position.clone(),
            orientation_v_b=self._last_orientation.clone(),
            linear_velocity_v_b=self._last_velocity.clone(),
            status=SourceStatus(
                timestamp_s=self._last_timestamp_s.clone(),
                age_s=(publish_timestamp - self._last_timestamp_s).clamp_min(0.0),
                valid=self._valid.clone(),
            ),
        )
