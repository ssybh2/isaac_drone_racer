"""World-frame alignment and asynchronous source assembly."""

from __future__ import annotations

import torch

from .frame_math import compose_transform_w_b, rotate_vector
from .state_estimate import ImuEstimate, SourceStatus, StateEstimate, VioEstimate


class StateEstimateAssembler:
    """Apply ``T_WV`` and publish a hardware-portable policy state snapshot."""

    def __init__(self, position_w_v: torch.Tensor, orientation_w_v: torch.Tensor) -> None:
        if position_w_v.ndim != 2 or position_w_v.shape[1] != 3:
            raise ValueError("position_w_v must have shape (num_envs, 3)")
        if orientation_w_v.shape != (position_w_v.shape[0], 4):
            raise ValueError("orientation_w_v must have shape (num_envs, 4)")
        if position_w_v.device != orientation_w_v.device:
            raise ValueError("alignment tensors must use the same device")
        self.num_envs = position_w_v.shape[0]
        self.device = position_w_v.device
        self.position_w_v = position_w_v.clone()
        self.orientation_w_v = orientation_w_v.clone()

    @classmethod
    def identity(cls, num_envs: int, device: str | torch.device) -> StateEstimateAssembler:
        position = torch.zeros(num_envs, 3, device=device)
        orientation = torch.zeros(num_envs, 4, device=device)
        orientation[:, 0] = 1.0
        return cls(position, orientation)

    def set_alignment(
        self,
        env_ids: torch.Tensor,
        position_w_v: torch.Tensor,
        orientation_w_v: torch.Tensor,
    ) -> None:
        env_ids = env_ids.to(device=self.device, dtype=torch.long)
        expected_count = env_ids.shape[0]
        if position_w_v.shape != (expected_count, 3):
            raise ValueError("position_w_v shape must match env_ids")
        if orientation_w_v.shape != (expected_count, 4):
            raise ValueError("orientation_w_v shape must match env_ids")
        self.position_w_v[env_ids] = position_w_v
        self.orientation_w_v[env_ids] = orientation_w_v

    def _timestamp_tensor(self, timestamp_s: float | torch.Tensor) -> torch.Tensor:
        if isinstance(timestamp_s, torch.Tensor):
            value = timestamp_s.to(device=self.device, dtype=torch.float32)
            if value.ndim == 0:
                return value.expand(self.num_envs).clone()
            if tuple(value.shape) != (self.num_envs,):
                raise ValueError(f"timestamp must have shape ({self.num_envs},)")
            return value
        return torch.full((self.num_envs,), float(timestamp_s), device=self.device)

    @staticmethod
    def _status_at_publish(source: SourceStatus, publish_timestamp_s: torch.Tensor) -> SourceStatus:
        return SourceStatus(
            timestamp_s=source.timestamp_s,
            age_s=(publish_timestamp_s - source.timestamp_s).clamp_min(0.0),
            valid=source.valid,
        )

    def assemble(
        self,
        vio: VioEstimate,
        imu: ImuEstimate,
        publish_timestamp_s: float | torch.Tensor,
    ) -> StateEstimate:
        publish_timestamp = self._timestamp_tensor(publish_timestamp_s)
        position_w_b, orientation_w_b = compose_transform_w_b(
            self.position_w_v,
            self.orientation_w_v,
            vio.position_v_b,
            vio.orientation_v_b,
        )
        linear_velocity_w_b = rotate_vector(self.orientation_w_v, vio.linear_velocity_v_b)
        return StateEstimate(
            publish_timestamp_s=publish_timestamp,
            position_w_v=self.position_w_v.clone(),
            orientation_w_v=self.orientation_w_v.clone(),
            position_w_b=position_w_b,
            orientation_w_b=orientation_w_b,
            linear_velocity_w_b=linear_velocity_w_b,
            angular_velocity_b=imu.angular_velocity_b,
            vio_status=self._status_at_publish(vio.status, publish_timestamp),
            imu_status=self._status_at_publish(imu.status, publish_timestamp),
        )
