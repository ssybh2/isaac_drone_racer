"""Typed tensor contracts shared by simulated and future hardware estimators."""

from __future__ import annotations

from dataclasses import dataclass

import torch


def _validate_tensor(
    name: str,
    value: torch.Tensor,
    shape: tuple[int, ...],
    device: torch.device,
    *,
    boolean: bool = False,
) -> None:
    if not isinstance(value, torch.Tensor) or tuple(value.shape) != shape:
        raise ValueError(f"{name} must have shape {shape}")
    if value.device != device:
        raise ValueError(f"{name} must be on device {device}")
    if boolean:
        if value.dtype != torch.bool:
            raise ValueError(f"{name} must use boolean dtype")
    elif not torch.isfinite(value).all():
        raise ValueError(f"{name} must contain only finite values")


def _validate_unit_quaternion(name: str, value: torch.Tensor) -> None:
    norms = torch.linalg.vector_norm(value, dim=-1)
    if not torch.allclose(norms, torch.ones_like(norms), atol=1.0e-4, rtol=1.0e-4):
        raise ValueError(f"{name} must contain a unit quaternion")


@dataclass
class SourceStatus:
    """Timing and freshness metadata for one asynchronous source."""

    timestamp_s: torch.Tensor
    age_s: torch.Tensor
    valid: torch.Tensor

    def validate(self, num_envs: int, device: torch.device) -> SourceStatus:
        _validate_tensor("timestamp_s", self.timestamp_s, (num_envs,), device)
        _validate_tensor("age_s", self.age_s, (num_envs,), device)
        _validate_tensor("valid", self.valid, (num_envs,), device, boolean=True)
        if torch.any(self.age_s < -1.0e-7):
            raise ValueError("age_s must be non-negative")
        return self


@dataclass
class GroundTruthState:
    """Simulator-only input accepted by fake sensor providers."""

    position_w_b: torch.Tensor
    orientation_w_b: torch.Tensor
    linear_velocity_w_b: torch.Tensor
    angular_velocity_b: torch.Tensor

    @property
    def num_envs(self) -> int:
        return self.position_w_b.shape[0]

    @property
    def device(self) -> torch.device:
        return self.position_w_b.device

    def validate(self) -> GroundTruthState:
        count = self.num_envs
        device = self.device
        _validate_tensor("position_w_b", self.position_w_b, (count, 3), device)
        _validate_tensor("orientation_w_b", self.orientation_w_b, (count, 4), device)
        _validate_tensor("linear_velocity_w_b", self.linear_velocity_w_b, (count, 3), device)
        _validate_tensor("angular_velocity_b", self.angular_velocity_b, (count, 3), device)
        _validate_unit_quaternion("orientation_w_b", self.orientation_w_b)
        return self


@dataclass
class VioEstimate:
    """Pose and velocity expressed in the local VIO frame V."""

    position_v_b: torch.Tensor
    orientation_v_b: torch.Tensor
    linear_velocity_v_b: torch.Tensor
    status: SourceStatus

    def validate(self) -> VioEstimate:
        count = self.position_v_b.shape[0]
        device = self.position_v_b.device
        _validate_tensor("position_v_b", self.position_v_b, (count, 3), device)
        _validate_tensor("orientation_v_b", self.orientation_v_b, (count, 4), device)
        _validate_tensor("linear_velocity_v_b", self.linear_velocity_v_b, (count, 3), device)
        _validate_unit_quaternion("orientation_v_b", self.orientation_v_b)
        self.status.validate(count, device)
        return self


@dataclass
class ImuEstimate:
    """Gyroscope estimate expressed in body frame B."""

    angular_velocity_b: torch.Tensor
    status: SourceStatus

    def validate(self) -> ImuEstimate:
        count = self.angular_velocity_b.shape[0]
        device = self.angular_velocity_b.device
        _validate_tensor("angular_velocity_b", self.angular_velocity_b, (count, 3), device)
        self.status.validate(count, device)
        return self


@dataclass
class StateEstimate:
    """World-aligned policy snapshot assembled from VIO and IMU sources."""

    publish_timestamp_s: torch.Tensor
    position_w_v: torch.Tensor
    orientation_w_v: torch.Tensor
    position_w_b: torch.Tensor
    orientation_w_b: torch.Tensor
    linear_velocity_w_b: torch.Tensor
    angular_velocity_b: torch.Tensor
    vio_status: SourceStatus
    imu_status: SourceStatus

    @property
    def num_envs(self) -> int:
        return self.position_w_b.shape[0]

    @property
    def device(self) -> torch.device:
        return self.publish_timestamp_s.device

    def validate(self) -> StateEstimate:
        count = self.num_envs
        device = self.device
        _validate_tensor("publish_timestamp_s", self.publish_timestamp_s, (count,), device)
        _validate_tensor("position_w_v", self.position_w_v, (count, 3), device)
        _validate_tensor("orientation_w_v", self.orientation_w_v, (count, 4), device)
        _validate_tensor("position_w_b", self.position_w_b, (count, 3), device)
        _validate_tensor("orientation_w_b", self.orientation_w_b, (count, 4), device)
        _validate_tensor("linear_velocity_w_b", self.linear_velocity_w_b, (count, 3), device)
        _validate_tensor("angular_velocity_b", self.angular_velocity_b, (count, 3), device)
        _validate_unit_quaternion("orientation_w_v", self.orientation_w_v)
        _validate_unit_quaternion("orientation_w_b", self.orientation_w_b)
        self.vio_status.validate(count, device)
        self.imu_status.validate(count, device)
        return self
