"""Pure PyTorch orchestration for Stage 1 state sources."""

from __future__ import annotations

import torch

from .fake_imu import FakeImu
from .fake_sensor_cfg import FakeImuCfg, FakeVioCfg
from .fake_vio import FakeVio
from .state_estimate import GroundTruthState, ImuEstimate, StateEstimate, VioEstimate
from .state_estimate_assembler import StateEstimateAssembler


class Stage1StatePipeline:
    """Own independently timed Fake VIO and Fake IMU providers.

    The class deliberately has no Isaac imports. Simulator and real-hardware
    adapters both feed the same typed inputs and consume the same StateEstimate.
    """

    def __init__(
        self,
        num_envs: int,
        device: str | torch.device,
        vio_cfg: FakeVioCfg,
        imu_cfg: FakeImuCfg,
        seed: int,
    ) -> None:
        self.num_envs = num_envs
        self.device = torch.device(device)
        self.vio = FakeVio(num_envs, self.device, vio_cfg, seed=seed + 1)
        self.imu = FakeImu(num_envs, self.device, imu_cfg, seed=seed + 2)
        self.assembler = StateEstimateAssembler.identity(num_envs, self.device)
        self.vio_ingest_count = 0
        self.imu_ingest_count = 0
        self._vio_estimate: VioEstimate | None = None
        self._imu_estimate: ImuEstimate | None = None
        self.current_estimate: StateEstimate | None = None

    def reset(
        self,
        env_ids: torch.Tensor,
        ground_truth: GroundTruthState,
        timestamp_s: float | torch.Tensor,
    ) -> StateEstimate:
        self._vio_estimate = self.vio.reset(env_ids, ground_truth, timestamp_s)
        self._imu_estimate = self.imu.reset(
            env_ids, ground_truth.angular_velocity_b, timestamp_s
        )
        return self.publish(timestamp_s)

    def ingest_imu(
        self, angular_velocity_b: torch.Tensor, timestamp_s: float | torch.Tensor
    ) -> ImuEstimate:
        self._imu_estimate = self.imu.update(angular_velocity_b, timestamp_s)
        self.imu_ingest_count += 1
        return self._imu_estimate

    def ingest_vio(
        self, ground_truth: GroundTruthState, timestamp_s: float | torch.Tensor
    ) -> VioEstimate:
        self._vio_estimate = self.vio.update(ground_truth, timestamp_s)
        self.vio_ingest_count += 1
        return self._vio_estimate

    def set_world_alignment(
        self,
        env_ids: torch.Tensor,
        position_w_v: torch.Tensor,
        orientation_w_v: torch.Tensor,
    ) -> None:
        self.assembler.set_alignment(env_ids, position_w_v, orientation_w_v)

    def publish(self, timestamp_s: float | torch.Tensor) -> StateEstimate:
        if self._vio_estimate is None or self._imu_estimate is None:
            raise RuntimeError("Stage1StatePipeline must be reset before publishing")
        # Re-evaluate ages at the common policy timestamp even if neither source
        # emitted a new sample since the previous publication.
        vio = self.vio.estimate(timestamp_s)
        imu = self.imu.estimate(timestamp_s)
        self.current_estimate = self.assembler.assemble(vio, imu, timestamp_s)
        return self.current_estimate
