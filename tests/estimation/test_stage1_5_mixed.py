import torch

from estimation.fake_imu import FakeImu
from estimation.fake_sensor_cfg import FakeSensorPipelineCfg
from estimation.fake_vio import FakeVio
from estimation.state_estimate import GroundTruthState


def _ground_truth(count: int = 64) -> GroundTruthState:
    position = torch.zeros(count, 3)
    orientation = torch.zeros(count, 4)
    orientation[:, 0] = 1.0
    velocity = torch.zeros(count, 3)
    angular_velocity = torch.zeros(count, 3)
    return GroundTruthState(position, orientation, velocity, angular_velocity)


def test_mixed_profile_spans_clean_to_slightly_harder_than_nominal():
    cfg = FakeSensorPipelineCfg.from_profile("mixed")

    assert cfg.profile == "mixed"
    assert cfg.vio.update_period_range_s is not None
    assert cfg.vio.update_period_range_s.low == 0.01
    assert cfg.vio.update_period_range_s.high == 0.02
    assert cfg.imu.update_period_range_s is not None
    assert cfg.imu.update_period_range_s.low == 0.0025
    assert cfg.imu.update_period_range_s.high == 0.005
    assert cfg.vio.position_noise_std_m.low == 0.0
    assert cfg.vio.position_noise_std_m.high > 0.03
    assert cfg.vio.latency_s.low == 0.0
    assert cfg.vio.latency_s.high > 0.04


def test_mixed_vio_samples_per_environment_periods_reproducibly():
    cfg = FakeSensorPipelineCfg.from_profile("mixed").vio
    first = FakeVio(64, "cpu", cfg, seed=123)
    second = FakeVio(64, "cpu", cfg, seed=123)

    first.reset(torch.arange(64), _ground_truth(), 0.0)
    second.reset(torch.arange(64), _ground_truth(), 0.0)

    periods = first.update_period_s
    torch.testing.assert_close(periods, second.update_period_s)
    assert torch.all(periods >= 0.01)
    assert torch.all(periods <= 0.02)
    assert torch.unique(periods).numel() > 1


def test_mixed_imu_samples_per_environment_periods_reproducibly():
    cfg = FakeSensorPipelineCfg.from_profile("mixed").imu
    first = FakeImu(64, "cpu", cfg, seed=321)
    second = FakeImu(64, "cpu", cfg, seed=321)
    gyro = torch.zeros(64, 3)

    first.reset(torch.arange(64), gyro, 0.0)
    second.reset(torch.arange(64), gyro, 0.0)

    periods = first.update_period_s
    torch.testing.assert_close(periods, second.update_period_s)
    assert torch.all(periods >= 0.0025)
    assert torch.all(periods <= 0.005)
    assert torch.unique(periods).numel() > 1
