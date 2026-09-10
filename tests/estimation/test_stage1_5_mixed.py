import torch

from estimation.fake_imu import FakeImu
from estimation.fake_sensor_cfg import FakeSensorPipelineCfg
from estimation.fake_vio import FakeVio
from estimation.state_estimate import GroundTruthState
from estimation.pipeline import Stage1StatePipeline


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


def test_mixed_pipeline_contains_joint_clean_episodes_and_keeps_them_clean():
    cfg = FakeSensorPipelineCfg.from_profile("mixed")
    assert cfg.clean_episode_probability > 0.0
    pipelines = [Stage1StatePipeline(256, "cpu", cfg.vio, cfg.imu, seed=12,
                 clean_episode_probability=cfg.clean_episode_probability) for _ in range(2)]
    gt = _ground_truth(256)
    for pipeline in pipelines:
        pipeline.reset(torch.arange(256), gt, 0.0)
    mask = pipelines[0].clean_episode_mask
    assert 32 < mask.sum() < 128
    torch.testing.assert_close(mask, pipelines[1].clean_episode_mask)
    pipeline = pipelines[0]
    for step in range(1, 41):
        gt.position_w_b[:] = step * 0.01
        gt.angular_velocity_b[:] = step * 0.1
        for substep in range(4):
            pipeline.ingest_imu(gt.angular_velocity_b, (step - 1) * 0.01 + (substep + 1) * 0.0025)
        pipeline.ingest_vio(gt, step * 0.01)
        estimate = pipeline.publish(step * 0.01)
        torch.testing.assert_close(estimate.position_w_b[mask], gt.position_w_b[mask])
        torch.testing.assert_close(estimate.angular_velocity_b[mask], gt.angular_velocity_b[mask])
        assert torch.all(estimate.vio_status.age_s[mask] == 0)
        assert torch.all(estimate.imu_status.age_s[mask] == 0)
    untouched = pipeline.clean_episode_mask[128:].clone()
    pipeline.reset(torch.arange(128), gt, 0.4)
    torch.testing.assert_close(pipeline.clean_episode_mask[128:], untouched)


def test_clean_episode_probability_rejects_invalid_values():
    import pytest

    cfg = FakeSensorPipelineCfg.from_profile("mixed")
    for probability in (-0.1, 1.1):
        with pytest.raises(ValueError):
            Stage1StatePipeline(2, "cpu", cfg.vio, cfg.imu, seed=12,
                                clean_episode_probability=probability)
