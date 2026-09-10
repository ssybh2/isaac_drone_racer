import pytest
import torch

from estimation.fake_sensor_cfg import FakeSensorPipelineCfg
from estimation.pipeline import Stage1StatePipeline
from estimation.state_estimate import GroundTruthState


@pytest.mark.parametrize("start", [0.0, 500.0, 5000.0])
def test_clean_sources_never_skip_an_update_during_long_runs(start):
    cfg = FakeSensorPipelineCfg.from_profile("clean")
    pipeline = Stage1StatePipeline(1, "cpu", cfg.vio, cfg.imu, seed=42)
    gt = GroundTruthState(torch.zeros(1, 3), torch.tensor([[1., 0., 0., 0.]]),
                          torch.zeros(1, 3), torch.zeros(1, 3))
    pipeline.reset(torch.arange(1), gt, start)
    for step in range(1, 401):
        timestamp = start + step * 0.0025
        gt.angular_velocity_b[:] = step
        pipeline.ingest_imu(gt.angular_velocity_b, timestamp)
        if step % 4 == 0:
            gt.position_w_b[:] = step
            pipeline.ingest_vio(gt, timestamp)
            estimate = pipeline.publish(timestamp)
            torch.testing.assert_close(estimate.position_w_b, gt.position_w_b)
            torch.testing.assert_close(estimate.angular_velocity_b, gt.angular_velocity_b)
            assert estimate.vio_status.age_s.item() == 0.0
            assert estimate.imu_status.age_s.item() == 0.0
