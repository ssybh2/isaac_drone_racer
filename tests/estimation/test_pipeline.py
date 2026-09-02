import torch

from estimation.fake_sensor_cfg import FakeImuCfg, FakeVioCfg
from estimation.pipeline import Stage1StatePipeline
from estimation.state_estimate import GroundTruthState


def ground_truth(position_x: tuple[float, ...] = (1.0, 2.0), gyro_x: float = 0.0) -> GroundTruthState:
    count = len(position_x)
    position = torch.zeros(count, 3)
    position[:, 0] = torch.tensor(position_x)
    orientation = torch.zeros(count, 4)
    orientation[:, 0] = 1.0
    velocity = torch.zeros(count, 3)
    velocity[:, 1] = 3.0
    angular_velocity = torch.zeros(count, 3)
    angular_velocity[:, 0] = gyro_x
    return GroundTruthState(position, orientation, velocity, angular_velocity)


def pipeline() -> Stage1StatePipeline:
    return Stage1StatePipeline(
        num_envs=2,
        device="cpu",
        vio_cfg=FakeVioCfg.clean(update_period_s=0.01),
        imu_cfg=FakeImuCfg.clean(update_period_s=0.0025),
        seed=17,
    )


def test_clean_pipeline_matches_ground_truth():
    value = pipeline()
    gt = ground_truth(gyro_x=0.7)

    estimate = value.reset(torch.arange(2), gt, timestamp_s=0.0)

    torch.testing.assert_close(estimate.position_w_b, gt.position_w_b)
    torch.testing.assert_close(estimate.orientation_w_b, gt.orientation_w_b)
    torch.testing.assert_close(estimate.linear_velocity_w_b, gt.linear_velocity_w_b)
    torch.testing.assert_close(estimate.angular_velocity_b, gt.angular_velocity_b)


def test_imu_and_vio_keep_independent_source_timestamps():
    value = pipeline()
    value.reset(torch.arange(2), ground_truth(), 0.0)

    for timestamp in (0.0025, 0.005, 0.0075, 0.01):
        value.ingest_imu(ground_truth(gyro_x=timestamp).angular_velocity_b, timestamp)
    value.ingest_vio(ground_truth((4.0, 5.0)), 0.01)
    estimate = value.publish(0.01)

    torch.testing.assert_close(estimate.vio_status.timestamp_s, torch.full((2,), 0.01))
    torch.testing.assert_close(estimate.imu_status.timestamp_s, torch.full((2,), 0.01))
    assert value.imu_ingest_count == 4
    assert value.vio_ingest_count == 1


def test_intermediate_publish_exposes_latest_independent_samples():
    value = pipeline()
    value.reset(torch.arange(2), ground_truth(), 0.0)

    value.ingest_imu(ground_truth(gyro_x=0.0025).angular_velocity_b, 0.0025)
    estimate = value.publish(0.0025)

    torch.testing.assert_close(estimate.vio_status.timestamp_s, torch.zeros(2))
    torch.testing.assert_close(estimate.vio_status.age_s, torch.full((2,), 0.0025))
    torch.testing.assert_close(estimate.imu_status.timestamp_s, torch.full((2,), 0.0025))


def test_pipeline_selective_reset_preserves_other_environment():
    value = pipeline()
    value.reset(torch.arange(2), ground_truth((0.0, 0.0)), 0.0)
    value.ingest_imu(ground_truth((0.0, 0.0), gyro_x=0.5).angular_velocity_b, 0.0025)
    value.ingest_vio(ground_truth((1.0, 2.0)), 0.01)
    before = value.publish(0.01)

    after = value.reset(torch.tensor([0]), ground_truth((8.0, 99.0), gyro_x=9.0), 0.02)

    torch.testing.assert_close(after.position_w_b[1], before.position_w_b[1])
    torch.testing.assert_close(after.angular_velocity_b[1], before.angular_velocity_b[1])


def test_pipeline_alignment_is_explicit_and_mutable():
    value = pipeline()
    value.reset(torch.arange(2), ground_truth(), 0.0)

    value.set_world_alignment(
        torch.tensor([0]),
        position_w_v=torch.tensor([[10.0, 0.0, 0.0]]),
        orientation_w_v=torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
    )
    estimate = value.publish(0.0)

    torch.testing.assert_close(estimate.position_w_b[0], torch.tensor([11.0, 0.0, 0.0]))
    torch.testing.assert_close(estimate.position_w_v[0], torch.tensor([10.0, 0.0, 0.0]))
