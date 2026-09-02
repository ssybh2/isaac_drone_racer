import torch

from estimation.fake_imu import FakeImu
from estimation.fake_sensor_cfg import FakeImuCfg, UniformRange


def gyro(x_values: tuple[float, ...] = (1.0, 2.0)) -> torch.Tensor:
    output = torch.zeros(len(x_values), 3)
    output[:, 0] = torch.tensor(x_values)
    output[:, 2] = -0.5
    return output


def test_clean_imu_matches_ground_truth_gyro():
    source = gyro()
    imu = FakeImu(2, "cpu", FakeImuCfg.clean(update_period_s=0.0025), seed=7)

    output = imu.reset(torch.arange(2), source, timestamp_s=0.0)

    torch.testing.assert_close(output.angular_velocity_b, source)
    assert output.status.valid.tolist() == [True, True]


def test_imu_fixed_episode_bias_is_applied():
    cfg = FakeImuCfg.clean()
    cfg.bias_radps = UniformRange(0.25, 0.25)
    imu = FakeImu(2, "cpu", cfg, seed=4)

    output = imu.reset(torch.arange(2), gyro(), 0.0)

    torch.testing.assert_close(output.angular_velocity_b, gyro() + 0.25)


def test_imu_white_noise_is_seed_reproducible():
    cfg = FakeImuCfg.clean()
    cfg.noise_std_radps = UniformRange(0.2, 0.2)
    first = FakeImu(2, "cpu", cfg, seed=19)
    second = FakeImu(2, "cpu", cfg, seed=19)

    first_output = first.reset(torch.arange(2), gyro(), 0.0)
    second_output = second.reset(torch.arange(2), gyro(), 0.0)

    torch.testing.assert_close(first_output.angular_velocity_b, second_output.angular_velocity_b)
    assert not torch.allclose(first_output.angular_velocity_b, gyro())


def test_imu_update_rate_marks_intermediate_snapshot_stale():
    imu = FakeImu(2, "cpu", FakeImuCfg.clean(update_period_s=0.005), seed=3)
    imu.reset(torch.arange(2), gyro(), 0.0)

    stale = imu.update(gyro((4.0, 5.0)), 0.0025)
    fresh = imu.update(gyro((6.0, 7.0)), 0.005)

    assert stale.status.valid.tolist() == [False, False]
    torch.testing.assert_close(stale.angular_velocity_b[:, 0], torch.tensor([1.0, 2.0]))
    assert fresh.status.valid.tolist() == [True, True]
    torch.testing.assert_close(fresh.angular_velocity_b[:, 0], torch.tensor([6.0, 7.0]))


def test_imu_latency_returns_exact_older_physics_sample():
    cfg = FakeImuCfg.clean(update_period_s=0.0025)
    cfg.latency_s = UniformRange(0.005, 0.005)
    imu = FakeImu(2, "cpu", cfg, seed=8)
    imu.reset(torch.arange(2), gyro((0.0, 10.0)), 0.0)

    imu.update(gyro((1.0, 11.0)), 0.0025)
    imu.update(gyro((2.0, 12.0)), 0.005)
    output = imu.update(gyro((3.0, 13.0)), 0.0075)

    torch.testing.assert_close(output.angular_velocity_b[:, 0], torch.tensor([1.0, 11.0]))
    torch.testing.assert_close(output.status.timestamp_s, torch.tensor([0.0025, 0.0025]))


def test_imu_dropout_holds_last_delivered_sample_and_increases_age():
    cfg = FakeImuCfg.clean(update_period_s=0.0025)
    cfg.dropout_probability = UniformRange(1.0, 1.0)
    imu = FakeImu(2, "cpu", cfg, seed=2)
    initial = imu.reset(torch.arange(2), gyro(), 0.0)

    output = imu.update(gyro((9.0, 9.0)), 0.0025)

    torch.testing.assert_close(output.angular_velocity_b, initial.angular_velocity_b)
    torch.testing.assert_close(output.status.age_s, torch.tensor([0.0025, 0.0025]))
    assert output.status.valid.tolist() == [False, False]


def test_imu_bias_random_walk_accumulates_without_motion():
    cfg = FakeImuCfg.clean(update_period_s=0.0025)
    cfg.bias_random_walk_std_radps_per_sqrt_s = UniformRange(0.5, 0.5)
    imu = FakeImu(2, "cpu", cfg, seed=29)
    initial = imu.reset(torch.arange(2), gyro((0.0, 0.0)), 0.0)

    output = imu.update(gyro((0.0, 0.0)), 0.0025)

    assert not torch.allclose(output.angular_velocity_b, initial.angular_velocity_b)


def test_imu_reset_of_one_environment_preserves_other_delivery():
    cfg = FakeImuCfg.clean(update_period_s=0.0025)
    cfg.bias_random_walk_std_radps_per_sqrt_s = UniformRange(0.5, 0.5)
    imu = FakeImu(2, "cpu", cfg, seed=11)
    imu.reset(torch.arange(2), gyro((0.0, 0.0)), 0.0)
    before = imu.update(gyro((0.0, 0.0)), 0.0025)

    after = imu.reset(torch.tensor([0]), gyro((5.0, 99.0)), 0.005)

    torch.testing.assert_close(after.angular_velocity_b[1], before.angular_velocity_b[1])
