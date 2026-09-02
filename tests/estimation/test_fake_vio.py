import torch

from estimation.fake_sensor_cfg import FakeVioCfg, UniformRange
from estimation.fake_vio import FakeVio
from estimation.state_estimate import GroundTruthState


def ground_truth(position_x: tuple[float, ...] = (1.0, 2.0)) -> GroundTruthState:
    count = len(position_x)
    orientation = torch.zeros(count, 4)
    orientation[:, 0] = 1.0
    position = torch.zeros(count, 3)
    position[:, 0] = torch.tensor(position_x)
    velocity = torch.zeros(count, 3)
    velocity[:, 1] = 3.0
    return GroundTruthState(position, orientation, velocity, torch.zeros(count, 3))


def test_clean_vio_matches_ground_truth():
    gt = ground_truth()
    vio = FakeVio(2, "cpu", FakeVioCfg.clean(update_period_s=0.01), seed=7)

    output = vio.reset(torch.arange(2), gt, timestamp_s=0.0)

    torch.testing.assert_close(output.position_v_b, gt.position_w_b)
    torch.testing.assert_close(output.orientation_v_b, gt.orientation_w_b)
    torch.testing.assert_close(output.linear_velocity_v_b, gt.linear_velocity_w_b)
    assert output.status.valid.tolist() == [True, True]


def test_vio_noise_is_seed_reproducible():
    cfg = FakeVioCfg.clean(update_period_s=0.01)
    cfg.position_noise_std_m = UniformRange(0.2, 0.2)
    first = FakeVio(2, "cpu", cfg, seed=19)
    second = FakeVio(2, "cpu", cfg, seed=19)

    first_output = first.reset(torch.arange(2), ground_truth(), 0.0)
    second_output = second.reset(torch.arange(2), ground_truth(), 0.0)

    torch.testing.assert_close(first_output.position_v_b, second_output.position_v_b)
    assert not torch.allclose(first_output.position_v_b, ground_truth().position_w_b)


def test_vio_orientation_noise_preserves_unit_quaternion():
    cfg = FakeVioCfg.clean(update_period_s=0.01)
    cfg.orientation_noise_std_rad = UniformRange(0.3, 0.3)
    vio = FakeVio(2, "cpu", cfg, seed=4)

    output = vio.reset(torch.arange(2), ground_truth(), 0.0)

    torch.testing.assert_close(torch.linalg.vector_norm(output.orientation_v_b, dim=-1), torch.ones(2))


def test_vio_update_rate_marks_intermediate_snapshot_stale():
    vio = FakeVio(2, "cpu", FakeVioCfg.clean(update_period_s=0.02), seed=3)
    vio.reset(torch.arange(2), ground_truth(), 0.0)

    stale = vio.update(ground_truth((4.0, 5.0)), 0.01)
    fresh = vio.update(ground_truth((6.0, 7.0)), 0.02)

    assert stale.status.valid.tolist() == [False, False]
    torch.testing.assert_close(stale.position_v_b[:, 0], torch.tensor([1.0, 2.0]))
    assert fresh.status.valid.tolist() == [True, True]
    torch.testing.assert_close(fresh.position_v_b[:, 0], torch.tensor([6.0, 7.0]))


def test_vio_latency_returns_exact_older_source_sample():
    cfg = FakeVioCfg.clean(update_period_s=0.01)
    cfg.latency_s = UniformRange(0.02, 0.02)
    vio = FakeVio(2, "cpu", cfg, seed=8)
    vio.reset(torch.arange(2), ground_truth((0.0, 10.0)), 0.0)

    vio.update(ground_truth((1.0, 11.0)), 0.01)
    vio.update(ground_truth((2.0, 12.0)), 0.02)
    output = vio.update(ground_truth((3.0, 13.0)), 0.03)

    torch.testing.assert_close(output.position_v_b[:, 0], torch.tensor([1.0, 11.0]))
    torch.testing.assert_close(output.status.timestamp_s, torch.tensor([0.01, 0.01]))


def test_vio_dropout_holds_last_delivered_sample_and_increases_age():
    cfg = FakeVioCfg.clean(update_period_s=0.01)
    cfg.dropout_probability = UniformRange(1.0, 1.0)
    vio = FakeVio(2, "cpu", cfg, seed=2)
    initial = vio.reset(torch.arange(2), ground_truth(), 0.0)

    output = vio.update(ground_truth((9.0, 9.0)), 0.01)

    torch.testing.assert_close(output.position_v_b, initial.position_v_b)
    torch.testing.assert_close(output.status.age_s, torch.tensor([0.01, 0.01]))
    assert output.status.valid.tolist() == [False, False]


def test_vio_random_walk_drift_accumulates_without_ground_truth_motion():
    cfg = FakeVioCfg.clean(update_period_s=0.01)
    cfg.position_drift_std_m_per_sqrt_s = UniformRange(0.5, 0.5)
    vio = FakeVio(2, "cpu", cfg, seed=29)
    initial = vio.reset(torch.arange(2), ground_truth((0.0, 0.0)), 0.0)

    output = vio.update(ground_truth((0.0, 0.0)), 0.01)

    assert not torch.allclose(output.position_v_b, initial.position_v_b)


def test_vio_random_walk_uses_source_period_not_last_ingestion_period():
    cfg = FakeVioCfg.clean(update_period_s=0.02)
    cfg.position_drift_std_m_per_sqrt_s = UniformRange(0.5, 0.5)
    with_intermediate_ingest = FakeVio(2, "cpu", cfg, seed=31)
    direct_source_update = FakeVio(2, "cpu", cfg, seed=31)
    with_intermediate_ingest.reset(torch.arange(2), ground_truth((0.0, 0.0)), 0.0)
    direct_source_update.reset(torch.arange(2), ground_truth((0.0, 0.0)), 0.0)

    with_intermediate_ingest.update(ground_truth((0.0, 0.0)), 0.01)
    with_intermediate_ingest.generator.manual_seed(101)
    direct_source_update.generator.manual_seed(101)
    with_intermediate_ingest.update(ground_truth((0.0, 0.0)), 0.02)
    direct_source_update.update(ground_truth((0.0, 0.0)), 0.02)

    torch.testing.assert_close(
        with_intermediate_ingest._position_drift, direct_source_update._position_drift
    )


def test_vio_reset_of_one_environment_preserves_other_delivery():
    cfg = FakeVioCfg.clean(update_period_s=0.01)
    cfg.position_drift_std_m_per_sqrt_s = UniformRange(0.5, 0.5)
    vio = FakeVio(2, "cpu", cfg, seed=11)
    vio.reset(torch.arange(2), ground_truth((0.0, 0.0)), 0.0)
    before = vio.update(ground_truth((0.0, 0.0)), 0.01)

    after = vio.reset(torch.tensor([0]), ground_truth((5.0, 99.0)), 0.02)

    torch.testing.assert_close(after.position_v_b[1], before.position_v_b[1])
