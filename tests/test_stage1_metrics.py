import math

from utils.stage1_metrics import EpisodeRecord, Stage1EpisodeAccumulator


def test_two_episode_summary_matches_hand_derived_metrics():
    accumulator = Stage1EpisodeAccumulator()
    accumulator.add(
        EpisodeRecord(
            completed=True,
            gates_passed=7,
            collision=False,
            flyaway=False,
            episode_return=100.0,
            duration_s=8.0,
            position_rmse_m=0.10,
            attitude_rmse_rad=0.02,
            velocity_rmse_mps=0.20,
            vio_age_s=(0.0, 0.01),
            imu_age_s=(0.0, 0.0025),
            vio_valid=(True, False),
            imu_valid=(True, True),
            vio_dropped=(False, True),
            imu_dropped=(False, False),
        )
    )
    accumulator.add(
        EpisodeRecord(
            completed=False,
            gates_passed=3,
            collision=True,
            flyaway=False,
            episode_return=20.0,
            duration_s=4.0,
            position_rmse_m=0.30,
            attitude_rmse_rad=0.06,
            velocity_rmse_mps=0.40,
            vio_age_s=(0.02, 0.03),
            imu_age_s=(0.005, 0.0075),
            vio_valid=(True, True),
            imu_valid=(False, True),
            vio_dropped=(False, False),
            imu_dropped=(True, False),
        )
    )

    summary = accumulator.summary()

    assert summary["episodes"] == 2
    assert summary["completion_rate"] == 0.5
    assert summary["collision_rate"] == 0.5
    assert summary["flyaway_rate"] == 0.0
    assert summary["mean_gates_passed"] == 5.0
    assert summary["mean_return"] == 60.0
    assert summary["mean_duration_s"] == 6.0
    assert summary["mean_position_rmse_m"] == 0.2
    assert summary["vio_dropout_fraction"] == 0.25
    assert summary["imu_dropout_fraction"] == 0.25
    assert summary["vio_fresh_fraction"] == 0.75
    assert summary["imu_fresh_fraction"] == 0.75
    assert math.isclose(summary["vio_age_p95_s"], 0.0285)
    assert math.isclose(summary["imu_age_p95_s"], 0.007125)


def test_empty_summary_is_well_defined():
    summary = Stage1EpisodeAccumulator().summary()

    assert summary["episodes"] == 0
    assert summary["completion_rate"] == 0.0
    assert summary["vio_age_p95_s"] == 0.0
