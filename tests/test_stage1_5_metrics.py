from utils.stage1_metrics import EpisodeRecord, Stage1EpisodeAccumulator


def _record(*, completed: bool, lap_time_s: float | None, failure_gate: int | None):
    return EpisodeRecord(
        completed=completed,
        gates_passed=7 if completed else 3,
        collision=not completed,
        flyaway=False,
        episode_return=100.0 if completed else 10.0,
        duration_s=lap_time_s if lap_time_s is not None else (8.0 if completed else 4.0),
        position_rmse_m=0.1,
        attitude_rmse_rad=0.02,
        velocity_rmse_mps=0.2,
        lap_time_s=lap_time_s,
        failure_gate=failure_gate,
    )


def test_first_lap_summary_reports_lap_time_and_failure_gate_counts():
    accumulator = Stage1EpisodeAccumulator()
    accumulator.add(_record(completed=True, lap_time_s=7.5, failure_gate=None))
    accumulator.add(_record(completed=True, lap_time_s=8.5, failure_gate=None))
    accumulator.add(_record(completed=False, lap_time_s=None, failure_gate=4))
    accumulator.add(_record(completed=False, lap_time_s=None, failure_gate=4))
    accumulator.add(_record(completed=False, lap_time_s=None, failure_gate=6))

    summary = accumulator.summary()

    assert summary["completion_rate"] == 0.4
    assert summary["first_lap_completion_rate"] == 0.4
    assert summary["mean_lap_time_s"] == 8.0
    assert summary["failure_gate_counts"] == {"4": 2, "6": 1}


def test_legacy_success_record_without_lap_time_remains_supported():
    accumulator = Stage1EpisodeAccumulator()
    accumulator.add(_record(completed=True, lap_time_s=None, failure_gate=None))

    summary = accumulator.summary()

    assert summary["completion_rate"] == 1.0
    assert summary["first_lap_completion_rate"] == 1.0
    assert summary["mean_lap_time_s"] == 0.0
