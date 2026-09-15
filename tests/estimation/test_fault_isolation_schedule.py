import pytest

from estimation.fault_isolation_schedule import resolve_motion_start_time_s


def test_absolute_motion_schedule_preserves_existing_behavior():
    assert resolve_motion_start_time_s(
        absolute_start_s=4.0,
        after_init_delay_s=None,
        initialized_time_s=None,
    ) == pytest.approx(4.0)


def test_init_relative_motion_waits_until_openvins_is_observed():
    assert resolve_motion_start_time_s(
        absolute_start_s=4.0,
        after_init_delay_s=1.0,
        initialized_time_s=None,
    ) is None


def test_init_relative_motion_uses_estimator_observation_time():
    assert resolve_motion_start_time_s(
        absolute_start_s=4.0,
        after_init_delay_s=1.0,
        initialized_time_s=12.41,
    ) == pytest.approx(13.41)


@pytest.mark.parametrize(
    ("absolute_start_s", "after_init_delay_s"),
    [(-0.1, None), (4.0, -0.1)],
)
def test_motion_schedule_rejects_negative_delays(absolute_start_s, after_init_delay_s):
    with pytest.raises(ValueError):
        resolve_motion_start_time_s(
            absolute_start_s=absolute_start_s,
            after_init_delay_s=after_init_delay_s,
            initialized_time_s=12.41,
        )
