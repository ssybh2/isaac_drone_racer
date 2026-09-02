import pytest

from estimation.fake_sensor_cfg import FakeSensorPipelineCfg, fake_sensor_profile


@pytest.mark.parametrize("name", ["clean", "mild", "nominal", "stress"])
def test_named_profiles_construct_valid_independent_sources(name: str):
    cfg = FakeSensorPipelineCfg.from_profile(name)

    assert cfg.profile == name
    assert cfg.vio.update_period_s >= 0.0025
    assert cfg.imu.update_period_s >= 0.0025


def test_clean_profile_has_no_corruption():
    cfg = FakeSensorPipelineCfg.from_profile("clean")

    assert cfg.vio.position_noise_std_m.high == 0.0
    assert cfg.vio.latency_s.high == 0.0
    assert cfg.vio.dropout_probability.high == 0.0
    assert cfg.imu.noise_std_radps.high == 0.0
    assert cfg.imu.latency_s.high == 0.0


def test_nominal_profile_matches_documented_envelopes():
    cfg = FakeSensorPipelineCfg.from_profile("nominal")

    assert (cfg.vio.position_noise_std_m.low, cfg.vio.position_noise_std_m.high) == (0.005, 0.03)
    assert (cfg.vio.latency_s.low, cfg.vio.latency_s.high) == (0.0, 0.04)
    assert (cfg.vio.dropout_probability.low, cfg.vio.dropout_probability.high) == (0.0, 0.03)
    assert (cfg.imu.noise_std_radps.low, cfg.imu.noise_std_radps.high) == (0.001, 0.01)
    assert (cfg.imu.latency_s.low, cfg.imu.latency_s.high) == (0.0, 0.005)


def test_stress_profile_is_stricter_than_nominal_and_unknown_is_rejected():
    nominal = FakeSensorPipelineCfg.from_profile("nominal")
    stress = FakeSensorPipelineCfg.from_profile("stress")

    assert stress.vio.position_noise_std_m.low >= nominal.vio.position_noise_std_m.high
    assert stress.vio.latency_s.high > nominal.vio.latency_s.high
    assert stress.imu.noise_std_radps.high > nominal.imu.noise_std_radps.high
    with pytest.raises(ValueError, match="unknown Fake Sensor profile"):
        fake_sensor_profile("mystery")
