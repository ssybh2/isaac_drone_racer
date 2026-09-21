from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def _text(relative: str) -> str:
    return (ROOT / relative).read_text(encoding="utf-8")


def test_swift_ctbr_action_contract_is_explicit():
    action = _text("tasks/drone_racer/mdp/swift_ctbr_action.py")

    expected = (
        "class SwiftCTBRAction(ActionTerm):",
        "collective_accel",
        "desired_rate = normalized[:, 1:4] * self._body_rate_max",
        "BodyRatePIDController",
        "allocate_wrench_rate_priority",
        "vehicle_mass_kg: float = 0.6076",
        "body_rate_max_radps: tuple[float, float, float] = (10.0, 10.0, 6.0)",
        "rate_kp: tuple[float, float, float] = (0.025, 0.025, 0.030)",
    )
    for token in expected:
        assert token in action

    # Zero thrust-channel action must map to hover acceleration instead of
    # directly representing a raw motor command.
    assert "thrust_action >= 0.0" in action
    assert "gravity + thrust_action * gravity" in action
    assert "thrust_action * (self._max_collective_accel - gravity)" in action


def test_rate_controller_uses_betaflight_style_d_term():
    controller = _text("dynamics/body_rate_controller.py")

    assert "- self.kd * measured_derivative" in controller
    assert "throttle_cut" in controller
    assert "self.integral = torch.where" in controller


def test_wrench_allocator_prioritizes_body_moments():
    allocation = _text("dynamics/allocation.py")

    expected = (
        "def allocate_wrench_rate_priority",
        "moment_wrench[:, 1:] = wrench[:, 1:]",
        "moment_scale",
        "requested_collective_per_rotor",
        "rotor_thrusts",
    )
    for token in expected:
        assert token in allocation


def test_ctbr_tasks_are_registered_and_separate_from_legacy_motor_rl():
    registry = _text("tasks/drone_racer/__init__.py")
    cfg = _text("tasks/drone_racer/drone_racer_swift_ctbr_env_cfg.py")

    assert "Isaac-Drone-Racer-Swift-CTBR-Control-v0" in registry
    assert "Isaac-Drone-Racer-Learned-Inertial-Swift-CTBR-v0" in registry
    assert "DroneRacerSwiftCTBRControlEnvCfg" in cfg
    assert "DroneRacerLearnedInertialSwiftCTBRRLCfg" in cfg
    assert "SwiftCTBRActionCfg" in cfg


def test_ctbr_smoke_covers_hover_roll_pitch_yaw():
    smoke = _text("scripts/rl/smoke_swift_ctbr_control.py")

    for token in (
        '"hover"',
        '("roll", 0',
        '("pitch", 1',
        '("yaw", 2',
        "tracking_rmse_radps",
        "z_drift_m",
    ):
        assert token in smoke
