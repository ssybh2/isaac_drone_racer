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



def test_swift_31d_observation_contract():
    obs = _text("tasks/drone_racer/mdp/learned_inertial_observations.py")
    cfg = _text("tasks/drone_racer/drone_racer_swift_ctbr_env_cfg.py")

    for token in (
        "def learned_inertial_swift_state",
        "R_wb = math_utils.matrix_from_quat",
        "return torch.cat((p_w, v_w, R_wb), dim=-1)",
        "def learned_next_gate_corners_relative_w",
        "relative_w.reshape(env.num_envs, 12)",
    ):
        assert token in obs

    for token in (
        "class LearnedInertialSwiftPolicyCfg",
        "platform_state = ObsTerm(func=mdp.learned_inertial_swift_state)",
        "next_gate_corners = ObsTerm(",
        "previous_action = ObsTerm(func=mdp.last_action)",
    ):
        assert token in cfg


def test_ctbr_gate_flight_smoke_is_estimator_driven():
    smoke = _text("scripts/rl/smoke_swift_ctbr_gate_flight.py")

    for token in (
        "raw_env.learned_inertial_state",
        "desired_body_rate",
        "_physical_ctbr_to_normalized",
        "mission_gate_passes",
        "truth_gate_passes",
        '"controller_uses_simulator_root_state": False',
    ):
        assert token in smoke



def test_swift_policy0_training_contract():
    registry = _text("tasks/drone_racer/__init__.py")
    cfg = _text("tasks/drone_racer/drone_racer_swift_ctbr_env_cfg.py")
    agent = _text("tasks/drone_racer/agents/skrl_swift_ctbr_cfg.yaml")
    rewards = _text("tasks/drone_racer/mdp/rewards.py")
    env = _text("tasks/drone_racer/swift_ctbr_racing_env.py")

    for token in (
        "Isaac-Drone-Racer-Swift-CTBR-Train-v0",
        "SwiftCTBRRacingEnv",
        "skrl_swift_ctbr_cfg.yaml",
    ):
        assert token in registry

    for token in (
        "class SwiftGTPolicyCfg",
        "class SwiftCTBRTrainingRewardsCfg",
        "self.scene.num_envs = 100",
        "self.episode_length_s = 15.0",
        "weight=100.0",
        "weight=2.0",
        "weight=-0.02",
        "weight=-0.01",
        "weight=-500.0",
    ):
        assert token in cfg

    for token in (
        "def swift_ctbr_body_rate_command_l2",
        "def swift_ctbr_command_delta_l2",
    ):
        assert token in rewards

    for token in (
        "separate: True",
        "layers: [128, 128]",
        "learning_rate: 3.0e-04",
        "discount_factor: 0.99",
        "ratio_clip: 0.2",
        "output: tanh(ACTIONS)",
        "rewards_shaper_scale: 1.0",
    ):
        assert token in agent

    assert "--checkpoint" not in agent.lower()
    assert "source_checkpoint" not in agent.lower()
    assert "gym.spaces.Box" in env
    assert "low=-1.0" in env and "high=1.0" in env



def test_swift_policy0_train_script_audit():
    train = _text("scripts/rl/train.py")

    for token in (
        "SWIFT_CTBR_POLICY0_TASKS = {",
        '"Isaac-Drone-Racer-Swift-CTBR-Train-v0"',
        '"Isaac-Drone-Racer-Swift-CTBR-Train-PassState-v0"',
        "def _audit_swift_ctbr_policy0_cfg",
        'str(policy_cfg.get("output")) != "tanh(ACTIONS)"',
        "expected_layers = [128, 128]",
        'float(agent["learning_rate"]) - 3.0e-4',
        'float(agent["discount_factor"]) - 0.99',
        'float(agent["ratio_clip"]) - 0.2',
        "_audit_swift_ctbr_policy0_cfg(env, agent_cfg)",
        '"Isaac-Drone-Racer-Learned-Inertial-Swift-CTBR-v0"',
    ):
        assert token in train



def test_swift_pass_state_curriculum_and_evaluator():
    commands = _text("tasks/drone_racer/mdp/commands.py")
    cfg = _text("tasks/drone_racer/drone_racer_swift_ctbr_env_cfg.py")
    registry = _text("tasks/drone_racer/__init__.py")
    evaluator = _text("scripts/rl/evaluate_swift_ctbr_policy0.py")

    for token in (
        "class SwiftPassStateGateTargetingCommand",
        "forward_speed_range_mps",
        "start_pos",
        "velocity_w = heading * speed",
        "class SwiftPassStateGateTargetingCommandCfg",
    ):
        assert token in commands

    assert "SwiftPassStateGateTargetingCommandCfg" in cfg
    assert "forward_speed_range_mps=(1.5, 3.0)" in cfg
    assert "Isaac-Drone-Racer-Swift-CTBR-Train-PassState-v0" in registry
    assert "skrl_swift_ctbr_passstate_cfg.yaml" in registry

    for token in (
        "SWIFT CTBR POLICY-0 EVALUATION SUMMARY",
        "gates_passed",
        "action_saturation_fraction",
        "full_lap_completion_rate",
    ):
        assert token in evaluator
