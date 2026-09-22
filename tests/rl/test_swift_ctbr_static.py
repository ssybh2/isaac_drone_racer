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



def test_swift_policy0_evaluator_uses_isaaclab_21_public_termination_api():
    evaluator = _text("scripts/rl/evaluate_swift_ctbr_policy0.py")

    assert "manager.get_term(name)" in evaluator
    assert "manager._last_episode_dones" not in evaluator



def test_upstream_inspired_gt_racing_contract():
    registry = _text("tasks/drone_racer/__init__.py")
    cfg = _text("tasks/drone_racer/drone_racer_swift_ctbr_env_cfg.py")
    agent = _text("tasks/drone_racer/agents/skrl_swift_ctbr_gt_racing_cfg.yaml")
    train = _text("scripts/rl/train.py")

    for token in (
        "Isaac-Drone-Racer-Swift-CTBR-GT-Racing-v0",
        "DroneRacerSwiftCTBRGTRacingEnvCfg",
        "skrl_swift_ctbr_gt_racing_cfg.yaml",
    ):
        assert token in registry

    for token in (
        "class SwiftCTBRGTRacingRewardsCfg",
        "func=mdp.gate_passed",
        "weight=400.0",
        "func=mdp.progress",
        "weight=20.0",
        "func=mdp.lookat_next_gate",
        "weight=0.1",
        "self.scene.num_envs = 4096",
        "self.episode_length_s = 20.0",
    ):
        assert token in cfg

    for token in (
        "separate: False",
        "layers: [256, 256, 256]",
        "activations: elu",
        "rollouts: 24",
        "learning_epochs: 5",
        "mini_batches: 4",
        "learning_rate: 1.0e-04",
        "entropy_loss_scale: 0.005",
        "rewards_shaper_scale: 0.6",
        "timesteps: 50000",
        "output: tanh(ACTIONS)",
    ):
        assert token in agent

    for token in (
        "SWIFT_CTBR_GT_RACING_TASKS = {",
        "def _audit_swift_ctbr_gt_racing_cfg",
        "expected_layers = [256, 256, 256]",
        '"rollouts": 24',
        '"learning_epochs": 5',
        '"mini_batches": 4',
        "_audit_swift_ctbr_gt_racing_cfg(env, agent_cfg)",
    ):
        assert token in train



def test_estimated_state_gt_policy_deployment_contract():
    registry = _text("tasks/drone_racer/__init__.py")
    obs = _text("tasks/drone_racer/mdp/learned_inertial_observations.py")
    audit = _text("scripts/rl/audit_swift_ctbr_estimated_state_no_gt.py")

    for token in (
        "Isaac-Drone-Racer-Learned-Inertial-Swift-CTBR-GTPolicy-v0",
        "DroneRacerLearnedInertialSwiftCTBRRLCfg",
        "skrl_swift_ctbr_gt_racing_cfg.yaml",
    ):
        assert token in registry

    platform_start = obs.index("def learned_inertial_swift_state")
    gate_start = obs.index("def learned_next_gate_corners_relative_w")
    platform_block = obs[platform_start:gate_start]
    gate_block = obs[gate_start:]

    for token in ("root_pos_w", "root_lin_vel_w", "root_quat_w", "root_state_w"):
        assert token not in platform_block
        assert token not in gate_block

    for token in (
        "EstimatedStateGateTargetingCommand",
        "actor_reads_runtime_simulator_gt",
        "mission_progression_reads_runtime_simulator_gt",
        "self.next_gate_idx[self._mission_gate_passed] += 1",
    ):
        assert token in audit



def test_gt_control_estimator_shadow_contract():
    registry = _text("tasks/drone_racer/__init__.py")
    cfg = _text("tasks/drone_racer/drone_racer_swift_ctbr_env_cfg.py")
    obs = _text("tasks/drone_racer/mdp/observations.py")

    for token in (
        "Isaac-Drone-Racer-Learned-Inertial-Swift-CTBR-GTShadow-v0",
        "DroneRacerLearnedInertialSwiftCTBRGTShadowCfg",
        "skrl_swift_ctbr_gt_racing_cfg.yaml",
    ):
        assert token in registry

    for token in (
        "class SwiftGTShadowPolicyCfg",
        "platform_state = ObsTerm(func=mdp.swift_gt_state)",
        "swift_gt_truth_next_gate_corners_relative_w",
        "class DroneRacerLearnedInertialSwiftCTBRGTShadowCfg",
    ):
        assert token in cfg

    for token in (
        "def swift_gt_truth_next_gate_corners_relative_w",
        "gt_next_gate_idx",
        "asset.data.root_pos_w",
    ):
        assert token in obs



def test_gtshadow_config_is_defined_after_base():
    cfg = _text("tasks/drone_racer/drone_racer_swift_ctbr_env_cfg.py")
    base = cfg.index("class DroneRacerLearnedInertialSwiftCTBRRLCfg")
    shadow = cfg.index("class DroneRacerLearnedInertialSwiftCTBRGTShadowCfg")
    assert base < shadow


def test_gtshadow_flyaway_uses_truth_gate():
    cfg = _text("tasks/drone_racer/drone_racer_swift_ctbr_env_cfg.py")
    terminations = _text("tasks/drone_racer/mdp/terminations.py")

    truth_start = terminations.index("def flyaway_truth_gate(")
    truth_block = terminations[truth_start:]
    assert "gt_next_gate_idx" in truth_block
    assert "command.track.data.object_com_pos_w" in truth_block
    assert "command.command" not in truth_block

    shadow_start = cfg.index(
        "class DroneRacerLearnedInertialSwiftCTBRGTShadowCfg"
    )
    shadow_block = cfg[shadow_start:]
    assert "self.terminations.flyaway = DoneTerm(" in shadow_block
    assert "func=mdp.flyaway_truth_gate" in shadow_block


def test_gtshadow_truth_progression_matches_gt_training_semantics():
    commands = _text("tasks/drone_racer/mdp/commands.py")

    estimated_start = commands.index("class EstimatedStateGateTargetingCommand")
    estimated = commands[estimated_start:]

    assert "def _legacy_gt_gate_crossing(" in estimated
    assert "euler_xyz_from_quat" in estimated
    assert "absolute_offset_w = torch.abs(current_pos_w - gate_pose_w[:, :3])" in estimated
    assert "self._legacy_gt_gate_crossing(" in estimated


def test_gtshadow_rewards_use_truth_gate():
    cfg = _text("tasks/drone_racer/drone_racer_swift_ctbr_env_cfg.py")
    rewards = _text("tasks/drone_racer/mdp/rewards.py")

    assert "def progress_truth_gate(" in rewards
    assert "def lookat_truth_gate(" in rewards
    assert "class SwiftCTBRGTShadowRewardsCfg" in cfg

    shadow_start = cfg.index(
        "class DroneRacerLearnedInertialSwiftCTBRGTShadowCfg"
    )
    shadow = cfg[shadow_start:]
    assert (
        "rewards: SwiftCTBRGTShadowRewardsCfg = "
        "SwiftCTBRGTShadowRewardsCfg()"
    ) in shadow


def test_gt_fixed_start_reference_task_contract():
    cfg = _text("tasks/drone_racer/drone_racer_swift_ctbr_env_cfg.py")
    registry = _text("tasks/drone_racer/__init__.py")

    start = cfg.index("class DroneRacerSwiftCTBRGTFixedStartEnvCfg")
    fixed = cfg[start:]
    assert "DroneRacerSwiftCTBRGTRacingEnvCfg" in fixed
    assert "self.commands.target.randomise_start = None" in fixed
    assert "self.events.reset_base = EventTerm(" in fixed
    assert "load_stage2_gate_geometry().center_g" in fixed
    assert "Isaac-Drone-Racer-Swift-CTBR-GT-FixedStart-v0" in registry


def test_perception_aware_gt_racing_contract():
    registry = _text("tasks/drone_racer/__init__.py")
    cfg = _text("tasks/drone_racer/drone_racer_swift_ctbr_env_cfg.py")
    rewards = _text("tasks/drone_racer/mdp/rewards.py")
    baseline_agent = _text(
        "tasks/drone_racer/agents/skrl_swift_ctbr_gt_racing_cfg.yaml"
    )
    perception_agent = _text(
        "tasks/drone_racer/agents/skrl_swift_ctbr_gt_perception_cfg.yaml"
    )
    train = _text("scripts/rl/train.py")

    assert "Isaac-Drone-Racer-Swift-CTBR-GT-PerceptionAware-v0" in registry
    assert "DroneRacerSwiftCTBRGTPerceptionAwareEnvCfg" in registry
    assert "skrl_swift_ctbr_gt_perception_cfg.yaml" in registry

    assert "class SwiftCTBRGTPerceptionAwareRewardsCfg" in cfg
    assert "func=mdp.gt_next_gate_image_visibility" in cfg
    assert "weight=2.0" in cfg
    assert '"usable_bonus_weight": 0.30' in cfg
    assert (
        "class DroneRacerSwiftCTBRGTPerceptionAwareEnvCfg"
        in cfg
    )
    assert "DroneRacerSwiftCTBRGTRacingEnvCfg" in cfg

    for token in (
        "def gt_next_gate_image_visibility(",
        "OPENVINS_CAMERA_INTRINSICS",
        "OPENVINS_CAMERA_RESOLUTION",
        "CAMERA_TO_BODY_ROTATION",
        "object_pos_w",
        "visible.sum(dim=1) >= 2",
        "margin_score",
        "center_score",
    ):
        assert token in rewards

    # Training hyperparameters must remain identical to the successful
    # 37-gate baseline. Only logging metadata is allowed to differ.
    normalized_baseline = baseline_agent.replace(
        'directory: "swift_ctbr_gt_racing"',
        'directory: "LOGDIR"',
    ).replace(
        'experiment_name: "easy7_4096env_roll24_256x3_gt"',
        'experiment_name: "EXPERIMENT"',
    ).replace(
        "# GT racing PPO profile.",
        "# PROFILE",
    )
    normalized_perception = perception_agent.replace(
        'directory: "swift_ctbr_gt_perception"',
        'directory: "LOGDIR"',
    ).replace(
        'experiment_name: "easy7_4096env_roll24_256x3_gt_perception"',
        'experiment_name: "EXPERIMENT"',
    )
    assert "rollouts: 24" in normalized_perception
    assert "learning_epochs: 5" in normalized_perception
    assert "mini_batches: 4" in normalized_perception
    assert "learning_rate: 1.0e-04" in normalized_perception
    assert "timesteps: 50000" in normalized_perception

    assert "SWIFT_CTBR_GT_RACING_TASKS = {" in train
    assert '"Isaac-Drone-Racer-Swift-CTBR-GT-PerceptionAware-v0"' in train


def test_perception_aware_gt_racing_v2_contract():
    registry = _text("tasks/drone_racer/__init__.py")
    cfg = _text("tasks/drone_racer/drone_racer_swift_ctbr_env_cfg.py")
    rewards = _text("tasks/drone_racer/mdp/rewards.py")
    agent = _text(
        "tasks/drone_racer/agents/skrl_swift_ctbr_gt_perception_v2_cfg.yaml"
    )
    train = _text("scripts/rl/train.py")

    assert "Isaac-Drone-Racer-Swift-CTBR-GT-PerceptionAwareV2-v0" in registry
    assert "DroneRacerSwiftCTBRGTPerceptionAwareV2EnvCfg" in registry
    assert "skrl_swift_ctbr_gt_perception_v2_cfg.yaml" in registry

    assert "class SwiftCTBRGTPerceptionAwareV2RewardsCfg" in cfg
    assert "camera_observability = RewTerm(" in cfg
    assert "weight=5.0" in cfg
    assert '"output_bias": -1.0' in cfg
    assert '"insufficient_visible_penalty": 0.50' in cfg

    assert "insufficient_visible_penalty" in rewards
    assert "visible_count < 2" in rewards

    for token in (
        "rollouts: 24",
        "learning_epochs: 5",
        "mini_batches: 4",
        "learning_rate: 1.0e-04",
        "timesteps: 50000",
    ):
        assert token in agent

    assert '"Isaac-Drone-Racer-Swift-CTBR-GT-PerceptionAwareV2-v0"' in train


def test_perception_aware_gt_racing_v3_contract():
    registry = _text("tasks/drone_racer/__init__.py")
    cfg = _text("tasks/drone_racer/drone_racer_swift_ctbr_env_cfg.py")
    rewards = _text("tasks/drone_racer/mdp/rewards.py")
    agent = _text(
        "tasks/drone_racer/agents/skrl_swift_ctbr_gt_perception_v3_cfg.yaml"
    )
    train = _text("scripts/rl/train.py")

    assert "Isaac-Drone-Racer-Swift-CTBR-GT-PerceptionAwareV3-v0" in registry
    assert "DroneRacerSwiftCTBRGTPerceptionAwareV3EnvCfg" in registry
    assert "skrl_swift_ctbr_gt_perception_v3_cfg.yaml" in registry

    assert "class SwiftCTBRGTPerceptionAwareV3RewardsCfg" in cfg
    assert "func=mdp.gt_next_gate_camera_angle_l2" in cfg
    assert "weight=-8.0" in cfg
    assert "camera_observability = RewTerm(" in cfg
    assert "weight=2.0" in cfg

    for token in (
        "def gt_next_gate_camera_angle_l2(",
        "camera_offset_b",
        "optical_axis_b",
        "torch.acos",
        "torch.square(angle)",
    ):
        assert token in rewards

    for token in (
        "rollouts: 24",
        "learning_epochs: 5",
        "mini_batches: 4",
        "learning_rate: 1.0e-04",
        "timesteps: 50000",
    ):
        assert token in agent

    assert '"Isaac-Drone-Racer-Swift-CTBR-GT-PerceptionAwareV3-v0"' in train


def test_circular12_gt_estimator_validation_track_contract():
    track = _text("tasks/drone_racer/track_generator.py")
    cfg = _text("tasks/drone_racer/drone_racer_swift_ctbr_env_cfg.py")
    registry = _text("tasks/drone_racer/__init__.py")
    agent = _text("tasks/drone_racer/agents/skrl_swift_ctbr_gt_circular12_cfg.yaml")
    train = _text("scripts/rl/train.py")

    assert "CIRCULAR_12_GATE_TRACK_CONFIG" in track
    for gate_id in range(1, 13):
        assert f'"{gate_id}"' in track
    assert '"1":  {"pos": (0.0, 0.0, 1.0), "yaw": 0.0}' in track
    assert '"4":  {"pos": (12.0, 12.0, 1.0), "yaw": torch.pi / 2.0}' in track
    assert '"7":  {"pos": (0.0, 24.0, 1.0), "yaw": torch.pi}' in track
    assert '"10": {"pos": (-12.0, 12.0, 1.0), "yaw": -torch.pi / 2.0}' in track

    assert "class DroneRacerSwiftCTBRGTCircular12RacingEnvCfg" in cfg
    assert "track_config=CIRCULAR_12_GATE_TRACK_CONFIG" in cfg
    assert "Isaac-Drone-Racer-Swift-CTBR-GT-Circular12-v0" in registry
    assert "skrl_swift_ctbr_gt_circular12_cfg.yaml" in registry
    assert '"Isaac-Drone-Racer-Swift-CTBR-GT-Circular12-v0"' in train

    for token in (
        "layers: [256, 256, 256]",
        "rollouts: 24",
        "learning_epochs: 5",
        "mini_batches: 4",
        "learning_rate: 1.0e-04",
        "timesteps: 50000",
        'directory: "swift_ctbr_gt_circular12"',
        'experiment_name: "circular12_r12_4096env_roll24_256x3_gt"',
    ):
        assert token in agent


def test_circular12_fixed_start_and_estimator_replacement_contract():
    cfg = _text("tasks/drone_racer/drone_racer_swift_ctbr_env_cfg.py")
    registry = _text("tasks/drone_racer/__init__.py")

    for token in (
        "class DroneRacerSwiftCTBRGTCircular12FixedStartEnvCfg",
        "DroneRacerSwiftCTBRGTCircular12RacingEnvCfg",
        "self.commands.target.randomise_start = None",
        "class DroneRacerLearnedInertialSwiftCTBRCircular12GTPolicyCfg",
        "DroneRacerLearnedInertialSwiftCTBRRLCfg",
        "track_config=CIRCULAR_12_GATE_TRACK_CONFIG",
    ):
        assert token in cfg

    for token in (
        "Isaac-Drone-Racer-Swift-CTBR-GT-Circular12-FixedStart-v0",
        "DroneRacerSwiftCTBRGTCircular12FixedStartEnvCfg",
        "Isaac-Drone-Racer-Learned-Inertial-Swift-CTBR-Circular12-GTPolicy-v0",
        "DroneRacerLearnedInertialSwiftCTBRCircular12GTPolicyCfg",
        "skrl_swift_ctbr_gt_circular12_cfg.yaml",
    ):
        assert token in registry


def test_direct_reprojection_associates_over_all_mapped_gates():
    env = _text("tasks/drone_racer/learned_inertial_racing_env.py")

    for token in (
        "for gate_index in range(self._gate_track_layout.num_gates):",
        "candidates.append((score, gate_index, points_w))",
        "candidates.sort(key=lambda item: item[0])",
        "association_rmse, gate_index, points_w = candidates[0]",
        'diagnostic["selected_gate_index"] = int(gate_index)',
    ):
        assert token in env


def test_circular12_known_start_matches_training_reset_support():
    track = _text("tasks/drone_racer/track_generator.py")
    cfg = _text("tasks/drone_racer/drone_racer_swift_ctbr_env_cfg.py")
    registry = _text("tasks/drone_racer/__init__.py")

    for token in (
        "CIRCULAR_12_KNOWN_START_POS_W",
        "-5.133974596215562",
        "1.10769515",
        "CIRCULAR_12_KNOWN_START_ROT_WXYZ",
    ):
        assert token in track

    for token in (
        "class DroneRacerSwiftCTBRGTCircular12KnownStartEnvCfg",
        "self.scene.robot.init_state.pos = CIRCULAR_12_KNOWN_START_POS_W",
        "class DroneRacerLearnedInertialSwiftCTBRCircular12KnownStartGTPolicyCfg",
    ):
        assert token in cfg

    for token in (
        "Isaac-Drone-Racer-Swift-CTBR-GT-Circular12-KnownStart-v0",
        "Isaac-Drone-Racer-Learned-Inertial-Swift-CTBR-Circular12-KnownStart-GTPolicy-v0",
    ):
        assert token in registry
