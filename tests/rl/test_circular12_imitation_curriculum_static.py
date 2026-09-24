from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_imitation_training_and_estimator_curriculum_are_registered():
    registry = (ROOT / "tasks/drone_racer/__init__.py").read_text()
    expected = [
        "Isaac-Drone-Racer-Swift-CTBR-GT-Circular12-ImitationFineTune-v0",
        "Isaac-Drone-Racer-Swift-CTBR-GT-Circular12-ImitationResidualNoise-v0",
        "Isaac-Drone-Racer-Learned-Inertial-Swift-CTBR-Circular12-Imitation-GTShadow-Color20-v0",
        "Isaac-Drone-Racer-Learned-Inertial-Swift-CTBR-Circular12-Imitation-Blend25-Color20-v0",
        "Isaac-Drone-Racer-Learned-Inertial-Swift-CTBR-Circular12-Imitation-Blend50-Color20-v0",
        "Isaac-Drone-Racer-Learned-Inertial-Swift-CTBR-Circular12-Imitation-Blend75-Color20-v0",
        "Isaac-Drone-Racer-Learned-Inertial-Swift-CTBR-Circular12-Imitation-EstStateTruthMission-Color20-v0",
        "Isaac-Drone-Racer-Learned-Inertial-Swift-CTBR-Circular12-Imitation-EstimatorMission-Color20-v0",
        "Isaac-Drone-Racer-Learned-Inertial-Swift-CTBR-Circular12-Imitation-EstimatorMission-Color20-V7-v0",
    ]
    for task in expected:
        assert task in registry


def test_imitation_task_keeps_tight_ctbr_authority_and_20deg_camera():
    cfg = (
        ROOT / "tasks/drone_racer/drone_racer_swift_ctbr_env_cfg.py"
    ).read_text()
    assert "body_rate_max_radps = (4.0, 4.0, 2.0)" in cfg
    assert "pitch_up_deg=20.0" in cfg
    assert "circular12_color20_h207_gateid_kprcnn_v1" in cfg


def test_train_auto_enables_camera_for_learned_inertial_curriculum():
    train = (ROOT / "scripts/rl/train.py").read_text()
    assert 'CAMERA_REQUIRED_TASK_PREFIXES' in train
    assert '"Isaac-Drone-Racer-Learned-Inertial-"' in train


def test_full_estimator_blend_does_not_fallback_to_gt():
    obs = (
        ROOT / "tasks/drone_racer/mdp/learned_inertial_observations.py"
    ).read_text()
    assert "if float(blend_alpha) >= 1.0 - 1.0e-12" in obs
    assert "q_identity[:, 0] = 1.0" in obs


def test_imitation_ppo_is_covered_by_gt_racing_contract_audit():
    train = (ROOT / "scripts/rl/train.py").read_text()
    assert (
        '"Isaac-Drone-Racer-Swift-CTBR-GT-Circular12-'
        'ImitationFineTune-v0"'
    ) in train


def test_stage_c_noise_robustness_task_is_registered():
    registry = (ROOT / "tasks/drone_racer/__init__.py").read_text()
    cfg = (
        ROOT / "tasks/drone_racer/drone_racer_swift_ctbr_env_cfg.py"
    ).read_text()
    obs = (
        ROOT / "tasks/drone_racer/mdp/learned_inertial_observations.py"
    ).read_text()
    assert (
        "Isaac-Drone-Racer-Swift-CTBR-GT-Circular12-"
        "ImitationNoiseRobust-v0"
    ) in registry
    assert "SwiftGTNoisePolicyCfg" in cfg
    assert "noisy_gt_swift_state" in obs
    assert "_circular12_noisy_gt_cache" in obs


def test_stage_e_uses_pure_estimator_platform_state_not_alpha_one_blend():
    cfg = (
        ROOT / "tasks/drone_racer/drone_racer_swift_ctbr_env_cfg.py"
    ).read_text()
    assert "class SwiftEstimatorTruthMissionPolicyCfg" in cfg
    assert "platform_state = ObsTerm(func=mdp.learned_inertial_swift_state)" in cfg
    assert "func=mdp.learned_truth_next_gate_corners_relative_w" in cfg

    observations = (
        ROOT / "tasks/drone_racer/mdp/learned_inertial_observations.py"
    ).read_text()
    assert "def learned_truth_next_gate_corners_relative_w" in observations


def test_expert_validation_uses_coordinated_reference_reset():
    registry = (ROOT / "tasks/drone_racer/__init__.py").read_text()
    cfg = (
        ROOT / "tasks/drone_racer/drone_racer_swift_ctbr_env_cfg.py"
    ).read_text()
    events = (ROOT / "tasks/drone_racer/mdp/events.py").read_text()
    assert (
        "Isaac-Drone-Racer-Swift-CTBR-GT-Circular12-"
        "ExpertValidation-v0"
    ) in registry
    assert "DroneRacerSwiftCTBRGTCircular12ExpertValidationEnvCfg" in cfg
    assert "reset_circular12_coordinated_state" in events
    assert "velocity[:, 5] = omega" in events
    assert "self.commands.target.randomise_start = None" in cfg


def test_expert_demo_uses_mild_reset_perturbations():
    registry = (ROOT / "tasks/drone_racer/__init__.py").read_text()
    cfg = (
        ROOT / "tasks/drone_racer/drone_racer_swift_ctbr_env_cfg.py"
    ).read_text()
    events = (ROOT / "tasks/drone_racer/mdp/events.py").read_text()
    assert (
        "Isaac-Drone-Racer-Swift-CTBR-GT-Circular12-ExpertDemo-v0"
    ) in registry
    assert "DroneRacerSwiftCTBRGTCircular12ExpertDemoEnvCfg" in cfg
    for token in (
        '"phase_jitter_rad"',
        '"radial_jitter_m"',
        '"height_jitter_m"',
        '"speed_jitter_mps"',
        '"attitude_jitter_rad"',
        '"angular_rate_jitter_radps"',
    ):
        assert token in cfg
    assert "def _sym_jitter" in events
