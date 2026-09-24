from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_imitation_training_and_estimator_curriculum_are_registered():
    registry = (ROOT / "tasks/drone_racer/__init__.py").read_text()
    expected = [
        "Isaac-Drone-Racer-Swift-CTBR-GT-Circular12-ImitationFineTune-v0",
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
