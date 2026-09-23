from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_color20_gate_id_gtshadow_task_uses_new_detector_and_camera():
    cfg = (
        ROOT / "tasks/drone_racer/drone_racer_swift_ctbr_env_cfg.py"
    ).read_text(encoding="utf-8")
    assert "DroneRacerLearnedInertialSwiftCTBRCircular12KnownStartGTShadowColor20VisionV1Cfg" in cfg
    assert "circular12_color20_h207_gateid_kprcnn_v1" in cfg
    assert "pitch_up_deg=20.0" in cfg
    assert "gate_camera_pitch_up_deg = 20.0" in cfg
    assert "gate_reprojection_use_checkpoint_sigma = True" in cfg
    assert "gate_debug_gt_diagnostics = True" in cfg


def test_color20_gate_id_gtshadow_task_is_registered():
    registry = (
        ROOT / "tasks/drone_racer/__init__.py"
    ).read_text(encoding="utf-8")
    assert (
        "Isaac-Drone-Racer-Learned-Inertial-Swift-CTBR-Circular12-"
        "KnownStart-GTShadow-Color20VisionV1-v0"
    ) in registry
