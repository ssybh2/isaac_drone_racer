from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_direct_reprojection_uses_gate_identity_when_confident():
    text = (
        ROOT / "tasks/drone_racer/learned_inertial_racing_env.py"
    ).read_text(encoding="utf-8")
    assert "gate_identity_min_confidence" in text
    assert "gate_index_from_id" in text
    assert "candidate_gate_indices = (identified_gate_index,)" in text
    assert "gate_identity_reprojection_inconsistent" in text


def test_detector_checkpoint_contract_supports_gate_id_classes():
    text = (
        ROOT / "perception/torchvision_keypoint_detector.py"
    ).read_text(encoding="utf-8")
    assert 'self.gate_identity_mode == "gate_id_class"' in text
    assert "gate_id=gate_id" in text
