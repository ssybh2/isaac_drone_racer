from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_direct_reprojection_uses_bounded_gate_identity_prior_with_fallback():
    text = (
        ROOT / "tasks/drone_racer/learned_inertial_racing_env.py"
    ).read_text(encoding="utf-8")
    assert "gate_identity_min_confidence" in text
    assert "gate_identity_preferred_max_rmse_px" in text
    assert "gate_identity_preference_margin_px" in text
    assert "identity_candidates" in text
    assert 'association_mode = "gate_id_preferred"' in text
    assert 'association_mode = "all_map_fallback"' in text
    assert "for gate_index in range(self._gate_track_layout.num_gates)" in text
    assert "gate_identity_fallback_used" in text


def test_detector_checkpoint_contract_supports_gate_id_classes():
    text = (
        ROOT / "perception/torchvision_keypoint_detector.py"
    ).read_text(encoding="utf-8")
    assert 'self.gate_identity_mode == "gate_id_class"' in text
    assert "gate_id=gate_id" in text
