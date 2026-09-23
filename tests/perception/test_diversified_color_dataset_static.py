from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
COLLECTOR = ROOT / "scripts/perception/collect_circular12_diversified_vision_dataset.py"


def test_diversified_collector_locks_camera_and_height_defaults():
    text = COLLECTOR.read_text(encoding="utf-8")
    assert 'parser.add_argument("--camera-pitch-up-deg", type=float, default=20.0)' in text
    assert 'parser.add_argument("--body-height-m", type=float, default=2.07)' in text


def test_diversified_collector_uses_disjoint_condition_splits():
    text = COLLECTOR.read_text(encoding="utf-8")
    assert "SPEEDS_MPS = (14.0, 15.5, 17.0, 18.5, 20.0, 21.5)" in text
    assert "PATH_RADII_M = (11.6, 12.0, 12.4)" in text
    assert "PHASE_OFFSETS_DEG = (0.0, 7.5, 15.0)" in text
    assert "VAL_CONDITIONS" in text
    assert "TEST_CONDITIONS" in text
    assert "all phase offsets stay in one split" in text


def test_diversified_collector_writes_training_layout_and_gate_ids():
    text = COLLECTOR.read_text(encoding="utf-8")
    assert 'root / "vision" / split / "images"' in text
    assert 'root / "vision" / split / "labels"' in text
    assert '"gate_id": int(identity.gate_id)' in text
    assert '"color_name": identity.color_name' in text
    assert '"mapped_gates": mapped_gates' in text
