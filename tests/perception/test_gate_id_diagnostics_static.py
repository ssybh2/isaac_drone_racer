from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CONFUSION = ROOT / "scripts/perception/evaluate_gate_id_confusion.py"
VIDEO = ROOT / "scripts/perception/visualize_gate_id_detector.py"


def test_gate_id_confusion_matches_by_geometry_before_identity():
    text = CONFUSION.read_text(encoding="utf-8")
    assert "confusion_counts.csv" in text
    assert "confusion_row_normalized.csv" in text
    assert "confusion_matrix.png" in text
    assert "confusion_report.json" in text
    assert "predicted Gate ID is scored after the geometry match" in text


def test_gate_id_video_renders_prediction_truth_identity_and_corners():
    text = VIDEO.read_text(encoding="utf-8")
    assert "P:{pred_text}" in text
    assert "T:G{truth_gate_id:02d}" in text
    assert "_draw_poly(" in text
    assert "--run-index" in text
    assert "--max-frames" in text


def test_gate_id_diagnostics_default_to_new_color20_checkpoint():
    expected = "circular12_color20_h207_gateid_kprcnn_v1"
    assert expected in CONFUSION.read_text(encoding="utf-8")
    assert expected in VIDEO.read_text(encoding="utf-8")
