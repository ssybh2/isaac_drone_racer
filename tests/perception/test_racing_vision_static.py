"""Static contract tests for the racing-vision data/retraining pipeline."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def _text(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def test_racing_vision_collector_keeps_control_gt_only():
    text = _text("scripts/perception/collect_racing_vision_dataset.py")
    assert "Isaac-Drone-Racer-Swift-CTBR-GT-Racing-v0" in text
    assert "stage2_reference_camera_cfg()" in text
    assert 'args_cli.enable_cameras = True' in text
    assert "--min-gates" in text
    assert "if keep:" in text
    assert "IsaacStage2DatasetCollector" in text
    assert "obs, _ = wrapped.reset()" in text
    assert "learned_inertial" not in text.lower()


def test_racing_vision_collector_splits_by_complete_episode():
    text = _text("scripts/perception/collect_racing_vision_dataset.py")
    assert "_split_for_success" in text
    assert '"train"' in text
    assert '"val"' in text
    assert '"test"' in text
    assert "accepted_episode_index" in text
    assert "dataset_split" in text


def test_racing_training_targets_runtime_hybrid():
    text = _text("scripts/perception/train_racing_vision_models.py")
    assert "TorchvisionStage2KeypointDataset" in text
    assert "GateKeypointNet" in text
    assert "racing_repeat" in text
    assert "rcnn_image_size" in text
    assert "guard_input_size" in text
    assert "usable_ge2_rate_given_gt_ge2" in text
    assert "false_usable_ge2_rate_given_gt_lt2" in text
    assert "torchvision_keypointrcnn_racing_best.pt" in text
    assert "gate_keypoint_net_racing_best.pt" in text


def test_hybrid_racing_eval_measures_availability():
    text = _text("scripts/perception/evaluate_racing_vision_hybrid.py")
    assert "VisibilityGuardedGateCornerDetector" in text
    assert "keypoint_confidence_threshold=0.0" in text
    assert "hybrid_availability_given_gt_ge2" in text
    assert "speed_buckets_mps" in text
    assert "body_rate_buckets_radps" in text
