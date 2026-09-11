import json

import cv2
import numpy as np
import torch

from perception.keypoint_detector import GateKeypointNet, Stage2KeypointDataset, TorchGateCornerDetector


def _write_sample(root, sample_id="sample"):
    (root / "images").mkdir(parents=True)
    (root / "labels").mkdir(parents=True)
    image = np.zeros((48, 64, 3), dtype=np.uint8)
    assert cv2.imwrite(str(root / "images" / f"{sample_id}.png"), image)
    label = {
        "corners_uv": [[8, 40], [56, 40], [56, 8], [8, 8]],
        "visible": [True, True, True, True],
        "camera": {"image_width": 64, "image_height": 48},
    }
    (root / "labels" / f"{sample_id}.json").write_text(json.dumps(label), encoding="utf-8")


def test_stage2_dataset_normalizes_pixels(tmp_path):
    _write_sample(tmp_path)
    image, corners, visible, sample_id = Stage2KeypointDataset(tmp_path, input_size=32)[0]
    assert image.shape == (3, 32, 32)
    assert sample_id == "sample"
    assert torch.all(visible == 1)
    torch.testing.assert_close(corners[0], torch.tensor([0.125, 40 / 48]))


def test_detector_checkpoint_implements_corner_protocol(tmp_path):
    model = GateKeypointNet(input_size=32, width=4)
    checkpoint = tmp_path / "detector.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "input_size": 32,
            "width": 4,
            "metadata": {"schema": "test"},
        },
        checkpoint,
    )
    detector = TorchGateCornerDetector(checkpoint)
    result = detector.detect(np.zeros((48, 64, 3), dtype=np.uint8))
    assert result.corners_uv.shape == (4, 2)
    assert result.visible.shape == (4,)
    assert np.all(np.isfinite(result.corners_uv))
