import json

import cv2
import numpy as np

from perception.racing_multigate_dataset import RacingMultiGateKeypointDataset


def _write_sample(root, sample_id, mapped_gates):
    (root / "images").mkdir(parents=True, exist_ok=True)
    (root / "labels").mkdir(parents=True, exist_ok=True)
    image = np.zeros((32, 32, 3), dtype=np.uint8)
    assert cv2.imwrite(str(root / "images" / f"{sample_id}.png"), image)
    payload = {
        "schema": "isaac_drone_racer.racing_estimator_frame.v1",
        "sample_id": sample_id,
        "mapped_gates": mapped_gates,
    }
    (root / "labels" / f"{sample_id}.json").write_text(
        json.dumps(payload),
        encoding="utf-8",
    )


def test_multigate_dataset_uses_all_eligible_mapped_gates(tmp_path):
    _write_sample(
        tmp_path,
        "frame",
        [
            {
                "gate_index": 2,
                "corners_uv": [[4, 20], [12, 20], [12, 8], [4, 8]],
                "visible": [True, True, True, True],
                "confidence": [1, 1, 1, 1],
            },
            {
                "gate_index": 7,
                "corners_uv": [[20, 24], [29, 24], [29, 12], [20, 12]],
                "visible": [True, True, False, False],
                "confidence": [1, 1, 0, 0],
            },
            {
                "gate_index": 9,
                "corners_uv": [[1, 1], [2, 1], [2, 2], [1, 2]],
                "visible": [True, False, False, False],
                "confidence": [1, 0, 0, 0],
            },
        ],
    )

    dataset = RacingMultiGateKeypointDataset(
        tmp_path,
        min_visible_corners=2,
        box_padding_px=2.0,
    )
    image, target, info = dataset[0]

    assert tuple(image.shape) == (3, 32, 32)
    assert tuple(target["boxes"].shape) == (2, 4)
    assert tuple(target["keypoints"].shape) == (2, 4, 3)
    assert target["labels"].tolist() == [1, 1]
    assert info.gate_indices == (2, 7)
    assert info.visible_masks.sum(axis=1).tolist() == [4, 2]


def test_multigate_dataset_keeps_negative_frames(tmp_path):
    _write_sample(
        tmp_path,
        "negative",
        [
            {
                "gate_index": 3,
                "corners_uv": [[0, 0], [0, 0], [0, 0], [0, 0]],
                "visible": [False, False, False, False],
                "confidence": [0, 0, 0, 0],
            }
        ],
    )

    dataset = RacingMultiGateKeypointDataset(
        tmp_path,
        min_visible_corners=2,
    )
    _, target, info = dataset[0]

    assert tuple(target["boxes"].shape) == (0, 4)
    assert tuple(target["keypoints"].shape) == (0, 4, 3)
    assert target["labels"].numel() == 0
    assert info.gate_indices == ()
