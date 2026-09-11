"""Torchvision Keypoint R-CNN comparison baseline for Stage2B."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.models.detection import keypointrcnn_resnet50_fpn

from .corner_detection import CornerObservation


def build_keypoint_rcnn(*, image_size: int = 128):
    """Build the maintained torchvision keypoint architecture without hidden downloads."""
    return keypointrcnn_resnet50_fpn(
        weights=None,
        weights_backbone=None,
        num_classes=2,
        num_keypoints=4,
        min_size=image_size,
        max_size=image_size,
        box_detections_per_img=10,
    )


class TorchvisionStage2KeypointDataset(Dataset):
    """Adapt Stage2 labels to torchvision's box/keypoint target contract."""

    def __init__(self, root: str | Path, sample_ids: list[str] | None = None):
        self.root = Path(root)
        if sample_ids is None:
            sample_ids = sorted(path.stem for path in (self.root / "labels").glob("*.json"))
        self.sample_ids = list(sample_ids)
        if not self.sample_ids:
            raise ValueError(f"No Stage2 samples found under {self.root}")

    def __len__(self) -> int:
        return len(self.sample_ids)

    def __getitem__(self, index: int):
        sample_id = self.sample_ids[index]
        label = json.loads((self.root / "labels" / f"{sample_id}.json").read_text(encoding="utf-8"))
        bgr = cv2.imread(str(self.root / "images" / f"{sample_id}.png"), cv2.IMREAD_COLOR)
        if bgr is None:
            raise RuntimeError(f"Unable to read Stage2 image {sample_id}")
        image = torch.from_numpy(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).copy()).permute(2, 0, 1).float().div_(255.0)

        height, width = bgr.shape[:2]
        corners = np.asarray(label["corners_uv"], dtype=np.float32)
        visible = np.asarray(label["visible"], dtype=bool)
        visible_corners = corners[visible]
        if len(visible_corners):
            x1, y1 = visible_corners.min(axis=0) - 8.0
            x2, y2 = visible_corners.max(axis=0) + 8.0
            box = [max(0.0, x1), max(0.0, y1), min(float(width - 1), x2), min(float(height - 1), y2)]
        else:
            box = [0.0, 0.0, float(width - 1), float(height - 1)]
        box[2] = max(box[2], box[0] + 1.0)
        box[3] = max(box[3], box[1] + 1.0)

        keypoints = np.zeros((4, 3), dtype=np.float32)
        keypoints[visible, :2] = corners[visible]
        keypoints[visible, 2] = 2.0
        target = {
            "boxes": torch.tensor([box], dtype=torch.float32),
            "labels": torch.ones(1, dtype=torch.int64),
            "keypoints": torch.from_numpy(keypoints).unsqueeze(0),
            "image_id": torch.tensor(index, dtype=torch.int64),
        }
        return image, target, sample_id


class TorchvisionGateCornerDetector:
    """Inference adapter exposing Keypoint R-CNN through CornerObservation."""

    def __init__(
        self,
        checkpoint_path: str | Path,
        *,
        device: str = "cpu",
        detection_threshold: float = 0.5,
    ):
        payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
        self.device = torch.device(device)
        self.image_size = int(payload["image_size"])
        self.model = build_keypoint_rcnn(image_size=self.image_size).to(self.device)
        self.model.load_state_dict(payload["model_state_dict"])
        self.model.eval()
        self.detection_threshold = float(detection_threshold)
        self.metadata = dict(payload["metadata"])

    @torch.inference_mode()
    def detect(self, rgb_image: np.ndarray, *, timestamp_s: float = 0.0) -> CornerObservation:
        image = np.asarray(rgb_image)
        if image.ndim != 3 or image.shape[-1] != 3:
            raise ValueError("rgb_image must have shape (H, W, 3)")
        tensor = torch.from_numpy(image.copy()).permute(2, 0, 1).float().div_(255.0).to(self.device)
        output = self.model([tensor])[0]
        if not len(output["scores"]) or float(output["scores"][0]) < self.detection_threshold:
            return CornerObservation(
                np.zeros((4, 2), dtype=np.float64),
                visible=np.zeros(4, dtype=bool),
                confidence=np.zeros(4, dtype=np.float64),
                timestamp_s=timestamp_s,
                source="torchvision_keypoint_rcnn",
            )
        confidence = torch.sigmoid(output["keypoints_scores"][0])
        return CornerObservation(
            corners_uv=output["keypoints"][0, :, :2].cpu().numpy(),
            visible=np.ones(4, dtype=bool),
            confidence=confidence.cpu().numpy(),
            timestamp_s=timestamp_s,
            source="torchvision_keypoint_rcnn",
        )
