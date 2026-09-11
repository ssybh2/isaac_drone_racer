"""Small supervised Stage2B gate-corner detector and dataset utilities."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset

from .corner_detection import CornerObservation


class GateKeypointNet(nn.Module):
    """Compact spatial heatmap network with ordered-corner visibility heads."""

    def __init__(self, input_size: int = 128, width: int = 32):
        super().__init__()
        self.input_size = int(input_size)
        self.width = int(width)
        self.enc1 = self._block(3, width, stride=2)
        self.enc2 = self._block(width, width * 2, stride=2)
        self.enc3 = self._block(width * 2, width * 4, stride=2)
        self.bottleneck = nn.Sequential(
            self._block(width * 4, width * 4),
            self._block(width * 4, width * 4),
        )
        self.dec2 = self._block(width * 6, width * 2)
        self.dec1 = self._block(width * 3, width)
        self.heatmap_head = nn.Conv2d(width, 4, 1)
        self.visibility_head = nn.Linear(width * 4, 4)

    @staticmethod
    def _block(in_channels: int, out_channels: int, *, stride: int = 1) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        feature1 = self.enc1(images)
        feature2 = self.enc2(feature1)
        feature3 = self.bottleneck(self.enc3(feature2))
        decoded2 = F.interpolate(feature3, size=feature2.shape[-2:], mode="bilinear", align_corners=False)
        decoded2 = self.dec2(torch.cat((decoded2, feature2), dim=1))
        decoded1 = F.interpolate(decoded2, size=feature1.shape[-2:], mode="bilinear", align_corners=False)
        decoded1 = self.dec1(torch.cat((decoded1, feature1), dim=1))
        heatmaps = self.heatmap_head(decoded1)

        batch, corners, height, width = heatmaps.shape
        probabilities = torch.softmax(heatmaps.reshape(batch, corners, -1), dim=-1).reshape_as(heatmaps)
        x_coordinates = torch.linspace(0.0, 1.0, width, device=images.device, dtype=images.dtype)
        y_coordinates = torch.linspace(0.0, 1.0, height, device=images.device, dtype=images.dtype)
        x = (probabilities.sum(dim=2) * x_coordinates).sum(dim=-1)
        y = (probabilities.sum(dim=3) * y_coordinates).sum(dim=-1)
        corners_normalized = torch.stack((x, y), dim=-1)

        pooled = F.adaptive_avg_pool2d(feature3, 1).flatten(1)
        visibility_logits = self.visibility_head(pooled)
        return corners_normalized, visibility_logits


class Stage2KeypointDataset(Dataset):
    """Read the versioned PNG/JSON samples emitted by Stage2DatasetWriter."""

    def __init__(self, root: str | Path, sample_ids: list[str] | None = None, *, input_size: int = 128):
        self.root = Path(root)
        self.input_size = int(input_size)
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
        image = cv2.imread(str(self.root / "images" / f"{sample_id}.png"), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"Unable to read Stage2 image {sample_id}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (self.input_size, self.input_size), interpolation=cv2.INTER_AREA)
        image_tensor = torch.from_numpy(image.copy()).permute(2, 0, 1).float().div_(255.0)

        width = float(label["camera"]["image_width"])
        height = float(label["camera"]["image_height"])
        corners = torch.as_tensor(label["corners_uv"], dtype=torch.float32)
        corners[:, 0] /= width
        corners[:, 1] /= height
        visible = torch.as_tensor(label["visible"], dtype=torch.float32)
        # Invisible coordinates are placeholders and never enter the corner loss.
        corners = torch.where(visible[:, None].bool(), corners, torch.zeros_like(corners))
        return image_tensor, corners, visible, sample_id


@dataclass(frozen=True)
class DetectorCheckpoint:
    input_size: int
    width: int
    state_dict: dict
    metadata: dict


def load_detector_checkpoint(path: str | Path, *, map_location: str | torch.device = "cpu") -> DetectorCheckpoint:
    payload = torch.load(path, map_location=map_location, weights_only=False)
    required = {"model_state_dict", "input_size", "width", "metadata"}
    missing = required.difference(payload)
    if missing:
        raise ValueError(f"Incomplete Stage2B detector checkpoint; missing {sorted(missing)}")
    return DetectorCheckpoint(
        input_size=int(payload["input_size"]),
        width=int(payload["width"]),
        state_dict=payload["model_state_dict"],
        metadata=dict(payload["metadata"]),
    )


class TorchGateCornerDetector:
    """Inference adapter implementing the shared GateCornerDetector protocol."""

    def __init__(self, checkpoint_path: str | Path, *, device: str = "cpu", visibility_threshold: float = 0.5):
        checkpoint = load_detector_checkpoint(checkpoint_path, map_location=device)
        self.device = torch.device(device)
        self.model = GateKeypointNet(checkpoint.input_size, checkpoint.width).to(self.device)
        self.model.load_state_dict(checkpoint.state_dict)
        self.model.eval()
        self.input_size = checkpoint.input_size
        self.visibility_threshold = float(visibility_threshold)
        self.metadata = checkpoint.metadata

    @torch.inference_mode()
    def detect(self, rgb_image: np.ndarray, *, timestamp_s: float = 0.0) -> CornerObservation:
        image = np.asarray(rgb_image)
        if image.ndim != 3 or image.shape[-1] != 3:
            raise ValueError("rgb_image must have shape (H, W, 3)")
        height, width = image.shape[:2]
        tensor = torch.from_numpy(image.copy()).permute(2, 0, 1).float().div_(255.0).unsqueeze(0)
        tensor = F.interpolate(tensor, (self.input_size, self.input_size), mode="bilinear", align_corners=False)
        corners, logits = self.model(tensor.to(self.device))
        confidence = torch.sigmoid(logits[0])
        corners = corners[0]
        corners[:, 0] *= width
        corners[:, 1] *= height
        return CornerObservation(
            corners_uv=corners.cpu().numpy(),
            visible=(confidence >= self.visibility_threshold).cpu().numpy(),
            confidence=confidence.cpu().numpy(),
            timestamp_s=timestamp_s,
            source="stage2b_gate_keypoint_net",
        )
