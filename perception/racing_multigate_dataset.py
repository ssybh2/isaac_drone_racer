"""Multi-instance gate dataset utilities for high-speed Circular-12 vision.

The racing collector stores labels for every mapped gate in each camera frame.
This module converts those map-wide labels into torchvision Keypoint R-CNN
targets while keeping episode-level train/val/test separation intact.

Only gates with at least min_visible_corners are trained as instances.
Frames with no eligible gate remain in the dataset as true negative images,
which is important for racing where the camera frequently points away from all
usable gates.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class MultiGateTargetInfo:
    sample_id: str
    gate_indices: tuple[int, ...]
    visible_masks: np.ndarray
    corners_uv: np.ndarray

    def __post_init__(self) -> None:
        visible = np.asarray(self.visible_masks, dtype=bool)
        corners = np.asarray(self.corners_uv, dtype=np.float64)
        if visible.ndim != 2 or visible.shape[1:] != (4,):
            raise ValueError("visible_masks must have shape (N, 4)")
        if corners.shape != (visible.shape[0], 4, 2):
            raise ValueError("corners_uv must have shape (N, 4, 2)")
        object.__setattr__(self, "visible_masks", visible)
        object.__setattr__(self, "corners_uv", corners)


def sample_ids(root: str | Path) -> list[str]:
    root = Path(root)
    ids = sorted(path.stem for path in (root / "labels").glob("*.json"))
    if not ids:
        raise ValueError(f"No racing vision labels found under {root}")
    return ids


def _instance_box(
    corners_uv: np.ndarray,
    visible: np.ndarray,
    *,
    width: int,
    height: int,
    padding_px: float,
) -> list[float]:
    points = np.asarray(corners_uv, dtype=np.float32)[np.asarray(visible, dtype=bool)]
    if len(points) < 1:
        raise ValueError("instance box requires at least one visible corner")
    x1, y1 = points.min(axis=0) - float(padding_px)
    x2, y2 = points.max(axis=0) + float(padding_px)
    box = [
        max(0.0, float(x1)),
        max(0.0, float(y1)),
        min(float(width - 1), float(x2)),
        min(float(height - 1), float(y2)),
    ]
    box[2] = max(box[2], box[0] + 1.0)
    box[3] = max(box[3], box[1] + 1.0)
    return box


class RacingMultiGateKeypointDataset(Dataset):
    """Torchvision dataset using mapped_gates instead of only the active gate."""

    def __init__(
        self,
        root: str | Path,
        sample_ids_override: list[str] | None = None,
        *,
        min_visible_corners: int = 2,
        box_padding_px: float = 10.0,
    ) -> None:
        self.root = Path(root)
        self.sample_ids = (
            sample_ids(self.root)
            if sample_ids_override is None
            else list(sample_ids_override)
        )
        if not self.sample_ids:
            raise ValueError(f"No samples found under {self.root}")
        self.min_visible_corners = int(min_visible_corners)
        self.box_padding_px = float(box_padding_px)
        if not 1 <= self.min_visible_corners <= 4:
            raise ValueError("min_visible_corners must be in [1, 4]")
        if self.box_padding_px < 0.0:
            raise ValueError("box_padding_px must be non-negative")

    def __len__(self) -> int:
        return len(self.sample_ids)

    def __getitem__(self, index: int):
        sample_id = self.sample_ids[index]
        label_path = self.root / "labels" / f"{sample_id}.json"
        image_path = self.root / "images" / f"{sample_id}.png"
        payload = json.loads(label_path.read_text(encoding="utf-8"))
        mapped = payload.get("mapped_gates")
        if mapped is None:
            raise ValueError(
                f"{label_path} has no mapped_gates; use the Circular-12 racing dataset"
            )

        bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise RuntimeError(f"Unable to read racing image {image_path}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        image = (
            torch.from_numpy(rgb.copy())
            .permute(2, 0, 1)
            .float()
            .div_(255.0)
        )
        height, width = bgr.shape[:2]

        boxes: list[list[float]] = []
        keypoints: list[np.ndarray] = []
        gate_indices: list[int] = []
        truth_corners: list[np.ndarray] = []
        truth_visible: list[np.ndarray] = []

        for gate in mapped:
            corners = np.asarray(gate["corners_uv"], dtype=np.float32).reshape(4, 2)
            visible = np.asarray(gate["visible"], dtype=bool).reshape(4)
            if int(visible.sum()) < self.min_visible_corners:
                continue

            boxes.append(
                _instance_box(
                    corners,
                    visible,
                    width=width,
                    height=height,
                    padding_px=self.box_padding_px,
                )
            )
            kp = np.zeros((4, 3), dtype=np.float32)
            kp[visible, :2] = corners[visible]
            kp[visible, 2] = 2.0
            keypoints.append(kp)
            gate_indices.append(int(gate["gate_index"]))
            truth_corners.append(corners.astype(np.float64))
            truth_visible.append(visible)

        count = len(boxes)
        if count:
            boxes_tensor = torch.as_tensor(boxes, dtype=torch.float32).reshape(count, 4)
            keypoints_tensor = torch.from_numpy(
                np.stack(keypoints, axis=0)
            ).float()
            area = (
                (boxes_tensor[:, 2] - boxes_tensor[:, 0])
                * (boxes_tensor[:, 3] - boxes_tensor[:, 1])
            )
            corners_array = np.stack(truth_corners, axis=0)
            visible_array = np.stack(truth_visible, axis=0)
        else:
            boxes_tensor = torch.zeros((0, 4), dtype=torch.float32)
            keypoints_tensor = torch.zeros((0, 4, 3), dtype=torch.float32)
            area = torch.zeros((0,), dtype=torch.float32)
            corners_array = np.zeros((0, 4, 2), dtype=np.float64)
            visible_array = np.zeros((0, 4), dtype=bool)

        # Torchvision class 0 is background. Circular-12 Gate IDs are 1..12,
        # so the detector classification head can learn color -> global Gate ID
        # directly while the keypoint head learns the four semantic corners.
        labels_tensor = torch.as_tensor(
            [gate_index + 1 for gate_index in gate_indices],
            dtype=torch.int64,
        )
        target = {
            "boxes": boxes_tensor,
            "labels": labels_tensor,
            "keypoints": keypoints_tensor,
            "image_id": torch.tensor(index, dtype=torch.int64),
            "area": area,
            "iscrowd": torch.zeros((count,), dtype=torch.int64),
        }
        info = MultiGateTargetInfo(
            sample_id=sample_id,
            gate_indices=tuple(gate_indices),
            visible_masks=visible_array,
            corners_uv=corners_array,
        )
        return image, target, info


def collate(batch):
    images, targets, infos = zip(*batch)
    return list(images), list(targets), list(infos)
