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


def build_keypoint_rcnn(*, image_size: int = 128, num_classes: int = 2):
    """Build the maintained torchvision keypoint architecture without hidden downloads."""
    return keypointrcnn_resnet50_fpn(
        weights=None,
        weights_backbone=None,
        num_classes=int(num_classes),
        num_keypoints=4,
        min_size=image_size,
        max_size=image_size,
        box_detections_per_img=10,
    )


def _decode_keypoint_visibility(
    corners_uv: np.ndarray,
    *,
    image_width: int,
    image_height: int,
    keypoint_logits: np.ndarray | None,
    instance_score: float,
    confidence_threshold: float,
    min_quad_area_px2: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Conservatively convert Keypoint R-CNN outputs into the Stage2 visibility contract.

    Torchvision's ``keypoints_scores`` are heatmap peak logits, not calibrated
    visibility probabilities.  We therefore use sigmoid(logit) only as a
    monotonic confidence heuristic, multiply it by the instance score, require
    the point to be physically inside the image, and reject the full
    observation if the ordered LB/RB/RT/LT polygon is not a plausible convex
    quadrilateral.

    This removes the previous unsafe ``visible=np.ones(4)`` behavior.  It does
    not claim to solve every hallucinated-corner case; downstream IPPE
    reprojection and Kalman innovation gates remain required.
    """
    corners = np.asarray(corners_uv, dtype=np.float64).reshape(4, 2)
    finite = np.all(np.isfinite(corners), axis=1)
    in_frame = (
        finite
        & (corners[:, 0] >= 0.0)
        & (corners[:, 0] < float(image_width))
        & (corners[:, 1] >= 0.0)
        & (corners[:, 1] < float(image_height))
    )

    if keypoint_logits is None:
        # Fail closed.  Current torchvision exposes keypoints_scores; silently
        # treating a missing confidence signal as four visible corners would
        # reintroduce the original closed-loop bug.
        confidence = np.zeros(4, dtype=np.float64)
    else:
        logits = np.asarray(keypoint_logits, dtype=np.float64).reshape(4)
        confidence = 1.0 / (1.0 + np.exp(-np.clip(logits, -60.0, 60.0)))
        confidence *= float(np.clip(instance_score, 0.0, 1.0))

    visible = in_frame & (confidence >= float(confidence_threshold))

    # PnP is only useful when the four ordered corners form a physical gate
    # opening.  Reject the whole observation rather than guessing which
    # hallucinated point caused an impossible polygon.
    if np.all(visible):
        contour = corners.astype(np.float32).reshape(-1, 1, 2)
        area = abs(float(cv2.contourArea(contour)))
        convex = bool(cv2.isContourConvex(contour))
        edge_lengths = np.linalg.norm(np.roll(corners, -1, axis=0) - corners, axis=1)
        if (not convex) or area < float(min_quad_area_px2) or np.any(edge_lengths < 1.0):
            visible[:] = False

    return visible, confidence


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
    """Inference adapter exposing Keypoint R-CNN gate instances.

    detect_all is the production multi-instance API. detect is retained for
    legacy single-gate callers and returns the highest-scoring usable instance,
    or an empty observation when no instance survives thresholding.
    """

    def __init__(
        self,
        checkpoint_path: str | Path,
        *,
        device: str = "cpu",
        detection_threshold: float = 0.5,
        keypoint_confidence_threshold: float = 0.5,
        min_quad_area_px2: float = 16.0,
        max_instances: int = 10,
    ):
        payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
        self.device = torch.device(device)
        self.image_size = int(payload["image_size"])
        self.metadata = dict(payload.get("metadata", {}))
        self.num_classes = int(
            payload.get(
                "num_classes",
                self.metadata.get("num_classes", 2),
            )
        )
        self.gate_identity_mode = str(
            self.metadata.get("gate_identity_mode", "class_agnostic")
        )
        self.model = build_keypoint_rcnn(
            image_size=self.image_size,
            num_classes=self.num_classes,
        ).to(self.device)
        self.model.load_state_dict(payload["model_state_dict"])
        self.model.eval()
        self.detection_threshold = float(detection_threshold)
        self.keypoint_confidence_threshold = float(keypoint_confidence_threshold)
        self.min_quad_area_px2 = float(min_quad_area_px2)
        self.max_instances = int(max_instances)
        if self.max_instances < 1:
            raise ValueError("max_instances must be positive")

    @staticmethod
    def _empty_observation(timestamp_s: float) -> CornerObservation:
        return CornerObservation(
            np.zeros((4, 2), dtype=np.float64),
            visible=np.zeros(4, dtype=bool),
            confidence=np.zeros(4, dtype=np.float64),
            timestamp_s=timestamp_s,
            source="torchvision_keypoint_rcnn",
        )

    @torch.inference_mode()
    def detect_all(
        self,
        rgb_image: np.ndarray,
        *,
        timestamp_s: float = 0.0,
    ) -> list[CornerObservation]:
        image = np.asarray(rgb_image)
        if image.ndim != 3 or image.shape[-1] != 3:
            raise ValueError("rgb_image must have shape (H, W, 3)")
        tensor = (
            torch.from_numpy(image.copy())
            .permute(2, 0, 1)
            .float()
            .div_(255.0)
            .to(self.device)
        )
        output = self.model([tensor])[0]
        scores = output.get("scores")
        if scores is None or not len(scores):
            return []

        raw_keypoint_scores = output.get("keypoints_scores")
        raw_labels = output.get("labels")
        observations: list[CornerObservation] = []
        limit = min(int(len(scores)), self.max_instances)
        for index in range(limit):
            instance_score = float(scores[index].detach().cpu())
            if instance_score < self.detection_threshold:
                break
            corners = (
                output["keypoints"][index, :, :2]
                .detach()
                .cpu()
                .numpy()
            )
            logits = None
            if raw_keypoint_scores is not None and len(raw_keypoint_scores) > index:
                logits = raw_keypoint_scores[index].detach().cpu().numpy()

            visible, confidence = _decode_keypoint_visibility(
                corners,
                image_width=int(image.shape[1]),
                image_height=int(image.shape[0]),
                keypoint_logits=logits,
                instance_score=instance_score,
                confidence_threshold=self.keypoint_confidence_threshold,
                min_quad_area_px2=self.min_quad_area_px2,
            )
            gate_id = None
            gate_id_confidence = None
            if (
                self.gate_identity_mode == "gate_id_class"
                and raw_labels is not None
                and len(raw_labels) > index
            ):
                predicted_label = int(raw_labels[index].detach().cpu())
                if 1 <= predicted_label <= 12:
                    gate_id = predicted_label
                    gate_id_confidence = float(
                        np.clip(instance_score, 0.0, 1.0)
                    )

            observations.append(
                CornerObservation(
                    corners_uv=corners,
                    visible=visible,
                    confidence=confidence,
                    timestamp_s=timestamp_s,
                    source=(
                        "torchvision_keypoint_rcnn:"
                        f"instance={index}:score={instance_score:.4f}"
                        + (
                            f":gate_id={gate_id}"
                            if gate_id is not None
                            else ""
                        )
                    ),
                    gate_id=gate_id,
                    gate_id_confidence=gate_id_confidence,
                )
            )
        return observations

    @torch.inference_mode()
    def detect(
        self,
        rgb_image: np.ndarray,
        *,
        timestamp_s: float = 0.0,
    ) -> CornerObservation:
        observations = self.detect_all(
            rgb_image,
            timestamp_s=timestamp_s,
        )
        if not observations:
            return self._empty_observation(timestamp_s)
        return observations[0]
