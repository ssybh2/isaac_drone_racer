"""Isaac-side Stage2B dataset collection built on Stage2A oracle projection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from .dataset import Stage2DatasetWriter
from .gate_geometry import GateGeometry
from .isaac_adapter import IsaacStage2TruthAdapter
from .perfect_gate_corner_sensor import PerfectGateCornerSensor


@dataclass
class IsaacStage2DatasetCollector:
    """Export RGB and exact projected gate corners for selected environments.

    Dataset labeling intentionally does not require PnP or a camera/body
    extrinsic. That keeps detector supervision independent from pose-recovery
    calibration. Geometric in-frame visibility is recorded; later dataset
    versions may additionally mark true occlusion using depth/segmentation.
    """

    env: object
    geometry: GateGeometry
    writer: Stage2DatasetWriter
    robot_name: str = "robot"
    track_name: str = "track"
    camera_name: str = "tiled_camera"
    command_name: str = "target"
    image_transform: Callable[[np.ndarray], tuple[np.ndarray, dict]] | None = None

    def __post_init__(self) -> None:
        self.adapter = IsaacStage2TruthAdapter(
            self.env,
            robot_name=self.robot_name,
            track_name=self.track_name,
            camera_name=self.camera_name,
            command_name=self.command_name,
        )

    def capture(self, sample_id: str, env_id: int = 0, *, extra: dict | None = None):
        snapshot = self.adapter.snapshot(env_id)
        T_cg_truth = snapshot.truth.T_wc.inverse() @ snapshot.truth.T_wg
        sensor = PerfectGateCornerSensor(self.geometry, snapshot.camera)
        oracle_corners = sensor.measure(T_cg_truth, timestamp_s=snapshot.truth.timestamp_s)
        rgb = self.adapter.rgb(env_id)
        sample_extra = dict(extra or {})
        if self.image_transform is not None:
            rgb, transform_metadata = self.image_transform(rgb)
            sample_extra["image_randomization"] = transform_metadata
        return self.writer.write_sample(
            sample_id,
            rgb,
            oracle_corners,
            snapshot.camera,
            gate_index=snapshot.gate_index,
            T_wg=snapshot.truth.T_wg,
            T_wc=snapshot.truth.T_wc,
            T_wb=snapshot.truth.T_wb,
            extra=sample_extra,
        )

    def capture_batch(self, sample_prefix: str, env_ids, *, extras=None):
        outputs = []
        env_ids = list(env_ids)
        if extras is not None and len(extras) != len(env_ids):
            raise ValueError("extras must have one entry per environment")
        for index, env_id in enumerate(env_ids):
            extra = None if extras is None else extras[index]
            outputs.append(
                self.capture(f"{sample_prefix}_env{int(env_id):04d}", int(env_id), extra=extra)
            )
        return outputs
