"""Shared Stage 2 perception architecture.

Stage2A and Stage2B intentionally differ only in the source of the four image
corners. Projection, PnP, frame transforms and diagnostics are shared.
"""

from .camera_model import CameraCalibration
from .corner_detection import CornerObservation, GateCornerDetector
from .dataset import Stage2DatasetWriter
from .dataset_collector import IsaacStage2DatasetCollector
from .gate_geometry import GateGeometry
from .keypoint_detector import GateKeypointNet, Stage2KeypointDataset, TorchGateCornerDetector
from .planar_pnp import OpenCvPlanarPnP, PnPBackend, PnPResult
from .pose_recovery import GatePoseRecovery, GatePoseSolution
from .rigid_transform import RigidTransform
from .stage2_calibration import load_stage2_gate_geometry, stage2_camera_to_body
from .stage2a_pipeline import Stage2APerceptionPipeline, Stage2AResult, Stage2ATruth
from .stage2b_pipeline import Stage2BPerceptionPipeline, Stage2BResult
from .swift_gate_measurement import (
    CornerPerturbationConfig,
    GatePoseMeasurement,
    GatePoseMeasurementBuilder,
)
from .swift_isaac_adapter import active_gate_index_from_isaac, track_layout_from_isaac
from .track_layout import GateAssociation, TrackLayout

__all__ = [
    "CameraCalibration",
    "CornerObservation",
    "CornerPerturbationConfig",
    "GateAssociation",
    "GateCornerDetector",
    "GateGeometry",
    "GateKeypointNet",
    "GatePoseMeasurement",
    "GatePoseMeasurementBuilder",
    "GatePoseRecovery",
    "GatePoseSolution",
    "IsaacStage2DatasetCollector",
    "OpenCvPlanarPnP",
    "PnPBackend",
    "PnPResult",
    "RigidTransform",
    "Stage2APerceptionPipeline",
    "Stage2AResult",
    "Stage2ATruth",
    "Stage2BPerceptionPipeline",
    "Stage2BResult",
    "Stage2DatasetWriter",
    "Stage2KeypointDataset",
    "TorchGateCornerDetector",
    "TrackLayout",
    "active_gate_index_from_isaac",
    "load_stage2_gate_geometry",
    "stage2_camera_to_body",
    "track_layout_from_isaac",
]
