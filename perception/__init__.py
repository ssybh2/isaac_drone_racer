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

__all__ = [
    "CameraCalibration",
    "CornerObservation",
    "GateCornerDetector",
    "GateGeometry",
    "GateKeypointNet",
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
    "load_stage2_gate_geometry",
    "stage2_camera_to_body",
]
