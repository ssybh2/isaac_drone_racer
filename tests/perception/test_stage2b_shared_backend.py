import numpy as np
import pytest

pytest.importorskip("cv2")

from perception.camera_model import CameraCalibration
from perception.corner_detection import CornerObservation
from perception.gate_geometry import GateGeometry
from perception.pose_recovery import GatePoseRecovery
from perception.rigid_transform import RigidTransform
from perception.stage2a_pipeline import Stage2APerceptionPipeline, Stage2ATruth
from perception.stage2b_pipeline import Stage2BPerceptionPipeline


class EchoOracleDetector:
    def __init__(self, observation):
        self.observation = observation

    def detect(self, rgb_image, *, timestamp_s=0.0):
        return CornerObservation(
            self.observation.corners_uv,
            visible=self.observation.visible,
            confidence=np.ones(4),
            timestamp_s=timestamp_s,
            source="echo_test_detector",
        )


def test_stage2b_detector_output_uses_same_pose_backend_as_stage2a():
    geometry = GateGeometry.rectangular_x_normal(1.5, 1.5, source="synthetic_test")
    camera = CameraCalibration(
        K=np.array([[600.0, 0.0, 320.0], [0.0, 600.0, 240.0], [0.0, 0.0, 1.0]]),
        image_width=640,
        image_height=480,
    )
    R_bc = np.array([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])
    T_bc = RigidTransform(R_bc, np.array([0.1, 0.0, 0.0]), to_frame="B", from_frame="C")
    T_wb = RigidTransform(np.eye(3), np.zeros(3), to_frame="W", from_frame="B")
    T_wc = T_wb @ T_bc
    T_wg = RigidTransform(np.eye(3), np.array([5.0, 0.0, 0.0]), to_frame="W", from_frame="G")

    stage2a = Stage2APerceptionPipeline(geometry, camera, T_bc)
    reference = stage2a.process_truth(Stage2ATruth(T_wg=T_wg, T_wc=T_wc, T_wb=T_wb))

    stage2b = Stage2BPerceptionPipeline(
        EchoOracleDetector(reference.oracle_corners),
        GatePoseRecovery(geometry, camera, T_bc),
    )
    result = stage2b.process(
        np.zeros((480, 640, 3), dtype=np.uint8),
        T_wg=T_wg,
        oracle_corners=reference.oracle_corners,
        T_wb_truth=T_wb,
    )

    assert result.metrics.corner_rmse_px == pytest.approx(0.0)
    assert result.metrics.body_translation_error_m < 1e-5
    assert result.metrics.body_rotation_error_rad < 1e-5
    assert np.allclose(result.solution.target_pos_b, reference.solution.target_pos_b, atol=1e-5)
