import numpy as np
import pytest

pytest.importorskip("cv2")

from perception.camera_model import CameraCalibration
from perception.gate_geometry import GateGeometry
from perception.rigid_transform import RigidTransform, rotation_error_rad, translation_error_m
from perception.stage2a_pipeline import Stage2APerceptionPipeline, Stage2ATruth


def _camera_to_body() -> RigidTransform:
    # Optical camera: x right, y down, z forward.
    # Body: x forward, y left, z up.
    R_bc = np.array(
        [
            [0.0, 0.0, 1.0],
            [-1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
        ]
    )
    return RigidTransform(R_bc, np.array([0.14, 0.0, 0.05]), to_frame="B", from_frame="C")


def _calibration() -> CameraCalibration:
    return CameraCalibration(
        K=np.array([[620.0, 0.0, 500.0], [0.0, 620.0, 500.0], [0.0, 0.0, 1.0]]),
        image_width=1000,
        image_height=1000,
    )


def test_perfect_projection_pnp_closes_body_camera_and_gate_pose():
    geometry = GateGeometry.rectangular_x_normal(1.5, 1.5, source="synthetic_test")
    T_bc = _camera_to_body()

    T_wb = RigidTransform.identity("W")
    # Relabel identity as world <- body.
    T_wb = RigidTransform(np.eye(3), np.zeros(3), to_frame="W", from_frame="B")
    T_wc = T_wb @ T_bc
    T_wg = RigidTransform(np.eye(3), np.array([4.0, 0.2, 1.0]), to_frame="W", from_frame="G")

    pipeline = Stage2APerceptionPipeline(geometry, _calibration(), T_bc)
    result = pipeline.process_truth(Stage2ATruth(T_wg=T_wg, T_wc=T_wc, T_wb=T_wb))

    assert result.oracle_corners.complete
    assert result.metrics.corner_reprojection_rmse_px < 1e-5
    assert result.metrics.gate_translation_error_m < 1e-5
    assert result.metrics.gate_rotation_error_rad < 1e-5
    assert result.metrics.camera_translation_error_m < 1e-5
    assert result.metrics.camera_rotation_error_rad < 1e-5
    assert result.metrics.body_translation_error_m < 1e-5
    assert result.metrics.body_rotation_error_rad < 1e-5
    assert result.metrics.extrinsic_translation_error_m < 1e-9
    assert result.metrics.extrinsic_rotation_error_rad < 1e-9

    assert translation_error_m(result.solution.T_wb_est, T_wb) < 1e-5
    assert rotation_error_rad(result.solution.T_wb_est, T_wb) < 1e-5


def test_wrong_camera_mount_is_visible_in_extrinsic_metric():
    geometry = GateGeometry.rectangular_x_normal(1.5, 1.5, source="synthetic_test")
    T_bc_truth = _camera_to_body()
    T_bc_wrong = RigidTransform(
        T_bc_truth.R,
        T_bc_truth.t + np.array([0.02, 0.0, 0.0]),
        to_frame="B",
        from_frame="C",
    )
    T_wb = RigidTransform(np.eye(3), np.zeros(3), to_frame="W", from_frame="B")
    T_wc = T_wb @ T_bc_truth
    T_wg = RigidTransform(np.eye(3), np.array([4.0, 0.0, 1.0]), to_frame="W", from_frame="G")

    pipeline = Stage2APerceptionPipeline(geometry, _calibration(), T_bc_wrong)
    result = pipeline.process_truth(Stage2ATruth(T_wg=T_wg, T_wc=T_wc, T_wb=T_wb))

    assert result.metrics.gate_translation_error_m < 1e-5
    assert result.metrics.extrinsic_translation_error_m == pytest.approx(0.02, abs=1e-8)
    assert result.metrics.body_translation_error_m == pytest.approx(0.02, abs=1e-5)
