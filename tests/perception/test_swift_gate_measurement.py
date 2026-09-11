import numpy as np
import pytest

pytest.importorskip("cv2")

from perception.camera_model import CameraCalibration, project_visible_with_calibration
from perception.corner_detection import CornerObservation
from perception.gate_geometry import GateGeometry
from perception.rigid_transform import RigidTransform, translation_error_m
from perception.stage2_calibration import stage2_camera_to_body
from perception.swift_gate_measurement import (
    CornerPerturbationConfig,
    GatePoseMeasurementBuilder,
)
from perception.track_layout import TrackLayout


def _camera() -> CameraCalibration:
    return CameraCalibration(
        K=np.array([[293.0, 0.0, 128.0], [0.0, 293.0, 128.0], [0.0, 0.0, 1.0]]),
        image_width=256,
        image_height=256,
    )


def _geometry() -> GateGeometry:
    return GateGeometry.rectangular_x_normal(1.524, 1.524, source="swift_test")


def _truth(distance_m: float):
    T_bc = stage2_camera_to_body()
    T_wb = RigidTransform(np.eye(3), np.zeros(3), to_frame="W", from_frame="B")
    T_wc = T_wb @ T_bc
    T_wg = RigidTransform(
        np.eye(3),
        np.array([distance_m, 0.0, 0.05]),
        to_frame="W",
        from_frame="G",
    )
    T_cg = T_wc.inverse() @ T_wg
    points_c = T_cg.transform_points(_geometry().object_points_g)
    corners, visible = project_visible_with_calibration(points_c, _camera())
    assert np.all(visible)
    return T_bc, T_wb, T_wg, CornerObservation(corners, visible=visible, source="oracle")


def test_known_twg_and_nominal_tcg_recover_world_body_pose():
    T_bc, T_wb, T_wg, observation = _truth(4.0)
    builder = GatePoseMeasurementBuilder(
        _geometry(),
        _camera(),
        T_bc,
        TrackLayout((T_wg,)),
        perturbation=CornerPerturbationConfig(
            num_samples=20,
            corner_sigma_px=0.0,
            min_valid_samples=20,
            seed=7,
        ),
    )

    measurement = builder.build(observation, gate_index=0)

    assert translation_error_m(measurement.T_wb_gate, T_wb) < 1.0e-5
    assert measurement.valid_sample_count == 20
    assert np.trace(measurement.position_covariance_w) < 1.0e-4


def test_same_pixel_uncertainty_produces_larger_world_covariance_far_away():
    def covariance_trace(distance_m: float) -> float:
        T_bc, _, T_wg, observation = _truth(distance_m)
        builder = GatePoseMeasurementBuilder(
            _geometry(),
            _camera(),
            T_bc,
            TrackLayout((T_wg,)),
            perturbation=CornerPerturbationConfig(
                num_samples=200,
                corner_sigma_px=2.0,
                min_valid_samples=150,
                seed=11,
            ),
        )
        return float(np.trace(builder.build(observation, gate_index=0).position_covariance_w))

    near = covariance_trace(3.0)
    far = covariance_trace(8.0)

    assert far > near
    assert far > 2.0 * near


def test_unlabeled_gate_can_be_associated_to_nearest_mapped_pose_from_vio():
    T_bc, T_wb, T_wg_true, observation = _truth(4.0)
    wrong_gate = RigidTransform(
        np.eye(3),
        np.array([20.0, 12.0, 0.05]),
        to_frame="W",
        from_frame="G",
    )
    layout = TrackLayout((wrong_gate, T_wg_true))
    builder = GatePoseMeasurementBuilder(
        _geometry(),
        _camera(),
        T_bc,
        layout,
        perturbation=CornerPerturbationConfig(
            num_samples=20,
            corner_sigma_px=0.0,
            min_valid_samples=20,
            seed=3,
        ),
    )

    measurement = builder.build(
        observation,
        gate_index=None,
        reference_position_w_b=T_wb.t,
    )

    assert measurement.gate_index == 1
