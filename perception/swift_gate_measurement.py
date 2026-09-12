"""Swift-style gate-derived pose measurement with uncertainty propagation.

The detector/IPPE estimate is not treated as truth. Each nominal observation is
accompanied by perturbed corner observations; re-running IPPE propagates pixel
uncertainty into a world-position covariance for the VIO drift filter.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .camera_model import CameraCalibration
from .corner_detection import CornerObservation
from .gate_geometry import GateGeometry
from .planar_pnp import OpenCvPlanarPnP, PnPBackend, PnPResult
from .rigid_transform import RigidTransform
from .track_layout import TrackLayout


@dataclass(frozen=True)
class CornerPerturbationConfig:
    num_samples: int = 20
    corner_sigma_px: float = 2.0
    min_valid_samples: int = 10
    covariance_floor_m2: float = 1.0e-6
    min_corner_confidence: float = 0.0
    # A finite default prevents gross IPPE fits from reaching the drift filter.
    # The downstream Mahalanobis gate remains the final consistency check.
    max_nominal_reprojection_rmse_px: float = 5.0
    seed: int = 1

    def __post_init__(self) -> None:
        if self.num_samples < 2:
            raise ValueError("num_samples must be at least 2")
        if self.corner_sigma_px < 0.0:
            raise ValueError("corner_sigma_px must be non-negative")
        if not 2 <= self.min_valid_samples <= self.num_samples:
            raise ValueError("min_valid_samples must be in [2, num_samples]")
        if self.covariance_floor_m2 < 0.0:
            raise ValueError("covariance_floor_m2 must be non-negative")
        if not 0.0 <= self.min_corner_confidence <= 1.0:
            raise ValueError("min_corner_confidence must be in [0, 1]")
        if self.max_nominal_reprojection_rmse_px <= 0.0:
            raise ValueError("max_nominal_reprojection_rmse_px must be positive")


@dataclass(frozen=True)
class GatePoseMeasurement:
    gate_index: int
    observation: CornerObservation
    nominal_pnp: PnPResult
    T_cg: RigidTransform
    T_wc_gate: RigidTransform
    T_wb_gate: RigidTransform
    position_covariance_w: np.ndarray
    sampled_positions_w_b: np.ndarray
    valid_sample_count: int

    def __post_init__(self) -> None:
        covariance = np.asarray(self.position_covariance_w, dtype=np.float64).reshape(3, 3)
        samples = np.asarray(self.sampled_positions_w_b, dtype=np.float64).reshape(-1, 3)
        if not np.all(np.isfinite(covariance)) or not np.all(np.isfinite(samples)):
            raise ValueError("GatePoseMeasurement contains non-finite uncertainty data")
        covariance = 0.5 * (covariance + covariance.T)
        if np.linalg.eigvalsh(covariance)[0] < -1.0e-9:
            raise ValueError("Gate pose covariance must be positive semidefinite")
        object.__setattr__(self, "position_covariance_w", covariance)
        object.__setattr__(self, "sampled_positions_w_b", samples)

    @property
    def position_w_b(self) -> np.ndarray:
        return self.T_wb_gate.t


class GatePoseMeasurementBuilder:
    """Convert learned gate corners into a mapped body-position measurement."""

    def __init__(
        self,
        geometry: GateGeometry,
        camera: CameraCalibration,
        T_bc: RigidTransform,
        track_layout: TrackLayout,
        *,
        perturbation: CornerPerturbationConfig | None = None,
        pnp_backend: PnPBackend | None = None,
    ) -> None:
        self.geometry = geometry
        self.camera = camera
        self.T_bc = T_bc
        self.track_layout = track_layout
        self.perturbation = perturbation or CornerPerturbationConfig()
        self.pnp_backend = pnp_backend or OpenCvPlanarPnP()
        self._rng = np.random.default_rng(self.perturbation.seed)

    def _solve(self, corners_uv: np.ndarray) -> PnPResult:
        return self.pnp_backend.solve(
            self.geometry.object_points_g,
            corners_uv,
            self.camera.K,
            self.camera.dist_coeffs,
        )

    def _validate_observation(self, observation: CornerObservation) -> None:
        if not observation.complete:
            raise ValueError("Swift gate-pose update requires all four visible corners")
        if not np.all(np.isfinite(observation.confidence)):
            raise ValueError("Gate-corner confidence contains non-finite values")
        if np.any(observation.confidence < self.perturbation.min_corner_confidence):
            raise ValueError("Gate-corner confidence is below the configured threshold")

    def _pnp_is_usable(self, result: PnPResult) -> bool:
        return bool(
            result.success
            and result.T_cg is not None
            and np.isfinite(result.reprojection_rmse_px)
            and result.reprojection_rmse_px <= self.perturbation.max_nominal_reprojection_rmse_px
        )

    def build(
        self,
        observation: CornerObservation,
        *,
        gate_index: int | None = None,
        reference_position_w_b=None,
    ) -> GatePoseMeasurement:
        self._validate_observation(observation)
        nominal = self._solve(observation.corners_uv)
        if not nominal.success or nominal.T_cg is None:
            raise RuntimeError(nominal.message or "Nominal gate IPPE failed")
        if not np.isfinite(nominal.reprojection_rmse_px):
            raise RuntimeError("Nominal gate IPPE returned non-finite reprojection error")
        if nominal.reprojection_rmse_px > self.perturbation.max_nominal_reprojection_rmse_px:
            raise RuntimeError(
                "Nominal gate IPPE reprojection error exceeds configured limit: "
                f"{nominal.reprojection_rmse_px:.3f}px > "
                f"{self.perturbation.max_nominal_reprojection_rmse_px:.3f}px"
            )

        if gate_index is None:
            if reference_position_w_b is None:
                raise ValueError(
                    "reference_position_w_b is required when gate_index is not supplied"
                )
            association = self.track_layout.associate_to_nearest_gate(
                nominal.T_cg,
                self.T_bc,
                reference_position_w_b,
            )
            gate_index = association.gate_index

        T_wc = self.track_layout.camera_pose_from_gate(gate_index, nominal.T_cg)
        T_wb = self.track_layout.body_pose_from_gate(gate_index, nominal.T_cg, self.T_bc)

        sampled_positions: list[np.ndarray] = []
        sigma = self.perturbation.corner_sigma_px
        for _ in range(self.perturbation.num_samples):
            perturbation = self._rng.normal(0.0, sigma, size=(4, 2))
            sample_pnp = self._solve(observation.corners_uv + perturbation)
            if not self._pnp_is_usable(sample_pnp):
                continue
            sample_T_wb = self.track_layout.body_pose_from_gate(
                gate_index,
                sample_pnp.T_cg,
                self.T_bc,
            )
            sampled_positions.append(sample_T_wb.t)

        valid_count = len(sampled_positions)
        if valid_count < self.perturbation.min_valid_samples:
            raise RuntimeError(
                f"Only {valid_count}/{self.perturbation.num_samples} perturbed IPPE "
                "samples passed pose/reprojection validation"
            )

        samples = np.asarray(sampled_positions, dtype=np.float64)
        covariance = np.cov(samples, rowvar=False, ddof=1)
        covariance = np.asarray(covariance, dtype=np.float64).reshape(3, 3)
        covariance = 0.5 * (covariance + covariance.T)
        covariance += np.eye(3, dtype=np.float64) * self.perturbation.covariance_floor_m2

        return GatePoseMeasurement(
            gate_index=gate_index,
            observation=observation,
            nominal_pnp=nominal,
            T_cg=nominal.T_cg,
            T_wc_gate=T_wc,
            T_wb_gate=T_wb,
            position_covariance_w=covariance,
            sampled_positions_w_b=samples,
            valid_sample_count=valid_count,
        )
