"""Swift-style gate-derived pose measurement with uncertainty propagation.

The detector/IPPE estimate is intentionally *not* treated as truth.  Following
Kaufmann et al. (Nature 2023), each nominal gate observation is accompanied by
20 perturbed corner observations.  Re-running IPPE on those perturbations
propagates image-plane corner uncertainty into a world-position measurement
covariance suitable for the VIO drift Kalman filter.
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
    """Configuration for the 20-sample Swift uncertainty propagation."""

    num_samples: int = 20
    corner_sigma_px: float = 2.0
    min_valid_samples: int = 10
    covariance_floor_m2: float = 1.0e-6
    min_corner_confidence: float = 0.0
    max_nominal_reprojection_rmse_px: float = float("inf")
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


@dataclass(frozen=True)
class GatePoseMeasurement:
    """One mapped gate observation expressed as a noisy body world pose."""

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
        object.__setattr__(self, "position_covariance_w", covariance)
        object.__setattr__(self, "sampled_positions_w_b", samples)

    @property
    def position_w_b(self) -> np.ndarray:
        return self.T_wb_gate.t


class GatePoseMeasurementBuilder:
    """Convert learned 2D gate corners into a mapped pose measurement.

    The nominal four corners produce ``T_cg`` through the existing Stage2 IPPE
    backend.  The known track map supplies ``T_wg``.  The camera/body world pose
    follows directly:

        T_wc = T_wg @ inverse(T_cg)
        T_wb = T_wc @ inverse(T_bc)

    The same transform is recomputed for 20 perturbed corner sets; the sample
    covariance of the resulting body positions is the measurement covariance R
    used by the Swift translational drift filter.
    """

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
        if np.any(observation.confidence < self.perturbation.min_corner_confidence):
            raise ValueError("Gate-corner confidence is below the configured threshold")

    def build(
        self,
        observation: CornerObservation,
        *,
        gate_index: int | None = None,
        reference_position_w_b=None,
    ) -> GatePoseMeasurement:
        """Build one mapped pose measurement and its sampled covariance.

        ``gate_index`` should be supplied when the active gate identity is known
        from the task.  If it is omitted, ``reference_position_w_b`` is required
        and the detection is associated to the closest mapped gate using the
        current VIO position, matching the Swift track-layout logic.
        """

        self._validate_observation(observation)
        nominal = self._solve(observation.corners_uv)
        if not nominal.success or nominal.T_cg is None:
            raise RuntimeError(nominal.message or "Nominal gate IPPE failed")
        if nominal.reprojection_rmse_px > self.perturbation.max_nominal_reprojection_rmse_px:
            raise RuntimeError(
                "Nominal gate IPPE reprojection error exceeds configured limit: "
                f"{nominal.reprojection_rmse_px:.3f}px"
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
            perturbed = observation.corners_uv + perturbation
            sample_pnp = self._solve(perturbed)
            if not sample_pnp.success or sample_pnp.T_cg is None:
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
                "samples were valid"
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
