"""Reference Swift-style Stage2 perception/VIO fusion runtime.

This module connects the repository's Stage2 detector/IPPE path to the
translational VIO-drift filter.  It deliberately remains single-vehicle and
NumPy/OpenCV based so correctness can be validated before a batched RL backend
is introduced.
"""

from __future__ import annotations

from dataclasses import dataclass

from perception.corner_detection import CornerObservation
from perception.swift_gate_measurement import GatePoseMeasurement, GatePoseMeasurementBuilder

from .swift_vio_drift import FusedWorldEstimate, VioDriftKalmanFilter, VioWorldEstimate


@dataclass(frozen=True)
class SwiftFusionResult:
    fused_state: FusedWorldEstimate
    gate_measurement: GatePoseMeasurement | None
    measurement_accepted: bool
    rejection_reason: str = ""


class SwiftPerceptionFusion:
    """Fuse mapped gate measurements with the high-rate VIO state."""

    def __init__(
        self,
        measurement_builder: GatePoseMeasurementBuilder,
        drift_filter: VioDriftKalmanFilter | None = None,
    ) -> None:
        self.measurement_builder = measurement_builder
        self.drift_filter = drift_filter or VioDriftKalmanFilter()

    def reset(self, timestamp_s: float | None = None) -> None:
        self.drift_filter.reset(timestamp_s)

    def step(
        self,
        vio: VioWorldEstimate,
        *,
        gate_observation: CornerObservation | None = None,
        gate_index: int | None = None,
    ) -> SwiftFusionResult:
        """Advance one VIO step and optionally apply a gate correction.

        Missing, partial, low-confidence, or failed IPPE measurements do not
        replace the VIO state.  They simply result in a prediction-only Kalman
        step.  This is intentionally different from feeding raw ``T_cg`` to the
        control policy.
        """

        if gate_observation is None:
            fused = self.drift_filter.step(vio)
            return SwiftFusionResult(fused, None, False, "no gate observation")

        try:
            measurement = self.measurement_builder.build(
                gate_observation,
                gate_index=gate_index,
                reference_position_w_b=vio.position_w_b,
            )
        except (ValueError, RuntimeError) as exc:
            fused = self.drift_filter.step(vio)
            return SwiftFusionResult(fused, None, False, str(exc))

        fused = self.drift_filter.step(
            vio,
            gate_positions_w_b=(measurement.position_w_b,),
            position_covariances_w=(measurement.position_covariance_w,),
        )
        return SwiftFusionResult(fused, measurement, True)
