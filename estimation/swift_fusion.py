"""Reference Swift-style gate/VIO fusion with camera-time alignment."""

from __future__ import annotations

from dataclasses import dataclass

from perception.corner_detection import CornerObservation
from perception.swift_gate_measurement import GatePoseMeasurement, GatePoseMeasurementBuilder

from .swift_vio_drift import FusedWorldEstimate, VioDriftKalmanFilter, VioWorldEstimate
from .vio_time_buffer import VioWorldEstimateBuffer


@dataclass(frozen=True)
class SwiftFusionResult:
    fused_state: FusedWorldEstimate
    gate_measurement: GatePoseMeasurement | None
    measurement_accepted: bool
    rejection_reason: str = ""
    innovation_mahalanobis2: tuple[float, ...] = ()


class SwiftPerceptionFusion:
    """Fuse mapped gate measurements with high-rate VIO safely.

    Gate observations are interpreted at their image timestamp, not at the
    current control timestamp. A short VIO history is used for interpolation.
    Measurements that arrive older than the already-advanced drift-filter state
    are rejected rather than being fused at the wrong time. This keeps the
    reference implementation causal; a future high-latency detector can add a
    fixed-lag out-of-sequence replay layer if needed.
    """

    def __init__(
        self,
        measurement_builder: GatePoseMeasurementBuilder,
        drift_filter: VioDriftKalmanFilter | None = None,
        *,
        vio_buffer: VioWorldEstimateBuffer | None = None,
    ) -> None:
        self.measurement_builder = measurement_builder
        self.drift_filter = drift_filter or VioDriftKalmanFilter()
        self.vio_buffer = vio_buffer or VioWorldEstimateBuffer(max_age_s=2.0)

    def reset(self, timestamp_s: float | None = None) -> None:
        self.drift_filter.reset(timestamp_s)
        self.vio_buffer.clear()

    def _prediction_only(self, vio: VioWorldEstimate, reason: str) -> SwiftFusionResult:
        fused = self.drift_filter.step(vio)
        return SwiftFusionResult(fused, None, False, reason)

    def step(
        self,
        vio: VioWorldEstimate,
        *,
        gate_observation: CornerObservation | None = None,
        gate_index: int | None = None,
    ) -> SwiftFusionResult:
        """Advance one VIO step and optionally apply a timestamp-aligned gate correction."""
        self.vio_buffer.push(vio)

        if gate_observation is None:
            return self._prediction_only(vio, "no gate observation")

        gate_timestamp_s = float(gate_observation.timestamp_s)
        if gate_timestamp_s > vio.timestamp_s + 1.0e-6:
            return self._prediction_only(
                vio,
                f"gate observation timestamp {gate_timestamp_s:.6f}s is in the future",
            )

        last_filter_time = self.drift_filter.last_timestamp_s
        if last_filter_time is not None and gate_timestamp_s < last_filter_time - 1.0e-6:
            return self._prediction_only(
                vio,
                "gate observation is older than the current drift-filter state; "
                "rejecting out-of-sequence update",
            )

        try:
            gate_vio = self.vio_buffer.interpolate(gate_timestamp_s)
        except ValueError as exc:
            return self._prediction_only(vio, str(exc))

        try:
            measurement = self.measurement_builder.build(
                gate_observation,
                gate_index=gate_index,
                # Association must use VIO at the camera timestamp, not the
                # latest control-cycle position.
                reference_position_w_b=gate_vio.position_w_b,
            )
        except (ValueError, RuntimeError) as exc:
            return self._prediction_only(vio, str(exc))

        accepted = self.drift_filter.update_from_gate_positions(
            gate_vio,
            gate_positions_w_b=(measurement.position_w_b,),
            position_covariances_w=(measurement.position_covariance_w,),
        )
        diagnostics = self.drift_filter.last_update_diagnostics

        # Bring the corrected drift state back to the current VIO/control time.
        if vio.timestamp_s > gate_vio.timestamp_s + 1.0e-9:
            self.drift_filter.predict(vio.timestamp_s)
        fused = self.drift_filter.corrected(vio)

        if accepted == 0:
            d2 = diagnostics.mahalanobis2[0] if diagnostics.mahalanobis2 else float("nan")
            return SwiftFusionResult(
                fused,
                measurement,
                False,
                f"gate/VIO innovation rejected by Mahalanobis gate (d2={d2:.3f})",
                diagnostics.mahalanobis2,
            )

        return SwiftFusionResult(
            fused,
            measurement,
            True,
            innovation_mahalanobis2=diagnostics.mahalanobis2,
        )
