"""Orchestration for learned relative-motion correction of raw VIO."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .learned_motion import DisplacementPrediction, DisplacementPredictor, LearnedMotionBuffer
from .learned_vio_drift import LearnedDisplacementMeasurement, LearnedVioDriftFilter
from .swift_vio_drift import VioWorldEstimate
from .vio_time_buffer import VioWorldEstimateBuffer


def body_motion_to_world(vio: VioWorldEstimate, *, gyro_b, thrust_b) -> tuple[np.ndarray, np.ndarray]:
    """Rotate body-frame gyro/thrust vectors into the world frame of ``vio``."""
    w, x, y, z = np.asarray(vio.orientation_w_b_wxyz, dtype=np.float64).reshape(4)
    R_wb = np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )
    gyro = np.asarray(gyro_b, dtype=np.float64).reshape(3)
    thrust = np.asarray(thrust_b, dtype=np.float64).reshape(3)
    if not np.all(np.isfinite(gyro)) or not np.all(np.isfinite(thrust)):
        raise ValueError("body motion vectors must be finite")
    return R_wb @ gyro, R_wb @ thrust


@dataclass(frozen=True)
class HybridVioCorrectionResult:
    raw: VioWorldEstimate
    corrected: VioWorldEstimate
    learned_update_attempted: bool = False
    learned_update_accepted: bool = False
    learned_update_rejected: bool = False
    skipped_windows: int = 0
    mahalanobis2: float | None = None
    prediction: DisplacementPrediction | None = None


class HybridLearnedVioCorrector:
    """Fuse learned displacement constraints into a translational VIO drift EKF.

    Motion samples are buffered independently from VIO samples. At each exact
    non-overlapping learned window boundary, raw VIO is interpolated to the
    boundary, the predictor produces physical displacement over the same
    interval, and the drift filter receives the relative drift measurement.
    """

    def __init__(
        self,
        predictor: DisplacementPredictor,
        *,
        window_time_s: float = 0.5,
        sample_rate_hz: float = 100.0,
        drift_filter: LearnedVioDriftFilter | None = None,
        sigma_position: float = 0.05,
        sigma_velocity: float = 0.1,
        innovation_gate_chi2: float | None = 16.26623619623813,
    ) -> None:
        self.predictor = predictor
        self.motion_buffer = LearnedMotionBuffer(
            window_time_s=window_time_s,
            sample_rate_hz=sample_rate_hz,
        )
        self.window_time_s = float(window_time_s)
        self.sample_rate_hz = float(sample_rate_hz)
        self.drift_filter = drift_filter or LearnedVioDriftFilter(
            sigma_position=sigma_position,
            sigma_velocity=sigma_velocity,
            innovation_gate_chi2=innovation_gate_chi2,
            nominal_rate_hz=sample_rate_hz,
        )
        self.vio_buffer = VioWorldEstimateBuffer(
            max_age_s=max(2.0, 4.0 * self.window_time_s),
            max_samples=max(512, int(np.ceil(8.0 * self.window_time_s * self.sample_rate_hz))),
        )
        self.window_start_timestamp_s: float | None = None
        self.last_result: HybridVioCorrectionResult | None = None

    def reset(self) -> None:
        self.motion_buffer.reset()
        self.vio_buffer.clear()
        self.drift_filter.reset()
        self.window_start_timestamp_s = None
        self.last_result = None

    def ingest_motion_sample(self, timestamp_s: float, *, gyro_w, thrust_w) -> None:
        self.motion_buffer.append(timestamp_s, gyro_w=gyro_w, thrust_w=thrust_w)

    def _initialize_from_vio(self, raw: VioWorldEstimate) -> HybridVioCorrectionResult:
        self.window_start_timestamp_s = float(raw.timestamp_s)
        self.drift_filter.reset(raw.timestamp_s, anchor_vio_position_w=raw.position_w_b)
        corrected = self.drift_filter.step(raw)
        result = HybridVioCorrectionResult(raw=raw, corrected=corrected)
        self.last_result = result
        return result

    def _advance_skipped_window(self, endpoint: VioWorldEstimate) -> None:
        # Progress prediction to the skipped boundary but deliberately inject no
        # learned measurement. Re-anchor so a single missing window is not
        # replayed indefinitely on every later VIO sample.
        self.drift_filter.step(endpoint)
        self.drift_filter.reanchor(endpoint)
        self.window_start_timestamp_s = float(endpoint.timestamp_s)
        self.motion_buffer.discard_before(endpoint.timestamp_s)

    def step(self, raw: VioWorldEstimate) -> HybridVioCorrectionResult:
        self.vio_buffer.push(raw)
        if self.window_start_timestamp_s is None:
            return self._initialize_from_vio(raw)

        attempted = False
        accepted = False
        rejected = False
        skipped = 0
        last_prediction: DisplacementPrediction | None = None
        last_d2: float | None = None
        epsilon = 1.0e-9

        while raw.timestamp_s + epsilon >= self.window_start_timestamp_s + self.window_time_s:
            start = float(self.window_start_timestamp_s)
            end = start + self.window_time_s
            try:
                endpoint = self.vio_buffer.interpolate(end)
            except ValueError:
                # We cannot safely create a measurement without VIO at the exact
                # learned-window boundary. Leave the filter at its current time;
                # the retained history may bracket the boundary on a later call.
                break

            try:
                window = self.motion_buffer.window(start, end)
            except ValueError:
                skipped += 1
                self._advance_skipped_window(endpoint)
                continue

            prediction = self.predictor.predict(window)
            last_prediction = prediction
            measurement = LearnedDisplacementMeasurement(
                displacement_w=prediction.displacement_w,
                covariance_w=prediction.covariance_w,
                start_timestamp_s=start,
                end_timestamp_s=end,
            )
            attempted = True
            self.drift_filter.step(endpoint, measurement=measurement)
            diagnostics = self.drift_filter.last_update_diagnostics
            last_d2 = diagnostics.mahalanobis2
            if diagnostics.accepted:
                accepted = True
                self.window_start_timestamp_s = end
            else:
                rejected = True
                # The drift state is still valid; only this network constraint
                # was rejected. Re-anchor without applying the bad measurement.
                self.drift_filter.reanchor(endpoint)
                self.window_start_timestamp_s = end
            self.motion_buffer.discard_before(end)

        # Learned updates are processed at exact historical boundaries before
        # propagating the drift state to the newest raw OpenVINS timestamp.
        corrected = self.drift_filter.step(raw)
        result = HybridVioCorrectionResult(
            raw=raw,
            corrected=corrected,
            learned_update_attempted=attempted,
            learned_update_accepted=accepted,
            learned_update_rejected=rejected,
            skipped_windows=skipped,
            mahalanobis2=last_d2,
            prediction=last_prediction,
        )
        self.last_result = result
        return result
