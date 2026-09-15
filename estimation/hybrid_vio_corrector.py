"""Orchestration for learned relative-motion correction of raw VIO."""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from .learned_motion import DisplacementPrediction, DisplacementPredictor, LearnedMotionBuffer, MotionWindow
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
    prediction_start_timestamp_s: float | None = None
    prediction_end_timestamp_s: float | None = None
    raw_window_displacement_w: np.ndarray | None = None
    learned_velocity_update_applied: bool = False
    learned_velocity_drift_measurement_w: np.ndarray | None = None
    learned_velocity_drift_measurement_covariance_w: np.ndarray | None = None
    learned_velocity_mahalanobis2: float | None = None
    learned_position_injection_norm_m: float = 0.0
    learned_position_release_norm_m: float = 0.0
    learned_position_pending_norm_m: float = 0.0
    absolute_position_update_applied: bool = False
    absolute_position_mahalanobis2: float | None = None
    learned_velocity_drift_before_w: np.ndarray | None = None
    learned_velocity_drift_after_w: np.ndarray | None = None
    absolute_velocity_drift_before_w: np.ndarray | None = None
    absolute_velocity_drift_after_w: np.ndarray | None = None
    raw_vio_jump_detected: bool = False
    raw_vio_jump_residual_w: np.ndarray | None = None
    raw_vio_jump_residual_m: float | None = None
    raw_vio_jump_compensation_w: np.ndarray | None = None


class HybridLearnedVioCorrector:
    """Fuse learned displacement constraints into a translational VIO drift EKF.

    Motion samples are buffered independently from VIO samples. At each exact
    non-overlapping learned window boundary, raw VIO is interpolated to the
    boundary, the predictor produces physical displacement over the same
    interval, and the drift filter receives the relative drift measurement.
    Sparse absolute position anchors can be fused separately without disturbing
    the learned-window timing contract.

    Optional raw-VIO jump isolation detects implausible sample-to-sample position
    frame shifts relative to the velocity-predicted displacement and removes the
    accumulated frame shift from the internal VIO stream before it reaches the
    drift filter. The original raw VIO is retained unchanged for diagnostics.

    In position-only learned-update mode, an optional explicit drift-velocity
    measurement can be formed from each accepted learned window as
    ``(dp_vio_isolated - dp_learned) / dt``. It updates only ``v_d_current``;
    subsequent normal propagation then carries the learned correction smoothly
    between learned-window boundaries without restoring cross-covariance-driven
    velocity jumps.

    A separate optional position-residual slew defers the common-mode position
    correction created by an accepted learned window and releases it at a
    bounded rate on subsequent VIO samples. The same common-mode shift is
    applied to anchor and current position drift, preserving the learned
    relative-position state exactly while avoiding a one-frame output snap.
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
        relative_position_only: bool = False,
        learned_drift_velocity_from_displacement: bool = False,
        learned_drift_velocity_sigma_floor_mps: float = 0.5,
        learned_position_residual_slew: bool = False,
        learned_position_residual_max_rate_mps: float = 4.0,
        raw_vio_jump_isolation: bool = False,
        raw_vio_jump_threshold_m: float = 0.5,
    ) -> None:
        if raw_vio_jump_threshold_m <= 0.0 or not np.isfinite(raw_vio_jump_threshold_m):
            raise ValueError("raw_vio_jump_threshold_m must be positive and finite")
        if (
            learned_drift_velocity_sigma_floor_mps <= 0.0
            or not np.isfinite(learned_drift_velocity_sigma_floor_mps)
        ):
            raise ValueError("learned_drift_velocity_sigma_floor_mps must be positive and finite")
        if (
            learned_position_residual_max_rate_mps <= 0.0
            or not np.isfinite(learned_position_residual_max_rate_mps)
        ):
            raise ValueError("learned_position_residual_max_rate_mps must be positive and finite")
        if learned_drift_velocity_from_displacement and not relative_position_only:
            raise ValueError(
                "learned_drift_velocity_from_displacement requires relative_position_only"
            )
        if learned_position_residual_slew and not relative_position_only:
            raise ValueError("learned_position_residual_slew requires relative_position_only")
        self.predictor = predictor
        self.motion_buffer = LearnedMotionBuffer(
            window_time_s=window_time_s,
            sample_rate_hz=sample_rate_hz,
        )
        self.body_motion_buffer = LearnedMotionBuffer(
            window_time_s=window_time_s,
            sample_rate_hz=sample_rate_hz,
        )
        self.window_time_s = float(window_time_s)
        self.sample_rate_hz = float(sample_rate_hz)
        self.relative_position_only = bool(relative_position_only)
        self.learned_drift_velocity_from_displacement = bool(
            learned_drift_velocity_from_displacement
        )
        self.learned_drift_velocity_sigma_floor_mps = float(
            learned_drift_velocity_sigma_floor_mps
        )
        self.learned_position_residual_slew = bool(learned_position_residual_slew)
        self.learned_position_residual_max_rate_mps = float(
            learned_position_residual_max_rate_mps
        )
        self.raw_vio_jump_isolation = bool(raw_vio_jump_isolation)
        self.raw_vio_jump_threshold_m = float(raw_vio_jump_threshold_m)
        self.drift_filter = drift_filter or LearnedVioDriftFilter(
            sigma_position=sigma_position,
            sigma_velocity=sigma_velocity,
            innovation_gate_chi2=innovation_gate_chi2,
            nominal_rate_hz=sample_rate_hz,
            relative_position_only=self.relative_position_only,
        )
        buffer_age_s = max(2.0, 4.0 * self.window_time_s)
        buffer_samples = max(512, int(np.ceil(8.0 * self.window_time_s * self.sample_rate_hz)))
        self.vio_buffer = VioWorldEstimateBuffer(
            max_age_s=buffer_age_s,
            max_samples=buffer_samples,
        )
        self.raw_vio_buffer = VioWorldEstimateBuffer(
            max_age_s=buffer_age_s,
            max_samples=buffer_samples,
        )
        self.window_start_timestamp_s: float | None = None
        self.last_result: HybridVioCorrectionResult | None = None
        self._previous_raw_vio: VioWorldEstimate | None = None
        self._last_isolated_vio: VioWorldEstimate | None = None
        self._last_position_slew_timestamp_s: float | None = None
        self.pending_position_drift_w = np.zeros(3, dtype=np.float64)
        self.raw_vio_jump_compensation_w = np.zeros(3, dtype=np.float64)
        self.raw_vio_jump_count = 0
        self.raw_vio_jump_max_residual_m = 0.0
        self.raw_vio_jump_total_compensation_m = 0.0

    def reset(self) -> None:
        self.motion_buffer.reset()
        self.body_motion_buffer.reset()
        self.vio_buffer.clear()
        self.raw_vio_buffer.clear()
        self.drift_filter.reset()
        self.window_start_timestamp_s = None
        self.last_result = None
        self._previous_raw_vio = None
        self._last_isolated_vio = None
        self._last_position_slew_timestamp_s = None
        self.pending_position_drift_w.fill(0.0)
        self.raw_vio_jump_compensation_w.fill(0.0)
        self.raw_vio_jump_count = 0
        self.raw_vio_jump_max_residual_m = 0.0
        self.raw_vio_jump_total_compensation_m = 0.0

    def ingest_motion_sample(self, timestamp_s: float, *, gyro_w, thrust_w) -> None:
        """Append an already world-frame motion sample."""
        self.motion_buffer.append(timestamp_s, gyro_w=gyro_w, thrust_w=thrust_w)

    def ingest_body_motion_sample(self, timestamp_s: float, *, gyro_b, thrust_b) -> None:
        """Append a timestamped body-frame sample for later VIO-attitude rotation."""
        self.body_motion_buffer.append(timestamp_s, gyro_w=gyro_b, thrust_w=thrust_b)

    def _prediction_window(self, start: float, end: float) -> MotionWindow:
        if len(self.body_motion_buffer) > 0:
            body = self.body_motion_buffer.window(start, end)
            world_features = np.empty_like(body.features)
            for index, timestamp in enumerate(body.timestamps_s):
                vio_at_sample = self.vio_buffer.interpolate(float(timestamp))
                gyro_w, thrust_w = body_motion_to_world(
                    vio_at_sample,
                    gyro_b=body.features[0:3, index],
                    thrust_b=body.features[3:6, index],
                )
                world_features[0:3, index] = gyro_w
                world_features[3:6, index] = thrust_w
            return MotionWindow(
                features=world_features,
                timestamps_s=body.timestamps_s,
                start_timestamp_s=body.start_timestamp_s,
                end_timestamp_s=body.end_timestamp_s,
            )
        return self.motion_buffer.window(start, end)

    def _velocity_measurement_covariance(
        self,
        displacement_covariance_w: np.ndarray,
        dt_s: float,
    ) -> np.ndarray:
        covariance = np.asarray(displacement_covariance_w, dtype=np.float64) / (dt_s * dt_s)
        covariance = 0.5 * (covariance + covariance.T)
        floor_variance = self.learned_drift_velocity_sigma_floor_mps**2
        min_eig = float(np.linalg.eigvalsh(covariance)[0])
        if min_eig < floor_variance:
            covariance = covariance + np.eye(3) * (floor_variance - min_eig)
        return covariance

    def _apply_position_drift_common_mode(self, delta_w: np.ndarray) -> None:
        delta = np.asarray(delta_w, dtype=np.float64).reshape(3)
        self.drift_filter.x[0:3] += delta
        self.drift_filter.x[3:6] += delta

    def _release_pending_position_drift(self, timestamp_s: float) -> float:
        timestamp = float(timestamp_s)
        if not self.learned_position_residual_slew:
            return 0.0
        if self._last_position_slew_timestamp_s is None:
            self._last_position_slew_timestamp_s = timestamp
            return 0.0
        dt_s = timestamp - self._last_position_slew_timestamp_s
        if dt_s < -1.0e-9:
            raise ValueError("Position-residual slew timestamps must be monotonic")
        self._last_position_slew_timestamp_s = timestamp
        if dt_s <= 1.0e-12:
            return 0.0
        pending_norm = float(np.linalg.norm(self.pending_position_drift_w))
        if pending_norm <= 1.0e-12:
            return 0.0
        max_release = self.learned_position_residual_max_rate_mps * dt_s
        release_norm = min(pending_norm, max_release)
        release = self.pending_position_drift_w * (release_norm / pending_norm)
        self._apply_position_drift_common_mode(release)
        self.pending_position_drift_w -= release
        if np.linalg.norm(self.pending_position_drift_w) <= 1.0e-12:
            self.pending_position_drift_w.fill(0.0)
        return float(release_norm)

    def _isolate_raw_vio_jump(
        self,
        raw: VioWorldEstimate,
    ) -> tuple[VioWorldEstimate, bool, np.ndarray | None, float | None]:
        detected = False
        residual: np.ndarray | None = None
        residual_m: float | None = None
        previous = self._previous_raw_vio
        if previous is not None:
            dt = float(raw.timestamp_s) - float(previous.timestamp_s)
            if dt > 1.0e-12:
                observed_dp = np.asarray(raw.position_w_b) - np.asarray(previous.position_w_b)
                expected_dp = (
                    0.5
                    * (
                        np.asarray(previous.linear_velocity_w_b, dtype=np.float64)
                        + np.asarray(raw.linear_velocity_w_b, dtype=np.float64)
                    )
                    * dt
                )
                residual = np.asarray(observed_dp - expected_dp, dtype=np.float64)
                residual_m = float(np.linalg.norm(residual))
                if self.raw_vio_jump_isolation and residual_m > self.raw_vio_jump_threshold_m:
                    detected = True
                    self.raw_vio_jump_compensation_w += residual
                    self.raw_vio_jump_count += 1
                    self.raw_vio_jump_max_residual_m = max(
                        self.raw_vio_jump_max_residual_m,
                        residual_m,
                    )
                    self.raw_vio_jump_total_compensation_m += residual_m
        self._previous_raw_vio = raw
        isolated = VioWorldEstimate(
            position_w_b=(
                np.asarray(raw.position_w_b, dtype=np.float64)
                - self.raw_vio_jump_compensation_w
            ),
            linear_velocity_w_b=np.asarray(raw.linear_velocity_w_b, dtype=np.float64).copy(),
            orientation_w_b_wxyz=np.asarray(raw.orientation_w_b_wxyz, dtype=np.float64).copy(),
            timestamp_s=raw.timestamp_s,
        )
        self._last_isolated_vio = isolated
        return isolated, detected, residual, residual_m

    def _initialize_from_vio(
        self,
        raw: VioWorldEstimate,
        isolated: VioWorldEstimate,
        *,
        jump_detected: bool = False,
        jump_residual_w: np.ndarray | None = None,
        jump_residual_m: float | None = None,
    ) -> HybridVioCorrectionResult:
        self.window_start_timestamp_s = float(isolated.timestamp_s)
        self._last_position_slew_timestamp_s = float(isolated.timestamp_s)
        self.drift_filter.reset(isolated.timestamp_s, anchor_vio_position_w=isolated.position_w_b)
        corrected = self.drift_filter.step(isolated)
        result = HybridVioCorrectionResult(
            raw=raw,
            corrected=corrected,
            learned_position_pending_norm_m=float(np.linalg.norm(self.pending_position_drift_w)),
            raw_vio_jump_detected=jump_detected,
            raw_vio_jump_residual_w=None if jump_residual_w is None else jump_residual_w.copy(),
            raw_vio_jump_residual_m=jump_residual_m,
            raw_vio_jump_compensation_w=self.raw_vio_jump_compensation_w.copy(),
        )
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
        self.body_motion_buffer.discard_before(endpoint.timestamp_s)

    def apply_absolute_position(
        self,
        raw: VioWorldEstimate,
        *,
        position_w_b: np.ndarray,
        covariance_w: np.ndarray,
    ) -> HybridVioCorrectionResult:
        """Fuse one absolute position measurement at the latest raw-VIO time."""
        if self.last_result is None or self._last_isolated_vio is None:
            isolated, detected, residual, residual_m = self._isolate_raw_vio_jump(raw)
            self.raw_vio_buffer.push(raw)
            self.vio_buffer.push(isolated)
            self._initialize_from_vio(
                raw,
                isolated,
                jump_detected=detected,
                jump_residual_w=residual,
                jump_residual_m=residual_m,
            )
        if self.last_result is None or self._last_isolated_vio is None:
            raise RuntimeError("Hybrid VIO corrector failed to initialize")
        if abs(float(raw.timestamp_s) - float(self.last_result.raw.timestamp_s)) > 1.0e-6:
            raise ValueError("Absolute position anchor must match the latest raw VIO timestamp")
        if abs(float(raw.timestamp_s) - float(self._last_isolated_vio.timestamp_s)) > 1.0e-6:
            raise ValueError("Absolute position anchor must match the latest isolated VIO timestamp")
        if self.learned_position_residual_slew:
            self.pending_position_drift_w.fill(0.0)
            self._last_position_slew_timestamp_s = float(raw.timestamp_s)
        velocity_before = self.drift_filter.current_velocity_drift_w
        corrected = self.drift_filter.apply_absolute_position(
            self._last_isolated_vio,
            position_w_b=position_w_b,
            covariance_w=covariance_w,
        )
        velocity_after = self.drift_filter.current_velocity_drift_w
        diagnostics = self.drift_filter.last_absolute_update_diagnostics
        result = replace(
            self.last_result,
            corrected=corrected,
            learned_position_pending_norm_m=float(np.linalg.norm(self.pending_position_drift_w)),
            absolute_position_update_applied=True,
            absolute_position_mahalanobis2=diagnostics.mahalanobis2,
            absolute_velocity_drift_before_w=velocity_before,
            absolute_velocity_drift_after_w=velocity_after,
        )
        self.last_result = result
        return result

    def step(self, raw: VioWorldEstimate) -> HybridVioCorrectionResult:
        isolated, jump_detected, jump_residual, jump_residual_m = self._isolate_raw_vio_jump(raw)
        self.raw_vio_buffer.push(raw)
        self.vio_buffer.push(isolated)
        if self.window_start_timestamp_s is None:
            return self._initialize_from_vio(
                raw,
                isolated,
                jump_detected=jump_detected,
                jump_residual_w=jump_residual,
                jump_residual_m=jump_residual_m,
            )

        attempted = False
        accepted = False
        rejected = False
        skipped = 0
        last_prediction: DisplacementPrediction | None = None
        last_prediction_start: float | None = None
        last_prediction_end: float | None = None
        last_raw_window_displacement: np.ndarray | None = None
        last_d2: float | None = None
        learned_velocity_update_applied = False
        learned_velocity_measurement: np.ndarray | None = None
        learned_velocity_measurement_covariance: np.ndarray | None = None
        learned_velocity_d2: float | None = None
        learned_velocity_before: np.ndarray | None = None
        learned_velocity_after: np.ndarray | None = None
        learned_position_injection_norm = 0.0
        epsilon = 1.0e-9

        while isolated.timestamp_s + epsilon >= self.window_start_timestamp_s + self.window_time_s:
            start = float(self.window_start_timestamp_s)
            end = start + self.window_time_s
            try:
                startpoint = self.vio_buffer.interpolate(start)
                endpoint = self.vio_buffer.interpolate(end)
            except ValueError:
                # We cannot safely create a measurement without VIO at the exact
                # learned-window boundary. Leave the filter at its current time;
                # the retained history may bracket the boundary on a later call.
                break

            try:
                window = self._prediction_window(start, end)
            except ValueError:
                skipped += 1
                self._advance_skipped_window(endpoint)
                continue

            prediction = self.predictor.predict(window)
            last_prediction = prediction
            last_prediction_start = start
            last_prediction_end = end
            try:
                raw_start = self.raw_vio_buffer.interpolate(start)
                raw_endpoint = self.raw_vio_buffer.interpolate(end)
            except ValueError:
                last_raw_window_displacement = None
            else:
                last_raw_window_displacement = (
                    np.asarray(raw_endpoint.position_w_b, dtype=np.float64)
                    - np.asarray(raw_start.position_w_b, dtype=np.float64)
                )
            measurement = LearnedDisplacementMeasurement(
                displacement_w=prediction.displacement_w,
                covariance_w=prediction.covariance_w,
                start_timestamp_s=start,
                end_timestamp_s=end,
            )
            attempted = True
            position_velocity_before = self.drift_filter.current_velocity_drift_w
            position_drift_before: np.ndarray | None = None
            if self.learned_position_residual_slew:
                self.drift_filter.predict(endpoint.timestamp_s)
                position_drift_before = self.drift_filter.current_position_drift_w
            self.drift_filter.step(endpoint, measurement=measurement)
            position_velocity_after = self.drift_filter.current_velocity_drift_w
            diagnostics = self.drift_filter.last_update_diagnostics
            last_d2 = diagnostics.mahalanobis2
            if diagnostics.accepted:
                accepted = True
                self.window_start_timestamp_s = end
                if self.learned_position_residual_slew:
                    if position_drift_before is None:
                        raise RuntimeError("Position residual slew pre-update state was not captured")
                    position_injection = (
                        self.drift_filter.current_position_drift_w - position_drift_before
                    )
                    learned_position_injection_norm = float(np.linalg.norm(position_injection))
                    self._apply_position_drift_common_mode(-position_injection)
                    self.pending_position_drift_w += position_injection
                if self.learned_drift_velocity_from_displacement:
                    dt_s = end - start
                    isolated_displacement = (
                        np.asarray(endpoint.position_w_b, dtype=np.float64)
                        - np.asarray(startpoint.position_w_b, dtype=np.float64)
                    )
                    learned_velocity_measurement = (
                        isolated_displacement
                        - np.asarray(prediction.displacement_w, dtype=np.float64)
                    ) / dt_s
                    learned_velocity_measurement_covariance = self._velocity_measurement_covariance(
                        prediction.covariance_w,
                        dt_s,
                    )
                    learned_velocity_before = self.drift_filter.current_velocity_drift_w
                    self.drift_filter.apply_velocity_drift_measurement(
                        velocity_drift_w=learned_velocity_measurement,
                        covariance_w=learned_velocity_measurement_covariance,
                    )
                    learned_velocity_after = self.drift_filter.current_velocity_drift_w
                    learned_velocity_d2 = (
                        self.drift_filter.last_velocity_update_diagnostics.mahalanobis2
                    )
                    learned_velocity_update_applied = True
                else:
                    learned_velocity_before = position_velocity_before
                    learned_velocity_after = position_velocity_after
            else:
                rejected = True
                learned_velocity_before = position_velocity_before
                learned_velocity_after = position_velocity_after
                # The drift state is still valid; only this network constraint
                # was rejected. Re-anchor without applying the bad measurement.
                self.drift_filter.reanchor(endpoint)
                self.window_start_timestamp_s = end
            self.motion_buffer.discard_before(end)
            self.body_motion_buffer.discard_before(end)

        # Learned updates are processed at exact historical boundaries before
        # propagating the drift state to the newest jump-isolated OpenVINS time.
        corrected = self.drift_filter.step(isolated)
        learned_position_release_norm = self._release_pending_position_drift(
            isolated.timestamp_s
        )
        if learned_position_release_norm > 0.0:
            corrected = self.drift_filter.corrected(isolated)
        learned_position_pending_norm = float(np.linalg.norm(self.pending_position_drift_w))
        result = HybridVioCorrectionResult(
            raw=raw,
            corrected=corrected,
            learned_update_attempted=attempted,
            learned_update_accepted=accepted,
            learned_update_rejected=rejected,
            skipped_windows=skipped,
            mahalanobis2=last_d2,
            prediction=last_prediction,
            prediction_start_timestamp_s=last_prediction_start,
            prediction_end_timestamp_s=last_prediction_end,
            raw_window_displacement_w=last_raw_window_displacement,
            learned_velocity_update_applied=learned_velocity_update_applied,
            learned_velocity_drift_measurement_w=(
                None
                if learned_velocity_measurement is None
                else learned_velocity_measurement.copy()
            ),
            learned_velocity_drift_measurement_covariance_w=(
                None
                if learned_velocity_measurement_covariance is None
                else learned_velocity_measurement_covariance.copy()
            ),
            learned_velocity_mahalanobis2=learned_velocity_d2,
            learned_position_injection_norm_m=learned_position_injection_norm,
            learned_position_release_norm_m=learned_position_release_norm,
            learned_position_pending_norm_m=learned_position_pending_norm,
            learned_velocity_drift_before_w=learned_velocity_before,
            learned_velocity_drift_after_w=learned_velocity_after,
            raw_vio_jump_detected=jump_detected,
            raw_vio_jump_residual_w=None if jump_residual is None else jump_residual.copy(),
            raw_vio_jump_residual_m=jump_residual_m,
            raw_vio_jump_compensation_w=self.raw_vio_jump_compensation_w.copy(),
        )
        self.last_result = result
        return result
