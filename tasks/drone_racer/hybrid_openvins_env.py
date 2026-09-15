"""Hybrid learned-motion constrained OpenVINS runtime for drone racing."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from estimation.hybrid_vio_corrector import HybridLearnedVioCorrector
from estimation.openvins_bridge import OpenVinsFrameAlignment
from estimation.swift_vio_drift import VioWorldEstimate

from .drone_racer_hybrid_openvins_env_cfg import DroneRacerHybridOpenVinsEnvCfg
from .swift_openvins_env import SwiftOpenVinsDiagnosticEnv, _to_numpy


class HybridSwiftOpenVinsDiagnosticEnv(SwiftOpenVinsDiagnosticEnv):
    """OpenVINS -> learned relative-motion correction -> Swift gate fusion.

    Raw OpenVINS is always retained in ``openvins_raw_vio_estimate``. When a
    learned checkpoint is configured, ``openvins_learned_vio_estimate`` is the
    learned-corrected translation/velocity state and the compatibility field
    ``openvins_vio_estimate`` points at that corrected state. Existing Swift
    mapped-gate fusion therefore receives the learned-corrected VIO state.
    """

    cfg: DroneRacerHybridOpenVinsEnvCfg

    def __init__(self, cfg: DroneRacerHybridOpenVinsEnvCfg, render_mode: str | None = None, **kwargs):
        self.openvins_raw_vio_estimate: VioWorldEstimate | None = None
        self.openvins_learned_vio_estimate: VioWorldEstimate | None = None
        self.learned_motion_corrector: HybridLearnedVioCorrector | None = None
        self.learned_motion_last_result = None
        super().__init__(cfg=cfg, render_mode=render_mode, **kwargs)
        self._initialize_optional_learned_motion()

    def _initialize_optional_learned_motion(self) -> None:
        checkpoint = self.cfg.learned_motion_checkpoint
        if checkpoint is None:
            return

        from estimation.learned_motion import TorchTcnDisplacementPredictor

        checkpoint_path = Path(checkpoint).expanduser().resolve()
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Learned motion checkpoint not found: {checkpoint_path}")
        predictor = TorchTcnDisplacementPredictor(
            checkpoint_path,
            device=self.cfg.learned_motion_device,
            variance_floor=self.cfg.learned_motion_variance_floor,
        )
        if not np.isclose(
            predictor.window_time_s,
            self.cfg.learned_motion_window_time_s,
            rtol=0.0,
            atol=1.0e-9,
        ):
            raise ValueError(
                "Learned motion checkpoint window_time_s does not match environment config: "
                f"checkpoint={predictor.window_time_s}, "
                f"config={self.cfg.learned_motion_window_time_s}"
            )
        if not np.isclose(
            predictor.sample_rate_hz,
            self.cfg.learned_motion_sample_rate_hz,
            rtol=0.0,
            atol=1.0e-9,
        ):
            raise ValueError(
                "Learned motion checkpoint sample_rate_hz does not match environment config: "
                f"checkpoint={predictor.sample_rate_hz}, "
                f"config={self.cfg.learned_motion_sample_rate_hz}"
            )

        self.learned_motion_corrector = HybridLearnedVioCorrector(
            predictor,
            window_time_s=self.cfg.learned_motion_window_time_s,
            sample_rate_hz=self.cfg.learned_motion_sample_rate_hz,
            sigma_position=self.cfg.learned_motion_sigma_position,
            sigma_velocity=self.cfg.learned_motion_sigma_velocity,
            innovation_gate_chi2=self.cfg.learned_motion_innovation_gate_chi2,
        )

    def _publish_imu_if_due(self) -> None:
        """Publish IMU to OpenVINS and retain timestamp-identical motion-model inputs."""
        timestamp_s = self._timestamp_s()
        if not self._openvins_rate_gate.imu_due(timestamp_s):
            return
        imu = self.scene["imu"]
        gyro_b = _to_numpy(imu.data.ang_vel_b[0]).astype(np.float64)
        accel_b = _to_numpy(imu.data.lin_acc_b[0]).astype(np.float64)
        self._openvins_bridge.publish_imu(
            angular_velocity_b=gyro_b,
            linear_acceleration_b=accel_b,
            timestamp_s=timestamp_s,
        )

        if self.learned_motion_corrector is not None:
            control_term = self.action_manager.get_term("control_action")
            processed = _to_numpy(control_term.processed_actions[0]).astype(np.float64).reshape(-1)
            if processed.size < 1 or not np.isfinite(processed[0]):
                raise RuntimeError("control_action does not expose a finite collective thrust")
            thrust_b = np.array([0.0, 0.0, float(processed[0])], dtype=np.float64)
            self.learned_motion_corrector.ingest_body_motion_sample(
                timestamp_s,
                gyro_b=gyro_b,
                thrust_b=thrust_b,
            )

    def _consume_openvins(self) -> None:
        sample = self._openvins_bridge.drain_latest(max_callbacks=32)
        if sample is None:
            return
        if (
            self._last_openvins_sample_timestamp_s is not None
            and sample.timestamp_s <= self._last_openvins_sample_timestamp_s + 1.0e-12
        ):
            return

        alignment_created = self.openvins_alignment is None
        if alignment_created:
            try:
                reference = self._alignment_truth_buffer.interpolate(sample.timestamp_s)
            except ValueError:
                return
            self.openvins_alignment = OpenVinsFrameAlignment.from_reference_pose(
                sample,
                reference_position_w_b=reference.position_w_b,
                reference_orientation_w_b_wxyz=reference.orientation_w_b_wxyz,
            )
            self._alignment_truth_buffer.clear()

        raw_world = self.openvins_alignment.to_world(sample)
        self.openvins_raw_vio_estimate = raw_world

        selected_world = raw_world
        if self.learned_motion_corrector is not None:
            self.learned_motion_last_result = self.learned_motion_corrector.step(raw_world)
            selected_world = self.learned_motion_last_result.corrected
            self.openvins_learned_vio_estimate = selected_world
        else:
            self.learned_motion_last_result = None
            self.openvins_learned_vio_estimate = None

        # Compatibility contract: all existing downstream consumers continue to
        # read this field. In hybrid mode it is the learned-corrected VIO state.
        self.openvins_vio_estimate = selected_world
        self.openvins_vio_buffer.push(selected_world)
        self._last_openvins_sample_timestamp_s = sample.timestamp_s

        if self.swift_fusion is not None:
            if alignment_created:
                self._pending_gate_frames.discard_before(selected_world.timestamp_s)
            pending = self._pending_gate_frames.pop_ready(selected_world.timestamp_s + 1.0e-6)
            if pending is None:
                self.swift_last_fusion_result = self.swift_fusion.step(selected_world)
            else:
                self.swift_last_fusion_result = self.swift_fusion.step(
                    selected_world,
                    gate_observation=pending.observation,
                    gate_index=pending.gate_index,
                )
                self._maybe_dump_rejection(
                    pending.observation,
                    pending.rgb,
                    self.swift_last_fusion_result,
                )
            self.swift_fused_estimate = self.swift_last_fusion_result.fused_state

    def _update_openvins_log(self) -> None:
        super()._update_openvins_log()
        log = self.extras.setdefault("log", {})
        log["LearnedMotion/enabled"] = float(self.learned_motion_corrector is not None)
        if self.openvins_raw_vio_estimate is not None and self.openvins_vio_estimate is not None:
            correction = (
                self.openvins_raw_vio_estimate.position_w_b
                - self.openvins_vio_estimate.position_w_b
            )
            log["LearnedMotion/position_correction_norm_m"] = float(np.linalg.norm(correction))
        if self.learned_motion_last_result is not None:
            result = self.learned_motion_last_result
            log["LearnedMotion/update_attempted"] = float(result.learned_update_attempted)
            log["LearnedMotion/update_accepted"] = float(result.learned_update_accepted)
            log["LearnedMotion/update_rejected"] = float(result.learned_update_rejected)
            log["LearnedMotion/skipped_windows"] = float(result.skipped_windows)
            if result.mahalanobis2 is not None:
                log["LearnedMotion/innovation_d2"] = float(result.mahalanobis2)

    def _reset_idx(self, env_ids) -> None:
        super()._reset_idx(env_ids)
        self.openvins_raw_vio_estimate = None
        self.openvins_learned_vio_estimate = None
        self.learned_motion_last_result = None
        if self.learned_motion_corrector is not None:
            self.learned_motion_corrector.reset()
