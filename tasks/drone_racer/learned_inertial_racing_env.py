"""OpenVINS-free learned-inertial racing runtime.

Estimator inputs after reset:
  * Isaac IMU measurements
  * applied collective thrust
  * TCN relative-displacement predictions
  * Stage2 camera gate PnP measurements + known track map

Simulator root pose is not used by the estimator after the fixed known start.
Ground truth remains available only to rewards/evaluation and supervised data
collection scripts.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from isaaclab.envs import ManagerBasedRLEnv

from estimation.learned_inertial_odometry import (
    LearnedInertialOdometry,
    protected_displacement_covariance,
    quat_wxyz_to_rotmat,
    rotmat_to_quat_wxyz,
)
from estimation.learned_motion import LearnedMotionBuffer, TorchTcnDisplacementPredictor

from .drone_racer_learned_inertial_env_cfg import DroneRacerLearnedInertialEnvCfg


def _np(value) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    return np.asarray(value)


class LearnedInertialRacingEnv(ManagerBasedRLEnv):
    """Single-stream sensor-faithful environment for IMO-style racing."""

    cfg: DroneRacerLearnedInertialEnvCfg

    def __init__(
        self,
        cfg: DroneRacerLearnedInertialEnvCfg,
        render_mode: str | None = None,
        **kwargs,
    ):
        self.learned_inertial_state = None
        self._lio: LearnedInertialOdometry | None = None
        self._motion_buffer: LearnedMotionBuffer | None = None
        self._motion_predictor = None
        self._learned_epoch_start_s: float | None = None
        self._next_clone_s: float | None = None
        self._next_learned_update_s: float | None = None
        self.swift_detector = None
        self._gate_builder = None
        self._last_camera_timestamp_s = -np.inf
        self._last_gate_measurement = None
        self._learned_update_count = 0
        self._learned_fusion_count = 0
        self._learned_update_skip_count = 0
        self._last_learned_innovation_w = None
        self._last_learned_prediction_w = None
        self._last_learned_measurement_w = None
        self._last_learned_measurement_source = None
        self._last_learned_predicted_rel_w = None
        self._last_learned_covariance_raw_w = None
        self._last_learned_covariance_used_w = None
        self._last_learned_window_start_s = None
        self._last_learned_window_end_s = None
        self._last_learned_update_timestamp_s = None
        self._last_learned_fused = False
        self._gate_attempt_count = 0
        self._gate_update_count = 0
        self._gate_reject_count = 0
        self._last_gate_mahalanobis2 = None
        self._imu_rng = None
        self._imu_accel_bias_b = np.zeros(3, dtype=np.float64)
        self._imu_gyro_bias_b = np.zeros(3, dtype=np.float64)
        self._last_imu_accel_b_meas = None
        self._last_imu_gyro_b_meas = None
        self._debug_truth_motion_history = {}
        super().__init__(cfg=cfg, render_mode=render_mode, **kwargs)
        if self.num_envs != 1:
            raise ValueError("LearnedInertialRacingEnv currently requires num_envs=1")

        learned_rate_hz = float(cfg.learned_update_rate_hz)
        window_s = float(cfg.learned_window_time_s)
        if learned_rate_hz <= 0.0:
            raise ValueError("learned_update_rate_hz must be positive")
        if window_s <= 0.0:
            raise ValueError("learned_window_time_s must be positive")

        fusion_rate_hz = (
            learned_rate_hz
            if cfg.learned_fusion_rate_hz is None
            else float(cfg.learned_fusion_rate_hz)
        )
        if fusion_rate_hz <= 0.0:
            raise ValueError("learned_fusion_rate_hz must be positive when set")
        if fusion_rate_hz > learned_rate_hz + 1.0e-12:
            raise ValueError(
                "learned_fusion_rate_hz cannot exceed learned_update_rate_hz"
            )
        fusion_stride = learned_rate_hz / fusion_rate_hz
        rounded_stride = int(round(fusion_stride))
        if rounded_stride < 1 or abs(fusion_stride - rounded_stride) > 1.0e-9:
            raise ValueError(
                "learned_update_rate_hz / learned_fusion_rate_hz must be an integer"
            )
        self._learned_fusion_stride = rounded_stride
        self._learned_fusion_rate_hz = fusion_rate_hz

        required_clones = int(np.ceil(window_s * learned_rate_hz - 1.0e-12)) + 1
        if int(cfg.learned_max_position_clones) < required_clones:
            raise ValueError(
                "learned_max_position_clones is too small for the configured fixed lag: "
                f"need at least {required_clones}"
            )

        imu_sigmas = (
            cfg.imu_accel_white_noise_sigma_mps2,
            cfg.imu_gyro_white_noise_sigma_radps,
            cfg.imu_accel_initial_bias_sigma_mps2,
            cfg.imu_gyro_initial_bias_sigma_radps,
            cfg.imu_accel_bias_rw_sigma_mps2_sqrt_s,
            cfg.imu_gyro_bias_rw_sigma_radps_sqrt_s,
            cfg.ekf_accel_noise_sigma,
            cfg.ekf_gyro_noise_sigma,
            cfg.ekf_accel_bias_rw_sigma,
            cfg.ekf_gyro_bias_rw_sigma,
        )
        if any(float(value) < 0.0 or not np.isfinite(float(value)) for value in imu_sigmas):
            raise ValueError("IMU corruption and EKF noise sigmas must be finite and non-negative")

        self._lio = LearnedInertialOdometry(
            accel_noise_sigma=float(cfg.ekf_accel_noise_sigma),
            gyro_noise_sigma=float(cfg.ekf_gyro_noise_sigma),
            accel_bias_rw_sigma=float(cfg.ekf_accel_bias_rw_sigma),
            gyro_bias_rw_sigma=float(cfg.ekf_gyro_bias_rw_sigma),
            max_position_clones=int(cfg.learned_max_position_clones),
            learned_kalman_gain_mode=str(cfg.learned_kalman_gain_mode),
        )
        self._motion_buffer = LearnedMotionBuffer(
            window_time_s=cfg.learned_window_time_s,
            sample_rate_hz=cfg.learned_sample_rate_hz,
        )
        if cfg.learned_motion_checkpoint is not None:
            checkpoint = Path(cfg.learned_motion_checkpoint).expanduser().resolve()
            if not checkpoint.exists():
                raise FileNotFoundError(f"learned motion checkpoint not found: {checkpoint}")
            self._motion_predictor = TorchTcnDisplacementPredictor(
                checkpoint,
                device=cfg.learned_motion_device,
            )
        self._initialize_detector_and_gate_builder()
        self._reset_estimator_from_known_start()

    def _timestamp_s(self) -> float:
        return float(self._sim_step_counter) * float(self.physics_dt)

    def _reset_estimator_from_known_start(self) -> None:
        # This pose is part of the task definition, not a runtime truth query.
        init = self.cfg.scene.robot.init_state
        self._lio.reset(
            timestamp_s=self._timestamp_s(),
            position_w_b=tuple(float(v) for v in init.pos),
            linear_velocity_w_b=(0.0, 0.0, 0.0),
            orientation_w_b_wxyz=tuple(float(v) for v in init.rot),
        )
        self._motion_buffer.reset()
        # Re-seed on reset so A/S/B/P/C fresh-process replay comparisons use
        # exactly the same synthetic sensor realization.
        self._imu_rng = np.random.default_rng(int(self.cfg.imu_noise_seed))
        self._imu_accel_bias_b = self._imu_rng.normal(
            0.0,
            float(self.cfg.imu_accel_initial_bias_sigma_mps2),
            size=3,
        )
        self._imu_gyro_bias_b = self._imu_rng.normal(
            0.0,
            float(self.cfg.imu_gyro_initial_bias_sigma_radps),
            size=3,
        )
        self._last_imu_accel_b_meas = None
        self._last_imu_gyro_b_meas = None
        self._debug_truth_motion_history = {}
        # Anchor the learned fixed-lag schedule on the first valid 100 Hz
        # motion sample, rather than inventing a thrust sample at reset.
        self._learned_epoch_start_s = None
        self._next_clone_s = None
        self._next_learned_update_s = None
        self._last_camera_timestamp_s = -np.inf
        self._last_gate_measurement = None
        self._learned_update_count = 0
        self._learned_fusion_count = 0
        self._learned_update_skip_count = 0
        self._last_learned_innovation_w = None
        self._last_learned_prediction_w = None
        self._last_learned_measurement_w = None
        self._last_learned_measurement_source = None
        self._last_learned_predicted_rel_w = None
        self._last_learned_covariance_raw_w = None
        self._last_learned_covariance_used_w = None
        self._last_learned_window_start_s = None
        self._last_learned_window_end_s = None
        self._last_learned_update_timestamp_s = None
        self._last_learned_fused = False
        self._last_gate_mahalanobis2 = None
        self.learned_inertial_state = self._lio.state()

    def _initialize_detector_and_gate_builder(self) -> None:
        if self.cfg.swift_detector_checkpoint is None:
            return

        from perception.camera_model import CameraCalibration
        from perception.stage2_calibration import load_stage2_gate_geometry, stage2_camera_to_body
        from perception.swift_gate_measurement import CornerPerturbationConfig, GatePoseMeasurementBuilder
        from perception.swift_isaac_adapter import track_layout_from_isaac
        from perception.torchvision_keypoint_detector import TorchvisionGateCornerDetector

        checkpoint = Path(self.cfg.swift_detector_checkpoint).expanduser().resolve()
        if not checkpoint.exists():
            raise FileNotFoundError(f"Swift detector checkpoint not found: {checkpoint}")

        camera = self.scene["tiled_camera"]
        K = _np(camera.data.intrinsic_matrices[0])
        image_height, image_width = (int(v) for v in camera.data.image_shape)
        calibration = CameraCalibration(
            K=K,
            image_width=image_width,
            image_height=image_height,
            distortion=None,
            model="pinhole",
        )
        self._gate_builder = GatePoseMeasurementBuilder(
            load_stage2_gate_geometry(),
            calibration,
            stage2_camera_to_body(),
            track_layout_from_isaac(self, env_id=0),
            perturbation=CornerPerturbationConfig(
                num_samples=20,
                corner_sigma_px=self.cfg.swift_corner_sigma_px,
            ),
        )

        coordinate_detector = TorchvisionGateCornerDetector(
            checkpoint,
            device=self.cfg.swift_detector_device,
            detection_threshold=self.cfg.swift_detection_threshold,
            keypoint_confidence_threshold=(
                0.0
                if self.cfg.swift_visibility_checkpoint is not None
                else self.cfg.swift_keypoint_confidence_threshold
            ),
        )
        if self.cfg.swift_visibility_checkpoint is None:
            self.swift_detector = coordinate_detector
        else:
            from perception.hybrid_keypoint_detector import VisibilityGuardedGateCornerDetector
            from perception.keypoint_detector import TorchGateCornerDetector

            visibility_path = Path(self.cfg.swift_visibility_checkpoint).expanduser().resolve()
            visibility_detector = TorchGateCornerDetector(
                visibility_path,
                device=self.cfg.swift_detector_device,
                visibility_threshold=self.cfg.swift_visibility_threshold,
            )
            self.swift_detector = VisibilityGuardedGateCornerDetector(
                coordinate_detector,
                visibility_detector,
                visibility_threshold=self.cfg.swift_visibility_threshold,
            )

    def _propagate_from_imu(self) -> None:
        timestamp_s = self._timestamp_s()
        if timestamp_s <= self._lio.timestamp_s + 1.0e-12:
            return

        dt = timestamp_s - float(self._lio.timestamp_s)
        imu = self.scene["imu"]
        accel_ideal = _np(imu.data.lin_acc_b[0]).astype(np.float64)
        gyro_ideal = _np(imu.data.ang_vel_b[0]).astype(np.float64)

        # Biases evolve as random walks; white noise is sampled independently
        # per simulated IMU sample. Defaults are all zero, reproducing the
        # previous ideal sensor path exactly.
        sqrt_dt = np.sqrt(dt)
        self._imu_accel_bias_b += (
            float(self.cfg.imu_accel_bias_rw_sigma_mps2_sqrt_s)
            * sqrt_dt
            * self._imu_rng.normal(size=3)
        )
        self._imu_gyro_bias_b += (
            float(self.cfg.imu_gyro_bias_rw_sigma_radps_sqrt_s)
            * sqrt_dt
            * self._imu_rng.normal(size=3)
        )
        accel_meas = (
            accel_ideal
            + self._imu_accel_bias_b
            + float(self.cfg.imu_accel_white_noise_sigma_mps2)
            * self._imu_rng.normal(size=3)
        )
        gyro_meas = (
            gyro_ideal
            + self._imu_gyro_bias_b
            + float(self.cfg.imu_gyro_white_noise_sigma_radps)
            * self._imu_rng.normal(size=3)
        )

        self._last_imu_accel_b_meas = accel_meas.copy()
        self._last_imu_gyro_b_meas = gyro_meas.copy()
        self._lio.propagate(
            gyro_b=gyro_meas,
            accel_b=accel_meas,
            timestamp_s=timestamp_s,
        )
        self.learned_inertial_state = self._lio.state()

    def _append_learned_motion_sample(self) -> None:
        """Append paper-style world-frame gyro + mass-normalized thrust."""
        timestamp_s = self._timestamp_s()
        imu = self.scene["imu"]
        gyro_b = (
            _np(imu.data.ang_vel_b[0]).astype(np.float64)
            if self._last_imu_gyro_b_meas is None
            else np.asarray(self._last_imu_gyro_b_meas, dtype=np.float64).copy()
        )

        # Allocation output channel 0 is collective force [N]. The FIVE_IN_DRONE
        # model mass is 0.6076 kg (same value used to derive thrust coefficient).
        control = self.action_manager.get_term("control_action")
        collective_force_n = float(_np(control.processed_actions[0])[0])
        mass_kg = float(self.cfg.vehicle_mass_kg)
        if mass_kg <= 0.0:
            raise ValueError("vehicle_mass_kg must be positive")
        thrust_b = np.array([0.0, 0.0, collective_force_n / mass_kg], dtype=np.float64)

        feature_frame = str(
            getattr(self._motion_predictor, "feature_frame", "world")
        )
        if feature_frame in ("body", "body_endpoint_gyro_aligned"):
            feature_gyro = gyro_b
            feature_thrust = thrust_b
        elif feature_frame == "world":
            if bool(self.cfg.learned_debug_truth_orientation_for_features):
                robot = self.scene["robot"]
                R_wb = quat_wxyz_to_rotmat(
                    _np(robot.data.root_quat_w[0]).astype(np.float64)
                )
            else:
                R_wb = self._lio.R
            feature_gyro = R_wb @ gyro_b
            feature_thrust = R_wb @ thrust_b
        else:
            raise RuntimeError(
                f"unsupported learned-motion feature frame: {feature_frame!r}"
            )

        self._motion_buffer.append(
            timestamp_s,
            gyro_w=feature_gyro,
            thrust_w=feature_thrust,
        )

        if bool(self.cfg.learned_debug_oracle_residual_fusion):
            robot = self.scene["robot"]
            key = round(float(timestamp_s), 9)
            self._debug_truth_motion_history[key] = (
                _np(robot.data.root_pos_w[0]).astype(np.float64).copy(),
                _np(robot.data.root_lin_vel_w[0]).astype(np.float64).copy(),
            )
            cutoff = float(timestamp_s) - 2.0 * float(self.cfg.learned_window_time_s)
            for old_key in list(self._debug_truth_motion_history):
                if old_key < cutoff:
                    del self._debug_truth_motion_history[old_key]

    def _maybe_update_learned_displacement(self) -> None:
        """Run 0.5 s overlapping TCN constraints at the configured update rate."""
        if self._motion_predictor is None:
            return

        now = self._timestamp_s()
        window_s = float(self.cfg.learned_window_time_s)
        update_period_s = 1.0 / float(self.cfg.learned_update_rate_hz)
        timing_tolerance_s = max(1.0e-6, 0.51 * float(self.step_dt))

        if self._learned_epoch_start_s is None:
            self._learned_epoch_start_s = now
            self._lio.clone_current_position()
            self._next_clone_s = now + update_period_s
            self._next_learned_update_s = now + window_s
            return

        # V6.2 follows the UZH/TLIO stochastic-cloning ordering: create the
        # endpoint clone first, then form the relative factor between the
        # historical start clone and this endpoint clone. Unlike the legacy
        # constrained-gain path, the endpoint clone participates in the same
        # full Kalman update and therefore cannot remain stale.
        clone_due_now = False
        if now + 1.0e-9 >= self._next_clone_s:
            clone_due_now = abs(now - self._next_clone_s) <= timing_tolerance_s
            if clone_due_now:
                self._lio.clone_current_position()
            else:
                # Never label a current state with a historical clone timestamp.
                self._learned_update_skip_count += 1
            self._next_clone_s += update_period_s

        if now + 1.0e-9 < self._next_learned_update_s:
            return

        scheduled_end_s = float(self._next_learned_update_s)
        start_s = scheduled_end_s - window_s
        self._next_learned_update_s += update_period_s

        # A delayed endpoint would pair the TCN window with the wrong current
        # EKF state, so drop it instead of fusing time-misaligned information.
        if (
            abs(now - scheduled_end_s) > timing_tolerance_s
            or not clone_due_now
        ):
            self._learned_update_skip_count += 1
            self._lio.marginalize_clones_before(start_s, inclusive=True)
            self._motion_buffer.discard_before(start_s)
            return

        try:
            window = self._motion_buffer.window(start_s, scheduled_end_s)
        except ValueError:
            self._learned_update_skip_count += 1
            self._lio.marginalize_clones_before(start_s, inclusive=True)
            self._motion_buffer.discard_before(start_s)
            return

        prediction = self._motion_predictor.predict(window)
        protected_covariance = protected_displacement_covariance(
            prediction.covariance_w,
            sigma_floor_xyz_m=self.cfg.learned_sigma_floor_xyz_m,
            covariance_scale=self.cfg.learned_covariance_scale,
        )
        covariance_multiplier = float(
            self.cfg.learned_measurement_covariance_multiplier
        )
        if covariance_multiplier <= 0.0 or not np.isfinite(covariance_multiplier):
            raise ValueError(
                "learned_measurement_covariance_multiplier must be positive and finite"
            )
        protected_covariance = protected_covariance * covariance_multiplier
        target_mode = str(
            getattr(self._motion_predictor, "target_mode", "displacement")
        )
        try:
            if target_mode == "kinematic_residual":
                predicted_rel = self._lio.predicted_clone_kinematic_residual(
                    start_timestamp_s=start_s,
                    end_timestamp_s=scheduled_end_s,
                    clone_tolerance_s=timing_tolerance_s,
                )
            elif target_mode in (
                "kinematic_residual_body_end",
                "kinematic_residual_body_end_gyro_aligned",
            ):
                predicted_rel = self._lio.predicted_clone_kinematic_residual_body_end(
                    start_timestamp_s=start_s,
                    end_timestamp_s=scheduled_end_s,
                    clone_tolerance_s=timing_tolerance_s,
                )
            elif target_mode == "kinematic_residual_body_end_gravity_compensated":
                predicted_rel = (
                    self._lio.predicted_clone_kinematic_residual_body_end_gravity_compensated(
                        start_timestamp_s=start_s,
                        end_timestamp_s=scheduled_end_s,
                        clone_tolerance_s=timing_tolerance_s,
                    )
                )
            elif target_mode == "displacement":
                predicted_rel = self._lio.predicted_clone_relative_displacement(
                    start_timestamp_s=start_s,
                    end_timestamp_s=scheduled_end_s,
                    clone_tolerance_s=timing_tolerance_s,
                )
            else:
                raise RuntimeError(
                    f"unsupported learned-motion target mode: {target_mode!r}"
                )

            measurement_w = np.asarray(
                prediction.displacement_w, dtype=np.float64
            ).copy()
            network_bias_m = np.asarray(
                self.cfg.learned_network_bias_m,
                dtype=np.float64,
            ).reshape(3)
            if not np.all(np.isfinite(network_bias_m)):
                raise ValueError("learned_network_bias_m must be finite")
            measurement_w = measurement_w - network_bias_m
            measurement_source = (
                "network_bias_calibrated"
                if np.any(np.abs(network_bias_m) > 0.0)
                else "network"
            )
            if bool(self.cfg.learned_debug_oracle_residual_fusion):
                if target_mode not in (
                    "kinematic_residual",
                    "kinematic_residual_body_end",
                    "kinematic_residual_body_end_gyro_aligned",
                    "kinematic_residual_body_end_gravity_compensated",
                ):
                    raise RuntimeError(
                        "oracle residual fusion requires a kinematic-residual checkpoint"
                    )
                start_key = round(float(start_s), 9)
                end_key = round(float(scheduled_end_s), 9)
                if (
                    start_key not in self._debug_truth_motion_history
                    or end_key not in self._debug_truth_motion_history
                ):
                    raise RuntimeError(
                        "oracle residual fusion is missing GT history for window"
                    )
                p_start_gt, v_start_gt = self._debug_truth_motion_history[start_key]
                p_end_gt, _ = self._debug_truth_motion_history[end_key]
                window_dt = float(scheduled_end_s - start_s)
                measurement_w = (
                    p_end_gt
                    - p_start_gt
                    - v_start_gt * window_dt
                )
                if target_mode == "kinematic_residual_body_end_gravity_compensated":
                    measurement_w = (
                        measurement_w
                        - 0.5 * self._lio.gravity_w * window_dt * window_dt
                    )
                if target_mode in (
                    "kinematic_residual_body_end",
                    "kinematic_residual_body_end_gyro_aligned",
                    "kinematic_residual_body_end_gravity_compensated",
                ):
                    robot = self.scene["robot"]
                    R_end_gt = quat_wxyz_to_rotmat(
                        _np(robot.data.root_quat_w[0]).astype(np.float64)
                    )
                    measurement_w = R_end_gt.T @ measurement_w
                    measurement_source = "oracle_kinematic_residual_body_end"
                else:
                    measurement_source = "oracle_kinematic_residual"

            innovation = measurement_w - predicted_rel
            should_fuse = (
                bool(self.cfg.learned_apply_displacement_updates)
                and self._learned_update_count % self._learned_fusion_stride == 0
            )
            if should_fuse:
                if target_mode == "kinematic_residual":
                    self._lio.update_learned_clone_kinematic_residual(
                        measurement_w,
                        protected_covariance,
                        start_timestamp_s=start_s,
                        end_timestamp_s=scheduled_end_s,
                        clone_tolerance_s=timing_tolerance_s,
                        marginalize_start_clone=True,
                    )
                elif target_mode in (
                    "kinematic_residual_body_end",
                    "kinematic_residual_body_end_gyro_aligned",
                ):
                    self._lio.update_learned_clone_kinematic_residual_body_end(
                        measurement_w,
                        protected_covariance,
                        start_timestamp_s=start_s,
                        end_timestamp_s=scheduled_end_s,
                        clone_tolerance_s=timing_tolerance_s,
                        marginalize_start_clone=True,
                    )
                elif target_mode == "kinematic_residual_body_end_gravity_compensated":
                    self._lio.update_learned_clone_kinematic_residual_body_end_gravity_compensated(
                        measurement_w,
                        protected_covariance,
                        start_timestamp_s=start_s,
                        end_timestamp_s=scheduled_end_s,
                        clone_tolerance_s=timing_tolerance_s,
                        marginalize_start_clone=True,
                    )
                else:
                    self._lio.update_learned_clone_displacement(
                        measurement_w,
                        protected_covariance,
                        start_timestamp_s=start_s,
                        end_timestamp_s=scheduled_end_s,
                        clone_tolerance_s=timing_tolerance_s,
                        marginalize_start_clone=True,
                    )
                self._learned_fusion_count += 1
            else:
                # Keep 20 Hz predictions for diagnostics, but fuse only a
                # statistically safer subset. The skipped window's start clone
                # is no longer needed after this prediction.
                self._lio.marginalize_clone_at_timestamp(
                    start_s,
                    tolerance_s=timing_tolerance_s,
                )
        except (KeyError, RuntimeError):
            self._learned_update_skip_count += 1
            self._lio.marginalize_clones_before(start_s, inclusive=True)
            self._motion_buffer.discard_before(start_s)
            return

        self._learned_update_count += 1
        self._last_learned_innovation_w = np.asarray(innovation, dtype=np.float64).copy()
        self._last_learned_prediction_w = np.asarray(
            prediction.displacement_w, dtype=np.float64
        ).copy()
        self._last_learned_measurement_w = np.asarray(
            measurement_w, dtype=np.float64
        ).copy()
        self._last_learned_measurement_source = str(measurement_source)
        self._last_learned_predicted_rel_w = np.asarray(
            predicted_rel, dtype=np.float64
        ).copy()
        self._last_learned_covariance_raw_w = np.asarray(
            prediction.covariance_w, dtype=np.float64
        ).copy()
        self._last_learned_covariance_used_w = np.asarray(
            protected_covariance, dtype=np.float64
        ).copy()
        self._last_learned_window_start_s = float(start_s)
        self._last_learned_window_end_s = float(scheduled_end_s)
        self._last_learned_update_timestamp_s = now
        self._last_learned_fused = bool(should_fuse)
        # discard_before retains one interpolation predecessor. The next
        # 0.5 s window begins only 0.05 s later, so the histories overlap.
        self._motion_buffer.discard_before(start_s)
        self.learned_inertial_state = self._lio.state()

    def _camera_due(self) -> bool:
        now = self._timestamp_s()
        return now - self._last_camera_timestamp_s >= (1.0 / 30.0) - 1.0e-9

    def _maybe_gate_update(self) -> None:
        if self.swift_detector is None or self._gate_builder is None or not self._camera_due():
            return
        self._last_camera_timestamp_s = self._timestamp_s()
        self._gate_attempt_count += 1

        camera = self.scene["tiled_camera"]
        rgb = _np(camera.data.output["rgb"][0])[..., :3]
        if rgb.dtype != np.uint8:
            scale = (
                255.0
                if np.issubdtype(rgb.dtype, np.floating)
                and float(np.nanmax(rgb)) <= 1.0 + 1e-6
                else 1.0
            )
            rgb = np.clip(rgb * scale, 0.0, 255.0).astype(np.uint8)

        try:
            observation = self.swift_detector.detect(
                np.ascontiguousarray(rgb),
                timestamp_s=self._timestamp_s(),
            )
            measurement = self._gate_builder.build(
                observation,
                gate_index=None,
                reference_position_w_b=self._lio.p,
            )
        except (ValueError, RuntimeError):
            self._gate_reject_count += 1
            return

        # Reject a visually plausible but globally inconsistent gate association.
        innovation_p = measurement.position_w_b - self._lio.p
        Ppp = self._lio.P[6:9, 6:9]
        S = Ppp + measurement.position_covariance_w
        try:
            d2 = float(innovation_p.T @ np.linalg.solve(S, innovation_p))
        except np.linalg.LinAlgError:
            self._gate_reject_count += 1
            return
        self._last_gate_mahalanobis2 = d2
        if d2 > float(self.cfg.gate_position_mahalanobis2_max):
            self._gate_reject_count += 1
            return

        # Gate PnP + mapped T_wg is an absolute body-pose observation.
        q_wb = rotmat_to_quat_wxyz(measurement.T_wb_gate.R)
        sigma_rad = np.deg2rad(float(self.cfg.gate_orientation_sigma_deg))
        self._lio.update_gate_pose(
            position_w_b=measurement.position_w_b,
            position_covariance_w=measurement.position_covariance_w,
            orientation_w_b_wxyz=(
                q_wb if bool(self.cfg.gate_use_orientation_update) else None
            ),
            orientation_covariance_rad2=np.eye(3) * sigma_rad * sigma_rad,
        )
        self._last_gate_measurement = measurement
        self._gate_update_count += 1
        self.learned_inertial_state = self._lio.state()

    def _update_log(self) -> None:
        log = self.extras.setdefault("log", {})
        log["LearnedIO/learned_updates"] = float(self._learned_update_count)
        log["LearnedIO/learned_fusions"] = float(self._learned_fusion_count)
        log["LearnedIO/learned_update_skips"] = float(self._learned_update_skip_count)
        log["LearnedIO/position_clones"] = float(self._lio.clone_count)
        log["LearnedIO/gate_attempts"] = float(self._gate_attempt_count)
        log["LearnedIO/gate_updates"] = float(self._gate_update_count)
        log["LearnedIO/gate_rejects"] = float(self._gate_reject_count)
        log["LearnedIO/has_checkpoint"] = float(self._motion_predictor is not None)
        log["LearnedIO/has_gate_detector"] = float(self.swift_detector is not None)
        log["LearnedIO/cov_trace"] = float(np.trace(self._lio.P[:15, :15]))
        log["LearnedIO/imu_accel_bias_norm"] = float(
            np.linalg.norm(self._imu_accel_bias_b)
        )
        log["LearnedIO/imu_gyro_bias_norm"] = float(
            np.linalg.norm(self._imu_gyro_bias_b)
        )

    def reset(self, seed=None, env_ids=None, options=None):
        obs, extras = super().reset(seed=seed, env_ids=env_ids, options=options)
        self._reset_estimator_from_known_start()
        self.obs_buf = self.observation_manager.compute()
        return self.obs_buf, extras

    def step(self, action: torch.Tensor):
        """Mirror IsaacLab v2.1 RL step with estimator updates inserted."""
        self.action_manager.process_action(action.to(self.device))
        self.recorder_manager.record_pre_step()
        is_rendering = self.sim.has_gui() or self.sim.has_rtx_sensors()

        rendered = False
        for _ in range(self.cfg.decimation):
            self._sim_step_counter += 1
            self.action_manager.apply_action()
            self.scene.write_data_to_sim()
            self.sim.step(render=False)
            rendered = False
            if self._sim_step_counter % self.cfg.sim.render_interval == 0 and is_rendering:
                self.sim.render()
                rendered = True
            self.scene.update(dt=self.physics_dt)
            self._propagate_from_imu()

        self._append_learned_motion_sample()
        self._maybe_update_learned_displacement()
        if rendered:
            self._maybe_gate_update()
        self.learned_inertial_state = self._lio.state()

        self.episode_length_buf += 1
        self.common_step_counter += 1
        self.reset_buf = self.termination_manager.compute()
        self.reset_terminated = self.termination_manager.terminated
        self.reset_time_outs = self.termination_manager.time_outs
        self.reward_buf = self.reward_manager.compute(dt=self.step_dt)

        if len(self.recorder_manager.active_terms) > 0:
            self.obs_buf = self.observation_manager.compute()
            self.recorder_manager.record_post_step()

        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_env_ids) > 0:
            self.recorder_manager.record_pre_reset(reset_env_ids)
            self._reset_idx(reset_env_ids)
            self.scene.write_data_to_sim()
            self.sim.forward()
            if self.sim.has_rtx_sensors() and self.cfg.rerender_on_reset:
                self.sim.render()
            self._reset_estimator_from_known_start()
            self.recorder_manager.record_post_reset(reset_env_ids)

        self.command_manager.compute(dt=self.step_dt)
        if "interval" in self.event_manager.available_modes:
            self.event_manager.apply(mode="interval", dt=self.step_dt)

        self._update_log()
        self.obs_buf = self.observation_manager.compute()
        return self.obs_buf, self.reward_buf, self.reset_terminated, self.reset_time_outs, self.extras
