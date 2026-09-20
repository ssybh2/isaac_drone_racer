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
        self._gate_geometry = None
        self._gate_camera_calibration = None
        self._gate_T_bc = None
        self._gate_track_layout = None
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
        self._last_learned_skip_reason = None
        self._last_learned_update_diagnostics = None
        self._gate_attempt_count = 0
        self._gate_update_count = 0
        self._gate_reject_count = 0
        self._last_gate_mahalanobis2 = None
        self._gate_diagnostics: list[dict[str, object]] = []
        self._gate_stress_rng = None
        self._gate_observation_queue: list[tuple[float, object, dict[str, object]]] = []
        self._gate_stress_frame_drop_count = 0
        self._gate_stress_corner_drop_count = 0
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

        stress_probabilities = (
            float(cfg.gate_stress_frame_drop_probability),
            float(cfg.gate_stress_corner_drop_probability),
        )
        if any(
            not np.isfinite(value) or value < 0.0 or value > 1.0
            for value in stress_probabilities
        ):
            raise ValueError("gate stress probabilities must be finite and in [0, 1]")
        stress_nonnegative = (
            float(cfg.gate_stress_burst_start_s),
            float(cfg.gate_stress_burst_duration_s),
            float(cfg.gate_stress_pixel_noise_sigma_px),
            float(cfg.gate_stress_latency_s),
        )
        if any(
            not np.isfinite(value) or value < 0.0
            for value in stress_nonnegative
        ):
            raise ValueError(
                "gate stress start/duration/noise/latency must be finite and non-negative"
            )

        filter_structure = str(
            getattr(cfg, "learned_filter_structure", "legacy_current_clone")
        )
        if filter_structure != "uzh_two_clone_full":
            raise ValueError(
                "V6.2 learned inertial environment requires "
                "learned_filter_structure='uzh_two_clone_full'"
            )
        if str(cfg.learned_kalman_gain_mode) != "full":
            raise ValueError(
                "V6.2 UZH-style two-clone fusion requires the full Kalman gain; "
                "legacy freeze_* gain masks are not valid runtime configurations"
            )
        oracle_modes = (
            bool(cfg.learned_debug_oracle_residual_fusion),
            bool(cfg.learned_debug_oracle_uzh_displacement_fusion),
            bool(cfg.learned_debug_oracle_body_end_displacement_fusion),
            bool(cfg.learned_debug_oracle_second_difference_fusion),
            bool(cfg.learned_debug_oracle_delta_velocity_fusion),
            bool(cfg.learned_debug_oracle_body_end_delta_velocity_fusion),
        )
        if sum(int(v) for v in oracle_modes) > 1:
            raise ValueError(
                "choose only one Oracle fusion mode: kinematic residual, "
                "exact UZH relative displacement, endpoint-body displacement, "
                "three-clone second difference, world delta velocity, "
                "or endpoint-body delta velocity"
            )

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
        self._last_learned_skip_reason = None
        self._last_learned_update_diagnostics = None
        self._gate_attempt_count = 0
        self._gate_update_count = 0
        self._gate_reject_count = 0
        self._last_gate_mahalanobis2 = None
        self._gate_diagnostics = []
        self._gate_stress_rng = np.random.default_rng(
            int(self.cfg.gate_stress_seed)
        )
        self._gate_observation_queue = []
        self._gate_stress_frame_drop_count = 0
        self._gate_stress_corner_drop_count = 0
        self.learned_inertial_state = self._lio.state()

    def _initialize_detector_and_gate_builder(self) -> None:
        if self.cfg.swift_detector_checkpoint is None:
            return

        from perception.camera_model import CameraCalibration
        from perception.stage2_calibration import load_stage2_gate_geometry, stage2_camera_to_body
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
        geometry = load_stage2_gate_geometry()
        T_bc = stage2_camera_to_body()
        track_layout = track_layout_from_isaac(self, env_id=0)
        self._gate_geometry = geometry
        self._gate_camera_calibration = calibration
        self._gate_T_bc = T_bc
        self._gate_track_layout = track_layout

        measurement_model = str(
            getattr(self.cfg, "gate_measurement_model", "pnp_pose")
        )
        if measurement_model not in ("pnp_pose", "direct_reprojection"):
            raise ValueError(
                "gate_measurement_model must be 'pnp_pose' or "
                "'direct_reprojection'"
            )
        if measurement_model == "pnp_pose":
            from perception.swift_gate_measurement import (
                CornerPerturbationConfig,
                GatePoseMeasurementBuilder,
            )

            self._gate_builder = GatePoseMeasurementBuilder(
                geometry,
                calibration,
                T_bc,
                track_layout,
                perturbation=CornerPerturbationConfig(
                    num_samples=20,
                    corner_sigma_px=self.cfg.swift_corner_sigma_px,
                ),
            )
        else:
            self._gate_builder = None

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

        if (
            bool(self.cfg.learned_debug_oracle_residual_fusion)
            or bool(self.cfg.learned_debug_oracle_uzh_displacement_fusion)
            or bool(self.cfg.learned_debug_oracle_body_end_displacement_fusion)
            or bool(self.cfg.learned_debug_oracle_second_difference_fusion)
            or bool(self.cfg.learned_debug_oracle_delta_velocity_fusion)
            or bool(self.cfg.learned_debug_oracle_body_end_delta_velocity_fusion)
        ):
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
        target_mode = str(
            getattr(self._motion_predictor, "target_mode", "displacement")
        )
        sigma_floor = (
            self.cfg.learned_delta_velocity_sigma_floor_xyz_mps
            if target_mode in (
                "delta_velocity_gravity_compensated",
                "delta_velocity_body_end_gyro_aligned",
            )
            else self.cfg.learned_sigma_floor_xyz_m
        )
        protected_covariance = protected_displacement_covariance(
            prediction.covariance_w,
            sigma_floor_xyz_m=sigma_floor,
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
        if (
            bool(self.cfg.learned_debug_oracle_delta_velocity_fusion)
            or bool(self.cfg.learned_debug_oracle_body_end_delta_velocity_fusion)
        ):
            sigma_v = float(self.cfg.learned_debug_oracle_delta_velocity_sigma_mps)
            if sigma_v <= 0.0 or not np.isfinite(sigma_v):
                raise ValueError(
                    "learned_debug_oracle_delta_velocity_sigma_mps must be positive and finite"
                )
            protected_covariance = np.eye(3, dtype=np.float64) * sigma_v * sigma_v
        if bool(self.cfg.learned_debug_oracle_uzh_displacement_fusion):
            fusion_target_mode = "displacement"
        elif bool(self.cfg.learned_debug_oracle_body_end_displacement_fusion):
            fusion_target_mode = "displacement_body_end"
        elif bool(self.cfg.learned_debug_oracle_second_difference_fusion):
            fusion_target_mode = "second_difference_gravity_compensated"
        elif bool(self.cfg.learned_debug_oracle_delta_velocity_fusion):
            fusion_target_mode = "delta_velocity_gravity_compensated"
        elif bool(self.cfg.learned_debug_oracle_body_end_delta_velocity_fusion):
            fusion_target_mode = "delta_velocity_body_end_gyro_aligned"
        else:
            fusion_target_mode = target_mode
        try:
            if fusion_target_mode == "kinematic_residual":
                predicted_rel = self._lio.predicted_clone_kinematic_residual(
                    start_timestamp_s=start_s,
                    end_timestamp_s=scheduled_end_s,
                    clone_tolerance_s=timing_tolerance_s,
                )
            elif fusion_target_mode in (
                "kinematic_residual_body_end",
                "kinematic_residual_body_end_gyro_aligned",
            ):
                predicted_rel = self._lio.predicted_clone_kinematic_residual_body_end(
                    start_timestamp_s=start_s,
                    end_timestamp_s=scheduled_end_s,
                    clone_tolerance_s=timing_tolerance_s,
                )
            elif fusion_target_mode == "kinematic_residual_body_end_gravity_compensated":
                predicted_rel = (
                    self._lio.predicted_clone_kinematic_residual_body_end_gravity_compensated(
                        start_timestamp_s=start_s,
                        end_timestamp_s=scheduled_end_s,
                        clone_tolerance_s=timing_tolerance_s,
                    )
                )
            elif fusion_target_mode == "displacement":
                predicted_rel = self._lio.predicted_clone_relative_displacement(
                    start_timestamp_s=start_s,
                    end_timestamp_s=scheduled_end_s,
                    clone_tolerance_s=timing_tolerance_s,
                )
            elif fusion_target_mode in (
                "displacement_body_end",
                "displacement_body_end_gyro_aligned",
            ):
                predicted_rel = (
                    self._lio.predicted_clone_relative_displacement_body_end(
                        start_timestamp_s=start_s,
                        end_timestamp_s=scheduled_end_s,
                        clone_tolerance_s=timing_tolerance_s,
                    )
                )
            elif fusion_target_mode == "second_difference_gravity_compensated":
                middle_s = 0.5 * (start_s + scheduled_end_s)
                predicted_rel = (
                    self._lio.predicted_clone_second_difference_gravity_compensated(
                        start_timestamp_s=start_s,
                        middle_timestamp_s=middle_s,
                        end_timestamp_s=scheduled_end_s,
                        clone_tolerance_s=timing_tolerance_s,
                    )
                )
            elif fusion_target_mode == "delta_velocity_gravity_compensated":
                predicted_rel = (
                    self._lio.predicted_clone_delta_velocity_gravity_compensated(
                        start_timestamp_s=start_s,
                        end_timestamp_s=scheduled_end_s,
                        clone_tolerance_s=timing_tolerance_s,
                    )
                )
            elif fusion_target_mode == "delta_velocity_body_end_gyro_aligned":
                predicted_rel = (
                    self._lio.predicted_clone_delta_velocity_body_end_gravity_compensated(
                        start_timestamp_s=start_s,
                        end_timestamp_s=scheduled_end_s,
                        clone_tolerance_s=timing_tolerance_s,
                    )
                )
            else:
                raise RuntimeError(
                    f"unsupported learned-motion fusion target mode: {fusion_target_mode!r}"
                )

            measurement_w = np.asarray(
                prediction.displacement_w, dtype=np.float64
            ).copy()
            network_bias = np.asarray(
                (
                    self.cfg.learned_delta_velocity_network_bias_mps
                    if target_mode in (
                        "delta_velocity_gravity_compensated",
                        "delta_velocity_body_end_gyro_aligned",
                    )
                    else self.cfg.learned_network_bias_m
                ),
                dtype=np.float64,
            ).reshape(3)
            if not np.all(np.isfinite(network_bias)):
                raise ValueError("learned network bias must be finite")
            measurement_w = measurement_w - network_bias
            measurement_source = (
                "network_bias_calibrated"
                if np.any(np.abs(network_bias) > 0.0)
                else "network"
            )
            if bool(self.cfg.learned_debug_oracle_uzh_displacement_fusion):
                start_key = round(float(start_s), 9)
                end_key = round(float(scheduled_end_s), 9)
                if (
                    start_key not in self._debug_truth_motion_history
                    or end_key not in self._debug_truth_motion_history
                ):
                    raise RuntimeError(
                        "UZH displacement Oracle is missing GT history for window"
                    )
                p_start_gt, _ = self._debug_truth_motion_history[start_key]
                p_end_gt, _ = self._debug_truth_motion_history[end_key]
                measurement_w = p_end_gt - p_start_gt
                measurement_source = "oracle_uzh_relative_displacement"

            if bool(self.cfg.learned_debug_oracle_body_end_displacement_fusion):
                start_key = round(float(start_s), 9)
                end_key = round(float(scheduled_end_s), 9)
                if (
                    start_key not in self._debug_truth_motion_history
                    or end_key not in self._debug_truth_motion_history
                ):
                    raise RuntimeError(
                        "endpoint-body displacement Oracle is missing GT history for window"
                    )
                p_start_gt, _ = self._debug_truth_motion_history[start_key]
                p_end_gt, _ = self._debug_truth_motion_history[end_key]
                robot = self.scene["robot"]
                R_end_gt = quat_wxyz_to_rotmat(
                    _np(robot.data.root_quat_w[0]).astype(np.float64)
                )
                measurement_w = R_end_gt.T @ (p_end_gt - p_start_gt)
                measurement_source = "oracle_body_end_relative_displacement"

            if bool(self.cfg.learned_debug_oracle_second_difference_fusion):
                start_key = round(float(start_s), 9)
                middle_s = 0.5 * (start_s + scheduled_end_s)
                middle_key = round(float(middle_s), 9)
                end_key = round(float(scheduled_end_s), 9)
                if (
                    start_key not in self._debug_truth_motion_history
                    or middle_key not in self._debug_truth_motion_history
                    or end_key not in self._debug_truth_motion_history
                ):
                    raise RuntimeError(
                        "second-difference Oracle is missing GT history for window"
                    )
                p_start_gt, _ = self._debug_truth_motion_history[start_key]
                p_middle_gt, _ = self._debug_truth_motion_history[middle_key]
                p_end_gt, _ = self._debug_truth_motion_history[end_key]
                half_dt = float(scheduled_end_s - middle_s)
                measurement_w = (
                    p_end_gt
                    - 2.0 * p_middle_gt
                    + p_start_gt
                    - self._lio.gravity_w * half_dt * half_dt
                )
                measurement_source = "oracle_second_difference_gravity_compensated"

            if bool(self.cfg.learned_debug_oracle_delta_velocity_fusion):
                start_key = round(float(start_s), 9)
                end_key = round(float(scheduled_end_s), 9)
                if (
                    start_key not in self._debug_truth_motion_history
                    or end_key not in self._debug_truth_motion_history
                ):
                    raise RuntimeError(
                        "delta-velocity Oracle is missing GT history for window"
                    )
                _, v_start_gt = self._debug_truth_motion_history[start_key]
                _, v_end_gt = self._debug_truth_motion_history[end_key]
                window_dt = float(scheduled_end_s - start_s)
                measurement_w = (
                    v_end_gt
                    - v_start_gt
                    - self._lio.gravity_w * window_dt
                )
                measurement_source = "oracle_delta_velocity_gravity_compensated"

            if bool(self.cfg.learned_debug_oracle_body_end_delta_velocity_fusion):
                start_key = round(float(start_s), 9)
                end_key = round(float(scheduled_end_s), 9)
                if (
                    start_key not in self._debug_truth_motion_history
                    or end_key not in self._debug_truth_motion_history
                ):
                    raise RuntimeError(
                        "endpoint-body delta-velocity Oracle is missing GT history for window"
                    )
                _, v_start_gt = self._debug_truth_motion_history[start_key]
                _, v_end_gt = self._debug_truth_motion_history[end_key]
                window_dt = float(scheduled_end_s - start_s)
                delta_v_specific_w = (
                    v_end_gt
                    - v_start_gt
                    - self._lio.gravity_w * window_dt
                )
                robot = self.scene["robot"]
                R_end_gt = quat_wxyz_to_rotmat(
                    _np(robot.data.root_quat_w[0]).astype(np.float64)
                )
                measurement_w = R_end_gt.T @ delta_v_specific_w
                measurement_source = (
                    "oracle_body_end_delta_velocity_gravity_compensated"
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
                if fusion_target_mode == "kinematic_residual":
                    self._lio.update_learned_clone_kinematic_residual(
                        measurement_w,
                        protected_covariance,
                        start_timestamp_s=start_s,
                        end_timestamp_s=scheduled_end_s,
                        clone_tolerance_s=timing_tolerance_s,
                        marginalize_start_clone=True,
                    )
                elif fusion_target_mode in (
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
                elif fusion_target_mode == "kinematic_residual_body_end_gravity_compensated":
                    self._lio.update_learned_clone_kinematic_residual_body_end_gravity_compensated(
                        measurement_w,
                        protected_covariance,
                        start_timestamp_s=start_s,
                        end_timestamp_s=scheduled_end_s,
                        clone_tolerance_s=timing_tolerance_s,
                        marginalize_start_clone=True,
                    )
                elif fusion_target_mode in (
                    "displacement_body_end",
                    "displacement_body_end_gyro_aligned",
                ):
                    self._lio.update_learned_clone_displacement_body_end(
                        measurement_w,
                        protected_covariance,
                        start_timestamp_s=start_s,
                        end_timestamp_s=scheduled_end_s,
                        clone_tolerance_s=timing_tolerance_s,
                        marginalize_start_clone=True,
                    )
                elif fusion_target_mode == "second_difference_gravity_compensated":
                    middle_s = 0.5 * (start_s + scheduled_end_s)
                    self._lio.update_learned_clone_second_difference_gravity_compensated(
                        measurement_w,
                        protected_covariance,
                        start_timestamp_s=start_s,
                        middle_timestamp_s=middle_s,
                        end_timestamp_s=scheduled_end_s,
                        clone_tolerance_s=timing_tolerance_s,
                        marginalize_start_clone=True,
                    )
                elif fusion_target_mode == "delta_velocity_gravity_compensated":
                    self._lio.update_learned_clone_delta_velocity_gravity_compensated(
                        measurement_w,
                        protected_covariance,
                        start_timestamp_s=start_s,
                        end_timestamp_s=scheduled_end_s,
                        clone_tolerance_s=timing_tolerance_s,
                        marginalize_start_clone=True,
                    )
                elif fusion_target_mode == "delta_velocity_body_end_gyro_aligned":
                    self._lio.update_learned_clone_delta_velocity_body_end_gravity_compensated(
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
                # Preserve this learned-factor diagnostic before the
                # subsequent visual update overwrites LIO.last_update_diagnostics.
                self._last_learned_update_diagnostics = (
                    None
                    if self._lio.last_update_diagnostics is None
                    else self._lio.last_update_diagnostics.copy()
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
        except (KeyError, RuntimeError) as exc:
            self._learned_update_skip_count += 1
            self._last_learned_skip_reason = f"{type(exc).__name__}: {exc}"
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
        self._last_learned_skip_reason = None
        # discard_before retains one interpolation predecessor. The next
        # 0.5 s window begins only 0.05 s later, so the histories overlap.
        self._motion_buffer.discard_before(start_s)
        self.learned_inertial_state = self._lio.state()

    def _camera_due(self) -> bool:
        now = self._timestamp_s()
        return now - self._last_camera_timestamp_s >= (1.0 / 30.0) - 1.0e-9

    def _maybe_gate_reprojection_update(self) -> None:
        """Fuse 2-D gate corners directly against the known 3-D gate map.

        V6.4 deliberately does not call IPPE/PnP on this path. Gate identity is
        selected by projecting every mapped gate through the current inertial
        state and choosing the smallest semantic-corner pixel residual.

        V6.5 can inject deterministic frame loss, burst dropout, corner erasure,
        pixel noise and uncompensated processing latency before this update.
        All stress hooks are neutral by default.
        """
        if (
            self.swift_detector is None
            or self._gate_geometry is None
            or self._gate_camera_calibration is None
            or self._gate_T_bc is None
            or self._gate_track_layout is None
            or not self._camera_due()
        ):
            return

        now = float(self._timestamp_s())
        self._last_camera_timestamp_s = now
        self._gate_attempt_count += 1
        capture_diagnostic: dict[str, object] = {
            "attempt_index": int(self._gate_attempt_count),
            "timestamp_s": now,
            "capture_timestamp_s": now,
            "process_timestamp_s": None,
            "measurement_age_s": None,
            "measurement_model": "direct_reprojection",
            "accepted": False,
            "reject_stage": None,
            "reject_reason": None,
            "expected_active_gate_index": None,
            "selected_gate_index": None,
            "association_matches_active_gate": None,
            "visible_corner_count": 0,
            "visible_corner_indices": None,
            "association_pixel_rmse_px": None,
            "association_second_best_rmse_px": None,
            "pixel_residual_rmse_px": None,
            "pixel_residual_radial_rmse_px": None,
            "pixel_sigma_px": float(self.cfg.gate_reprojection_sigma_px),
            "huber_weights": None,
            "corner_normalized_innovation": None,
            "normalized_nis": None,
            "pre_update_position_error_norm_m": None,
            "post_update_position_error_norm_m": None,
            "pre_update_orientation_error_deg": None,
            "post_update_orientation_error_deg": None,
            "exception_type": None,
            "exception_message": None,
            "stress_frame_dropped": False,
            "stress_burst_active": False,
            "stress_pixel_noise_sigma_px": float(
                self.cfg.gate_stress_pixel_noise_sigma_px
            ),
            "stress_corner_drop_count": 0,
            "stress_latency_s": float(self.cfg.gate_stress_latency_s),
            "latency_compensation_enabled": bool(
                self.cfg.gate_reprojection_compensate_latency
            ),
            "latency_compensation_used": False,
            "capture_clone_timestamp_s": None,
        }

        try:
            from perception.swift_isaac_adapter import active_gate_index_from_isaac

            capture_diagnostic["expected_active_gate_index"] = int(
                active_gate_index_from_isaac(self, env_id=0)
            )
        except (AttributeError, KeyError, RuntimeError, ValueError):
            pass

        burst_start = float(self.cfg.gate_stress_burst_start_s)
        burst_duration = float(self.cfg.gate_stress_burst_duration_s)
        burst_active = (
            burst_duration > 0.0
            and burst_start <= now < burst_start + burst_duration
        )
        random_drop = bool(
            float(self.cfg.gate_stress_frame_drop_probability) > 0.0
            and self._gate_stress_rng.random()
            < float(self.cfg.gate_stress_frame_drop_probability)
        )
        frame_dropped = bool(burst_active or random_drop)
        capture_diagnostic["stress_burst_active"] = bool(burst_active)
        capture_diagnostic["stress_frame_dropped"] = frame_dropped

        if frame_dropped:
            self._gate_reject_count += 1
            self._gate_stress_frame_drop_count += 1
            capture_diagnostic["reject_stage"] = "stress"
            capture_diagnostic["reject_reason"] = "stress_frame_dropout"
            self._gate_diagnostics.append(capture_diagnostic)
        else:
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
                    timestamp_s=now,
                )
            except (ValueError, RuntimeError) as exc:
                self._gate_reject_count += 1
                capture_diagnostic["reject_stage"] = "detector"
                capture_diagnostic["reject_reason"] = "detector_exception"
                capture_diagnostic["exception_type"] = type(exc).__name__
                capture_diagnostic["exception_message"] = str(exc)
                self._gate_diagnostics.append(capture_diagnostic)
            else:
                from perception.corner_detection import CornerObservation

                corners_uv = np.asarray(
                    observation.corners_uv, dtype=np.float64
                ).copy()
                visible = np.asarray(
                    observation.visible, dtype=bool
                ).reshape(4).copy()
                confidence = np.asarray(
                    observation.confidence, dtype=np.float64
                ).reshape(4).copy()

                pixel_noise_sigma = float(
                    self.cfg.gate_stress_pixel_noise_sigma_px
                )
                if pixel_noise_sigma > 0.0:
                    corners_uv += self._gate_stress_rng.normal(
                        loc=0.0,
                        scale=pixel_noise_sigma,
                        size=corners_uv.shape,
                    )

                corner_drop_probability = float(
                    self.cfg.gate_stress_corner_drop_probability
                )
                if corner_drop_probability > 0.0:
                    drop_draw = (
                        self._gate_stress_rng.random(4)
                        < corner_drop_probability
                    )
                    effective_drop = visible & drop_draw
                    dropped_count = int(np.sum(effective_drop))
                    if dropped_count:
                        visible[effective_drop] = False
                        self._gate_stress_corner_drop_count += dropped_count
                        capture_diagnostic["stress_corner_drop_count"] = (
                            dropped_count
                        )

                stressed_observation = CornerObservation(
                    corners_uv=corners_uv,
                    visible=visible,
                    confidence=confidence,
                    timestamp_s=now,
                    source=f"{observation.source}+v6.5_stress",
                )

                latency_s = float(self.cfg.gate_stress_latency_s)
                if (
                    latency_s > 0.0
                    and bool(self.cfg.gate_reprojection_compensate_latency)
                ):
                    timestamps = self._lio.clone_timestamps_s
                    tolerance = float(
                        self.cfg.gate_reprojection_latency_clone_tolerance_s
                    )
                    if (
                        not timestamps
                        or abs(float(timestamps[-1]) - now) > tolerance
                    ):
                        self._lio.clone_current_position()
                    capture_diagnostic["latency_compensation_used"] = True
                    capture_diagnostic["capture_clone_timestamp_s"] = now

                self._gate_observation_queue.append(
                    (now, stressed_observation, capture_diagnostic)
                )

        latency_s = float(self.cfg.gate_stress_latency_s)
        if not self._gate_observation_queue:
            return
        capture_time_s, observation, diagnostic = self._gate_observation_queue[0]
        if now - capture_time_s + 1.0e-12 < latency_s:
            return
        self._gate_observation_queue.pop(0)
        diagnostic["process_timestamp_s"] = now
        diagnostic["measurement_age_s"] = float(now - capture_time_s)

        visible_indices = np.flatnonzero(
            np.asarray(observation.visible, dtype=bool).reshape(4)
        )
        diagnostic["visible_corner_count"] = int(len(visible_indices))
        diagnostic["visible_corner_indices"] = visible_indices.tolist()
        min_visible = int(self.cfg.gate_reprojection_min_visible_corners)
        if len(visible_indices) < min_visible:
            self._gate_reject_count += 1
            diagnostic["reject_stage"] = "visibility"
            diagnostic["reject_reason"] = "insufficient_visible_corners"
            self._gate_diagnostics.append(diagnostic)
            return

        observed_uv = np.asarray(
            observation.corners_uv, dtype=np.float64
        )[visible_indices]
        K = np.asarray(
            self._gate_camera_calibration.K, dtype=np.float64
        ).reshape(3, 3)
        R_bc = np.asarray(self._gate_T_bc.R, dtype=np.float64).reshape(3, 3)
        t_bc = np.asarray(self._gate_T_bc.t, dtype=np.float64).reshape(3)

        candidates: list[tuple[float, int, np.ndarray]] = []
        for gate_index in range(self._gate_track_layout.num_gates):
            T_wg = self._gate_track_layout.gate_pose(gate_index)
            all_points_w = T_wg.transform_points(
                self._gate_geometry.object_points_g
            )
            points_w = np.asarray(all_points_w, dtype=np.float64)[visible_indices]
            try:
                if bool(diagnostic.get("latency_compensation_used", False)):
                    predicted_uv, _ = (
                        self._lio.predict_gate_corner_reprojection_at_clone(
                            points_w,
                            K,
                            R_bc,
                            t_bc,
                            clone_timestamp_s=float(capture_time_s),
                            clone_tolerance_s=float(
                                self.cfg.gate_reprojection_latency_clone_tolerance_s
                            ),
                            min_depth_m=float(
                                self.cfg.gate_reprojection_min_depth_m
                            ),
                        )
                    )
                else:
                    predicted_uv, _ = self._lio.predict_gate_corner_reprojection(
                        points_w,
                        K,
                        R_bc,
                        t_bc,
                        min_depth_m=float(
                            self.cfg.gate_reprojection_min_depth_m
                        ),
                    )
            except (KeyError, ValueError, RuntimeError):
                continue
            residual = observed_uv - predicted_uv
            score = float(
                np.sqrt(
                    np.mean(np.sum(residual * residual, axis=1))
                )
            )
            if np.isfinite(score):
                candidates.append((score, gate_index, points_w))

        if not candidates:
            self._gate_reject_count += 1
            diagnostic["reject_stage"] = "association"
            diagnostic["reject_reason"] = "no_projectable_mapped_gate"
            self._gate_diagnostics.append(diagnostic)
            return

        candidates.sort(key=lambda item: item[0])
        association_rmse, gate_index, points_w = candidates[0]
        diagnostic["association_pixel_rmse_px"] = association_rmse
        diagnostic["association_second_best_rmse_px"] = (
            None if len(candidates) < 2 else float(candidates[1][0])
        )
        diagnostic["selected_gate_index"] = int(gate_index)
        expected_gate = diagnostic["expected_active_gate_index"]
        if expected_gate is not None:
            diagnostic["association_matches_active_gate"] = bool(
                int(gate_index) == int(expected_gate)
            )

        if association_rmse > float(
            self.cfg.gate_reprojection_association_max_rmse_px
        ):
            self._gate_reject_count += 1
            diagnostic["reject_stage"] = "association"
            diagnostic["reject_reason"] = "pixel_association_gate"
            self._gate_diagnostics.append(diagnostic)
            return

        gt_position = None
        gt_R = None
        if bool(self.cfg.gate_debug_gt_diagnostics):
            robot = self.scene["robot"]
            gt_position = _np(robot.data.root_pos_w[0]).astype(np.float64).copy()
            gt_quat = _np(robot.data.root_quat_w[0]).astype(np.float64).copy()
            gt_R = quat_wxyz_to_rotmat(gt_quat)
            diagnostic["pre_update_position_error_norm_m"] = float(
                np.linalg.norm(self._lio.p - gt_position)
            )
            relative_R = self._lio.R @ gt_R.T
            cosine = float(
                np.clip((np.trace(relative_R) - 1.0) * 0.5, -1.0, 1.0)
            )
            diagnostic["pre_update_orientation_error_deg"] = float(
                np.degrees(np.arccos(cosine))
            )

        try:
            residual_matrix = self._lio.update_gate_corner_reprojection(
                observed_uv,
                points_w,
                K,
                R_bc,
                t_bc,
                pixel_sigma_px=float(
                    self.cfg.gate_reprojection_sigma_px
                ),
                huber_delta_sigma=float(
                    self.cfg.gate_reprojection_huber_delta_sigma
                ),
                min_depth_m=float(
                    self.cfg.gate_reprojection_min_depth_m
                ),
                max_normalized_nis=(
                    None
                    if self.cfg.gate_reprojection_max_normalized_nis is None
                    else float(
                        self.cfg.gate_reprojection_max_normalized_nis
                    )
                ),
                clone_timestamp_s=(
                    float(capture_time_s)
                    if bool(diagnostic.get("latency_compensation_used", False))
                    else None
                ),
                clone_tolerance_s=float(
                    self.cfg.gate_reprojection_latency_clone_tolerance_s
                ),
            )
        except (KeyError, ValueError, RuntimeError) as exc:
            self._gate_reject_count += 1
            message = str(exc)
            diagnostic["reject_stage"] = "reprojection_update"
            diagnostic["reject_reason"] = (
                "reprojection_nis_gate"
                if "normalized NIS" in message
                else "reprojection_update_failure"
            )
            diagnostic["exception_type"] = type(exc).__name__
            diagnostic["exception_message"] = message
            self._gate_diagnostics.append(diagnostic)
            return

        update_diag = self._lio.last_update_diagnostics or {}
        diagnostic["pixel_residual_rmse_px"] = float(
            np.sqrt(np.mean(residual_matrix * residual_matrix))
        )
        diagnostic["pixel_residual_radial_rmse_px"] = float(
            np.sqrt(
                np.mean(
                    np.sum(
                        residual_matrix * residual_matrix,
                        axis=1,
                    )
                )
            )
        )
        weights = update_diag.get("huber_weights")
        if weights is not None:
            diagnostic["huber_weights"] = np.asarray(
                weights, dtype=np.float64
            ).tolist()
        normalized_corner = update_diag.get("corner_normalized_innovation")
        if normalized_corner is not None:
            diagnostic["corner_normalized_innovation"] = np.asarray(
                normalized_corner, dtype=np.float64
            ).tolist()
        if update_diag.get("preupdate_normalized_nis") is not None:
            diagnostic["normalized_nis"] = float(
                update_diag["preupdate_normalized_nis"]
            )

        self._gate_update_count += 1
        self.learned_inertial_state = self._lio.state()
        diagnostic["accepted"] = True

        if gt_position is not None and gt_R is not None:
            diagnostic["post_update_position_error_norm_m"] = float(
                np.linalg.norm(self._lio.p - gt_position)
            )
            relative_R = self._lio.R @ gt_R.T
            cosine = float(
                np.clip((np.trace(relative_R) - 1.0) * 0.5, -1.0, 1.0)
            )
            diagnostic["post_update_orientation_error_deg"] = float(
                np.degrees(np.arccos(cosine))
            )

        self._gate_diagnostics.append(diagnostic)

    def _maybe_gate_update(self) -> None:
        if str(getattr(self.cfg, "gate_measurement_model", "pnp_pose")) == "direct_reprojection":
            self._maybe_gate_reprojection_update()
            return
        if self.swift_detector is None or self._gate_builder is None or not self._camera_due():
            return
        self._last_camera_timestamp_s = self._timestamp_s()
        self._gate_attempt_count += 1

        # Keep the deployed estimator GT-free by default. The optional GT fields
        # below are evaluation-only diagnostics and never feed the EKF update.
        diagnostic: dict[str, object] = {
            "attempt_index": int(self._gate_attempt_count),
            "timestamp_s": float(self._timestamp_s()),
            "accepted": False,
            "reject_stage": None,
            "reject_reason": None,
            "expected_active_gate_index": None,
            "selected_gate_index": None,
            "association_matches_active_gate": None,
            "reprojection_rmse_px": None,
            "camera_to_gate_range_m": None,
            "position_covariance_diag_m2": None,
            "position_sigma_xyz_m": None,
            "position_covariance_eigenvalues_m2": None,
            "position_nees_vs_gt": None,
            "pnp_position_error_w_m": None,
            "pnp_position_error_norm_m": None,
            "pnp_orientation_error_deg": None,
            "mahalanobis2": None,
            "pre_update_position_error_norm_m": None,
            "post_update_position_error_norm_m": None,
            "pre_update_orientation_error_deg": None,
            "post_update_orientation_error_deg": None,
            "exception_type": None,
            "exception_message": None,
            "oracle_corner_complete": None,
            "detector_corner_error_px_by_corner": None,
            "detector_corner_residual_xy_px": None,
            "detector_corner_error_px_rmse": None,
            "detector_corner_error_px_mean": None,
            "detector_corner_error_px_max": None,
            "detector_gate_width_error_px": None,
            "detector_gate_height_error_px": None,
            "detector_gate_centroid_error_xy_px": None,
            "oracle_pnp_reprojection_rmse_px": None,
            "oracle_pnp_gate_translation_error_m": None,
            "oracle_pnp_gate_rotation_error_deg": None,
            "oracle_pnp_body_translation_error_m": None,
            "oracle_pnp_body_rotation_error_deg": None,
            "oracle_extrinsic_translation_error_m": None,
            "oracle_extrinsic_rotation_error_deg": None,
            "detector_pnp_gate_translation_error_m": None,
            "detector_pnp_gate_rotation_error_deg": None,
            "detector_pnp_body_position_error_full_pose_m": None,
            "detector_pnp_body_position_error_gt_attitude_m": None,
            "detector_pnp_body_position_error_ekf_attitude_m": None,
        }

        try:
            from perception.swift_isaac_adapter import active_gate_index_from_isaac

            diagnostic["expected_active_gate_index"] = int(
                active_gate_index_from_isaac(self, env_id=0)
            )
        except (AttributeError, KeyError, RuntimeError, ValueError):
            # Active-gate identity is mission-state metadata, not required for
            # perception or filtering. Keep the audit running if unavailable.
            pass

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
        except (ValueError, RuntimeError) as exc:
            self._gate_reject_count += 1
            diagnostic["reject_stage"] = "detector"
            diagnostic["reject_reason"] = "detector_exception"
            diagnostic["exception_type"] = type(exc).__name__
            diagnostic["exception_message"] = str(exc)
            self._gate_diagnostics.append(diagnostic)
            return

        audit_snapshot = None
        audit_T_cg_truth = None
        if bool(self.cfg.gate_debug_gt_diagnostics):
            # Decisive Stage2 decomposition on the exact live frame:
            # simulator truth -> perfect projected pixels -> the same PnP stack.
            # This is audit-only. Neither oracle pixels nor oracle poses are
            # ever supplied to the runtime estimator update.
            try:
                from perception.isaac_adapter import IsaacStage2TruthAdapter
                from perception.stage2a_pipeline import Stage2APerceptionPipeline

                snapshot = IsaacStage2TruthAdapter(self).snapshot(
                    env_id=0,
                    timestamp_s=self._timestamp_s(),
                )
                audit_snapshot = snapshot
                oracle_pipeline = Stage2APerceptionPipeline(
                    self._gate_builder.geometry,
                    snapshot.camera,
                    self._gate_builder.T_bc,
                )
                T_cg_truth = snapshot.truth.T_wc.inverse() @ snapshot.truth.T_wg
                audit_T_cg_truth = T_cg_truth
                oracle_corners = oracle_pipeline.project_perfect_corners(
                    T_cg_truth,
                    timestamp_s=self._timestamp_s(),
                )
                diagnostic["oracle_corner_complete"] = bool(oracle_corners.complete)

                detector_corners = np.asarray(
                    observation.corners_uv, dtype=np.float64
                )
                oracle_corner_uv = np.asarray(
                    oracle_corners.corners_uv, dtype=np.float64
                )
                corner_residual_xy = detector_corners - oracle_corner_uv
                corner_errors_px = np.linalg.norm(corner_residual_xy, axis=1)
                diagnostic["detector_corner_error_px_by_corner"] = (
                    corner_errors_px.tolist()
                )
                diagnostic["detector_corner_residual_xy_px"] = (
                    corner_residual_xy.tolist()
                )

                def _quad_width_height_centroid(corners):
                    width = 0.5 * (
                        np.linalg.norm(corners[1] - corners[0])
                        + np.linalg.norm(corners[2] - corners[3])
                    )
                    height = 0.5 * (
                        np.linalg.norm(corners[3] - corners[0])
                        + np.linalg.norm(corners[2] - corners[1])
                    )
                    return float(width), float(height), np.mean(corners, axis=0)

                det_w, det_h, det_center = _quad_width_height_centroid(
                    detector_corners
                )
                ora_w, ora_h, ora_center = _quad_width_height_centroid(
                    oracle_corner_uv
                )
                diagnostic["detector_gate_width_error_px"] = float(det_w - ora_w)
                diagnostic["detector_gate_height_error_px"] = float(det_h - ora_h)
                diagnostic["detector_gate_centroid_error_xy_px"] = (
                    det_center - ora_center
                ).tolist()
                diagnostic["detector_corner_error_px_rmse"] = float(
                    np.sqrt(np.mean(corner_errors_px**2))
                )
                diagnostic["detector_corner_error_px_mean"] = float(
                    np.mean(corner_errors_px)
                )
                diagnostic["detector_corner_error_px_max"] = float(
                    np.max(corner_errors_px)
                )

                if oracle_corners.complete:
                    oracle_result = oracle_pipeline.process_truth(snapshot.truth)
                    metrics = oracle_result.metrics
                    diagnostic["oracle_pnp_reprojection_rmse_px"] = float(
                        metrics.corner_reprojection_rmse_px
                    )
                    diagnostic["oracle_pnp_gate_translation_error_m"] = float(
                        metrics.gate_translation_error_m
                    )
                    diagnostic["oracle_pnp_gate_rotation_error_deg"] = float(
                        np.degrees(metrics.gate_rotation_error_rad)
                    )
                    if metrics.body_translation_error_m is not None:
                        diagnostic["oracle_pnp_body_translation_error_m"] = float(
                            metrics.body_translation_error_m
                        )
                    if metrics.body_rotation_error_rad is not None:
                        diagnostic["oracle_pnp_body_rotation_error_deg"] = float(
                            np.degrees(metrics.body_rotation_error_rad)
                        )
                    if metrics.extrinsic_translation_error_m is not None:
                        diagnostic["oracle_extrinsic_translation_error_m"] = float(
                            metrics.extrinsic_translation_error_m
                        )
                    if metrics.extrinsic_rotation_error_rad is not None:
                        diagnostic["oracle_extrinsic_rotation_error_deg"] = float(
                            np.degrees(metrics.extrinsic_rotation_error_rad)
                        )
            except (AttributeError, KeyError, RuntimeError, ValueError, np.linalg.LinAlgError) as exc:
                diagnostic["oracle_corner_audit_exception_type"] = type(exc).__name__
                diagnostic["oracle_corner_audit_exception_message"] = str(exc)

        try:
            measurement = self._gate_builder.build(
                observation,
                gate_index=None,
                reference_position_w_b=self._lio.p,
            )
        except (ValueError, RuntimeError) as exc:
            self._gate_reject_count += 1
            message = str(exc)
            message_lower = message.lower()
            if "reprojection" in message_lower:
                reason = "reprojection_gate"
            elif "all four visible corners" in message_lower:
                reason = "incomplete_corners"
            elif "confidence" in message_lower:
                reason = "corner_confidence"
            elif "perturbed ippe" in message_lower:
                reason = "perturbation_support"
            elif "ippe" in message_lower or "pnp" in message_lower:
                reason = "pnp_failure"
            else:
                reason = "measurement_builder_exception"
            diagnostic["reject_stage"] = "measurement_builder"
            diagnostic["reject_reason"] = reason
            diagnostic["exception_type"] = type(exc).__name__
            diagnostic["exception_message"] = message
            self._gate_diagnostics.append(diagnostic)
            return

        diagnostic["selected_gate_index"] = int(measurement.gate_index)

        if audit_snapshot is not None and audit_T_cg_truth is not None:
            gate_translation_error = (
                np.asarray(measurement.T_cg.t, dtype=np.float64)
                - np.asarray(audit_T_cg_truth.t, dtype=np.float64)
            )
            diagnostic["detector_pnp_gate_translation_error_m"] = float(
                np.linalg.norm(gate_translation_error)
            )
            gate_relative_R = measurement.T_cg.R @ audit_T_cg_truth.R.T
            gate_cosine = float(
                np.clip((np.trace(gate_relative_R) - 1.0) * 0.5, -1.0, 1.0)
            )
            diagnostic["detector_pnp_gate_rotation_error_deg"] = float(
                np.degrees(np.arccos(gate_cosine))
            )

            gt_body_position = np.asarray(
                audit_snapshot.truth.T_wb.t, dtype=np.float64
            )
            diagnostic["detector_pnp_body_position_error_full_pose_m"] = float(
                np.linalg.norm(
                    np.asarray(measurement.position_w_b, dtype=np.float64)
                    - gt_body_position
                )
            )

            # Position reconstruction using only PnP translation t_cg, with
            # either GT attitude (diagnostic lower bound) or the pre-update EKF
            # attitude (deployable formulation). This isolates the range-lever
            # amplification caused by noisy planar-PnP orientation.
            p_wg = np.asarray(audit_snapshot.truth.T_wg.t, dtype=np.float64)
            t_cg = np.asarray(measurement.T_cg.t, dtype=np.float64)
            R_bc = np.asarray(self._gate_builder.T_bc.R, dtype=np.float64)
            t_bc = np.asarray(self._gate_builder.T_bc.t, dtype=np.float64)

            def _body_position_from_attitude(R_wb):
                R_wb = np.asarray(R_wb, dtype=np.float64).reshape(3, 3)
                R_wc = R_wb @ R_bc
                p_wc = p_wg - R_wc @ t_cg
                return p_wc - R_wb @ t_bc

            p_wb_gt_attitude = _body_position_from_attitude(
                audit_snapshot.truth.T_wb.R
            )
            p_wb_ekf_attitude = _body_position_from_attitude(self._lio.R)
            diagnostic["detector_pnp_body_position_error_gt_attitude_m"] = float(
                np.linalg.norm(p_wb_gt_attitude - gt_body_position)
            )
            diagnostic["detector_pnp_body_position_error_ekf_attitude_m"] = float(
                np.linalg.norm(p_wb_ekf_attitude - gt_body_position)
            )

        expected_gate = diagnostic["expected_active_gate_index"]
        if expected_gate is not None:
            diagnostic["association_matches_active_gate"] = bool(
                int(measurement.gate_index) == int(expected_gate)
            )
        diagnostic["reprojection_rmse_px"] = float(
            measurement.nominal_pnp.reprojection_rmse_px
        )
        diagnostic["camera_to_gate_range_m"] = float(
            np.linalg.norm(measurement.T_cg.t)
        )
        covariance_w = np.asarray(
            measurement.position_covariance_w, dtype=np.float64
        ).reshape(3, 3)
        covariance_w = 0.5 * (covariance_w + covariance_w.T)
        diagnostic["position_covariance_diag_m2"] = np.diag(covariance_w).tolist()
        diagnostic["position_sigma_xyz_m"] = np.sqrt(
            np.maximum(np.diag(covariance_w), 0.0)
        ).tolist()
        diagnostic["position_covariance_eigenvalues_m2"] = np.linalg.eigvalsh(
            covariance_w
        ).tolist()

        gt_position = None
        gt_R = None
        if bool(self.cfg.gate_debug_gt_diagnostics):
            robot = self.scene["robot"]
            gt_position = _np(robot.data.root_pos_w[0]).astype(np.float64).copy()
            gt_quat = _np(robot.data.root_quat_w[0]).astype(np.float64).copy()
            gt_R = quat_wxyz_to_rotmat(gt_quat)

            pnp_error_w = np.asarray(
                measurement.position_w_b, dtype=np.float64
            ) - gt_position
            diagnostic["pnp_position_error_w_m"] = pnp_error_w.tolist()
            diagnostic["pnp_position_error_norm_m"] = float(
                np.linalg.norm(pnp_error_w)
            )
            try:
                diagnostic["position_nees_vs_gt"] = float(
                    pnp_error_w.T @ np.linalg.solve(covariance_w, pnp_error_w)
                )
            except np.linalg.LinAlgError:
                diagnostic["position_nees_vs_gt"] = None

            relative_R = measurement.T_wb_gate.R @ gt_R.T
            cosine = float(
                np.clip((np.trace(relative_R) - 1.0) * 0.5, -1.0, 1.0)
            )
            diagnostic["pnp_orientation_error_deg"] = float(
                np.degrees(np.arccos(cosine))
            )

            diagnostic["pre_update_position_error_norm_m"] = float(
                np.linalg.norm(self._lio.p - gt_position)
            )
            pre_relative_R = self._lio.R @ gt_R.T
            pre_cosine = float(
                np.clip((np.trace(pre_relative_R) - 1.0) * 0.5, -1.0, 1.0)
            )
            diagnostic["pre_update_orientation_error_deg"] = float(
                np.degrees(np.arccos(pre_cosine))
            )

        # Reject a visually plausible but globally inconsistent gate association.
        innovation_p = measurement.position_w_b - self._lio.p
        Ppp = self._lio.P[6:9, 6:9]
        S = Ppp + covariance_w
        try:
            d2 = float(innovation_p.T @ np.linalg.solve(S, innovation_p))
        except np.linalg.LinAlgError as exc:
            self._gate_reject_count += 1
            diagnostic["reject_stage"] = "mahalanobis"
            diagnostic["reject_reason"] = "innovation_covariance_singular"
            diagnostic["exception_type"] = type(exc).__name__
            diagnostic["exception_message"] = str(exc)
            self._gate_diagnostics.append(diagnostic)
            return
        self._last_gate_mahalanobis2 = d2
        diagnostic["mahalanobis2"] = d2
        if d2 > float(self.cfg.gate_position_mahalanobis2_max):
            self._gate_reject_count += 1
            diagnostic["reject_stage"] = "mahalanobis"
            diagnostic["reject_reason"] = "position_mahalanobis_gate"
            self._gate_diagnostics.append(diagnostic)
            return

        # Gate PnP + mapped T_wg is an absolute body-pose observation.
        q_wb = rotmat_to_quat_wxyz(measurement.T_wb_gate.R)
        sigma_rad = np.deg2rad(float(self.cfg.gate_orientation_sigma_deg))
        self._lio.update_gate_pose(
            position_w_b=measurement.position_w_b,
            position_covariance_w=covariance_w,
            orientation_w_b_wxyz=(
                q_wb if bool(self.cfg.gate_use_orientation_update) else None
            ),
            orientation_covariance_rad2=np.eye(3) * sigma_rad * sigma_rad,
        )
        self._last_gate_measurement = measurement
        self._gate_update_count += 1
        self.learned_inertial_state = self._lio.state()

        diagnostic["accepted"] = True
        if gt_position is not None and gt_R is not None:
            diagnostic["post_update_position_error_norm_m"] = float(
                np.linalg.norm(self._lio.p - gt_position)
            )
            post_relative_R = self._lio.R @ gt_R.T
            post_cosine = float(
                np.clip((np.trace(post_relative_R) - 1.0) * 0.5, -1.0, 1.0)
            )
            diagnostic["post_update_orientation_error_deg"] = float(
                np.degrees(np.arccos(post_cosine))
            )
        self._gate_diagnostics.append(diagnostic)

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
