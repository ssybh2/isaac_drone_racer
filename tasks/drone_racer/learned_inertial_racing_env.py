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
        self._window_start_s: float | None = None
        self._next_learned_update_s: float | None = None
        self.swift_detector = None
        self._gate_builder = None
        self._last_camera_timestamp_s = -np.inf
        self._last_gate_measurement = None
        self._learned_update_count = 0
        self._gate_update_count = 0
        super().__init__(cfg=cfg, render_mode=render_mode, **kwargs)
        if self.num_envs != 1:
            raise ValueError("LearnedInertialRacingEnv currently requires num_envs=1")

        self._lio = LearnedInertialOdometry()
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
        self._window_start_s = None
        self._next_learned_update_s = None
        self._last_camera_timestamp_s = -np.inf
        self._last_gate_measurement = None
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
        imu = self.scene["imu"]
        self._lio.propagate(
            gyro_b=_np(imu.data.ang_vel_b[0]),
            accel_b=_np(imu.data.lin_acc_b[0]),
            timestamp_s=timestamp_s,
        )
        self.learned_inertial_state = self._lio.state()

    def _append_learned_motion_sample(self) -> None:
        """Append paper-style world-frame gyro + mass-normalized thrust."""
        timestamp_s = self._timestamp_s()
        imu = self.scene["imu"]
        gyro_b = _np(imu.data.ang_vel_b[0]).astype(np.float64)

        # Allocation output channel 0 is collective force [N]. The FIVE_IN_DRONE
        # model mass is 0.6076 kg (same value used to derive thrust coefficient).
        control = self.action_manager.get_term("control_action")
        collective_force_n = float(_np(control.processed_actions[0])[0])
        mass_kg = 0.6076
        thrust_b = np.array([0.0, 0.0, collective_force_n / mass_kg], dtype=np.float64)

        R_wb = self._lio.R
        self._motion_buffer.append(
            timestamp_s,
            gyro_w=R_wb @ gyro_b,
            thrust_w=R_wb @ thrust_b,
        )

    def _maybe_update_learned_displacement(self) -> None:
        if self._motion_predictor is None:
            return
        now = self._timestamp_s()
        window_s = float(self.cfg.learned_window_time_s)

        if self._window_start_s is None:
            # First non-overlapping window. This MVP intentionally uses one
            # position clone at a time; a multi-clone 20 Hz update is the next
            # fidelity upgrade toward the paper's full EKF.
            self._window_start_s = now
            self._lio.begin_displacement_window()
            self._next_learned_update_s = now + window_s
            return

        if now + 1.0e-9 < self._next_learned_update_s:
            return

        try:
            window = self._motion_buffer.window(self._window_start_s, self._window_start_s + window_s)
        except ValueError:
            return
        prediction = self._motion_predictor.predict(window)
        self._lio.update_learned_displacement(
            prediction.displacement_w,
            prediction.covariance_w,
        )
        self._learned_update_count += 1
        self._motion_buffer.discard_before(now - window_s)

        self._window_start_s = now
        self._next_learned_update_s = now + window_s
        self._lio.begin_displacement_window()
        self.learned_inertial_state = self._lio.state()

    def _camera_due(self) -> bool:
        now = self._timestamp_s()
        return now - self._last_camera_timestamp_s >= (1.0 / 30.0) - 1.0e-9

    def _maybe_gate_update(self) -> None:
        if self.swift_detector is None or self._gate_builder is None or not self._camera_due():
            return
        self._last_camera_timestamp_s = self._timestamp_s()

        camera = self.scene["tiled_camera"]
        rgb = _np(camera.data.output["rgb"][0])[..., :3]
        if rgb.dtype != np.uint8:
            scale = 255.0 if np.issubdtype(rgb.dtype, np.floating) and float(np.nanmax(rgb)) <= 1.0 + 1e-6 else 1.0
            rgb = np.clip(rgb * scale, 0.0, 255.0).astype(np.uint8)

        observation = self.swift_detector.detect(
            np.ascontiguousarray(rgb),
            timestamp_s=self._timestamp_s(),
        )
        try:
            measurement = self._gate_builder.build(
                observation,
                gate_index=None,
                reference_position_w_b=self._lio.p,
            )
        except (ValueError, RuntimeError):
            return

        # Gate PnP + mapped T_wg is an absolute body-pose observation.
        q_wb = rotmat_to_quat_wxyz(measurement.T_wb_gate.R)
        sigma_rad = np.deg2rad(float(self.cfg.gate_orientation_sigma_deg))
        self._lio.update_gate_pose(
            position_w_b=measurement.position_w_b,
            position_covariance_w=measurement.position_covariance_w,
            orientation_w_b_wxyz=q_wb,
            orientation_covariance_rad2=np.eye(3) * sigma_rad * sigma_rad,
        )
        self._last_gate_measurement = measurement
        self._gate_update_count += 1
        self.learned_inertial_state = self._lio.state()

    def _update_log(self) -> None:
        log = self.extras.setdefault("log", {})
        log["LearnedIO/learned_updates"] = float(self._learned_update_count)
        log["LearnedIO/gate_updates"] = float(self._gate_update_count)
        log["LearnedIO/has_checkpoint"] = float(self._motion_predictor is not None)
        log["LearnedIO/has_gate_detector"] = float(self.swift_detector is not None)
        log["LearnedIO/cov_trace"] = float(np.trace(self._lio.P[:15, :15]))

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
