"""Single-vehicle Isaac/OpenVINS diagnostic runtime.

This is intentionally not a PPO training environment. It wires the calibrated
Isaac RGB + IMU streams to the external ROS2 OpenVINS process at source rates,
performs one-time V->W alignment, and exposes the latest world-aligned VIO state
for downstream Swift fusion diagnostics.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from isaaclab.envs import ManagerBasedRLEnv

from estimation.openvins_bridge import (
    OpenVinsFrameAlignment,
    OpenVinsRos2Bridge,
    OpenVinsSensorRateGate,
)
from estimation.swift_vio_drift import VioWorldEstimate
from estimation.vio_time_buffer import VioWorldEstimateBuffer

from .drone_racer_swift_perception_env_cfg import DroneRacerSwiftPerceptionEnvCfg


def _to_numpy(value) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    return np.asarray(value)


class SwiftOpenVinsDiagnosticEnv(ManagerBasedRLEnv):
    """Run one Isaac vehicle against an external ``ov_msckf`` ROS2 process."""

    cfg: DroneRacerSwiftPerceptionEnvCfg

    def __init__(self, cfg: DroneRacerSwiftPerceptionEnvCfg, render_mode: str | None = None, **kwargs):
        if cfg.swift_rejection_dump_limit < 0:
            raise ValueError("swift_rejection_dump_limit must be non-negative")

        self.openvins_vio_estimate: VioWorldEstimate | None = None
        self.openvins_alignment: OpenVinsFrameAlignment | None = None
        self._openvins_bridge: OpenVinsRos2Bridge | None = None
        self._openvins_rate_gate = OpenVinsSensorRateGate(imu_hz=200.0, camera_hz=30.0)
        self._alignment_truth_buffer = VioWorldEstimateBuffer(max_age_s=5.0, max_samples=4096)
        self.openvins_vio_buffer = VioWorldEstimateBuffer(max_age_s=2.0, max_samples=4096)
        self._last_openvins_sample_timestamp_s: float | None = None
        self._camera_contract_validated = False
        self._rejection_dump_count = 0
        self.swift_fusion = None
        self.swift_detector = None
        self.swift_fused_estimate = None
        self.swift_last_fusion_result = None
        self._latest_gate_observation = None
        self._latest_gate_index: int | None = None
        self._latest_gate_rgb: np.ndarray | None = None
        super().__init__(cfg=cfg, render_mode=render_mode, **kwargs)
        if self.num_envs != 1:
            raise ValueError("SwiftOpenVinsDiagnosticEnv requires exactly one Isaac environment")
        self._openvins_bridge = OpenVinsRos2Bridge()
        self._record_alignment_truth()

    def _initialize_optional_perception_fusion(self) -> None:
        checkpoint = self.cfg.swift_detector_checkpoint
        if checkpoint is None:
            return

        from estimation.swift_fusion import SwiftPerceptionFusion
        from perception.camera_model import CameraCalibration
        from perception.stage2_calibration import load_stage2_gate_geometry, stage2_camera_to_body
        from perception.swift_gate_measurement import GatePoseMeasurementBuilder
        from perception.swift_isaac_adapter import track_layout_from_isaac
        from perception.torchvision_keypoint_detector import TorchvisionGateCornerDetector

        checkpoint_path = Path(checkpoint).expanduser().resolve()
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Swift detector checkpoint not found: {checkpoint_path}")

        camera = self.scene["tiled_camera"]
        K = _to_numpy(camera.data.intrinsic_matrices[0])
        image_height, image_width = (int(v) for v in camera.data.image_shape)
        calibration = CameraCalibration(
            K=K,
            image_width=image_width,
            image_height=image_height,
            distortion=None,
            model="pinhole",
        )
        builder = GatePoseMeasurementBuilder(
            load_stage2_gate_geometry(),
            calibration,
            stage2_camera_to_body(),
            track_layout_from_isaac(self, env_id=0),
        )
        self.swift_detector = TorchvisionGateCornerDetector(
            checkpoint_path,
            device=self.cfg.swift_detector_device,
            detection_threshold=self.cfg.swift_detection_threshold,
            keypoint_confidence_threshold=self.cfg.swift_keypoint_confidence_threshold,
        )
        self.swift_fusion = SwiftPerceptionFusion(builder)

    def _timestamp_s(self) -> float:
        return float(self._sim_step_counter) * float(self.physics_dt)

    def _truth_vio_state(self) -> VioWorldEstimate:
        robot = self.scene["robot"]
        return VioWorldEstimate(
            position_w_b=_to_numpy(robot.data.root_pos_w[0]),
            linear_velocity_w_b=_to_numpy(robot.data.root_lin_vel_w[0]),
            orientation_w_b_wxyz=_to_numpy(robot.data.root_quat_w[0]),
            timestamp_s=self._timestamp_s(),
        )

    def _record_alignment_truth(self) -> None:
        # Simulator truth is permitted only for one-time V->W alignment. Stop
        # retaining it immediately after alignment has been established.
        if self.openvins_alignment is None:
            self._alignment_truth_buffer.push(self._truth_vio_state())

    def _validate_camera_contract(self) -> None:
        """Verify runtime Isaac intrinsics match the pinned OpenVINS YAML contract."""
        if self._camera_contract_validated:
            return

        from perception.stage2_calibration import (
            OPENVINS_CAMERA_RESOLUTION,
            openvins_camera_matrix,
        )

        camera = self.scene["tiled_camera"]
        image_height, image_width = (int(v) for v in camera.data.image_shape)
        expected_width, expected_height = OPENVINS_CAMERA_RESOLUTION
        if (image_width, image_height) != (expected_width, expected_height):
            raise RuntimeError(
                "Isaac/OpenVINS camera resolution mismatch: "
                f"runtime={(image_width, image_height)}, "
                f"expected={(expected_width, expected_height)}"
            )

        runtime_K = _to_numpy(camera.data.intrinsic_matrices[0]).astype(np.float64)
        expected_K = openvins_camera_matrix()
        if not np.allclose(runtime_K, expected_K, rtol=0.0, atol=1.0e-3):
            raise RuntimeError(
                "Isaac/OpenVINS camera intrinsics mismatch. Runtime K is\n"
                f"{runtime_K}\nwhile config/openvins/swift_sim/kalibr_imucam_chain.yaml "
                f"expects\n{expected_K}. Regenerate/revalidate calibration before running VIO."
            )
        self._camera_contract_validated = True

    def _publish_imu_if_due(self) -> None:
        timestamp_s = self._timestamp_s()
        if not self._openvins_rate_gate.imu_due(timestamp_s):
            return
        imu = self.scene["imu"]
        self._openvins_bridge.publish_imu(
            angular_velocity_b=_to_numpy(imu.data.ang_vel_b[0]),
            linear_acceleration_b=_to_numpy(imu.data.lin_acc_b[0]),
            timestamp_s=timestamp_s,
        )

    def _publish_camera_if_due(self) -> None:
        timestamp_s = self._timestamp_s()
        if not self._openvins_rate_gate.camera_due(timestamp_s):
            return
        self._validate_camera_contract()
        camera = self.scene["tiled_camera"]
        rgb = _to_numpy(camera.data.output["rgb"][0])[..., :3]
        if rgb.dtype != np.uint8:
            # Isaac camera output is normally uint8; fail-safe conversion keeps
            # the ROS contract explicit if a future Isaac version changes it.
            if np.issubdtype(rgb.dtype, np.floating):
                scale = 255.0 if float(np.nanmax(rgb)) <= 1.0 + 1.0e-6 else 1.0
                rgb = np.clip(rgb * scale, 0.0, 255.0).astype(np.uint8)
            else:
                rgb = np.clip(rgb, 0, 255).astype(np.uint8)
        rgb = np.ascontiguousarray(rgb)
        self._openvins_bridge.publish_rgb(rgb, timestamp_s=timestamp_s)
        if self.swift_detector is None and self.cfg.swift_detector_checkpoint is not None:
            self._initialize_optional_perception_fusion()
        if self.swift_detector is not None:
            self._latest_gate_observation = self.swift_detector.detect(
                rgb, timestamp_s=timestamp_s
            )
            self._latest_gate_rgb = rgb.copy()
            if self.cfg.swift_use_oracle_gate_index:
                # Controlled ablation only. Normal estimator operation must not
                # obtain gate identity from the task command manager.
                from perception.swift_isaac_adapter import active_gate_index_from_isaac

                self._latest_gate_index = active_gate_index_from_isaac(self, env_id=0)
            else:
                # Leave the detection unlabeled. GatePoseMeasurementBuilder then
                # associates it against the known track using VIO at this exact
                # camera timestamp.
                self._latest_gate_index = None

    def _maybe_dump_rejection(self, observation, rgb, result) -> None:
        """Persist a rejected detector/IPPE/fusion frame for threshold tuning."""
        dump_dir = self.cfg.swift_rejection_dump_dir
        if (
            dump_dir is None
            or observation is None
            or rgb is None
            or result.measurement_accepted
            or self._rejection_dump_count >= self.cfg.swift_rejection_dump_limit
        ):
            return

        try:
            import cv2
        except ImportError as exc:
            raise ImportError(
                "Rejected-frame export requires OpenCV (cv2), which is already required by the Stage2 PnP path."
            ) from exc

        output_dir = Path(dump_dir).expanduser().resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        timestamp_s = float(observation.timestamp_s)
        time_tag = f"{timestamp_s:.6f}".replace(".", "_")
        stem = f"reject_{self._rejection_dump_count:05d}_t{time_tag}"
        image_path = output_dir / f"{stem}.png"
        metadata_path = output_dir / f"{stem}.json"

        bgr = cv2.cvtColor(np.ascontiguousarray(rgb), cv2.COLOR_RGB2BGR)
        if not cv2.imwrite(str(image_path), bgr):
            raise RuntimeError(f"Unable to write rejected Swift frame: {image_path}")

        measurement = result.gate_measurement
        payload = {
            "schema": "isaac_drone_racer.swift_rejected_gate_observation.v1",
            "timestamp_s": timestamp_s,
            "source": str(observation.source),
            "corners_uv": np.asarray(observation.corners_uv, dtype=float).tolist(),
            "visible": np.asarray(observation.visible, dtype=bool).tolist(),
            "confidence": np.asarray(observation.confidence, dtype=float).tolist(),
            "rejection_reason": str(result.rejection_reason),
            "innovation_mahalanobis2": [
                float(value) for value in result.innovation_mahalanobis2
            ],
            "oracle_gate_identity_enabled": bool(self.cfg.swift_use_oracle_gate_index),
            "measurement_gate_index": None if measurement is None else int(measurement.gate_index),
            "nominal_reprojection_rmse_px": (
                None
                if measurement is None
                else float(measurement.nominal_pnp.reprojection_rmse_px)
            ),
            "image_file": image_path.name,
        }
        metadata_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        self._rejection_dump_count += 1

    def _consume_openvins(self) -> None:
        # OpenVINS publishes propagated odometry from its 200 Hz IMU callback,
        # while this diagnostic control loop runs at 100 Hz. Drain the ROS
        # subscription queue and keep the newest state; one spin_once per
        # control step would otherwise accumulate stale odometry indefinitely.
        sample = self._openvins_bridge.drain_latest(max_callbacks=32)
        if sample is None:
            return
        if (
            self._last_openvins_sample_timestamp_s is not None
            and sample.timestamp_s <= self._last_openvins_sample_timestamp_s + 1.0e-12
        ):
            return

        if self.openvins_alignment is None:
            try:
                reference = self._alignment_truth_buffer.interpolate(sample.timestamp_s)
            except ValueError:
                # Wait until the matching simulator truth timestamp is retained;
                # never align an old VIO sample to the current vehicle pose.
                return
            self.openvins_alignment = OpenVinsFrameAlignment.from_reference_pose(
                sample,
                reference_position_w_b=reference.position_w_b,
                reference_orientation_w_b_wxyz=reference.orientation_w_b_wxyz,
            )
            self._alignment_truth_buffer.clear()

        world = self.openvins_alignment.to_world(sample)
        self.openvins_vio_estimate = world
        self.openvins_vio_buffer.push(world)
        self._last_openvins_sample_timestamp_s = sample.timestamp_s

        if self.swift_fusion is not None:
            observation = self._latest_gate_observation
            gate_index = self._latest_gate_index
            gate_rgb = self._latest_gate_rgb
            if observation is not None and observation.timestamp_s > world.timestamp_s + 1.0e-6:
                # OpenVINS may trail the just-rendered camera frame by one ROS
                # spin. Keep the observation and matching RGB pending until VIO
                # brackets its timestamp instead of discarding it as a future
                # measurement.
                self.swift_last_fusion_result = self.swift_fusion.step(world)
            else:
                self._latest_gate_observation = None
                self._latest_gate_index = None
                self._latest_gate_rgb = None
                self.swift_last_fusion_result = self.swift_fusion.step(
                    world,
                    gate_observation=observation,
                    gate_index=gate_index,
                )
                self._maybe_dump_rejection(
                    observation,
                    gate_rgb,
                    self.swift_last_fusion_result,
                )
            self.swift_fused_estimate = self.swift_last_fusion_result.fused_state

    def _update_openvins_log(self) -> None:
        log = self.extras.setdefault("log", {})
        log["OpenVINS/aligned"] = float(self.openvins_alignment is not None)
        log["OpenVINS/camera_contract_validated"] = float(self._camera_contract_validated)
        log["OpenVINS/drained_odom_callbacks"] = float(
            0 if self._openvins_bridge is None else self._openvins_bridge.last_drain_count
        )
        log["SwiftFusion/oracle_gate_identity"] = float(self.cfg.swift_use_oracle_gate_index)
        log["SwiftFusion/rejection_dump_count"] = float(self._rejection_dump_count)
        if self.openvins_vio_estimate is None:
            log["OpenVINS/age_s"] = float("nan")
        else:
            log["OpenVINS/age_s"] = max(
                0.0, self._timestamp_s() - self.openvins_vio_estimate.timestamp_s
            )
        if self.swift_last_fusion_result is not None:
            log["SwiftFusion/measurement_accepted"] = float(
                self.swift_last_fusion_result.measurement_accepted
            )
            if self.swift_last_fusion_result.innovation_mahalanobis2:
                log["SwiftFusion/innovation_d2"] = float(
                    self.swift_last_fusion_result.innovation_mahalanobis2[0]
                )

    def step(self, action: torch.Tensor):
        """Mirror IsaacLab v2.1 step while inserting source-rate ROS transport."""
        self.action_manager.process_action(action.to(self.device))
        self.recorder_manager.record_pre_step()
        is_rendering = self.sim.has_gui() or self.sim.has_rtx_sensors()

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
            self._record_alignment_truth()
            self._publish_imu_if_due()
            if rendered:
                self._publish_camera_if_due()

        self._consume_openvins()

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
            # A simulator teleport invalidates the external VIO trajectory.
            # Re-arm local alignment and make the invalidation visible. For a
            # clean diagnostic run restart ov_msckf when this occurs.
            self.openvins_alignment = None
            self.openvins_vio_estimate = None
            self.openvins_vio_buffer.clear()
            self._alignment_truth_buffer.clear()
            if self.swift_fusion is not None:
                self.swift_fusion.reset()
            self.swift_fused_estimate = None
            self.swift_last_fusion_result = None
            self._latest_gate_observation = None
            self._latest_gate_index = None
            self._latest_gate_rgb = None
            self._openvins_rate_gate.reset()
            self._last_openvins_sample_timestamp_s = None
            self._record_alignment_truth()
            self.extras.setdefault("log", {})["OpenVINS/restart_required"] = 1.0
            self.recorder_manager.record_post_reset(reset_env_ids)

        self.command_manager.compute(dt=self.step_dt)
        if "interval" in self.event_manager.available_modes:
            self.event_manager.apply(mode="interval", dt=self.step_dt)
        self._update_openvins_log()
        self.obs_buf = self.observation_manager.compute()
        return self.obs_buf, self.reward_buf, self.reset_terminated, self.reset_time_outs, self.extras

    def close(self):
        if self._openvins_bridge is not None:
            self._openvins_bridge.close()
            self._openvins_bridge = None
        return super().close()
