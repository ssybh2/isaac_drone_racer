"""Isaac Lab lifecycle adapter for the Stage 1 estimator pipeline."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
from isaaclab.envs import ManagerBasedRLEnv

from estimation.fake_sensor_cfg import FakeSensorPipelineCfg
from estimation.pipeline import Stage1StatePipeline
from estimation.state_estimate import GroundTruthState

from .drone_racer_stage1_env_cfg import DroneRacerStage1EnvCfg


class Stage1DroneRacerEnv(ManagerBasedRLEnv):
    """Feed simulator truth only into Fake VIO/Fake IMU provider boundaries."""

    cfg: DroneRacerStage1EnvCfg

    def __init__(self, cfg: DroneRacerStage1EnvCfg, render_mode: str | None = None, **kwargs):
        # ObservationManager probes term dimensions inside the parent constructor.
        self.stage1_state_estimate = None
        self._stage1_pipeline = None
        super().__init__(cfg=cfg, render_mode=render_mode, **kwargs)

        self._validate_fake_sensor_periods(cfg.fake_sensors)
        self._stage1_pipeline = self._make_stage1_pipeline(
            cfg.fake_sensors, int(cfg.seed or 0) + cfg.fake_sensors.seed_offset
        )
        env_ids = torch.arange(self.num_envs, dtype=torch.long, device=self.device)
        self.stage1_state_estimate = self._stage1_pipeline.reset(
            env_ids, self._fake_sensor_ground_truth(), self._stage1_timestamp_s()
        )
        self.stage1_evaluation_snapshot = None

    def _validate_fake_sensor_periods(self, fake_sensors: FakeSensorPipelineCfg) -> None:
        if fake_sensors.vio.min_update_period_s + 1.0e-9 < self.physics_dt:
            raise ValueError("Fake VIO update period cannot be below the physics ingestion period")
        if fake_sensors.imu.min_update_period_s + 1.0e-9 < self.physics_dt:
            raise ValueError("Fake IMU update period cannot be below the physics ingestion period")

    def _make_stage1_pipeline(
        self, fake_sensors: FakeSensorPipelineCfg, seed: int
    ) -> Stage1StatePipeline:
        return Stage1StatePipeline(
            num_envs=self.num_envs,
            device=self.device,
            vio_cfg=fake_sensors.vio,
            imu_cfg=fake_sensors.imu,
            seed=seed,
        )

    def set_fake_sensor_profile(self, profile: str, seed: int | None = None) -> None:
        """Switch evaluation profiles without rebuilding the Isaac scene."""
        fake_sensors = FakeSensorPipelineCfg.from_profile(profile)
        self._validate_fake_sensor_periods(fake_sensors)
        self.cfg.fake_sensors = fake_sensors
        source_seed = int((self.cfg.seed or 0) if seed is None else seed) + fake_sensors.seed_offset
        self._stage1_pipeline = self._make_stage1_pipeline(fake_sensors, source_seed)
        env_ids = torch.arange(self.num_envs, dtype=torch.long, device=self.device)
        self.stage1_state_estimate = self._stage1_pipeline.reset(
            env_ids, self._fake_sensor_ground_truth(), self._stage1_timestamp_s()
        )

    @property
    def stage1_diagnostics(self) -> dict[str, int]:
        return {
            "imu_ingest_count": self._stage1_pipeline.imu_ingest_count,
            "vio_ingest_count": self._stage1_pipeline.vio_ingest_count,
        }

    def _stage1_timestamp_s(self) -> float:
        return self._sim_step_counter * self.physics_dt

    def _fake_sensor_ground_truth(self) -> GroundTruthState:
        """The sole Isaac drone-state boundary used by Stage 1 fake sensors."""
        robot = self.scene["robot"]
        return GroundTruthState(
            position_w_b=robot.data.root_pos_w,
            orientation_w_b=robot.data.root_quat_w,
            linear_velocity_w_b=robot.data.root_lin_vel_w,
            angular_velocity_b=robot.data.root_ang_vel_b,
        )

    def _stage1_error_tensors(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        estimate = self.stage1_state_estimate
        ground_truth = self._fake_sensor_ground_truth()
        quaternion_dot = torch.sum(
            estimate.orientation_w_b * ground_truth.orientation_w_b, dim=-1
        ).abs()
        attitude_error = 2.0 * torch.acos(quaternion_dot.clamp(0.0, 1.0))
        position_error = torch.linalg.vector_norm(
            estimate.position_w_b - ground_truth.position_w_b, dim=-1
        )
        velocity_error = torch.linalg.vector_norm(
            estimate.linear_velocity_w_b - ground_truth.linear_velocity_w_b, dim=-1
        )
        return position_error, attitude_error, velocity_error

    def _capture_stage1_evaluation_snapshot(self) -> None:
        """Capture pre-reset truth diagnostics for out-of-policy evaluation."""
        position_error, attitude_error, velocity_error = self._stage1_error_tensors()
        target = self.command_manager.get_term("target")
        term_dones = self.termination_manager._term_dones
        self.stage1_evaluation_snapshot = {
            "gate_passed": target.gate_passed.clone(),
            "next_gate_index": target.next_gate_idx.clone(),
            "position_error_m": position_error,
            "attitude_error_rad": attitude_error,
            "velocity_error_mps": velocity_error,
            "vio_age_s": self.stage1_state_estimate.vio_status.age_s,
            "imu_age_s": self.stage1_state_estimate.imu_status.age_s,
            "vio_valid": self.stage1_state_estimate.vio_status.valid,
            "imu_valid": self.stage1_state_estimate.imu_status.valid,
            "vio_dropped": self._stage1_pipeline.vio.dropped,
            "imu_dropped": self._stage1_pipeline.imu.dropped,
            "collision": term_dones["collision"].clone(),
            "flyaway": term_dones["flyaway"].clone(),
        }

    def _update_stage1_log(self) -> None:
        """Expose estimator health without feeding diagnostics into the policy."""
        estimate = self.stage1_state_estimate
        position_error, attitude_error, velocity_error = self._stage1_error_tensors()
        log = self.extras.setdefault("log", {})
        log["Stage1/position_error_mean_m"] = position_error.mean()
        log["Stage1/attitude_error_mean_rad"] = attitude_error.mean()
        log["Stage1/velocity_error_mean_mps"] = velocity_error.mean()
        log["Stage1/vio_age_mean_s"] = estimate.vio_status.age_s.mean()
        log["Stage1/imu_age_mean_s"] = estimate.imu_status.age_s.mean()
        log["Stage1/vio_fresh_fraction"] = estimate.vio_status.valid.float().mean()
        log["Stage1/imu_fresh_fraction"] = estimate.imu_status.valid.float().mean()

    def reset(
        self,
        seed: int | None = None,
        env_ids: Sequence[int] | None = None,
        options: dict[str, Any] | None = None,
    ):
        if env_ids is None:
            estimator_env_ids = torch.arange(
                self.num_envs, dtype=torch.long, device=self.device
            )
        else:
            estimator_env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        _, extras = super().reset(seed=seed, env_ids=env_ids, options=options)
        self.stage1_state_estimate = self._stage1_pipeline.reset(
            estimator_env_ids,
            self._fake_sensor_ground_truth(),
            self._stage1_timestamp_s(),
        )
        self.obs_buf = self.observation_manager.compute()
        return self.obs_buf, extras

    def step(self, action: torch.Tensor):
        """Run the pinned v2.1.0 RL step with estimator lifecycle insertions.

        Control flow mirrors:
        IsaacLab v2.1.0/source/isaaclab/isaaclab/envs/manager_based_rl_env.py
        ``ManagerBasedRLEnv.step``. Keep this method synchronized when upgrading
        Isaac Lab.
        """
        self.action_manager.process_action(action.to(self.device))
        self.recorder_manager.record_pre_step()
        is_rendering = self.sim.has_gui() or self.sim.has_rtx_sensors()

        for _ in range(self.cfg.decimation):
            self._sim_step_counter += 1
            self.action_manager.apply_action()
            self.scene.write_data_to_sim()
            self.sim.step(render=False)
            if self._sim_step_counter % self.cfg.sim.render_interval == 0 and is_rendering:
                self.sim.render()
            self.scene.update(dt=self.physics_dt)
            ground_truth = self._fake_sensor_ground_truth()
            self._stage1_pipeline.ingest_imu(
                ground_truth.angular_velocity_b, self._stage1_timestamp_s()
            )

        timestamp_s = self._stage1_timestamp_s()
        self._stage1_pipeline.ingest_vio(self._fake_sensor_ground_truth(), timestamp_s)
        self.stage1_state_estimate = self._stage1_pipeline.publish(timestamp_s)

        self.episode_length_buf += 1
        self.common_step_counter += 1
        self.reset_buf = self.termination_manager.compute()
        self.reset_terminated = self.termination_manager.terminated
        self.reset_time_outs = self.termination_manager.time_outs
        self.reward_buf = self.reward_manager.compute(dt=self.step_dt)
        self._capture_stage1_evaluation_snapshot()

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
            self.stage1_state_estimate = self._stage1_pipeline.reset(
                reset_env_ids, self._fake_sensor_ground_truth(), timestamp_s
            )
            self.recorder_manager.record_post_reset(reset_env_ids)

        self.command_manager.compute(dt=self.step_dt)
        if "interval" in self.event_manager.available_modes:
            self.event_manager.apply(mode="interval", dt=self.step_dt)
        self.stage1_state_estimate = self._stage1_pipeline.publish(timestamp_s)
        self._update_stage1_log()
        self.obs_buf = self.observation_manager.compute()
        return self.obs_buf, self.reward_buf, self.reset_terminated, self.reset_time_outs, self.extras
