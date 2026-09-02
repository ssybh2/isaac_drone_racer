"""Isaac Lab lifecycle adapter for the Stage 1 estimator pipeline."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
from isaaclab.envs import ManagerBasedRLEnv

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

        if cfg.fake_sensors.vio.update_period_s + 1.0e-9 < self.physics_dt:
            raise ValueError("Fake VIO update period cannot be below the physics ingestion period")
        if cfg.fake_sensors.imu.update_period_s + 1.0e-9 < self.physics_dt:
            raise ValueError("Fake IMU update period cannot be below the physics ingestion period")
        self._stage1_pipeline = Stage1StatePipeline(
            num_envs=self.num_envs,
            device=self.device,
            vio_cfg=cfg.fake_sensors.vio,
            imu_cfg=cfg.fake_sensors.imu,
            seed=int(cfg.seed or 0) + cfg.fake_sensors.seed_offset,
        )
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
        self.obs_buf = self.observation_manager.compute()
        return self.obs_buf, self.reward_buf, self.reset_terminated, self.reset_time_outs, self.extras
