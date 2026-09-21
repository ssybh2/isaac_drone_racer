# Copyright (c) 2025, Kousheek Chakraborty
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaaclab.assets import Articulation
from isaaclab.managers import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass

from dynamics import Allocation, BodyRatePIDController, Motor
from utils.logger import log

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class SwiftCTBRAction(ActionTerm):
    """Swift-style collective-thrust and body-rate action term.

    Normalized actor action a in [-1, 1]^4 is interpreted as:

    - a[0]: mass-normalized collective thrust around hover.
      0 means hover acceleration, +1 means full available static thrust,
      and sufficiently negative commands cut collective thrust.
    - a[1:4]: desired body rates around body x/y/z.

    A Betaflight-inspired PID converts body-rate error to moments. A
    rate-priority inverse mixer then converts [T, tau_x, tau_y, tau_z] to
    bounded rotor speeds before the existing motor/allocation dynamics are
    applied.
    """

    cfg: "SwiftCTBRActionCfg"

    def __init__(self, cfg: "SwiftCTBRActionCfg", env: "ManagerBasedRLEnv") -> None:
        super().__init__(cfg, env)

        self.cfg = cfg
        self._robot: Articulation = env.scene[self.cfg.asset_name]
        self._body_id = self._robot.find_bodies("body")[0]

        self._raw_actions = torch.zeros(self.num_envs, 4, device=self.device)
        self._ctbr_command = torch.zeros(self.num_envs, 4, device=self.device)
        self._processed_actions = torch.zeros(self.num_envs, 4, device=self.device)
        self._omega_ref = torch.zeros(self.num_envs, 4, device=self.device)
        self._omega_real = torch.zeros(self.num_envs, 4, device=self.device)
        self._rotor_thrusts = torch.zeros(self.num_envs, 4, device=self.device)
        self._thrust = torch.zeros(self.num_envs, 1, 3, device=self.device)
        self._moment = torch.zeros(self.num_envs, 1, 3, device=self.device)

        self._allocation = Allocation(
            num_envs=self.num_envs,
            arm_length=self.cfg.arm_length,
            thrust_coeff=self.cfg.thrust_coef,
            drag_coeff=self.cfg.drag_coef,
            device=self.device,
            dtype=self._raw_actions.dtype,
        )
        self._motor = Motor(
            num_envs=self.num_envs,
            taus=self.cfg.taus,
            init=self.cfg.init,
            max_rate=self.cfg.max_rate,
            min_rate=self.cfg.min_rate,
            dt=env.physics_dt,
            use=self.cfg.use_motor_model,
            device=self.device,
            dtype=self._raw_actions.dtype,
        )

        control_dt = float(env.step_dt)
        self._rate_controller = BodyRatePIDController(
            num_envs=self.num_envs,
            dt=control_dt,
            kp=self.cfg.rate_kp,
            ki=self.cfg.rate_ki,
            kd=self.cfg.rate_kd,
            integral_limit=self.cfg.rate_integral_limit,
            moment_limit=self.cfg.rate_moment_limit_nm,
            device=self.device,
            dtype=self._raw_actions.dtype,
        )

        self._max_collective_accel = (
            4.0
            * float(self.cfg.thrust_coef)
            * float(self.cfg.omega_max) ** 2
            / float(self.cfg.vehicle_mass_kg)
        )
        if self._max_collective_accel <= self.cfg.gravity_mps2:
            raise ValueError(
                "CTBR actuator must provide more than 1g maximum collective acceleration"
            )

        self._body_rate_max = torch.tensor(
            self.cfg.body_rate_max_radps,
            dtype=self._raw_actions.dtype,
            device=self.device,
        ).view(1, 3)

    @property
    def action_dim(self) -> int:
        return 4

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        """Actual [collective thrust, body moments] applied after allocation."""
        return self._processed_actions

    @property
    def ctbr_command(self) -> torch.Tensor:
        """Physical command [collective_accel, p_cmd, q_cmd, r_cmd]."""
        return self._ctbr_command

    @property
    def has_debug_vis_implementation(self) -> bool:
        return False

    def process_actions(self, actions: torch.Tensor) -> None:
        if actions.shape != (self.num_envs, 4):
            raise ValueError(
                f"SwiftCTBRAction expected {(self.num_envs, 4)}, got {tuple(actions.shape)}"
            )

        self._raw_actions.copy_(actions)
        normalized = torch.clamp(actions, -1.0, 1.0)

        # Piecewise hover-centred thrust map:
        #   -1 -> 0 thrust, 0 -> hover (1 g), +1 -> platform maximum.
        # This keeps the physically important hover point at zero policy output
        # without wasting the negative third of the normalized action range.
        gravity = float(self.cfg.gravity_mps2)
        thrust_action = normalized[:, 0]
        collective_accel = torch.where(
            thrust_action >= 0.0,
            gravity
            + thrust_action * (self._max_collective_accel - gravity),
            gravity + thrust_action * gravity,
        ).clamp(0.0, self._max_collective_accel)

        desired_rate = normalized[:, 1:4] * self._body_rate_max
        self._ctbr_command[:, 0] = collective_accel
        self._ctbr_command[:, 1:4] = desired_rate

        throttle_cut = collective_accel <= float(self.cfg.integral_reset_accel_mps2)
        moment_cmd = self._rate_controller.compute(
            desired_rate,
            self._robot.data.root_ang_vel_b,
            throttle_cut=throttle_cut,
        )

        desired_wrench = torch.zeros_like(self._processed_actions)
        desired_wrench[:, 0] = collective_accel * float(self.cfg.vehicle_mass_kg)
        desired_wrench[:, 1:4] = moment_cmd

        omega_ref, rotor_thrusts = self._allocation.allocate_wrench_rate_priority(
            desired_wrench,
            omega_max=float(self.cfg.omega_max),
        )
        omega_real = self._motor.compute(omega_ref)
        actual_wrench = self._allocation.compute(omega_real)

        self._omega_ref.copy_(omega_ref)
        self._omega_real.copy_(omega_real)
        self._rotor_thrusts.copy_(rotor_thrusts)
        self._processed_actions.copy_(actual_wrench)

        log(
            self._env,
            ["ctbr_c", "ctbr_p", "ctbr_q", "ctbr_r"],
            self._ctbr_command,
        )
        log(
            self._env,
            ["ctbr_w1", "ctbr_w2", "ctbr_w3", "ctbr_w4"],
            self._omega_real,
        )

    def apply_actions(self) -> None:
        self._thrust[:, 0, 2] = self._processed_actions[:, 0]
        self._moment[:, 0, :] = self._processed_actions[:, 1:4]
        self._robot.set_external_force_and_torque(
            self._thrust,
            self._moment,
            body_ids=self._body_id,
        )

    def reset(self, env_ids) -> None:
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._robot._ALL_INDICES

        self._raw_actions[env_ids] = 0.0
        self._ctbr_command[env_ids] = 0.0
        self._processed_actions[env_ids] = 0.0
        self._omega_ref[env_ids] = 0.0
        self._omega_real[env_ids] = 0.0
        self._rotor_thrusts[env_ids] = 0.0

        self._rate_controller.reset(env_ids)
        self._motor.reset(env_ids)


@configclass
class SwiftCTBRActionCfg(ActionTermCfg):
    """Configuration for SwiftCTBRAction."""

    class_type: type[ActionTerm] = SwiftCTBRAction

    asset_name: str = "robot"

    vehicle_mass_kg: float = 0.6076
    gravity_mps2: float = 9.81
    arm_length: float = 0.035
    drag_coef: float = 1.5e-9
    thrust_coef: float = 2.25e-7
    omega_max: float = 5145.0

    taus: tuple[float, float, float, float] = (0.0001, 0.0001, 0.0001, 0.0001)
    init: tuple[float, float, float, float] = (2572.5, 2572.5, 2572.5, 2572.5)
    max_rate: tuple[float, float, float, float] = (50000.0, 50000.0, 50000.0, 50000.0)
    min_rate: tuple[float, float, float, float] = (-50000.0, -50000.0, -50000.0, -50000.0)
    use_motor_model: bool = False

    body_rate_max_radps: tuple[float, float, float] = (10.0, 10.0, 6.0)

    rate_kp: tuple[float, float, float] = (0.025, 0.025, 0.030)
    rate_ki: tuple[float, float, float] = (0.004, 0.004, 0.001)
    rate_kd: tuple[float, float, float] = (0.00035, 0.00035, 0.00010)
    rate_integral_limit: tuple[float, float, float] = (2.0, 2.0, 1.5)
    rate_moment_limit_nm: tuple[float, float, float] = (0.28, 0.28, 0.14)

    integral_reset_accel_mps2: float = 0.5
