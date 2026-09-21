# Copyright (c) 2025, Kousheek Chakraborty
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch


class BodyRatePIDController:
    """Betaflight-inspired body-rate controller.

    The controller tracks body-rate commands in rad/s and produces body moments
    in N*m. The D term is pure damping on measured body-rate change, matching
    the Swift/Betaflight convention where the D-term reference is zero.
    """

    def __init__(
        self,
        num_envs: int,
        dt: float,
        kp: tuple[float, float, float],
        ki: tuple[float, float, float],
        kd: tuple[float, float, float],
        integral_limit: tuple[float, float, float],
        moment_limit: tuple[float, float, float],
        device: str,
        dtype: torch.dtype,
    ) -> None:
        if num_envs <= 0:
            raise ValueError("num_envs must be positive")
        if dt <= 0.0:
            raise ValueError("dt must be positive")

        self.num_envs = int(num_envs)
        self.dt = float(dt)
        self.device = device
        self.dtype = dtype

        self.kp = torch.tensor(kp, device=device, dtype=dtype).view(1, 3)
        self.ki = torch.tensor(ki, device=device, dtype=dtype).view(1, 3)
        self.kd = torch.tensor(kd, device=device, dtype=dtype).view(1, 3)
        self.integral_limit = torch.tensor(
            integral_limit, device=device, dtype=dtype
        ).view(1, 3)
        self.moment_limit = torch.tensor(
            moment_limit, device=device, dtype=dtype
        ).view(1, 3)

        if torch.any(self.kp < 0.0) or torch.any(self.ki < 0.0) or torch.any(self.kd < 0.0):
            raise ValueError("PID gains must be non-negative")
        if torch.any(self.integral_limit < 0.0):
            raise ValueError("integral limits must be non-negative")
        if torch.any(self.moment_limit <= 0.0):
            raise ValueError("moment limits must be positive")

        self.integral = torch.zeros(
            self.num_envs, 3, device=device, dtype=dtype
        )
        self.previous_rate = torch.zeros_like(self.integral)
        self.initialized = torch.zeros(
            self.num_envs, 1, device=device, dtype=torch.bool
        )

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        if env_ids is None:
            self.integral.zero_()
            self.previous_rate.zero_()
            self.initialized.zero_()
            return
        self.integral[env_ids] = 0.0
        self.previous_rate[env_ids] = 0.0
        self.initialized[env_ids] = False

    def compute(
        self,
        desired_rate: torch.Tensor,
        measured_rate: torch.Tensor,
        *,
        throttle_cut: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if desired_rate.shape != (self.num_envs, 3):
            raise ValueError(
                f"desired_rate must have shape {(self.num_envs, 3)}"
            )
        if measured_rate.shape != (self.num_envs, 3):
            raise ValueError(
                f"measured_rate must have shape {(self.num_envs, 3)}"
            )

        error = desired_rate - measured_rate

        self.integral += error * self.dt
        self.integral = torch.maximum(
            torch.minimum(self.integral, self.integral_limit),
            -self.integral_limit,
        )

        if throttle_cut is not None:
            mask = throttle_cut.reshape(self.num_envs, 1).to(dtype=torch.bool)
            self.integral = torch.where(
                mask,
                torch.zeros_like(self.integral),
                self.integral,
            )

        measured_derivative = torch.zeros_like(measured_rate)
        initialized_mask = self.initialized.expand(-1, 3)
        measured_derivative = torch.where(
            initialized_mask,
            (measured_rate - self.previous_rate) / self.dt,
            measured_derivative,
        )

        # Swift/Betaflight convention: D-term reference is zero, so D provides
        # damping on measured rate rather than differentiating command changes.
        moment = (
            self.kp * error
            + self.ki * self.integral
            - self.kd * measured_derivative
        )
        moment = torch.maximum(
            torch.minimum(moment, self.moment_limit),
            -self.moment_limit,
        )

        self.previous_rate.copy_(measured_rate)
        self.initialized.fill_(True)
        return moment
