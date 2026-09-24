"""Pure coordinated-turn reference for the high-quality Circular-12 template.

The previous 20-degree-camera / 2.07-m experiment directly wrote the ideal
pose and velocity into Isaac Sim. This module extracts only the reference
mathematics. It never writes simulator state and therefore cannot "cheat" the
vehicle dynamics.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class Circular12ReferenceConfig:
    radius_m: float = 12.0
    center_x_m: float = 0.0
    center_y_m: float = 12.0
    height_m: float = 2.07
    speed_mps: float = 17.712658128452922
    gravity_mps2: float = 9.81
    initial_phase_rad: float = -math.pi / 2.0

    def __post_init__(self) -> None:
        if self.radius_m <= 0.0:
            raise ValueError("radius_m must be positive")
        if self.speed_mps <= 0.0:
            raise ValueError("speed_mps must be positive")
        if self.gravity_mps2 <= 0.0:
            raise ValueError("gravity_mps2 must be positive")


@dataclass(frozen=True)
class Circular12ReferenceState:
    phase_rad: float
    position_w: np.ndarray
    velocity_w: np.ndarray
    acceleration_w: np.ndarray
    rotation_wb: np.ndarray
    angular_velocity_w: np.ndarray
    body_rate_b: np.ndarray
    collective_accel_mps2: float


class Circular12Reference:
    """Analytic steady, coordinated circular-flight reference."""

    def __init__(self, cfg: Circular12ReferenceConfig | None = None) -> None:
        self.cfg = cfg or Circular12ReferenceConfig()

    @property
    def omega_radps(self) -> float:
        return self.cfg.speed_mps / self.cfg.radius_m

    @property
    def bank_rad(self) -> float:
        return -math.atan2(
            self.cfg.speed_mps**2 / self.cfg.radius_m,
            self.cfg.gravity_mps2,
        )

    @property
    def bank_deg(self) -> float:
        return math.degrees(self.bank_rad)

    @property
    def collective_accel_mps2(self) -> float:
        return math.hypot(
            self.cfg.gravity_mps2,
            self.cfg.speed_mps**2 / self.cfg.radius_m,
        )

    def phase_from_position(self, position_w: np.ndarray) -> float:
        p = np.asarray(position_w, dtype=np.float64).reshape(3)
        return math.atan2(
            float(p[1]) - self.cfg.center_y_m,
            float(p[0]) - self.cfg.center_x_m,
        )

    def sample_time(self, t_s: float) -> Circular12ReferenceState:
        phase = self.cfg.initial_phase_rad + self.omega_radps * float(t_s)
        return self.sample_phase(phase)

    def sample_phase(self, phase_rad: float) -> Circular12ReferenceState:
        c = math.cos(float(phase_rad))
        s = math.sin(float(phase_rad))
        radius = self.cfg.radius_m
        speed = self.cfg.speed_mps

        radial = np.array([c, s, 0.0], dtype=np.float64)
        tangent = np.array([-s, c, 0.0], dtype=np.float64)
        inward = -radial

        position_w = np.array(
            [
                self.cfg.center_x_m + radius * c,
                self.cfg.center_y_m + radius * s,
                self.cfg.height_m,
            ],
            dtype=np.float64,
        )
        velocity_w = speed * tangent
        acceleration_w = (speed**2 / radius) * inward

        # Body +Z is the thrust direction. Body +X follows the local tangent.
        thrust_accel_w = acceleration_w + np.array(
            [0.0, 0.0, self.cfg.gravity_mps2], dtype=np.float64
        )
        z_des = thrust_accel_w / np.linalg.norm(thrust_accel_w)
        y_des = np.cross(z_des, tangent)
        y_des /= np.linalg.norm(y_des)
        x_des = np.cross(y_des, z_des)
        x_des /= np.linalg.norm(x_des)
        rotation_wb = np.column_stack((x_des, y_des, z_des))

        angular_velocity_w = np.array(
            [0.0, 0.0, self.omega_radps], dtype=np.float64
        )
        body_rate_b = rotation_wb.T @ angular_velocity_w

        return Circular12ReferenceState(
            phase_rad=float(phase_rad),
            position_w=position_w,
            velocity_w=velocity_w,
            acceleration_w=acceleration_w,
            rotation_wb=rotation_wb,
            angular_velocity_w=angular_velocity_w,
            body_rate_b=body_rate_b,
            collective_accel_mps2=self.collective_accel_mps2,
        )
