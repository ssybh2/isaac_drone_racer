"""Hybrid learned-motion/OpenVINS/Swift diagnostic configuration."""

from __future__ import annotations

from isaaclab.utils import configclass

from .drone_racer_swift_perception_env_cfg import DroneRacerSwiftPerceptionEnvCfg


@configclass
class DroneRacerHybridOpenVinsEnvCfg(DroneRacerSwiftPerceptionEnvCfg):
    """Enable an optional learned relative-motion constraint ahead of gate fusion.

    Leaving ``learned_motion_checkpoint`` as ``None`` preserves the raw OpenVINS
    behavior. Supplying a checkpoint activates the TCN displacement predictor,
    learned translational-drift EKF, and then the existing Swift mapped-gate
    absolute correction layer.
    """

    learned_motion_checkpoint: str | None = None
    learned_motion_device: str = "cuda"
    learned_motion_window_time_s: float = 0.5
    learned_motion_sample_rate_hz: float = 100.0
    learned_motion_sigma_position: float = 0.05
    learned_motion_sigma_velocity: float = 0.1
    learned_motion_innovation_gate_chi2: float | None = 16.26623619623813
    learned_motion_variance_floor: float = 1.0e-6

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.learned_motion_window_time_s <= 0.0:
            raise ValueError("learned_motion_window_time_s must be positive")
        if self.learned_motion_sample_rate_hz <= 0.0:
            raise ValueError("learned_motion_sample_rate_hz must be positive")
        if self.learned_motion_sigma_position < 0.0 or self.learned_motion_sigma_velocity < 0.0:
            raise ValueError("learned motion process-noise terms must be non-negative")
        if (
            self.learned_motion_innovation_gate_chi2 is not None
            and self.learned_motion_innovation_gate_chi2 <= 0.0
        ):
            raise ValueError("learned_motion_innovation_gate_chi2 must be positive or None")
        if self.learned_motion_variance_floor <= 0.0:
            raise ValueError("learned_motion_variance_floor must be positive")
