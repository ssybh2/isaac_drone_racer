"""Hybrid learned-motion/OpenVINS/Swift diagnostic configuration."""

from __future__ import annotations

from isaaclab.utils import configclass

from .drone_racer_swift_perception_env_cfg import DroneRacerSwiftPerceptionEnvCfg


@configclass
class DroneRacerHybridOpenVinsEnvCfg(DroneRacerSwiftPerceptionEnvCfg):
    """Enable optional learned motion and explicit diagnostic absolute anchors.

    Leaving ``learned_motion_checkpoint`` as ``None`` preserves raw OpenVINS
    behavior. Supplying a checkpoint activates the TCN displacement predictor,
    learned translational-drift EKF, and then the existing Swift mapped-gate
    absolute correction layer.

    ``oracle_absolute_position_enabled`` is an Isaac-only diagnostic switch. It
    injects sparse noisy simulator-truth position as an absolute measurement to
    measure the upper bound of a future real absolute source such as gate PnP,
    GPS, or UWB. It must remain disabled for sensor-faithful evaluation.
    """

    learned_motion_checkpoint: str | None = None
    learned_motion_device: str = "cuda"
    learned_motion_window_time_s: float = 0.5
    learned_motion_sample_rate_hz: float = 100.0
    learned_motion_sigma_position: float = 0.05
    learned_motion_sigma_velocity: float = 0.1
    learned_motion_innovation_gate_chi2: float | None = 16.26623619623813
    learned_motion_variance_floor: float = 1.0e-6
    learned_motion_relative_position_only: bool = False
    learned_motion_drift_velocity_from_displacement: bool = False
    learned_motion_drift_velocity_sigma_floor_mps: float = 0.5
    learned_motion_position_residual_slew: bool = False
    learned_motion_position_residual_max_rate_mps: float = 4.0
    learned_motion_raw_vio_jump_isolation: bool = False
    learned_motion_raw_vio_jump_threshold_m: float = 0.5

    oracle_absolute_position_enabled: bool = False
    oracle_position_rate_hz: float = 2.0
    oracle_position_sigma_m: float = 0.10
    oracle_position_seed: int = 0

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
        if self.learned_motion_drift_velocity_sigma_floor_mps <= 0.0:
            raise ValueError("learned_motion_drift_velocity_sigma_floor_mps must be positive")
        if self.learned_motion_position_residual_max_rate_mps <= 0.0:
            raise ValueError("learned_motion_position_residual_max_rate_mps must be positive")
        if (
            self.learned_motion_drift_velocity_from_displacement
            and not self.learned_motion_relative_position_only
        ):
            raise ValueError(
                "learned_motion_drift_velocity_from_displacement requires "
                "learned_motion_relative_position_only"
            )
        if (
            self.learned_motion_position_residual_slew
            and not self.learned_motion_relative_position_only
        ):
            raise ValueError(
                "learned_motion_position_residual_slew requires "
                "learned_motion_relative_position_only"
            )
        if self.learned_motion_raw_vio_jump_threshold_m <= 0.0:
            raise ValueError("learned_motion_raw_vio_jump_threshold_m must be positive")
        if self.oracle_position_rate_hz <= 0.0:
            raise ValueError("oracle_position_rate_hz must be positive")
        if self.oracle_position_sigma_m < 0.0:
            raise ValueError("oracle_position_sigma_m must be non-negative")
