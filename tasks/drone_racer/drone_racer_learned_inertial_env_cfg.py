"""Configuration for OpenVINS-free learned-inertial drone racing."""

from __future__ import annotations

from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.utils import configclass

from . import mdp
from .drone_racer_env_cfg import ActionsCfg
from .drone_racer_swift_perception_env_cfg import DroneRacerSwiftPerceptionEnvCfg


@configclass
class LearnedInertialPolicyCfg(ObsGroup):
    """Sensor/map observations available to the racing policy."""

    drone_state = ObsTerm(func=mdp.learned_inertial_drone_state)
    target_pos_b = ObsTerm(
        func=mdp.learned_target_pos_b,
        params={"command_name": "target"},
    )
    actions = ObsTerm(func=mdp.last_action)

    def __post_init__(self) -> None:
        self.enable_corruption = False
        self.concatenate_terms = True


@configclass
class LearnedInertialObservationsCfg:
    policy: LearnedInertialPolicyCfg = LearnedInertialPolicyCfg()
    # Rewards may use simulator truth during training, but the actor observation
    # intentionally has no GT pose. Add an asymmetric privileged critic later
    # only as a deliberate training ablation.
    critic = None


@configclass
class DroneRacerLearnedInertialEnvCfg(DroneRacerSwiftPerceptionEnvCfg):
    """Fixed-start IMU+TCN+EKF+mapped-gate configuration.

    This keeps the calibrated Stage2 camera/gate detector path but removes the
    external OpenVINS dependency from the runtime estimator.
    """

    observations: LearnedInertialObservationsCfg = LearnedInertialObservationsCfg()
    actions: ActionsCfg = ActionsCfg()

    learned_motion_checkpoint: str | None = None
    learned_motion_device: str = "cuda"
    learned_window_time_s: float = 0.5
    learned_sample_rate_hz: float = 100.0
    # Paper-style overlapping relative-displacement updates: a 0.5 s history
    # window is evaluated every 0.05 s, requiring a fixed-lag clone bank.
    learned_update_rate_hz: float = 20.0
    learned_max_position_clones: int = 11
    # Held-out IMO traces exposed rare horizontal covariance collapse
    # (millimetre sigma with roughly 0.2 m error). Keep learned heteroscedastic
    # uncertainty, but prevent implausibly confident EKF measurements.
    learned_sigma_floor_xyz_m: tuple[float, float, float] = (0.10, 0.10, 0.01)
    learned_covariance_scale: float = 1.25
    # Evaluation/debug switch: still run the 20 Hz network and clone schedule
    # but do not inject its displacement into the EKF. This enables shadow-mode
    # measurement diagnostics without changing the deployed input contract.
    learned_apply_displacement_updates: bool = True

    vehicle_mass_kg: float = 0.6076
    # The PnP builder already estimates world-position covariance from corner
    # perturbations. Until rotational uncertainty is propagated explicitly,
    # use a conservative fixed rotation sigma for the gate orientation update.
    gate_orientation_sigma_deg: float = 5.0
    gate_use_orientation_update: bool = True
    gate_position_mahalanobis2_max: float = 16.27  # chi2(3), ~99.9%

    # Deployment rule: simulator GT is used only once at reset to provide the
    # known fixed initial pose. It is never read by policy observations or
    # estimator updates after initialization.
    initialize_from_fixed_start_truth: bool = True

    def __post_init__(self) -> None:
        super().__post_init__()
        # Sensor-faithful first implementation is single-environment because
        # Stage2 detector/PnP is currently single-stream. Vectorized learned-IO
        # training is a later optimization, not a reason to leak GT state.
        self.scene.num_envs = 1
        self.commands.target.randomise_start = None
        self.events.push_robot = None
