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
    # Overlapping learned-motion predictions: a 0.5 s history window is
    # evaluated every 0.05 s. V6.2 stores UZH-style stochastic [R, v, p]
    # clones. Each learned factor is formed between a historical start clone
    # and a cloned endpoint; the full Kalman gain updates the correlated state.
    learned_update_rate_hz: float = 20.0
    # Optional lower EKF fusion rate while keeping the TCN evaluated at
    # learned_update_rate_hz. For a 0.5 s window, 2 Hz gives non-overlapping
    # measurements and avoids double-counting the same sensor history. None
    # preserves the legacy behavior and fuses every prediction.
    learned_fusion_rate_hz: float | None = None
    learned_max_position_clones: int = 11
    # Held-out IMO traces exposed rare horizontal covariance collapse
    # (millimetre sigma with roughly 0.2 m error). Keep learned heteroscedastic
    # uncertainty, but prevent implausibly confident EKF measurements.
    learned_sigma_floor_xyz_m: tuple[float, float, float] = (0.10, 0.10, 0.01)
    learned_covariance_scale: float = 1.25
    # Extra diagnostic/runtime multiplier applied to the final learned
    # measurement covariance. This is separate from the per-axis sigma floor
    # and lets us test whether overlapping windows are over-counting
    # information without changing the trained network.
    learned_measurement_covariance_multiplier: float = 1.0
    # Offline-calibrated network mean error, defined as E[prediction - truth]
    # in the learned target frame [m]. Runtime network measurements subtract
    # this vector before fusion. Keep zero unless estimated from an independent
    # train/validation split; never fit it on the evaluation replay.
    learned_network_bias_m: tuple[float, float, float] = (0.0, 0.0, 0.0)
    # Diagnostic-only learned relative-motion Kalman-gain constraint. "full"
    # preserves the production path. The freeze modes zero selected gain rows
    # before both state injection and Joseph covariance update, allowing us to
    # isolate legacy current-vs-clone gain behavior. V6.2's UZH-style
    # two-clone learned factor deliberately bypasses these masks and always
    # uses the full covariance-consistent Kalman gain. Absolute gate pose
    # updates also use the full gain.
    learned_kalman_gain_mode: str = "full"
    learned_filter_structure: str = "uzh_two_clone_full"
    # Evaluation/debug switch: still run the 20 Hz network and clone schedule
    # but do not inject its displacement into the EKF. This enables shadow-mode
    # measurement diagnostics without changing the deployed input contract.
    learned_apply_displacement_updates: bool = True
    # Diagnostic-only oracle measurement source. When enabled, the TCN still
    # runs for shadow accuracy metrics, but EKF fusion receives the exact GT
    # kinematic residual for the same window. This must never be enabled for
    # training/deployment; it isolates EKF measurement-model correctness.
    learned_debug_oracle_residual_fusion: bool = False
    # Diagnostic-only exact-paper factor. Keep running the V6.1 network for
    # shadow metrics, but fuse GT (p_j - p_i) through the pure UZH two-clone
    # relative-position Jacobian H=[-I,+I]. This isolates the stochastic-clone
    # EKF structure from the V6.1 kinematic-residual measurement extension.
    learned_debug_oracle_uzh_displacement_fusion: bool = False
    # Diagnostic-only gauge-invariant full displacement factor in the endpoint
    # body frame: z = R_j^T (p_j - p_i). This isolates the V6.1 start-velocity
    # subtraction from the body-frame representation.
    learned_debug_oracle_body_end_displacement_fusion: bool = False
    # Diagnostic-only three-clone acceleration-like factor:
    # z = p_end - 2*p_mid + p_start - g*half_dt^2.
    # With equally spaced clones this removes absolute position and constant
    # initial velocity while keeping a position-only Jacobian [+I,-2I,+I].
    learned_debug_oracle_second_difference_fusion: bool = False
    # Diagnostic only: rotate TCN gyro/thrust features with simulator truth
    # attitude instead of EKF attitude. If this materially improves fusion, it
    # proves the current world-frame learned measurement is state-dependent in
    # a way the EKF Jacobian does not model.
    learned_debug_truth_orientation_for_features: bool = False

    # Synthetic onboard-IMU corruption for estimator robustness experiments.
    # White-noise sigmas are per simulated sample. Bias random-walk sigmas use
    # units per sqrt(second), so the increment is sigma * sqrt(dt) * N(0, 1).
    # All defaults are zero to preserve the existing ideal-IMU baseline.
    imu_noise_seed: int = 0
    imu_accel_white_noise_sigma_mps2: float = 0.0
    imu_gyro_white_noise_sigma_radps: float = 0.0
    imu_accel_initial_bias_sigma_mps2: float = 0.0
    imu_gyro_initial_bias_sigma_radps: float = 0.0
    imu_accel_bias_rw_sigma_mps2_sqrt_s: float = 0.0
    imu_gyro_bias_rw_sigma_radps_sqrt_s: float = 0.0

    # EKF process-noise assumptions. These are deliberately independent of the
    # injected sensor corruption so consistency/mismatch can be ablated.
    ekf_accel_noise_sigma: float = 0.01
    ekf_gyro_noise_sigma: float = 0.001
    ekf_accel_bias_rw_sigma: float = 0.001
    ekf_gyro_bias_rw_sigma: float = 0.0001

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
