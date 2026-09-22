"""Configuration for OpenVINS-free learned-inertial drone racing."""

from __future__ import annotations

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.utils import configclass

from . import mdp
from .drone_racer_env_cfg import ActionsCfg, CommandsCfg
from .drone_racer_swift_perception_env_cfg import (
    DroneRacerSwiftPerceptionEnvCfg,
    SwiftPerceptionRewardsCfg,
)
from .track_generator import EASY_7_GATE_TRACK_CONFIG, generate_track


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
class LearnedInertialCommandsCfg(CommandsCfg):
    """Actor mission target advanced from the learned-inertial estimate."""

    target = mdp.EstimatedStateGateTargetingCommandCfg(
        asset_name="robot",
        track_name="track",
        randomise_start=None,
        record_fpv=False,
        resampling_time_range=(1e9, 1e9),
        debug_vis=True,
    )


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
    commands: LearnedInertialCommandsCfg = LearnedInertialCommandsCfg()

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
    # Delta-velocity checkpoints predict m/s rather than metres and therefore
    # require a unit-consistent covariance floor. Start conservatively at the
    # Oracle sigma that produced stable full-gain updates; calibrate from heldout
    # network errors before final deployment.
    learned_delta_velocity_sigma_floor_xyz_mps: tuple[float, float, float] = (
        0.05,
        0.05,
        0.05,
    )
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
    learned_delta_velocity_network_bias_mps: tuple[float, float, float] = (
        0.0,
        0.0,
        0.0,
    )
    # Runtime fusion parameterization for endpoint-body delta-velocity
    # checkpoints. "body_end" preserves the original nonlinear measurement
    # h=R_end^T[(v_end-v_start)-g*dt]. "world_nominal" rotates the network
    # measurement/covariance by the nominal endpoint attitude and fuses the
    # equivalent world-frame velocity-only factor. The latter deliberately
    # prevents the learned factor from acting as a strong attitude
    # pseudo-measurement; it is used after Circular-12 Oracle tests showed the
    # body-end coupling destabilizes the EKF even with exact measurements.
    learned_delta_velocity_body_end_fusion_frame: str = "body_end"
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
    # Diagnostic-only two-clone velocity increment:
    # z = (v_end - v_start) - g*dt. This removes the unknown initial velocity
    # from the learned target while keeping a first-difference velocity-only H.
    learned_debug_oracle_delta_velocity_fusion: bool = False
    # Gauge-invariant endpoint-body version of the delta-velocity Oracle:
    # z = R_end^T[(v_end-v_start)-g*dt].
    learned_debug_oracle_body_end_delta_velocity_fusion: bool = False
    # Evaluation-only online audit. The TCN runs on the exact production
    # runtime window/features, but its prediction is compared against the
    # simulator-truth target for the same timestamps. This never changes the
    # estimator state and is intended to distinguish offline/runtime feature
    # mismatch from Kalman-fusion effects.
    learned_debug_online_truth_audit: bool = False
    learned_debug_oracle_delta_velocity_sigma_mps: float = 0.05
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

    # Gate measurement model. "pnp_pose" preserves the V6.3 Swift-style
    # detector->IPPE->absolute-pose path for ablations. V6.4 production
    # experiments use "direct_reprojection": detected gate pixels are compared
    # directly against mapped 3-D gate corners inside the inertial EKF.
    gate_measurement_model: str = "pnp_pose"
    # Pixel-domain uncertainty from the independent Stage2 detector
    # calibration. This is a per-coordinate sigma, not a 2-D radial RMSE.
    gate_reprojection_sigma_px: float = 0.85
    # Require at least two semantic corners, matching the intended partial-gate
    # direct-reprojection use case. Four-corner completeness is no longer a
    # prerequisite for an EKF visual update.
    gate_reprojection_min_visible_corners: int = 2
    # Association is performed without PnP: every mapped gate is projected
    # through the current inertial state and scored in pixel space.
    gate_reprojection_association_max_rmse_px: float = 80.0
    gate_reprojection_min_depth_m: float = 0.05
    # Robust innovation weighting and a permissive final consistency gate.
    gate_reprojection_huber_delta_sigma: float = 2.5
    gate_reprojection_max_normalized_nis: float | None = 25.0

    # V6.5 perception robustness stress hooks. Defaults are exactly neutral,
    # so production-style V6.4 behavior is preserved unless an evaluator
    # explicitly enables corruption. These hooks operate on camera/gate
    # observations only; they never alter simulator truth or policy rewards.
    gate_stress_seed: int = 0
    # Independent random full-frame loss probability at each 30 Hz camera
    # sample. A deterministic burst can be superimposed below.
    gate_stress_frame_drop_probability: float = 0.0
    gate_stress_burst_start_s: float = 10.0
    gate_stress_burst_duration_s: float = 0.0
    # Additional isotropic Gaussian corruption applied to detector corner
    # pixels before association/fusion. The EKF measurement sigma remains the
    # configured gate_reprojection_sigma_px, intentionally exposing mismatch.
    gate_stress_pixel_noise_sigma_px: float = 0.0
    # Independent per-semantic-corner visibility erasure after the detector.
    gate_stress_corner_drop_probability: float = 0.0
    # Processing delay applied after corner detection. The delayed measurement
    # is fused against the current state (no OOSM rewind), intentionally testing
    # robustness to uncompensated perception latency.
    gate_stress_latency_s: float = 0.0

    # V6.6 delayed-vision handling. When enabled, camera-time [R,v,p] clones
    # are kept and delayed pixel residuals are evaluated at the capture-time
    # clone. The full stochastic-clone covariance then corrects the current
    # state through cross-correlation.
    gate_reprojection_compensate_latency: bool = True
    gate_reprojection_latency_clone_tolerance_s: float = 1.0e-6

    # Evaluation-only audit switch. When enabled, the runtime records simulator
    # truth alongside Gate-PnP measurements so detector/PnP/association quality
    # can be diagnosed. Truth is never fed into the estimator update itself.
    gate_debug_gt_diagnostics: bool = False

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


@configclass
class DroneRacerLearnedInertialRLCfg(DroneRacerLearnedInertialEnvCfg):
    """Frozen deployment-faithful estimator configuration for PPO integration.

    This profile intentionally encodes the V6.4/V6.5/V6.6 settings validated
    before RL so that ordinary training does not depend on evaluator-only CLI
    overrides.
    """

    # Make the Swift-2023 perception-aware reward explicit for the PPO task.
    # The functional form is exp(lambda_3 * delta_cam**4), where delta_cam is
    # the angle between the calibrated camera optical axis and the next gate.
    rewards: SwiftPerceptionRewardsCfg = SwiftPerceptionRewardsCfg()

    def __post_init__(self) -> None:
        super().__post_init__()

        # Phase-2 curriculum: train the learned-inertial PPO policy on the
        # easier seven-gate loop. The original expert track remains unchanged
        # in DroneRacerSceneCfg and is therefore still available to the other
        # tasks/evaluation profiles.
        self.scene.track = generate_track(track_config=EASY_7_GATE_TRACK_CONFIG)

        # Validated learned inertial model.
        self.learned_motion_checkpoint = (
            "artifacts/imo_tcn/model_v6_2_body_delta_velocity_balanced.pt"
        )
        self.learned_motion_device = "cuda"
        self.learned_update_rate_hz = 20.0
        self.learned_fusion_rate_hz = 2.0
        self.learned_measurement_covariance_multiplier = 1.0

        # Nominal sensor corruption used by the successful V6.4/V6.5
        # validation campaign.
        self.imu_accel_white_noise_sigma_mps2 = 0.01
        self.imu_gyro_white_noise_sigma_radps = 0.001
        self.imu_accel_bias_rw_sigma_mps2_sqrt_s = 0.001
        self.imu_gyro_bias_rw_sigma_radps_sqrt_s = 0.0001

        # Production gate perception: detector + visibility guard + mapped
        # direct pixel reprojection. PnP is not part of the RL estimator path.
        self.swift_detector_checkpoint = (
            "artifacts/stage2_next_steps_20260911/checkpoints/"
            "torchvision_keypointrcnn_best.pt"
        )
        self.swift_visibility_checkpoint = (
            "artifacts/stage2_next_steps_20260911/checkpoints/"
            "gate_keypoint_net_best.pt"
        )
        self.gate_measurement_model = "direct_reprojection"
        self.gate_reprojection_sigma_px = 0.85
        self.gate_reprojection_min_visible_corners = 2
        self.gate_reprojection_association_max_rmse_px = 80.0
        self.gate_reprojection_min_depth_m = 0.05
        self.gate_reprojection_huber_delta_sigma = 2.5
        self.gate_reprojection_max_normalized_nis = 25.0

        # Stress hooks are disabled for the nominal PPO task. They can be
        # re-enabled later as domain-randomization/curriculum experiments.
        self.gate_stress_frame_drop_probability = 0.0
        self.gate_stress_burst_duration_s = 0.0
        self.gate_stress_pixel_noise_sigma_px = 0.0
        self.gate_stress_corner_drop_probability = 0.0
        self.gate_stress_latency_s = 0.0

        # No oracle/debug estimator inputs in the actor training task.
        self.gate_debug_gt_diagnostics = False
        self.learned_debug_oracle_residual_fusion = False
        self.learned_debug_oracle_uzh_displacement_fusion = False
        self.learned_debug_oracle_body_end_displacement_fusion = False
        self.learned_debug_oracle_second_difference_fusion = False
        self.learned_debug_oracle_delta_velocity_fusion = False
        self.learned_debug_oracle_body_end_delta_velocity_fusion = False
        self.learned_debug_truth_orientation_for_features = False

        # Fixed known start is part of the deployment contract used throughout
        # the estimator validation. Mission progression itself is estimator
        # driven by EstimatedStateGateTargetingCommand.
        self.initialize_from_fixed_start_truth = True
        self.commands.target.randomise_start = None

        # The diagnostic Swift task intentionally disabled root-state reset,
        # which is fine for one-shot estimator replays but wrong for episodic
        # RL. Restore an exact fixed-start reset so simulator state and the
        # fixed-start estimator reset remain synchronized after termination.
        self.events.reset_base = EventTerm(
            func=mdp.reset_root_state_uniform,
            mode="reset",
            params={
                "pose_range": {
                    "x": (0.0, 0.0),
                    "y": (0.0, 0.0),
                    "z": (0.0, 0.0),
                    "roll": (0.0, 0.0),
                    "pitch": (0.0, 0.0),
                    "yaw": (0.0, 0.0),
                },
                "velocity_range": {
                    "x": (0.0, 0.0),
                    "y": (0.0, 0.0),
                    "z": (0.0, 0.0),
                    "roll": (0.0, 0.0),
                    "pitch": (0.0, 0.0),
                    "yaw": (0.0, 0.0),
                },
            },
        )
