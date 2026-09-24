"""Swift-style CTBR control configurations.

The control-only task isolates collective-thrust/body-rate control from the
learned estimator so hover and rate tracking can be tuned before PPO.
The learned-inertial CTBR task keeps the validated TCN + Stage2 + SC-EKF stack
but swaps the old direct-motor action for the new CTBR low-level controller.
"""

from __future__ import annotations

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

from perception.stage2_calibration import load_stage2_gate_geometry

from . import mdp
from .drone_racer_env_cfg import DroneRacerEnvCfg, DroneRacerEnvCfg_PLAY, RewardsCfg
from .drone_racer_learned_inertial_env_cfg import DroneRacerLearnedInertialRLCfg
from .drone_racer_stage2_env_cfg import stage2_reference_camera_cfg
from .track_generator import CIRCULAR_12_GATE_TRACK_CONFIG, CIRCULAR_12_KNOWN_START_POS_W, CIRCULAR_12_KNOWN_START_ROT_WXYZ, EASY_7_GATE_TRACK_CONFIG, generate_track





@configclass
class SwiftGTPolicyCfg(ObsGroup):
    """Vectorized ideal-training observation with the same 31D deployment layout."""

    platform_state = ObsTerm(func=mdp.swift_gt_state)
    next_gate_corners = ObsTerm(
        func=mdp.swift_gt_next_gate_corners_relative_w,
        params={"command_name": "target"},
    )
    previous_action = ObsTerm(func=mdp.last_action)

    def __post_init__(self) -> None:
        self.enable_corruption = False
        self.concatenate_terms = True


@configclass
class SwiftGTObservationsCfg:
    policy: SwiftGTPolicyCfg = SwiftGTPolicyCfg()
    critic = None


@configclass
class SwiftCTBRTrainingRewardsCfg(RewardsCfg):
    """Swift-2023 reward at 100 Hz, converted to IsaacLab dt-scaled weights.

    IsaacLab multiplies reward-term weights by step_dt=0.01. Therefore the
    published discrete-time coefficients are divided by 0.01 here:
      lambda1=1.0, lambda2=0.02, lambda3=-10,
      lambda4=-2e-4, lambda5=-1e-4, crash=5.
    """

    terminating = RewTerm(func=mdp.is_terminated, weight=-500.0)
    progress = RewTerm(
        func=mdp.progress,
        weight=100.0,
        params={"command_name": "target"},
    )
    lookat_next = RewTerm(
        func=mdp.swift_perception_awareness,
        weight=2.0,
        params={
            "command_name": "target",
            "lambda_3": -10.0,
            "camera_pos_b": (0.14, 0.0, 0.05),
            "camera_optical_axis_b": (1.0, 0.0, 0.0),
        },
    )
    body_rate_command = RewTerm(
        func=mdp.swift_ctbr_body_rate_command_l2,
        weight=-0.02,
        params={"action_name": "control_action"},
    )
    command_smoothness = RewTerm(
        func=mdp.swift_ctbr_command_delta_l2,
        weight=-0.01,
        params={"action_name": "control_action"},
    )

    # Disable legacy terms that are not part of Swift's Eq. (7)-(9).
    ang_vel_l2 = None
    gate_passed = None





@configclass
class SwiftCTBRGTRacingRewardsCfg(RewardsCfg):
    """GT-racing reward adapted from kousheekc/isaac_drone_racer.

    The important difference from policy-0 v1/v2 is that gate crossing is an
    explicit objective again. mdp.gate_passed returns +1 for a valid pass and
    -1 for crossing the gate plane outside the opening, so weight=400 supplies
    both a strong pass reward and a strong miss penalty.
    """

    terminating = RewTerm(func=mdp.is_terminated, weight=-500.0)
    ang_vel_l2 = RewTerm(func=mdp.ang_vel_l2, weight=-0.0001)
    progress = RewTerm(
        func=mdp.progress,
        weight=20.0,
        params={"command_name": "target"},
    )
    gate_passed = RewTerm(
        func=mdp.gate_passed,
        weight=400.0,
        params={"command_name": "target"},
    )
    lookat_next = RewTerm(
        func=mdp.lookat_next_gate,
        weight=0.1,
        params={"command_name": "target", "std": 0.5},
    )


@configclass
class SwiftCTBRGTStableRacingRewardsCfg(SwiftCTBRGTRacingRewardsCfg):
    """Circular-12 CTBR racing reward that suppresses gratuitous body spin.

    The current GT policy can exploit the CTBR rate action because the legacy
    angular-velocity penalty is effectively negligible relative to gate-pass
    reward.  Keep large bank angles legal (they are physically required for
    high-speed circular flight), but make sustained angular rate and violent
    rate-command changes expensive and strengthen forward/gate alignment.
    """

    ang_vel_l2 = RewTerm(func=mdp.ang_vel_l2, weight=-0.02)
    lookat_next = RewTerm(
        func=mdp.lookat_next_gate,
        weight=0.5,
        params={"command_name": "target", "std": 0.5},
    )
    body_rate_command = RewTerm(
        func=mdp.swift_ctbr_body_rate_command_l2,
        weight=-0.01,
        params={"action_name": "control_action"},
    )
    command_smoothness = RewTerm(
        func=mdp.swift_ctbr_command_delta_l2,
        weight=-0.002,
        params={"action_name": "control_action"},
    )


@configclass
class SwiftCTBRGTStableHeadingRewardsCfg(SwiftCTBRGTStableRacingRewardsCfg):
    """Discourage the nose spin seen in otherwise successful Circular-12 laps."""

    forward_velocity_heading = RewTerm(
        func=mdp.gt_forward_velocity_heading_error,
        weight=-2.0,
        params={"min_speed_mps": 5.0},
    )


@configclass
class SwiftCTBRGTStableMultiGateRewardsCfg(SwiftCTBRGTStableHeadingRewardsCfg):
    """Reward sustained usable frames from any mapped gate at the 40-degree mount."""

    multi_gate_continuity = RewTerm(
        func=mdp.gt_multigate_camera_continuity,
        weight=-4.0,
        params={
            "pitch_up_deg": 40.0,
            "margin_px": 16.0,
            "capture_every_steps": 4,
            "max_blackout_frames": 25,
        },
    )


@configclass
class SwiftCTBRGTPerceptionAwareRewardsCfg(SwiftCTBRGTRacingRewardsCfg):
    """Keep the successful GT racing objective and add camera observability.

    The policy observation remains exactly the same 31-D GT vector. This reward
    is training-only privileged information: calibrated gate corners are
    projected analytically into the production 256x256 camera image and the
    policy is rewarded for keeping the next gate usable by perception.
    """

    camera_visibility = RewTerm(
        func=mdp.gt_next_gate_image_visibility,
        weight=2.0,
        params={
            "command_name": "target",
            "margin_px": 24.0,
            "center_sigma": 1.0,
            "center_weight": 0.25,
            "coverage_weight": 0.25,
            "margin_weight": 0.20,
            "usable_bonus_weight": 0.30,
        },
    )


@configclass
class SwiftCTBRGTPerceptionAwareV2RewardsCfg(SwiftCTBRGTRacingRewardsCfg):
    """Penalty-only camera observability shaping for high-speed GT racing.

    V1 used a small positive visibility reward and the policy mainly learned to
    race faster. V2 makes camera loss explicitly costly while keeping perfect
    visibility near zero, so there is no incentive to hover just to accumulate
    perception reward.
    """

    camera_observability = RewTerm(
        func=mdp.gt_next_gate_image_visibility,
        weight=5.0,
        params={
            "command_name": "target",
            "margin_px": 24.0,
            "center_sigma": 1.0,
            "center_weight": 0.20,
            "coverage_weight": 0.25,
            "margin_weight": 0.15,
            "usable_bonus_weight": 0.40,
            "output_bias": -1.0,
            "insufficient_visible_penalty": 0.50,
        },
    )


@configclass
class SwiftCTBRGTPerceptionAwareV3RewardsCfg(SwiftCTBRGTRacingRewardsCfg):
    """Dense directional camera shaping plus an explicit >=2-corner constraint.

    V2's mostly binary visibility penalty improved racing speed but barely
    changed observability. V3 adds a squared 3-D camera-boresight angle penalty,
    which remains informative even when the gate is far outside the image.
    """

    camera_angle_l2 = RewTerm(
        func=mdp.gt_next_gate_camera_angle_l2,
        weight=-8.0,
        params={"command_name": "target"},
    )
    camera_observability = RewTerm(
        func=mdp.gt_next_gate_image_visibility,
        weight=2.0,
        params={
            "command_name": "target",
            "margin_px": 24.0,
            "center_sigma": 1.0,
            "center_weight": 0.20,
            "coverage_weight": 0.25,
            "margin_weight": 0.15,
            "usable_bonus_weight": 0.40,
            "output_bias": -1.0,
            "insufficient_visible_penalty": 0.50,
        },
    )


@configclass
class SwiftCTBRGTShadowRewardsCfg(SwiftCTBRGTRacingRewardsCfg):
    """GT-racing rewards evaluated against the truth-only shadow mission."""

    progress = RewTerm(
        func=mdp.progress_truth_gate,
        weight=20.0,
        params={"command_name": "target"},
    )
    lookat_next = RewTerm(
        func=mdp.lookat_truth_gate,
        weight=0.1,
        params={"command_name": "target", "std": 0.5},
    )


@configclass
class SwiftGTShadowPolicyCfg(ObsGroup):
    """GT-control observation for estimator-shadow racing diagnostics."""

    platform_state = ObsTerm(func=mdp.swift_gt_state)
    next_gate_corners = ObsTerm(
        func=mdp.swift_gt_truth_next_gate_corners_relative_w,
        params={"command_name": "target"},
    )
    previous_action = ObsTerm(func=mdp.last_action)

    def __post_init__(self) -> None:
        self.enable_corruption = False
        self.concatenate_terms = True


@configclass
class SwiftGTShadowObservationsCfg:
    policy: SwiftGTShadowPolicyCfg = SwiftGTShadowPolicyCfg()
    critic = None


@configclass
class LearnedInertialSwiftPolicyCfg(ObsGroup):
    """Swift-style 31D actor observation from estimator + known gate map."""

    platform_state = ObsTerm(func=mdp.learned_inertial_swift_state)
    next_gate_corners = ObsTerm(
        func=mdp.learned_next_gate_corners_relative_w,
        params={"command_name": "target"},
    )
    previous_action = ObsTerm(func=mdp.last_action)

    def __post_init__(self) -> None:
        self.enable_corruption = False
        self.concatenate_terms = True


@configclass
class LearnedInertialSwiftObservationsCfg:
    policy: LearnedInertialSwiftPolicyCfg = LearnedInertialSwiftPolicyCfg()
    critic = None


@configclass
class SwiftCTBRActionsCfg:
    control_action: mdp.SwiftCTBRActionCfg = mdp.SwiftCTBRActionCfg()


@configclass
class DroneRacerSwiftCTBRControlEnvCfg(DroneRacerEnvCfg_PLAY):
    """Minimal physics task used to tune hover and body-rate tracking."""

    actions: SwiftCTBRActionsCfg = SwiftCTBRActionsCfg()

    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.num_envs = 1
        self.scene.robot.init_state.pos = (-4.0, 0.0, 3.0)
        self.commands.target.randomise_start = None
        self.episode_length_s = 30.0

        # Exact deterministic reset: no random pose/rate disturbance while
        # tuning the low-level controller.
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





@configclass
class DroneRacerSwiftCTBRTrainEnvCfg(DroneRacerEnvCfg):
    """Vectorized ideal policy-0 training task, analogous to Swift simulation."""

    observations: SwiftGTObservationsCfg = SwiftGTObservationsCfg()
    actions: SwiftCTBRActionsCfg = SwiftCTBRActionsCfg()
    rewards: SwiftCTBRTrainingRewardsCfg = SwiftCTBRTrainingRewardsCfg()

    def __post_init__(self) -> None:
        super().__post_init__()

        # Swift trains 100 agents in parallel for 1,500 control steps/episode.
        self.scene.num_envs = 100
        self.episode_length_s = 15.0

        # Use the same easy seven-gate curriculum geometry as the validated
        # deployment stack, but keep training fully vectorized and sensor-free.
        self.scene.track = generate_track(track_config=EASY_7_GATE_TRACK_CONFIG)

        # Preserve the original policy-0 v1 reset semantics for checkpoint
        # reproducibility: random predecessor gate, but zero initial velocity.
        self.commands.target.randomise_start = True
        self.commands.target.debug_vis = False

        # No random external pushes in policy-0. Swift's first-stage training
        # uses the nominal simulator; empirical residuals are introduced only
        # during the later fine-tuning stage.
        self.events.push_robot = None








@configclass
class DroneRacerSwiftCTBRGTRacingEnvCfg(DroneRacerSwiftCTBRTrainEnvCfg):
    """Massively parallel GT racing task inspired by kousheekc's baseline.

    This stage deliberately uses simulator GT for the 31D actor observation.
    Its job is only to learn a strong racing policy. The final deployment task
    later swaps the GT state source for the learned-inertial / visual estimator
    while preserving the same policy interface and CTBR action semantics.
    """

    rewards: SwiftCTBRGTRacingRewardsCfg = SwiftCTBRGTRacingRewardsCfg()

    def __post_init__(self) -> None:
        super().__post_init__()

        # Match the original Isaac Drone Racer massively-parallel regime.
        self.scene.num_envs = 4096
        self.episode_length_s = 20.0

        # Keep the current Easy-7 curriculum geometry while learning the GT
        # racing policy. We can promote the trained policy to the Expert track
        # after it demonstrates consistent multi-gate / full-lap behavior.
        self.scene.track = generate_track(track_config=EASY_7_GATE_TRACK_CONFIG)

        # Match the original baseline's random-gate reset behavior. No camera,
        # IMU, VIO or EKF is used by the actor during this GT racing stage.
        self.commands.target.randomise_start = True
        self.commands.target.debug_vis = False
        self.events.push_robot = None


@configclass
class DroneRacerSwiftCTBRGTCircular12RacingEnvCfg(
    DroneRacerSwiftCTBRGTRacingEnvCfg
):
    """GT-only upper-bound racing task on the camera-compatible 12-gate loop.

    The 31-D actor observation, CTBR action interface, PPO profile and racing
    reward remain identical to the successful GT baseline.  Only the known
    track geometry changes.  This deliberately avoids perception-aware reward
    shaping while we establish whether the denser circular map naturally gives
    the production camera enough gate-corner observability.
    """

    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.track = generate_track(
            track_config=CIRCULAR_12_GATE_TRACK_CONFIG
        )


@configclass
class DroneRacerSwiftCTBRGTCircular12StableRacingEnvCfg(
    DroneRacerSwiftCTBRGTCircular12RacingEnvCfg
):
    """Circular-12 GT task with anti-spin CTBR limits and reward shaping."""

    rewards: SwiftCTBRGTStableRacingRewardsCfg = SwiftCTBRGTStableRacingRewardsCfg()

    def __post_init__(self) -> None:
        super().__post_init__()
        # Preserve aggressive racing authority while removing the 10 rad/s
        # roll/pitch and 6 rad/s yaw rates that enabled sustained tumbling.
        self.actions.control_action.body_rate_max_radps = (6.0, 6.0, 3.0)


@configclass
class DroneRacerSwiftCTBRGTCircular12StableHeadingEnvCfg(
    DroneRacerSwiftCTBRGTCircular12StableRacingEnvCfg
):
    """Stable-v1 dynamics with a forward-flight heading objective."""

    rewards: SwiftCTBRGTStableHeadingRewardsCfg = SwiftCTBRGTStableHeadingRewardsCfg()


@configclass
class DroneRacerSwiftCTBRGTCircular12StableMultiGateEnvCfg(
    DroneRacerSwiftCTBRGTCircular12StableHeadingEnvCfg
):
    """GT racing with a 25-Hz multi-gate camera-continuity objective."""

    rewards: SwiftCTBRGTStableMultiGateRewardsCfg = SwiftCTBRGTStableMultiGateRewardsCfg()


@configclass
class DroneRacerSwiftCTBRGTCircular12StableKnownStartEnvCfg(
    DroneRacerSwiftCTBRGTCircular12StableRacingEnvCfg
):
    """Stable anti-spin Circular-12 benchmark from the exact known start."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.num_envs = 1
        self.episode_length_s = 20.0
        self.scene.robot.init_state.pos = CIRCULAR_12_KNOWN_START_POS_W
        self.scene.robot.init_state.rot = CIRCULAR_12_KNOWN_START_ROT_WXYZ
        self.commands.target.randomise_start = None
        self.commands.target.debug_vis = False
        self.events.push_robot = None

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


@configclass
class DroneRacerSwiftCTBRGTPerceptionAwareEnvCfg(
    DroneRacerSwiftCTBRGTRacingEnvCfg
):
    """37-gate GT racing curriculum with explicit image-space visibility reward.

    Every training specification inherited from the successful GT racing task
    stays unchanged: Easy-7, 4096 environments, 20 s episodes, the same 31-D
    GT observation, CTBR action semantics and PPO configuration. Only the
    reward adds the calibrated camera-observability term.
    """

    rewards: SwiftCTBRGTPerceptionAwareRewardsCfg = (
        SwiftCTBRGTPerceptionAwareRewardsCfg()
    )


@configclass
class DroneRacerSwiftCTBRGTPerceptionAwareV2EnvCfg(
    DroneRacerSwiftCTBRGTRacingEnvCfg
):
    """Matched 37-gate GT task with strong penalty-only camera observability."""

    rewards: SwiftCTBRGTPerceptionAwareV2RewardsCfg = (
        SwiftCTBRGTPerceptionAwareV2RewardsCfg()
    )


@configclass
class DroneRacerSwiftCTBRGTPerceptionAwareV3EnvCfg(
    DroneRacerSwiftCTBRGTRacingEnvCfg
):
    """Matched GT racing task with dense camera-direction shaping."""

    rewards: SwiftCTBRGTPerceptionAwareV3RewardsCfg = (
        SwiftCTBRGTPerceptionAwareV3RewardsCfg()
    )


@configclass
class DroneRacerSwiftCTBRGTCircular12FixedStartEnvCfg(
    DroneRacerSwiftCTBRGTCircular12RacingEnvCfg
):
    """Pure-GT fixed-start reference on the Circular-12 estimator test track."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.num_envs = 1
        self.episode_length_s = 20.0

        gate_opening_center_g = load_stage2_gate_geometry().center_g
        self.scene.robot.init_state.pos = (
            -4.0,
            0.0,
            1.0 + float(gate_opening_center_g[2]),
        )
        self.scene.robot.init_state.rot = (1.0, 0.0, 0.0, 0.0)
        self.commands.target.randomise_start = None
        self.commands.target.debug_vis = False
        self.events.push_robot = None

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


@configclass
class DroneRacerSwiftCTBRGTCircular12KnownStartEnvCfg(
    DroneRacerSwiftCTBRGTCircular12RacingEnvCfg
):
    """GT Circular-12 benchmark from an exact training-support known start.

    Unlike the older four-metres-before-gate diagnostic start, this initial
    state is the zero-jitter sample of the same predecessor-gate reset used by
    GT training.  It is therefore suitable for a controlled GT-vs-estimator
    comparison without introducing a separate policy distribution shift.
    """

    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.num_envs = 1
        self.episode_length_s = 20.0
        self.scene.robot.init_state.pos = CIRCULAR_12_KNOWN_START_POS_W
        self.scene.robot.init_state.rot = CIRCULAR_12_KNOWN_START_ROT_WXYZ
        self.commands.target.randomise_start = None
        self.commands.target.debug_vis = False
        self.events.push_robot = None

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


@configclass
class DroneRacerSwiftCTBRGTFixedStartEnvCfg(DroneRacerSwiftCTBRGTRacingEnvCfg):
    """Pure-GT fixed-start reference for GTShadow policy-isolation tests.

    This task uses the successful GT actor/command/reward/control path but the
    exact same known fixed start as the learned-inertial deployment task. It
    contains no camera, IMU, learned model, detector, or EKF, so it cleanly
    measures whether the frozen GT policy itself is robust to the deployment
    start-state distribution.
    """

    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.num_envs = 1
        self.episode_length_s = 20.0

        gate_opening_center_g = load_stage2_gate_geometry().center_g
        self.scene.robot.init_state.pos = (
            -4.0,
            0.0,
            1.0 + float(gate_opening_center_g[2]),
        )
        self.scene.robot.init_state.rot = (1.0, 0.0, 0.0, 0.0)
        self.commands.target.randomise_start = None
        self.commands.target.debug_vis = False
        self.events.push_robot = None

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


@configclass
class DroneRacerSwiftCTBRPassStateTrainEnvCfg(DroneRacerSwiftCTBRTrainEnvCfg):
    """Policy-0 v2: Swift-like through-gate momentum reset curriculum."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.commands.target = mdp.SwiftPassStateGateTargetingCommandCfg(
            asset_name="robot",
            track_name="track",
            randomise_start=True,
            record_fpv=False,
            resampling_time_range=(1.0e9, 1.0e9),
            debug_vis=False,
            forward_speed_range_mps=(1.5, 3.0),
            post_gate_offset_m=1.0,
        )


@configclass
class DroneRacerLearnedInertialSwiftCTBRRLCfg(DroneRacerLearnedInertialRLCfg):
    """Full sensor-faithful learned-inertial stack with Swift CTBR actuation."""

    observations: LearnedInertialSwiftObservationsCfg = LearnedInertialSwiftObservationsCfg()
    actions: SwiftCTBRActionsCfg = SwiftCTBRActionsCfg()

    def __post_init__(self) -> None:
        super().__post_init__()
        # Keep one sensor-faithful environment for final closed-loop validation.
        # PPO training will later use a separate vectorized residual/noise task.
        self.scene.num_envs = 1


@configclass
class DroneRacerLearnedInertialSwiftCTBRCircular12GTPolicyCfg(
    DroneRacerLearnedInertialSwiftCTBRRLCfg
):
    """Frozen Circular-12 GT policy driven only by the production estimator.

    This is the controlled GT-replacement experiment: policy architecture,
    checkpoint, CTBR action and known track map are unchanged.  Only the
    platform-state source changes from simulator truth to learned_inertial_state
    (IMU + learned motion + SC-EKF + mapped-gate reprojection).
    """

    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.track = generate_track(
            track_config=CIRCULAR_12_GATE_TRACK_CONFIG
        )


@configclass
class DroneRacerLearnedInertialSwiftCTBRCircular12KnownStartGTPolicyCfg(
    DroneRacerLearnedInertialSwiftCTBRCircular12GTPolicyCfg
):
    """Estimator-driven Circular-12 task from the matched known GT start."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.robot.init_state.pos = CIRCULAR_12_KNOWN_START_POS_W
        self.scene.robot.init_state.rot = CIRCULAR_12_KNOWN_START_ROT_WXYZ
        self.commands.target.randomise_start = None


@configclass
class DroneRacerLearnedInertialSwiftCTBRGTShadowCfg(
    DroneRacerLearnedInertialSwiftCTBRRLCfg
):
    """GT-controlled racing while the learned estimator runs in shadow.

    The actor receives simulator-truth 31D observations and truth-only mission
    progression, reproducing the successful GT racing trajectory distribution.
    TCN + IMU + SC-EKF + Stage2 continue running in the background so their
    errors can be measured without feeding back into the control policy.
    """

    observations: SwiftGTShadowObservationsCfg = SwiftGTShadowObservationsCfg()
    rewards: SwiftCTBRGTShadowRewardsCfg = SwiftCTBRGTShadowRewardsCfg()

    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.num_envs = 1
        self.episode_length_s = 20.0

        self.terminations.flyaway = DoneTerm(
            func=mdp.flyaway_truth_gate,
            params={"command_name": "target", "distance": 20.0},
        )




@configclass
class DroneRacerLearnedInertialSwiftCTBRCircular12KnownStartGTPolicyV7Cfg(
    DroneRacerLearnedInertialSwiftCTBRCircular12KnownStartGTPolicyCfg
):
    """Circular-12 estimator-driven GT-policy task with only the TCN upgraded to V7."""

    def __post_init__(self) -> None:
        super().__post_init__()
        # Controlled ablation: preserve camera, EKF, fusion cadence, covariance
        # protection, CTBR, map and policy checkpoint.  Change only the learned
        # motion checkpoint from the mild-motion V6.2 model to the racing V7 model.
        self.learned_motion_checkpoint = (
            "artifacts/imo_tcn/model_v7_circular12_racing.pt"
        )


@configclass
class DroneRacerLearnedInertialSwiftCTBRCircular12KnownStartGTShadowV7Cfg(
    DroneRacerLearnedInertialSwiftCTBRGTShadowCfg
):
    """GT-controlled Circular-12 shadow run using the V7 racing TCN.

    The actor stays on simulator truth while the full production estimator runs
    in the background.  This isolates estimator integration from closed-loop
    feedback before V7 is allowed to drive the frozen GT racing policy.
    """

    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.track = generate_track(
            track_config=CIRCULAR_12_GATE_TRACK_CONFIG
        )
        self.scene.robot.init_state.pos = CIRCULAR_12_KNOWN_START_POS_W
        self.scene.robot.init_state.rot = CIRCULAR_12_KNOWN_START_ROT_WXYZ
        self.commands.target.randomise_start = None
        self.learned_motion_checkpoint = (
            "artifacts/imo_tcn/model_v7_circular12_racing.pt"
        )


@configclass
class DroneRacerLearnedInertialSwiftCTBRCircular12KnownStartGTShadowV7NoVisionCfg(
    DroneRacerLearnedInertialSwiftCTBRCircular12KnownStartGTShadowV7Cfg
):
    """V7 GT-shadow ablation with camera fusion disabled.

    This keeps IMU propagation + V7 learned delta-velocity fusion unchanged and
    removes the sparse legacy detector updates, isolating inertial/learned
    integration from perception.
    """

    def __post_init__(self) -> None:
        super().__post_init__()
        self.swift_detector_checkpoint = None
        self.swift_visibility_checkpoint = None


@configclass
class DroneRacerLearnedInertialSwiftCTBRCircular12KnownStartGTShadowV7ImuOnlyCfg(
    DroneRacerLearnedInertialSwiftCTBRCircular12KnownStartGTShadowV7NoVisionCfg
):
    """GT-shadow IMU-only control experiment; V7 runs but is not fused."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.learned_apply_displacement_updates = False


@configclass
class DroneRacerLearnedInertialSwiftCTBRCircular12KnownStartGTShadowV7OnlineAuditCfg(
    DroneRacerLearnedInertialSwiftCTBRCircular12KnownStartGTShadowV7ImuOnlyCfg
):
    """IMU-only GT-shadow with V7 online prediction-vs-truth diagnostics."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.learned_debug_online_truth_audit = True


@configclass
class DroneRacerLearnedInertialSwiftCTBRCircular12KnownStartGTShadowV7OracleDVCfg(
    DroneRacerLearnedInertialSwiftCTBRCircular12KnownStartGTShadowV7NoVisionCfg
):
    """GT-shadow exact endpoint-body delta-velocity fusion without vision.

    The V7 network still runs for shadow diagnostics, but the EKF receives the
    exact GT measurement for the identical 0.5 s clone window.  If this mode
    diverges, the fault is in clone/EKF delta-velocity integration rather than
    the learned model.
    """

    def __post_init__(self) -> None:
        super().__post_init__()
        self.learned_debug_oracle_body_end_delta_velocity_fusion = True


@configclass
class DroneRacerLearnedInertialSwiftCTBRCircular12KnownStartGTShadowV7OracleWorldDVCfg(
    DroneRacerLearnedInertialSwiftCTBRCircular12KnownStartGTShadowV7NoVisionCfg
):
    """GT-shadow exact WORLD-frame gravity-compensated delta-velocity fusion.

    This removes the endpoint-body attitude term from the learned factor while
    preserving the same two-clone velocity factor and fusion cadence.  Comparing
    this against OracleDV isolates body-frame attitude coupling/Jacobian effects.
    """

    def __post_init__(self) -> None:
        super().__post_init__()
        self.learned_debug_oracle_delta_velocity_fusion = True


@configclass
class DroneRacerLearnedInertialSwiftCTBRCircular12KnownStartGTShadowV7WorldProjectedCfg(
    DroneRacerLearnedInertialSwiftCTBRCircular12KnownStartGTShadowV7NoVisionCfg
):
    """V7 network fused through the stable world-frame velocity-only factor."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.learned_delta_velocity_body_end_fusion_frame = "world_nominal"


@configclass
class DroneRacerLearnedInertialSwiftCTBRCircular12KnownStartGTShadowV7WorldProjectedFreezeAttBiasCfg(
    DroneRacerLearnedInertialSwiftCTBRCircular12KnownStartGTShadowV7WorldProjectedCfg
):
    """World-projected V7 with learned DV prevented from correcting attitude/bias."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.learned_delta_velocity_gain_mode = "freeze_attitude_bias"


@configclass
class DroneRacerLearnedInertialSwiftCTBRCircular12KnownStartGTShadowV7CalibratedFreezeAttBiasCfg(
    DroneRacerLearnedInertialSwiftCTBRCircular12KnownStartGTShadowV7WorldProjectedCfg
):
    """Validation-bias calibrated V7 with attitude/bias protected from learned DV."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.learned_delta_velocity_gain_mode = "freeze_attitude_bias"
        self.learned_delta_velocity_calibration_path = (
            "artifacts/imo_tcn/model_v7_circular12_racing.pt."
            "fusion_calibration.json"
        )


@configclass
class DroneRacerLearnedInertialSwiftCTBRCircular12KnownStartGTShadowV7CalibratedVelocityOnlyCfg(
    DroneRacerLearnedInertialSwiftCTBRCircular12KnownStartGTShadowV7WorldProjectedCfg
):
    """Validation-bias calibrated V7 with learned DV restricted to velocity states."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.learned_delta_velocity_gain_mode = "freeze_position_attitude_bias"
        self.learned_delta_velocity_calibration_path = (
            "artifacts/imo_tcn/model_v7_circular12_racing.pt."
            "fusion_calibration.json"
        )


@configclass
class DroneRacerLearnedInertialSwiftCTBRCircular12KnownStartGTShadowMultiGateVisionCfg(
    DroneRacerLearnedInertialSwiftCTBRCircular12KnownStartGTShadowV7ImuOnlyCfg
):
    """GT-shadow Circular-12 estimator with IMU propagation + new multi-gate vision."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.tiled_camera = stage2_reference_camera_cfg(
            pitch_up_deg=40.0
        )
        self.gate_camera_pitch_up_deg = 40.0
        self.swift_detector_checkpoint = (
            "artifacts/racing_vision/circular12_pitch40_multigate/"
            "torchvision_keypointrcnn_multigate_best.pt"
        )
        self.swift_visibility_checkpoint = None
        self.swift_detection_threshold = 0.35
        self.swift_keypoint_confidence_threshold = 0.35
        self.gate_measurement_model = "direct_reprojection"
        self.gate_reprojection_use_checkpoint_sigma = True
        self.gate_reprojection_min_visible_corners = 2
        self.gate_reprojection_association_max_rmse_px = 80.0
        self.gate_reprojection_huber_delta_sigma = 2.5
        self.gate_reprojection_max_normalized_nis = 25.0


@configclass
class DroneRacerLearnedInertialSwiftCTBRCircular12KnownStartGTShadowColor20VisionV1Cfg(
    DroneRacerLearnedInertialSwiftCTBRCircular12KnownStartGTShadowV7ImuOnlyCfg
):
    """GT-controlled Circular-12 shadow run with the Color20 Gate-ID detector.

    The actor remains on simulator truth, while the estimator uses IMU
    propagation plus the newly retrained 20-degree, 2.07-m diversified visual
    detector. Learned-motion fusion stays disabled in this first integration
    check so the effect of visual map corrections is isolated.
    """

    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.tiled_camera = stage2_reference_camera_cfg(
            pitch_up_deg=20.0
        )
        self.gate_camera_pitch_up_deg = 20.0
        self.swift_detector_checkpoint = (
            "artifacts/racing_vision/"
            "circular12_color20_h207_gateid_kprcnn_v1/"
            "torchvision_keypointrcnn_multigate_best.pt"
        )
        self.swift_visibility_checkpoint = None
        self.swift_detection_threshold = 0.35
        self.swift_keypoint_confidence_threshold = 0.35
        self.gate_measurement_model = "direct_reprojection"
        self.gate_reprojection_use_checkpoint_sigma = True
        self.gate_reprojection_min_visible_corners = 2
        self.gate_reprojection_association_max_rmse_px = 80.0
        self.gate_identity_min_confidence = 0.50
        self.gate_identity_preferred_max_rmse_px = 15.0
        self.gate_identity_preference_margin_px = 4.0
        self.gate_reprojection_huber_delta_sigma = 2.5
        self.gate_reprojection_max_normalized_nis = 25.0
        self.gate_debug_gt_diagnostics = True


@configclass
class DroneRacerLearnedInertialSwiftCTBRCircular12KnownStartGTPolicyMultiGateVisionCfg(
    DroneRacerLearnedInertialSwiftCTBRCircular12KnownStartGTPolicyV7Cfg
):
    """Estimator-driven frozen GT policy using IMU + multi-gate reprojection only."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.learned_apply_displacement_updates = False
        self.scene.tiled_camera = stage2_reference_camera_cfg(
            pitch_up_deg=40.0
        )
        self.gate_camera_pitch_up_deg = 40.0
        self.swift_detector_checkpoint = (
            "artifacts/racing_vision/circular12_pitch40_multigate/"
            "torchvision_keypointrcnn_multigate_best.pt"
        )
        self.swift_visibility_checkpoint = None
        self.swift_detection_threshold = 0.35
        self.swift_keypoint_confidence_threshold = 0.35
        self.gate_measurement_model = "direct_reprojection"
        self.gate_reprojection_use_checkpoint_sigma = True
        self.gate_reprojection_min_visible_corners = 2
        self.gate_reprojection_association_max_rmse_px = 80.0
        self.gate_reprojection_huber_delta_sigma = 2.5
        self.gate_reprojection_max_normalized_nis = 25.0


# ---------------------------------------------------------------------------
# Circular-12 imitation -> PPO -> estimator replacement curriculum
# ---------------------------------------------------------------------------

@configclass
class SwiftCTBRGTImitationFineTuneRewardsCfg(SwiftCTBRGTRacingRewardsCfg):
    """Stage-A 14 m/s PPO reward with a decaying non-tumbling expert anchor.

    Keep the first PPO stage exactly on the validated BC/DAgger operating
    point. Faster 16 and 17.7 m/s stages must be explicit later curriculum
    steps because the 31-D actor observation does not contain a target-speed
    command.
    """

    ang_vel_l2 = None
    lookat_next = None

    expert_anchor = RewTerm(
        func=mdp.circular12_expert_anchor_l2,
        weight=-1.0,
        params={
            "target_speed_mps": 14.0,
            "start_scale": 4.0,
            "end_scale": 0.25,
            "anneal_steps": 24000,
            "action_name": "control_action",
        },
    )
    coordinated_attitude = RewTerm(
        func=mdp.circular12_coordinated_attitude_l2,
        weight=-4.0,
        params={"target_speed_mps": 14.0},
    )
    coordinated_body_rate = RewTerm(
        func=mdp.circular12_coordinated_body_rate_l2,
        weight=-0.5,
        params={"target_speed_mps": 14.0},
    )
    radius_error = RewTerm(
        func=mdp.circular12_radius_error_l2,
        weight=-1.0,
        params={"radius_m": 12.0, "center_xy": (0.0, 12.0)},
    )
    height_error = RewTerm(
        func=mdp.circular12_height_error_l2,
        weight=-2.0,
        params={"height_m": 2.07},
    )
    speed_error = RewTerm(
        func=mdp.circular12_speed_error_l2,
        weight=-0.2,
        params={"target_speed_mps": 14.0},
    )
    body_rate_command = RewTerm(
        func=mdp.swift_ctbr_body_rate_command_l2,
        weight=-0.005,
        params={"action_name": "control_action"},
    )
    command_smoothness = RewTerm(
        func=mdp.swift_ctbr_command_delta_l2,
        weight=-0.002,
        params={"action_name": "control_action"},
    )


@configclass
class DroneRacerSwiftCTBRGTCircular12ImitationFineTuneEnvCfg(
    DroneRacerSwiftCTBRGTCircular12RacingEnvCfg
):
    """BC/DAgger warm-started PPO with tighter CTBR authority."""

    rewards: SwiftCTBRGTImitationFineTuneRewardsCfg = (
        SwiftCTBRGTImitationFineTuneRewardsCfg()
    )

    def __post_init__(self) -> None:
        super().__post_init__()
        self.actions.control_action.body_rate_max_radps = (4.0, 4.0, 2.0)

        # Stage-A PPO must begin on the same physically coordinated 14 m/s
        # distribution that produced the validated zero-tumble BC/DAgger
        # checkpoint. The legacy GT-racing reset starts from zero velocity with
        # up to +/-45 deg attitude perturbations, which is intentionally outside
        # this first fine-tuning curriculum stage.
        self.commands.target.randomise_start = None
        self.events.reset_base = EventTerm(
            func=mdp.reset_circular12_coordinated_state,
            mode="reset",
            params={
                "target_speed_mps": 14.0,
                "radius_m": 12.0,
                "center_xy": (0.0, 12.0),
                "height_m": 2.07,
                "phase_rad": -7.0 * 3.141592653589793 / 12.0,
                "gravity_mps2": 9.81,
                "phase_jitter_rad": 5.0 * 3.141592653589793 / 180.0,
                "radial_jitter_m": 0.20,
                "height_jitter_m": 0.10,
                "speed_jitter_mps": 0.50,
                "attitude_jitter_rad": 3.0 * 3.141592653589793 / 180.0,
                "angular_rate_jitter_radps": 0.10,
                "asset_cfg_name": "robot",
            },
        )

        # PPO exploration can occasionally miss a gate even when initialized
        # from the validated zero-tumble BC. Keep the miss recorded for the
        # racing reward, but advance the actor-visible target so a single miss
        # cannot create the stale-behind-gate observation failure mode already
        # eliminated during the imitation stage.
        self.commands.target.advance_target_on_miss = True


@configclass
class SwiftGTNoisePolicyCfg(ObsGroup):
    """Stage C: GT state corrupted by configurable estimator-like residuals."""

    platform_state = ObsTerm(
        func=mdp.noisy_gt_swift_state,
        params={
            "position_std_m": 0.08,
            "velocity_std_mps": 0.08,
            "attitude_std_deg": 1.0,
        },
    )
    next_gate_corners = ObsTerm(
        func=mdp.noisy_gt_next_gate_corners_relative_w,
        params={
            "command_name": "target",
            "position_std_m": 0.08,
            "velocity_std_mps": 0.08,
            "attitude_std_deg": 1.0,
        },
    )
    previous_action = ObsTerm(func=mdp.last_action)

    def __post_init__(self) -> None:
        self.enable_corruption = False
        self.concatenate_terms = True


@configclass
class SwiftGTNoiseObservationsCfg:
    policy: SwiftGTNoisePolicyCfg = SwiftGTNoisePolicyCfg()
    critic = None


@configclass
class DroneRacerSwiftCTBRGTCircular12ImitationNoiseRobustEnvCfg(
    DroneRacerSwiftCTBRGTCircular12ImitationFineTuneEnvCfg
):
    """Stage C: vectorized PPO robustness to estimator-like observation error."""

    observations: SwiftGTNoiseObservationsCfg = SwiftGTNoiseObservationsCfg()


@configclass
class DroneRacerSwiftCTBRGTCircular12ExpertValidationEnvCfg(
    DroneRacerSwiftCTBRGTCircular12ImitationFineTuneEnvCfg
):
    """Single-env clean free-flight validation/demo task for the CTBR expert.

    Unlike PPO training, this task does not use random predecessor-gate resets.
    It initializes exactly once onto a coordinated circular state immediately
    before Gate 1. The expert must then sustain the flight through physics.
    """

    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.num_envs = 1
        self.episode_length_s = 20.0
        self.commands.target.randomise_start = None
        self.commands.target.debug_vis = False
        # A missed gate is still a miss for scoring, but keeping that gate as
        # the actor target after its plane is already behind the vehicle creates
        # a stale observation that the circular expert itself does not follow.
        # Advance the control target so safety recovery remains well-posed.
        self.commands.target.advance_target_on_miss = True
        self.events.push_robot = None
        self.events.reset_base = EventTerm(
            func=mdp.reset_circular12_coordinated_state,
            mode="reset",
            params={
                "target_speed_mps": 14.0,
                "radius_m": 12.0,
                "center_xy": (0.0, 12.0),
                "height_m": 2.07,
                "phase_rad": -7.0 * 3.141592653589793 / 12.0,
                "gravity_mps2": 9.81,
                "asset_cfg_name": "robot",
            },
        )


@configclass
class DroneRacerSwiftCTBRGTCircular12ExpertDemoEnvCfg(
    DroneRacerSwiftCTBRGTCircular12ExpertValidationEnvCfg
):
    """Expert demonstration task with mild recoverable reset perturbations."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.events.reset_base.params.update(
            {
                "phase_jitter_rad": 5.0 * 3.141592653589793 / 180.0,
                "radial_jitter_m": 0.20,
                "height_jitter_m": 0.10,
                "speed_jitter_mps": 0.50,
                "attitude_jitter_rad": 3.0 * 3.141592653589793 / 180.0,
                "angular_rate_jitter_radps": 0.10,
            }
        )


@configclass
class SwiftBlend25PolicyCfg(ObsGroup):
    platform_state = ObsTerm(
        func=mdp.blended_inertial_swift_state,
        params={"blend_alpha": 0.25},
    )
    next_gate_corners = ObsTerm(
        func=mdp.blended_truth_next_gate_corners_relative_w,
        params={"blend_alpha": 0.25, "command_name": "target"},
    )
    previous_action = ObsTerm(func=mdp.last_action)

    def __post_init__(self) -> None:
        self.enable_corruption = False
        self.concatenate_terms = True


@configclass
class SwiftBlend50PolicyCfg(SwiftBlend25PolicyCfg):
    platform_state = ObsTerm(
        func=mdp.blended_inertial_swift_state,
        params={"blend_alpha": 0.50},
    )
    next_gate_corners = ObsTerm(
        func=mdp.blended_truth_next_gate_corners_relative_w,
        params={"blend_alpha": 0.50, "command_name": "target"},
    )


@configclass
class SwiftBlend75PolicyCfg(SwiftBlend25PolicyCfg):
    platform_state = ObsTerm(
        func=mdp.blended_inertial_swift_state,
        params={"blend_alpha": 0.75},
    )
    next_gate_corners = ObsTerm(
        func=mdp.blended_truth_next_gate_corners_relative_w,
        params={"blend_alpha": 0.75, "command_name": "target"},
    )


@configclass
class SwiftBlend100PolicyCfg(SwiftBlend25PolicyCfg):
    platform_state = ObsTerm(
        func=mdp.blended_inertial_swift_state,
        params={"blend_alpha": 1.00},
    )
    next_gate_corners = ObsTerm(
        func=mdp.blended_truth_next_gate_corners_relative_w,
        params={"blend_alpha": 1.00, "command_name": "target"},
    )


@configclass
class SwiftBlend25ObservationsCfg:
    policy: SwiftBlend25PolicyCfg = SwiftBlend25PolicyCfg()
    critic = None


@configclass
class SwiftBlend50ObservationsCfg:
    policy: SwiftBlend50PolicyCfg = SwiftBlend50PolicyCfg()
    critic = None


@configclass
class SwiftBlend75ObservationsCfg:
    policy: SwiftBlend75PolicyCfg = SwiftBlend75PolicyCfg()
    critic = None


@configclass
class SwiftBlend100ObservationsCfg:
    policy: SwiftBlend100PolicyCfg = SwiftBlend100PolicyCfg()
    critic = None


@configclass
class DroneRacerLearnedInertialSwiftCTBRCircular12ImitationGTShadowColor20Cfg(
    DroneRacerLearnedInertialSwiftCTBRCircular12KnownStartGTShadowColor20VisionV1Cfg
):
    """Stage B: GT actor; IMU + SC-EKF + Color20 run only in shadow."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.actions.control_action.body_rate_max_radps = (4.0, 4.0, 2.0)
        # V7 remains non-actuating here, but collect online prediction-vs-truth
        # evidence on the new stable trajectory before any learned fusion.
        self.learned_debug_online_truth_audit = True


@configclass
class DroneRacerLearnedInertialSwiftCTBRCircular12ImitationBlend25Color20Cfg(
    DroneRacerLearnedInertialSwiftCTBRCircular12ImitationGTShadowColor20Cfg
):
    """Stage D25: 25% estimator platform state, truth mission progression."""
    observations: SwiftBlend25ObservationsCfg = SwiftBlend25ObservationsCfg()


@configclass
class DroneRacerLearnedInertialSwiftCTBRCircular12ImitationBlend50Color20Cfg(
    DroneRacerLearnedInertialSwiftCTBRCircular12ImitationGTShadowColor20Cfg
):
    """Stage D50: 50% estimator platform state, truth mission progression."""
    observations: SwiftBlend50ObservationsCfg = SwiftBlend50ObservationsCfg()


@configclass
class DroneRacerLearnedInertialSwiftCTBRCircular12ImitationBlend75Color20Cfg(
    DroneRacerLearnedInertialSwiftCTBRCircular12ImitationGTShadowColor20Cfg
):
    """Stage D75: 75% estimator platform state, truth mission progression."""
    observations: SwiftBlend75ObservationsCfg = SwiftBlend75ObservationsCfg()


@configclass
class SwiftEstimatorTruthMissionPolicyCfg(ObsGroup):
    """Stage E: estimator platform state with truth-only mission gate index."""

    platform_state = ObsTerm(func=mdp.learned_inertial_swift_state)
    next_gate_corners = ObsTerm(
        func=mdp.learned_truth_next_gate_corners_relative_w,
        params={"command_name": "target"},
    )
    previous_action = ObsTerm(func=mdp.last_action)

    def __post_init__(self) -> None:
        self.enable_corruption = False
        self.concatenate_terms = True


@configclass
class SwiftEstimatorTruthMissionObservationsCfg:
    policy: SwiftEstimatorTruthMissionPolicyCfg = (
        SwiftEstimatorTruthMissionPolicyCfg()
    )
    critic = None


@configclass
class DroneRacerLearnedInertialSwiftCTBRCircular12ImitationEstStateTruthMissionColor20Cfg(
    DroneRacerLearnedInertialSwiftCTBRCircular12ImitationGTShadowColor20Cfg
):
    """Stage E: estimator p/v/R with truth-only mission gate progression."""
    observations: SwiftEstimatorTruthMissionObservationsCfg = (
        SwiftEstimatorTruthMissionObservationsCfg()
    )


def _configure_circular12_coordinated14_known_start(cfg) -> None:
    """Make physical reset and no-GT estimator reset exactly coincide."""

    cfg.scene.num_envs = 1
    cfg.episode_length_s = 20.0
    cfg.commands.target.randomise_start = None
    cfg.commands.target.advance_target_on_miss = True
    cfg.commands.target.debug_vis = False
    cfg.events.push_robot = None

    # Nominal coordinated Circular-12 state:
    # phase=-105 deg, r=12 m, center=(0, 12), h=2.07 m, speed=14 m/s.
    # This is task-definition knowledge, not a runtime simulator-truth read.
    cfg.scene.robot.init_state.pos = (
        -3.1058285412302475,
        0.4088900845311798,
        2.07,
    )
    cfg.scene.robot.init_state.rot = (
        0.8628651271905666,
        -0.4882895508025350,
        0.06428453890898132,
        -0.11359834907570406,
    )
    cfg.scene.robot.init_state.lin_vel = (
        13.522961568046956,
        -3.6234666314352886,
        0.0,
    )
    cfg.scene.robot.init_state.ang_vel = (
        0.0,
        0.0,
        1.1666666666666667,
    )
    cfg.events.reset_base = EventTerm(
        func=mdp.reset_circular12_coordinated_state,
        mode="reset",
        params={
            "target_speed_mps": 14.0,
            "radius_m": 12.0,
            "center_xy": (0.0, 12.0),
            "height_m": 2.07,
            "phase_rad": -7.0 * 3.141592653589793 / 12.0,
            "gravity_mps2": 9.81,
            "asset_cfg_name": "robot",
        },
    )


@configclass
class DroneRacerLearnedInertialSwiftCTBRCircular12ImitationGTShadowCoordinated14Color20Cfg(
    DroneRacerLearnedInertialSwiftCTBRCircular12ImitationGTShadowColor20Cfg
):
    """Strict A/B baseline: GT 31-D actor input on coordinated 14 m/s reset."""

    def __post_init__(self) -> None:
        super().__post_init__()
        _configure_circular12_coordinated14_known_start(self)


@configclass
class DroneRacerLearnedInertialSwiftCTBRCircular12ImitationGTShadowCoordinated14Color20OracleAssociationCfg(
    DroneRacerLearnedInertialSwiftCTBRCircular12ImitationGTShadowCoordinated14Color20Cfg
):
    """Evaluation-only oracle gate association; GT actor remains in control.

    Detector pixels and the SC-EKF reprojection update remain real. Only the
    mapped-gate association is fixed to the simulator-truth active gate index.
    """

    def __post_init__(self) -> None:
        super().__post_init__()
        self.gate_debug_force_truth_active_gate_association = True


@configclass
class DroneRacerLearnedInertialSwiftCTBRCircular12ImitationEstStateTruthMissionCoordinated14Color20Cfg(
    DroneRacerLearnedInertialSwiftCTBRCircular12ImitationEstStateTruthMissionColor20Cfg
):
    """Strict A/B treatment: estimator p/v/R, truth-only gate progression."""

    def __post_init__(self) -> None:
        super().__post_init__()
        _configure_circular12_coordinated14_known_start(self)


def _configure_color20_gate_updates(cfg) -> None:
    cfg.scene.tiled_camera = stage2_reference_camera_cfg(pitch_up_deg=20.0)
    cfg.gate_camera_pitch_up_deg = 20.0
    cfg.swift_detector_checkpoint = (
        "artifacts/racing_vision/"
        "circular12_color20_h207_gateid_kprcnn_v1/"
        "torchvision_keypointrcnn_multigate_best.pt"
    )
    cfg.swift_visibility_checkpoint = None
    cfg.swift_detection_threshold = 0.35
    cfg.swift_keypoint_confidence_threshold = 0.35
    cfg.gate_measurement_model = "direct_reprojection"
    cfg.gate_reprojection_use_checkpoint_sigma = True
    cfg.gate_reprojection_min_visible_corners = 2
    cfg.gate_reprojection_association_max_rmse_px = 80.0
    cfg.gate_identity_min_confidence = 0.50
    cfg.gate_identity_preferred_max_rmse_px = 15.0
    cfg.gate_identity_preference_margin_px = 4.0
    cfg.gate_reprojection_huber_delta_sigma = 2.5
    cfg.gate_reprojection_max_normalized_nis = 25.0
    cfg.gate_debug_gt_diagnostics = True


@configclass
class DroneRacerLearnedInertialSwiftCTBRCircular12ImitationEstimatorMissionColor20Cfg(
    DroneRacerLearnedInertialSwiftCTBRCircular12KnownStartGTPolicyCfg
):
    """Stage F: estimator state + estimated mission, learned-motion fusion OFF."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.actions.control_action.body_rate_max_radps = (4.0, 4.0, 2.0)
        self.learned_apply_displacement_updates = False
        _configure_color20_gate_updates(self)


@configclass
class DroneRacerLearnedInertialSwiftCTBRCircular12ImitationEstimatorMissionLegacyCoordinated14Color20Cfg(
    DroneRacerLearnedInertialSwiftCTBRCircular12ImitationEstimatorMissionColor20Cfg
):
    """Strict no-truth-index A/B with training-compatible mission semantics.

    Estimator p/v/R and estimator-driven mission progression are actor-visible.
    Simulator truth is retained only for reward/evaluation bookkeeping. The
    historical training gate-crossing semantics are preserved so the only
    control-path change from Stage E is removal of the truth-only mission index.
    """

    def __post_init__(self) -> None:
        super().__post_init__()
        _configure_circular12_coordinated14_known_start(self)
        self.commands.target.mission_crossing_mode = "legacy_training"


@configclass
class DroneRacerLearnedInertialSwiftCTBRCircular12ImitationEstimatorMissionColor20V7Cfg(
    DroneRacerLearnedInertialSwiftCTBRCircular12ImitationEstimatorMissionColor20Cfg
):
    """Optional last stage: enable V7 only after its shadow/offline audit passes."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.learned_motion_checkpoint = (
            "artifacts/imo_tcn/model_v7_circular12_racing.pt"
        )
        self.learned_apply_displacement_updates = True


@configclass
class SwiftImitationResidualNoisePolicyCfg(ObsGroup):
    """Stage C actor input: GT corrupted by estimator-scale residuals."""

    platform_state = ObsTerm(
        func=mdp.swift_gt_state_with_residual_noise,
        params={
            "position_std_m": 0.08,
            "velocity_std_mps": 0.06,
            "attitude_std_deg": 0.5,
        },
    )
    next_gate_corners = ObsTerm(
        func=mdp.swift_gt_next_gate_corners_relative_noisy_w,
        params={
            "command_name": "target",
            "position_std_m": 0.08,
            "velocity_std_mps": 0.06,
            "attitude_std_deg": 0.5,
        },
    )
    previous_action = ObsTerm(func=mdp.last_action)

    def __post_init__(self) -> None:
        self.enable_corruption = False
        self.concatenate_terms = True


@configclass
class SwiftImitationResidualNoiseObservationsCfg:
    policy: SwiftImitationResidualNoisePolicyCfg = (
        SwiftImitationResidualNoisePolicyCfg()
    )
    critic = None


@configclass
class DroneRacerSwiftCTBRGTCircular12ImitationResidualNoiseEnvCfg(
    DroneRacerSwiftCTBRGTCircular12ImitationFineTuneEnvCfg
):
    """Stage C: vectorized GT PPO with estimator-scale observation residuals.

    The bootstrap standard deviations are explicit placeholders. Replace them
    with residual statistics measured under the new non-tumbling GTShadow
    trajectory before the final robustness fine-tune.
    """

    observations: SwiftImitationResidualNoiseObservationsCfg = (
        SwiftImitationResidualNoiseObservationsCfg()
    )
