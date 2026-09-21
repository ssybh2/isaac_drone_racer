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
from .track_generator import EASY_7_GATE_TRACK_CONFIG, generate_track





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


