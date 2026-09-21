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
from isaaclab.utils import configclass

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
