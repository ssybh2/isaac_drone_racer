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
from isaaclab.utils import configclass

from . import mdp
from .drone_racer_env_cfg import DroneRacerEnvCfg_PLAY
from .drone_racer_learned_inertial_env_cfg import DroneRacerLearnedInertialRLCfg





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
class DroneRacerLearnedInertialSwiftCTBRRLCfg(DroneRacerLearnedInertialRLCfg):
    """Full sensor-faithful learned-inertial stack with Swift CTBR actuation."""

    observations: LearnedInertialSwiftObservationsCfg = LearnedInertialSwiftObservationsCfg()
    actions: SwiftCTBRActionsCfg = SwiftCTBRActionsCfg()

    def __post_init__(self) -> None:
        super().__post_init__()
        # Keep one sensor-faithful environment for final closed-loop validation.
        # PPO training will later use a separate vectorized residual/noise task.
        self.scene.num_envs = 1
