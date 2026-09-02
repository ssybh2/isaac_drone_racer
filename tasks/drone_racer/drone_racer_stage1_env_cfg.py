"""Stage 1 environment configuration with estimator-backed drone state."""

from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.utils import configclass

from estimation.fake_sensor_cfg import FakeSensorPipelineCfg

from . import mdp
from .drone_racer_env_cfg import (
    DroneRacerEnvCfg,
    DroneRacerEnvCfg_PLAY,
    ObservationsCfg,
)


@configclass
class Stage1ObservationsCfg:
    """Keep the 20-value Stage 0 contract while replacing only drone state."""

    @configclass
    class PolicyCfg(ObsGroup):
        estimated_drone_state = ObsTerm(func=mdp.estimated_drone_state)
        # Stage 1 intentionally retains oracle gate-relative guidance.
        target_pos_b = ObsTerm(func=mdp.target_pos_b, params={"command_name": "target"})
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
    critic: ObservationsCfg.CriticCfg = ObservationsCfg.CriticCfg()


@configclass
class DroneRacerStage1EnvCfg(DroneRacerEnvCfg):
    observations: Stage1ObservationsCfg = Stage1ObservationsCfg()
    fake_sensors: FakeSensorPipelineCfg = FakeSensorPipelineCfg.from_profile("clean")


@configclass
class DroneRacerStage1EnvCfg_PLAY(DroneRacerEnvCfg_PLAY):
    observations: Stage1ObservationsCfg = Stage1ObservationsCfg()
    fake_sensors: FakeSensorPipelineCfg = FakeSensorPipelineCfg.from_profile("clean")
