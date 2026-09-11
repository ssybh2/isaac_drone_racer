"""Camera-enabled Stage2 calibration/dataset environment config."""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.sensors import TiledCameraCfg
from isaaclab.utils import configclass

from .drone_racer_env_cfg import DroneRacerEnvCfg, DroneRacerSceneCfg


def stage2_reference_camera_cfg() -> TiledCameraCfg:
    """Pinhole reference camera for Stage2A calibration and Stage2B labels.

    The mount position mirrors the existing repository camera placeholder.
    Treat it as a parameter to validate with the Stage2A extrinsic metric.
    """
    return TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/body/camera",
        offset=TiledCameraCfg.OffsetCfg(
            pos=(0.14, 0.0, 0.05),
            rot=(1.0, 0.0, 0.0, 0.0),
            convention="world",
        ),
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(),
        width=1000,
        height=1000,
    )


@configclass
class DroneRacerStage2DataEnvCfg(DroneRacerEnvCfg):
    """Small camera-enabled environment used for calibration and dataset export."""

    scene: DroneRacerSceneCfg = DroneRacerSceneCfg(num_envs=32, env_spacing=0.0)

    def __post_init__(self) -> None:
        super().__post_init__()
        # Base DroneRacerEnvCfg disables the camera for RL throughput. Stage2
        # data/calibration explicitly turns it back on with a pinhole model so
        # oracle projection and PnP share one camera model.
        self.scene.tiled_camera = stage2_reference_camera_cfg()
