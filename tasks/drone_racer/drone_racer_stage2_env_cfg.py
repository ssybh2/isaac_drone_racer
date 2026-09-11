"""Camera-enabled Stage2 calibration/dataset environment config."""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.sensors import TiledCameraCfg
from isaaclab.utils import configclass

from .drone_racer_env_cfg import DroneRacerEnvCfg, DroneRacerSceneCfg

from perception.stage2_calibration import (
    CAMERA_MODEL,
    CAMERA_OFFSET_CONVENTION,
    CAMERA_OFFSET_POS_B,
    CAMERA_OFFSET_ROT_WXYZ,
)


def stage2_reference_camera_cfg() -> TiledCameraCfg:
    """Pinhole reference camera for Stage2A calibration and Stage2B labels.

    The mount and optical transform are authoritative values from
    ``perception.stage2_calibration`` and are checked against Isaac truth by
    the Stage2A overlay diagnostic.
    """
    if CAMERA_MODEL != "pinhole":
        raise ValueError(f"Stage2 reference camera requires pinhole calibration, got {CAMERA_MODEL!r}")
    return TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/body/camera",
        offset=TiledCameraCfg.OffsetCfg(
            pos=CAMERA_OFFSET_POS_B,
            rot=CAMERA_OFFSET_ROT_WXYZ,
            convention=CAMERA_OFFSET_CONVENTION,
        ),
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(),
        width=1000,
        height=1000,
        # Stage2 labels and extrinsic metrics require the pose associated with
        # every rendered frame. TiledCamera otherwise exposes only its
        # initialization pose even while the drone moves.
        return_latest_camera_pose=True,
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
