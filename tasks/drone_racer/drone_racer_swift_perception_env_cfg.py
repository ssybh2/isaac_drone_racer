"""Single-vehicle Swift perception diagnostic environment.

This config intentionally enables the calibrated pinhole camera and Isaac IMU
for OpenVINS/perception validation. It is not the large-scale PPO training
configuration.
"""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.sensors import ImuCfg, TiledCameraCfg
from isaaclab.utils import configclass

from perception.stage2_calibration import (
    CAMERA_MODEL,
    CAMERA_OFFSET_CONVENTION,
    CAMERA_OFFSET_POS_B,
    CAMERA_OFFSET_ROT_WXYZ,
)

from . import mdp
from .drone_racer_env_cfg import DroneRacerEnvCfg, DroneRacerSceneCfg, RewardsCfg


def swift_openvins_camera_cfg() -> TiledCameraCfg:
    """256x256 pinhole camera matching the validated Stage2 calibration."""
    if CAMERA_MODEL != "pinhole":
        raise ValueError(f"Swift/OpenVINS camera requires pinhole calibration, got {CAMERA_MODEL!r}")
    return TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/body/camera",
        offset=TiledCameraCfg.OffsetCfg(
            pos=CAMERA_OFFSET_POS_B,
            rot=CAMERA_OFFSET_ROT_WXYZ,
            convention=CAMERA_OFFSET_CONVENTION,
        ),
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(),
        width=256,
        height=256,
        return_latest_camera_pose=True,
    )


@configclass
class SwiftPerceptionRewardsCfg(RewardsCfg):
    """Existing task reward with Swift's camera-aware term replacing the legacy look-at term."""

    lookat_next = RewTerm(
        func=mdp.swift_perception_awareness,
        weight=0.02,
        params={
            "command_name": "target",
            "lambda_3": -10.0,
            "camera_pos_b": CAMERA_OFFSET_POS_B,
            "camera_optical_axis_b": (1.0, 0.0, 0.0),
        },
    )


@configclass
class DroneRacerSwiftPerceptionEnvCfg(DroneRacerEnvCfg):
    """One-env camera+IMU configuration for real OpenVINS fusion diagnostics."""

    scene: DroneRacerSceneCfg = DroneRacerSceneCfg(num_envs=1, env_spacing=0.0)
    rewards: SwiftPerceptionRewardsCfg = SwiftPerceptionRewardsCfg()

    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.tiled_camera = swift_openvins_camera_cfg()
        self.scene.imu = ImuCfg(
            prim_path="{ENV_REGEX_NS}/Robot/body",
            debug_vis=False,
        )
        self.events.push_robot = None
