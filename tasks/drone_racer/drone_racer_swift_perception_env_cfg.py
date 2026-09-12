"""Single-vehicle Swift perception/OpenVINS diagnostic configuration."""

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
    """One-env camera+IMU config for OpenVINS and optional detector/fusion diagnostics."""

    scene: DroneRacerSceneCfg = DroneRacerSceneCfg(num_envs=1, env_spacing=0.0)
    rewards: SwiftPerceptionRewardsCfg = SwiftPerceptionRewardsCfg()

    # Optional learned detector. Leave None for pure Isaac<->OpenVINS transport
    # validation. Supplying a checkpoint enables detector->IPPE->drift fusion,
    # but this diagnostic path is not authorized for PPO training yet.
    swift_detector_checkpoint: str | None = None
    swift_detector_device: str = "cuda"
    swift_detection_threshold: float = 0.5
    swift_keypoint_confidence_threshold: float = 0.5

    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.tiled_camera = swift_openvins_camera_cfg()
        self.scene.imu = ImuCfg(
            prim_path="{ENV_REGEX_NS}/Robot/body",
            debug_vis=False,
        )
        self.events.push_robot = None
