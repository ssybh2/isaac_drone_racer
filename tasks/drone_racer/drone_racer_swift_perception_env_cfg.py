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
    OPENVINS_CAMERA_RESOLUTION,
    load_stage2_gate_geometry,
)

from . import mdp
from .drone_racer_env_cfg import DroneRacerEnvCfg, DroneRacerSceneCfg, RewardsCfg


def swift_openvins_camera_cfg() -> TiledCameraCfg:
    if CAMERA_MODEL != "pinhole":
        raise ValueError(f"Swift/OpenVINS camera requires pinhole calibration, got {CAMERA_MODEL!r}")
    image_width, image_height = OPENVINS_CAMERA_RESOLUTION
    return TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/body/camera",
        offset=TiledCameraCfg.OffsetCfg(
            pos=CAMERA_OFFSET_POS_B,
            rot=CAMERA_OFFSET_ROT_WXYZ,
            convention=CAMERA_OFFSET_CONVENTION,
        ),
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(),
        width=image_width,
        height=image_height,
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
    # Optional compact model used only as a learned visibility guard. The
    # seed-2 validation calibration selected 0.75 (0.89% partial false accepts,
    # 97.8% complete-sample recall); the R-CNN still owns corner coordinates.
    swift_visibility_checkpoint: str | None = None
    swift_detector_device: str = "cuda"
    swift_detection_threshold: float = 0.5
    swift_keypoint_confidence_threshold: float = 0.5
    swift_visibility_threshold: float = 0.75
    # Robust pixel-noise calibration on the seed-2 validation corpus after
    # hybrid visibility gating (see artifacts/swift1_detector_calibration).
    swift_corner_sigma_px: float = 0.8472250465393066

    # False is the sensor-faithful default: an unlabeled detected gate is
    # associated against the known track using the timestamp-aligned VIO pose.
    # True is only for controlled diagnostics that intentionally use Isaac's
    # task-level next_gate_idx as an oracle identity to isolate association error.
    swift_use_oracle_gate_index: bool = False

    # Optional failure corpus for tuning detector visibility/reprojection/NIS
    # gates. Each consumed rejected camera observation is saved as RGB + JSON
    # with its reason and innovation diagnostics until the configured limit.
    swift_rejection_dump_dir: str | None = None
    swift_rejection_dump_limit: int = 200

    def __post_init__(self) -> None:
        super().__post_init__()
        # Deterministic, level pose four metres before gate 0. Three metres is
        # close enough for the outer right frame to be cropped at 256x256,
        # which correctly fails the calibrated all-corners-visible guard.
        # Gate actor origins are at floor level, so derive the camera-visible
        # opening center from the authoritative geometry instead of using
        # actor z.
        # The base racing command randomizes both the previous gate and
        # attitude during reset, which is unsuitable for OpenVINS' stationary
        # initialization window.
        gate_opening_center_g = load_stage2_gate_geometry().center_g
        self.scene.robot.init_state.pos = (
            -4.0,
            0.0,
            1.0 + float(gate_opening_center_g[2]),
        )
        self.scene.robot.init_state.rot = (1.0, 0.0, 0.0, 0.0)
        self.commands.target.randomise_start = None
        self.scene.tiled_camera = swift_openvins_camera_cfg()
        self.scene.imu = ImuCfg(
            prim_path="{ENV_REGEX_NS}/Robot/body",
            debug_vis=False,
        )
        self.events.push_robot = None
