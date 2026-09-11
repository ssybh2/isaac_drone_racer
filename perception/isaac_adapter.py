"""Isaac Lab boundary for Stage 2 perception truth and camera calibration.

All simulator-specific API access lives here. The geometry/PnP core remains
testable without launching Isaac Sim.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .camera_model import CameraCalibration
from .rigid_transform import RigidTransform
from .stage2a_pipeline import Stage2ATruth


def _to_numpy(value) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    return np.asarray(value)


def _transform_from_isaac_pose(position, quaternion_wxyz, *, to_frame: str, from_frame: str) -> RigidTransform:
    """Convert the pinned Isaac Lab wxyz pose into a named transform.

    Quaternion handling is isolated here so an Isaac Lab API upgrade only
    requires changing this adapter.
    """
    return RigidTransform.from_pose_wxyz(
        _to_numpy(position),
        _to_numpy(quaternion_wxyz),
        to_frame=to_frame,
        from_frame=from_frame,
    )


@dataclass(frozen=True)
class IsaacStage2Snapshot:
    truth: Stage2ATruth
    camera: CameraCalibration
    gate_index: int


class IsaacStage2TruthAdapter:
    """Read the active gate, body, and optical camera truth for one environment."""

    def __init__(
        self,
        env,
        *,
        robot_name: str = "robot",
        track_name: str = "track",
        camera_name: str = "tiled_camera",
        command_name: str = "target",
    ):
        self.env = env
        self.robot_name = robot_name
        self.track_name = track_name
        self.camera_name = camera_name
        self.command_name = command_name

    def _camera(self):
        sensors = getattr(self.env.scene, "sensors", {})
        if self.camera_name not in sensors:
            raise RuntimeError(
                f"Stage2 requires enabled scene sensor {self.camera_name!r}. "
                "The base DroneRacerEnvCfg currently disables tiled_camera in __post_init__."
            )
        return sensors[self.camera_name]

    def snapshot(self, env_id: int = 0, *, timestamp_s: float | None = None) -> IsaacStage2Snapshot:
        robot = self.env.scene[self.robot_name]
        track = self.env.scene[self.track_name]
        camera = self._camera()
        command = self.env.command_manager.get_term(self.command_name)

        # Isaac Lab v2.1 TiledCamera stores its initialization pose unless the
        # pose buffer is explicitly refreshed. The config flag documents that
        # Stage2 requires the latest frame pose, while this pinned-version shim
        # keeps RGB labels aligned with a moving drone.
        if getattr(camera.cfg, "return_latest_camera_pose", False) and hasattr(camera, "_update_poses"):
            camera._update_poses(camera._ALL_INDICES)

        gate_index = int(_to_numpy(command.next_gate_idx[env_id]).item())

        # IMPORTANT: use actor/link pose, not COM pose. PnP object points are
        # calibrated in the gate asset actor frame.
        track_data = track.data
        gate_pos_all = getattr(track_data, "object_pos_w", None)
        gate_quat_all = getattr(track_data, "object_quat_w", None)
        if gate_pos_all is None or gate_quat_all is None:
            gate_pos_all = getattr(track_data, "object_link_pos_w", None)
            gate_quat_all = getattr(track_data, "object_link_quat_w", None)
        if gate_pos_all is None or gate_quat_all is None:
            raise RuntimeError(
                "RigidObjectCollection does not expose actor/link gate pose. "
                "Do not replace this with object_com_pos_w; calibrate against the actor frame."
            )

        T_wg = _transform_from_isaac_pose(
            gate_pos_all[env_id, gate_index],
            gate_quat_all[env_id, gate_index],
            to_frame="W",
            from_frame="G",
        )
        T_wb = _transform_from_isaac_pose(
            robot.data.root_pos_w[env_id],
            robot.data.root_quat_w[env_id],
            to_frame="W",
            from_frame="B",
        )

        cam_data = camera.data
        if not hasattr(cam_data, "pos_w") or not hasattr(cam_data, "quat_w_ros"):
            raise RuntimeError(
                "Camera data must expose pos_w and quat_w_ros so Stage2 uses "
                "the OpenCV/ROS optical convention (+X right, +Y down, +Z forward)."
            )
        T_wc = _transform_from_isaac_pose(
            cam_data.pos_w[env_id],
            cam_data.quat_w_ros[env_id],
            to_frame="W",
            from_frame="C",
        )

        K = _to_numpy(cam_data.intrinsic_matrices[env_id])
        image_shape = tuple(int(v) for v in cam_data.image_shape)
        if len(image_shape) != 2:
            raise RuntimeError(f"Unexpected Isaac camera image_shape={image_shape!r}")
        image_height, image_width = image_shape

        calibration = CameraCalibration(
            K=K,
            image_width=image_width,
            image_height=image_height,
            distortion=None,
            model="pinhole",
        )

        if timestamp_s is None:
            physics_dt = float(getattr(self.env, "physics_dt", 0.0))
            sim_step = int(getattr(self.env, "_sim_step_counter", 0))
            timestamp_s = sim_step * physics_dt

        return IsaacStage2Snapshot(
            truth=Stage2ATruth(
                T_wg=T_wg,
                T_wc=T_wc,
                T_wb=T_wb,
                timestamp_s=float(timestamp_s),
            ),
            camera=calibration,
            gate_index=gate_index,
        )

    def rgb(self, env_id: int = 0) -> np.ndarray:
        camera = self._camera()
        output = camera.data.output
        if "rgb" not in output:
            raise RuntimeError("Stage2B dataset export requires camera data_type 'rgb'")
        image = _to_numpy(output["rgb"][env_id])
        if image.ndim != 3 or image.shape[-1] not in (3, 4):
            raise RuntimeError(f"Unexpected RGB image shape {image.shape}")
        return image[..., :3]
