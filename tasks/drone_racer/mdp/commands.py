# Copyright (c) 2025, Kousheek Chakraborty
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# This project uses the IsaacLab framework (https://github.com/isaac-sim/IsaacLab),
# which is licensed under the BSD-3-Clause License.

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import MISSING
from typing import TYPE_CHECKING

import cv2
import isaaclab.utils.math as math_utils
import torch
from isaaclab.assets import Articulation, RigidObjectCollection
from isaaclab.managers import CommandTerm, CommandTermCfg, SceneEntityCfg
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.sensors import TiledCamera
from isaaclab.utils import configclass

from .events import reset_after_prev_gate

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


class GateTargetingCommand(CommandTerm):
    """Command generator that generates a pose command from a uniform distribution."""

    cfg: GateTargetingCommandCfg
    """Configuration for the command generator."""

    def __init__(self, cfg: GateTargetingCommandCfg, env: ManagerBasedEnv):
        """Initialize the command generator class.

        Args:
            cfg: The configuration parameters for the command generator.
            env: The environment object.
        """
        # initialize the base class
        super().__init__(cfg, env)

        self.cfg = cfg

        # FPV video recording
        if self.cfg.record_fpv:
            self.video_id = 0
            self.fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self.sensor_cfg: SceneEntityCfg = SceneEntityCfg("tiled_camera")
            self.sensor: TiledCamera = self._env.scene.sensors[self.sensor_cfg.name]

        # extract the robot and track for which the command is generated
        self.robot: Articulation = env.scene[cfg.asset_name]
        self.track: RigidObjectCollection = env.scene[cfg.track_name]
        self.gate_size = cfg.gate_size
        self.num_gates = self.track.num_objects

        # create buffers
        # -- commands: (x, y, z, qw, qx, qy, qz) in simulation world frame
        self.env_ids = torch.arange(self.num_envs, device=self.device)
        self.prev_robot_pos_w = self.robot.data.root_pos_w
        self._gate_missed = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._gate_passed = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.next_gate_idx = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)
        self.next_gate_w = torch.zeros(self.num_envs, 7, device=self.device)

    def __str__(self) -> str:
        msg = "GateTargetingCommand:\n"
        msg += f"\tCommand dimension: {tuple(self.command.shape[1:])}\n"
        msg += f"\tResampling time range: {self.cfg.resampling_time_range}\n"
        return msg

    """
    Properties
    """

    @property
    def command(self) -> torch.Tensor:
        """The desired pose command. Shape is (num_envs, 7).

        The first three elements correspond to the position, followed by the quaternion orientation in (w, x, y, z).
        """
        return self.next_gate_w

    @property
    def gate_missed(self) -> torch.Tensor:
        return self._gate_missed

    @property
    def gate_passed(self) -> torch.Tensor:
        return self._gate_passed

    @property
    def previous_pos(self) -> torch.Tensor:
        return self.prev_robot_pos_w

    """
    Implementation specific functions.
    """

    def _update_metrics(self):
        pass

    def _resample_command(self, env_ids: Sequence[int]):
        # Release and reinitialize video writer only after the first iteration
        if hasattr(self, "out") and self.cfg.record_fpv:
            self.out.release()
            print(f"FPV video saved as fpv_{self.video_id}.mp4")
            self.video_id += 1

        if self.cfg.record_fpv:
            self.out = cv2.VideoWriter(f"fpv_{self.video_id}.mp4", self.fourcc, 100, (1000, 1000))

        if self.cfg.randomise_start is None:
            self.next_gate_idx[env_ids] = 0

        else:
            if self.cfg.randomise_start:
                self.next_gate_idx[env_ids] = torch.randint(
                    low=0, high=self.num_gates, size=(len(env_ids),), device=self.device, dtype=torch.int32
                )
            else:
                self.next_gate_idx[env_ids] = 1

            gate_indices = self.next_gate_idx - 1
            gate_positions = self.track.data.object_com_pos_w[self.env_ids, gate_indices]
            gate_orientations = self.track.data.object_quat_w[self.env_ids, gate_indices]
            gate_w = torch.cat([gate_positions, gate_orientations], dim=1)

            reset_after_prev_gate(
                env=self._env,
                env_ids=env_ids,
                gate_pose=gate_w,
                pose_range={
                    "x": (-0.5, 0.5),
                    "y": (-0.5, 0.5),
                    "z": (-0.5, 0.5),
                    "roll": (-torch.pi / 4, torch.pi / 4),
                    "pitch": (-torch.pi / 4, torch.pi / 4),
                    "yaw": (-torch.pi / 4, torch.pi / 4),
                },
                velocity_range={
                    "x": (0.0, 0.0),
                    "y": (0.0, 0.0),
                    "z": (0.0, 0.0),
                    "roll": (0.0, 0.0),
                    "pitch": (0.0, 0.0),
                    "yaw": (0.0, 0.0),
                },
                asset_cfg_name=self.cfg.asset_name,
            )

    def _update_command(self):
        if self.cfg.record_fpv:
            image = self.sensor.data.output["rgb"][0].cpu().numpy()
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            self.out.write(image)

        next_gate_positions = self.track.data.object_com_pos_w[self.env_ids, self.next_gate_idx]
        next_gate_orientations = self.track.data.object_quat_w[self.env_ids, self.next_gate_idx]
        self.next_gate_w = torch.cat([next_gate_positions, next_gate_orientations], dim=1)

        # Gate passing logic
        (roll, pitch, yaw) = math_utils.euler_xyz_from_quat(self.next_gate_w[:, 3:7])
        normal = torch.stack([torch.cos(yaw), torch.sin(yaw)], dim=1)
        pos_old_projected = (self.prev_robot_pos_w[:, 0] - self.next_gate_w[:, 0]) * normal[:, 0] + (
            self.prev_robot_pos_w[:, 1] - self.next_gate_w[:, 1]
        ) * normal[:, 1]
        pos_new_projected = (self.robot.data.root_pos_w[:, 0] - self.next_gate_w[:, 0]) * normal[:, 0] + (
            self.robot.data.root_pos_w[:, 1] - self.next_gate_w[:, 1]
        ) * normal[:, 1]
        passed_gate_plane = (pos_old_projected < 0) & (pos_new_projected > 0)

        self._gate_passed = passed_gate_plane & (
            torch.all(torch.abs(self.robot.data.root_pos_w - self.next_gate_w[:, :3]) < (self.gate_size / 2), dim=1)
        )

        self._gate_missed = passed_gate_plane & (
            torch.any(torch.abs(self.robot.data.root_pos_w - self.next_gate_w[:, :3]) > (self.gate_size / 2), dim=1)
        )

        # Update next gate target for the envs that passed the gate
        self.next_gate_idx[self._gate_passed] += 1
        self.next_gate_idx = self.next_gate_idx % self.num_gates

        self.prev_robot_pos_w = self.robot.data.root_pos_w

    def _set_debug_vis_impl(self, debug_vis: bool):
        # create markers if necessary for the first time
        if debug_vis:
            if not hasattr(self, "target_visualizer"):
                # -- goal pose
                self.target_visualizer = VisualizationMarkers(self.cfg.target_visualizer_cfg)
                # -- current body pose
                self.drone_visualizer = VisualizationMarkers(self.cfg.drone_visualizer_cfg)
            # set their visibility to true
            self.target_visualizer.set_visibility(True)
            self.drone_visualizer.set_visibility(True)
        else:
            if hasattr(self, "target_visualizer"):
                self.target_visualizer.set_visibility(False)
                self.drone_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        # check if robot is initialized
        # note: this is needed in-case the robot is de-initialized. we can't access the data
        if not self.robot.is_initialized:
            return
        # update the markers
        self.target_visualizer.visualize(self.next_gate_w[:, :3], self.next_gate_w[:, 3:])
        self.drone_visualizer.visualize(self.robot.data.root_pos_w, self.robot.data.root_quat_w)



class EstimatedStateGateTargetingCommand(GateTargetingCommand):
    """Gate mission state advanced only from the learned-inertial estimate.

    The actor-visible target index must not be advanced from Isaac root-state
    truth.  Simulator truth is retained in separate buffers exclusively for
    reward/evaluation bookkeeping.

    This command assumes the mapped gate frame +X axis is the gate-plane
    normal, matching the existing track/yaw convention.
    """

    cfg: EstimatedStateGateTargetingCommandCfg

    def __init__(
        self,
        cfg: EstimatedStateGateTargetingCommandCfg,
        env: ManagerBasedEnv,
    ):
        super().__init__(cfg, env)
        self._mission_gate_passed = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self._mission_gate_missed = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self._gt_gate_passed = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self._gt_gate_missed = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        # Keep truth-only evaluation progression completely independent from
        # the actor-visible estimator-driven mission index.  Without this,
        # an estimator crossing the gate plane one control step early advances
        # next_gate_idx and makes the truth path evaluate the wrong gate on the
        # following step.
        self._gt_next_gate_idx = torch.zeros(
            self.num_envs, dtype=torch.int32, device=self.device
        )
        self._mission_gate_pass_count = torch.zeros(
            self.num_envs, dtype=torch.int32, device=self.device
        )
        self._gt_gate_pass_count = torch.zeros(
            self.num_envs, dtype=torch.int32, device=self.device
        )
        self._prev_estimated_pos_w = self._known_start_position_w()
        self.prev_robot_pos_w = self.robot.data.root_pos_w.clone()

    @property
    def gate_passed(self) -> torch.Tensor:
        """Ground-truth pass flag for reward/evaluation only."""
        return self._gt_gate_passed

    @property
    def gate_missed(self) -> torch.Tensor:
        """Ground-truth miss flag for reward/evaluation only."""
        return self._gt_gate_missed

    @property
    def mission_gate_passed(self) -> torch.Tensor:
        """Estimator-derived gate-pass flag that advances actor mission state."""
        return self._mission_gate_passed

    @property
    def mission_gate_missed(self) -> torch.Tensor:
        return self._mission_gate_missed

    @property
    def gt_next_gate_idx(self) -> torch.Tensor:
        """Truth-only gate index for reward/evaluation diagnostics."""
        return self._gt_next_gate_idx

    @property
    def mission_gate_pass_count(self) -> torch.Tensor:
        """Per-episode estimator-driven gate-pass count for diagnostics."""
        return self._mission_gate_pass_count

    @property
    def gt_gate_pass_count(self) -> torch.Tensor:
        """Per-episode truth gate-pass count for diagnostics."""
        return self._gt_gate_pass_count

    def _known_start_position_w(self) -> torch.Tensor:
        """Return the task-defined initial position without reading runtime GT."""
        init_pos = torch.tensor(
            self._env.cfg.scene.robot.init_state.pos,
            dtype=torch.float32,
            device=self.device,
        ).view(1, 3)
        return init_pos.expand(self.num_envs, 3) + self._env.scene.env_origins

    def _estimated_position_w(self) -> torch.Tensor:
        state = getattr(self._env, "learned_inertial_state", None)
        if state is None:
            return self._known_start_position_w()
        position = torch.as_tensor(
            state.position_w_b,
            dtype=torch.float32,
            device=self.device,
        ).view(1, 3)
        if self.num_envs != 1:
            raise RuntimeError(
                "EstimatedStateGateTargetingCommand currently requires num_envs=1"
            )
        return position

    @staticmethod
    def _gate_crossing(
        previous_pos_w: torch.Tensor,
        current_pos_w: torch.Tensor,
        gate_pose_w: torch.Tensor,
        gate_size: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        gate_pos_w = gate_pose_w[:, :3]
        gate_quat_w = gate_pose_w[:, 3:7]
        gate_quat_inv = math_utils.quat_inv(gate_quat_w)

        previous_g = math_utils.quat_apply(
            gate_quat_inv, previous_pos_w - gate_pos_w
        )
        current_g = math_utils.quat_apply(
            gate_quat_inv, current_pos_w - gate_pos_w
        )

        crossed_plane = (previous_g[:, 0] < 0.0) & (current_g[:, 0] >= 0.0)
        half_size = 0.5 * float(gate_size)
        inside_opening = (
            (torch.abs(current_g[:, 1]) < half_size)
            & (torch.abs(current_g[:, 2]) < half_size)
        )
        return (
            crossed_plane & inside_opening,
            crossed_plane & ~inside_opening,
        )

    def _resample_command(self, env_ids: Sequence[int]):
        super()._resample_command(env_ids)
        self._mission_gate_passed[env_ids] = False
        self._mission_gate_missed[env_ids] = False
        self._gt_gate_passed[env_ids] = False
        self._gt_gate_missed[env_ids] = False
        self._gt_next_gate_idx[env_ids] = self.next_gate_idx[env_ids]
        self._mission_gate_pass_count[env_ids] = 0
        self._gt_gate_pass_count[env_ids] = 0
        known_start = self._known_start_position_w()
        self._prev_estimated_pos_w[env_ids] = known_start[env_ids]
        # This truth buffer is never exposed to the actor.  It exists only so
        # training reward/evaluation can compare actual gate passage.
        self.prev_robot_pos_w[env_ids] = self.robot.data.root_pos_w[env_ids]

    def _update_command(self):
        if self.cfg.record_fpv:
            image = self.sensor.data.output["rgb"][0].cpu().numpy()
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            self.out.write(image)

        gate_indices = self.next_gate_idx.to(dtype=torch.long)
        gate_positions = self.track.data.object_com_pos_w[
            self.env_ids, gate_indices
        ]
        gate_orientations = self.track.data.object_quat_w[
            self.env_ids, gate_indices
        ]
        active_gate_w = torch.cat((gate_positions, gate_orientations), dim=1)

        estimated_pos_w = self._estimated_position_w()
        self._mission_gate_passed, self._mission_gate_missed = (
            self._gate_crossing(
                self._prev_estimated_pos_w,
                estimated_pos_w,
                active_gate_w,
                self.gate_size,
            )
        )

        # Ground truth is deliberately kept on a separate reward/evaluation
        # path with its *own* gate index and never controls next_gate_idx.
        # This prevents a one-step estimator/GT plane-crossing timing difference
        # from permanently desynchronizing truth gate-pass accounting.
        gt_gate_indices = self._gt_next_gate_idx.to(dtype=torch.long)
        gt_gate_positions = self.track.data.object_com_pos_w[
            self.env_ids, gt_gate_indices
        ]
        gt_gate_orientations = self.track.data.object_quat_w[
            self.env_ids, gt_gate_indices
        ]
        gt_active_gate_w = torch.cat(
            (gt_gate_positions, gt_gate_orientations), dim=1
        )
        current_gt_pos_w = self.robot.data.root_pos_w
        self._gt_gate_passed, self._gt_gate_missed = self._gate_crossing(
            self.prev_robot_pos_w,
            current_gt_pos_w,
            gt_active_gate_w,
            self.gate_size,
        )

        self._mission_gate_pass_count += self._mission_gate_passed.to(torch.int32)
        self._gt_gate_pass_count += self._gt_gate_passed.to(torch.int32)

        self.next_gate_idx[self._mission_gate_passed] += 1
        self.next_gate_idx %= self.num_gates

        self._gt_next_gate_idx[self._gt_gate_passed] += 1
        self._gt_next_gate_idx %= self.num_gates

        # Publish the actor-visible target from the estimator-driven mission
        # index immediately after any transition.
        gate_indices = self.next_gate_idx.to(dtype=torch.long)
        gate_positions = self.track.data.object_com_pos_w[
            self.env_ids, gate_indices
        ]
        gate_orientations = self.track.data.object_quat_w[
            self.env_ids, gate_indices
        ]
        self.next_gate_w = torch.cat((gate_positions, gate_orientations), dim=1)

        self._prev_estimated_pos_w = estimated_pos_w.clone()
        self.prev_robot_pos_w = current_gt_pos_w.clone()

class SwiftPassStateGateTargetingCommand(GateTargetingCommand):
    """Swift-style random-gate initialization with through-gate momentum.

    Swift initializes each training episode near a state previously observed
    while passing a random gate. We approximate that policy-0 curriculum here
    without requiring a prerecorded trajectory: place the vehicle just after
    the previous gate, point it toward the next gate, and initialize it with a
    non-zero forward velocity plus bounded pose/rate perturbations.

    This command is training-only. Deployment continues to use the
    estimator-driven mission command.
    """

    cfg: "SwiftPassStateGateTargetingCommandCfg"

    def _resample_command(self, env_ids: Sequence[int]):
        if self.cfg.randomise_start is None:
            return super()._resample_command(env_ids)

        ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        if ids.numel() == 0:
            return

        if self.cfg.randomise_start:
            self.next_gate_idx[ids] = torch.randint(
                low=0,
                high=self.num_gates,
                size=(ids.numel(),),
                device=self.device,
                dtype=torch.int32,
            )
        else:
            self.next_gate_idx[ids] = 1

        prev_idx = (self.next_gate_idx[ids] - 1) % self.num_gates
        next_idx = self.next_gate_idx[ids]

        prev_pos = self.track.data.object_com_pos_w[ids, prev_idx]
        prev_quat = self.track.data.object_quat_w[ids, prev_idx]
        next_pos = self.track.data.object_com_pos_w[ids, next_idx]

        prev_normal_b = torch.tensor(
            [1.0, 0.0, 0.0],
            dtype=prev_pos.dtype,
            device=self.device,
        ).expand(ids.numel(), 3)
        prev_normal_w = math_utils.quat_apply(prev_quat, prev_normal_b)

        # Start just after the previously passed gate.
        base_pos = prev_pos + float(self.cfg.post_gate_offset_m) * prev_normal_w

        pos_jitter = torch.empty(
            ids.numel(), 3, device=self.device, dtype=prev_pos.dtype
        )
        pos_jitter[:, 0].uniform_(
            -float(self.cfg.position_jitter_m[0]),
            float(self.cfg.position_jitter_m[0]),
        )
        pos_jitter[:, 1].uniform_(
            -float(self.cfg.position_jitter_m[1]),
            float(self.cfg.position_jitter_m[1]),
        )
        pos_jitter[:, 2].uniform_(
            -float(self.cfg.position_jitter_m[2]),
            float(self.cfg.position_jitter_m[2]),
        )
        start_pos = base_pos + pos_jitter

        to_next = next_pos - start_pos
        horizontal = to_next.clone()
        horizontal[:, 2] = 0.0
        horizontal_norm = torch.linalg.norm(horizontal, dim=1, keepdim=True)
        fallback = prev_normal_w.clone()
        fallback[:, 2] = 0.0
        fallback = fallback / torch.clamp(
            torch.linalg.norm(fallback, dim=1, keepdim=True),
            min=1.0e-6,
        )
        heading = torch.where(
            horizontal_norm > 1.0e-6,
            horizontal / torch.clamp(horizontal_norm, min=1.0e-6),
            fallback,
        )

        base_yaw = torch.atan2(heading[:, 1], heading[:, 0])
        yaw_jitter = torch.empty(
            ids.numel(), device=self.device, dtype=prev_pos.dtype
        ).uniform_(
            -float(self.cfg.yaw_jitter_rad),
            float(self.cfg.yaw_jitter_rad),
        )
        roll = torch.empty_like(base_yaw).uniform_(
            -float(self.cfg.roll_pitch_jitter_rad),
            float(self.cfg.roll_pitch_jitter_rad),
        )
        pitch = torch.empty_like(base_yaw).uniform_(
            -float(self.cfg.roll_pitch_jitter_rad),
            float(self.cfg.roll_pitch_jitter_rad),
        )
        yaw = base_yaw + yaw_jitter
        quat = math_utils.quat_from_euler_xyz(roll, pitch, yaw)

        speed = torch.empty(
            ids.numel(), 1, device=self.device, dtype=prev_pos.dtype
        ).uniform_(
            float(self.cfg.forward_speed_range_mps[0]),
            float(self.cfg.forward_speed_range_mps[1]),
        )
        velocity_w = heading * speed
        velocity_w[:, 2] += torch.empty(
            ids.numel(), device=self.device, dtype=prev_pos.dtype
        ).uniform_(
            -float(self.cfg.vertical_speed_jitter_mps),
            float(self.cfg.vertical_speed_jitter_mps),
        )

        angular_velocity = torch.empty(
            ids.numel(), 3, device=self.device, dtype=prev_pos.dtype
        ).uniform_(
            -float(self.cfg.body_rate_jitter_radps),
            float(self.cfg.body_rate_jitter_radps),
        )

        self.robot.write_root_pose_to_sim(
            torch.cat((start_pos, quat), dim=-1),
            env_ids=ids,
        )
        self.robot.write_root_velocity_to_sim(
            torch.cat((velocity_w, angular_velocity), dim=-1),
            env_ids=ids,
        )

        # Keep progress bookkeeping synchronized with the reset pose even if
        # only a subset of vectorized environments is being reset.
        previous = self.robot.data.root_pos_w.clone()
        previous[ids] = start_pos
        self.prev_robot_pos_w = previous


@configclass
class GateTargetingCommandCfg(CommandTermCfg):
    """Configuration for gate targeting command generator."""

    class_type: type = GateTargetingCommand

    asset_name: str = MISSING
    """Name of the asset in the environment for which the commands are generated."""

    track_name: str = MISSING
    """Name of the track in the environment for which the commands are generated."""

    randomise_start: bool | None = None
    """If True, the starting gate is randomised at every reset."""

    record_fpv: bool = False
    """If True, the first-person view (FPV) camera is recorded during the simulation."""

    gate_size: float = 1.5
    """Size of the gate in meters. This is used to determine if the drone has passed through the gate."""

    target_visualizer_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(prim_path="/Visuals/Command/goal_pose")
    """The configuration for the goal pose visualization marker. Defaults to FRAME_MARKER_CFG."""

    drone_visualizer_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(prim_path="/Visuals/Command/body_pose")
    """The configuration for the current pose visualization marker. Defaults to FRAME_MARKER_CFG."""

    # Set the scale of the visualization markers to (0.1, 0.1, 0.1)
    target_visualizer_cfg.markers["frame"].scale = (0.0001, 0.0001, 0.0001)
    drone_visualizer_cfg.markers["frame"].scale = (0.0001, 0.0001, 0.0001)





@configclass
class SwiftPassStateGateTargetingCommandCfg(GateTargetingCommandCfg):
    """Training-only approximation of Swift's gate-pass state initialization."""

    class_type: type = SwiftPassStateGateTargetingCommand

    post_gate_offset_m: float = 1.0
    position_jitter_m: tuple[float, float, float] = (0.35, 0.35, 0.25)
    forward_speed_range_mps: tuple[float, float] = (1.5, 3.0)
    vertical_speed_jitter_mps: float = 0.35
    yaw_jitter_rad: float = 0.35
    roll_pitch_jitter_rad: float = 0.20
    body_rate_jitter_radps: float = 0.5


@configclass
class EstimatedStateGateTargetingCommandCfg(GateTargetingCommandCfg):
    """Deployment-faithful gate mission state for learned-inertial RL."""

    class_type: type = EstimatedStateGateTargetingCommand
