# Copyright (c) 2025, Kousheek Chakraborty
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# This project uses the IsaacLab framework (https://github.com/isaac-sim/IsaacLab),
# which is licensed under the BSD-3-Clause License.

from __future__ import annotations

from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
import torch
from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg

from perception.stage2_calibration import (
    CAMERA_OFFSET_POS_B,
    CAMERA_TO_BODY_ROTATION,
    OPENVINS_CAMERA_INTRINSICS,
    OPENVINS_CAMERA_RESOLUTION,
    load_stage2_gate_geometry,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


# Loaded once at module import. The reward runs at 100 Hz over 4096 environments,
# so the calibrated gate geometry must not be re-read from disk every step.
_STAGE2_GATE_CORNERS_G = tuple(
    tuple(float(v) for v in point)
    for point in load_stage2_gate_geometry().object_points_g.tolist()
)
_GT_CAMERA_TENSOR_CACHE: dict[tuple[str, torch.dtype], tuple[torch.Tensor, ...]] = {}


def _gt_camera_reward_tensors(
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Cache tiny calibrated tensors so 4096-env PPO does no CPU->GPU copy per step."""
    key = (str(device), dtype)
    cached = _GT_CAMERA_TENSOR_CACHE.get(key)
    if cached is None:
        cached = (
            torch.as_tensor(
                _STAGE2_GATE_CORNERS_G,
                dtype=dtype,
                device=device,
            ),
            torch.as_tensor(
                CAMERA_OFFSET_POS_B,
                dtype=dtype,
                device=device,
            ),
            torch.as_tensor(
                CAMERA_TO_BODY_ROTATION,
                dtype=dtype,
                device=device,
            ),
        )
        _GT_CAMERA_TENSOR_CACHE[key] = cached
    return cached


def pos_error_l2(
    env: ManagerBasedRLEnv,
    command_name: str,
    target_pos: list | None = None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize asset pos from its target pos using L2 squared kernel."""

    asset: RigidObject = env.scene[asset_cfg.name]

    if target_pos is None:
        target_pos = env.command_manager.get_term(command_name).command
        target_pos_tensor = target_pos[:, :3]
    else:
        target_pos_tensor = (
            torch.tensor(target_pos, dtype=torch.float32, device=asset.device).repeat(env.num_envs, 1)
            + env.scene.env_origins
        )

    return torch.sum(torch.square(asset.data.root_pos_w - target_pos_tensor), dim=1)


def pos_error_tanh(
    env: ManagerBasedRLEnv,
    std: float,
    command_name: str | None = None,
    target_pos: list | None = None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize asset pos from its target pos using L2 squared kernel."""

    asset: RigidObject = env.scene[asset_cfg.name]

    if target_pos is None:
        target_pos = env.command_manager.get_term(command_name).command
        target_pos_tensor = target_pos[:, :3]
    else:
        target_pos_tensor = (
            torch.tensor(target_pos, dtype=torch.float32, device=asset.device).repeat(env.num_envs, 1)
            + env.scene.env_origins
        )

    distance = torch.norm(asset.data.root_pos_w - target_pos_tensor, dim=1)
    return 1 - torch.tanh(distance / std)


def progress(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward reduction in distance to the active gate."""

    asset: RigidObject = env.scene[asset_cfg.name]

    target_pos = env.command_manager.get_term(command_name).command[:, :3]
    previous_pos = env.command_manager.get_term(command_name).previous_pos
    current_pos = asset.data.root_pos_w

    prev_distance = torch.norm(previous_pos - target_pos, dim=1)
    current_distance = torch.norm(current_pos - target_pos, dim=1)

    return prev_distance - current_distance


def progress_truth_gate(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """GTShadow-only progress reward toward the truth mission gate."""
    asset: RigidObject = env.scene[asset_cfg.name]
    command = env.command_manager.get_term(command_name)

    if not hasattr(command, "gt_next_gate_idx"):
        raise AttributeError(
            "progress_truth_gate requires a command exposing gt_next_gate_idx"
        )

    gate_indices = command.gt_next_gate_idx.to(dtype=torch.long)
    env_ids = torch.arange(env.num_envs, device=asset.device)
    target_pos = command.track.data.object_com_pos_w[env_ids, gate_indices]
    previous_pos = command.previous_pos
    current_pos = asset.data.root_pos_w

    prev_distance = torch.norm(previous_pos - target_pos, dim=1)
    current_distance = torch.norm(current_pos - target_pos, dim=1)
    return prev_distance - current_distance


def gate_passed(
    env: ManagerBasedRLEnv,
    command_name: str | None = None,
) -> torch.Tensor:
    """Reward for passing a gate."""
    missed = (-1.0) * env.command_manager.get_term(command_name).gate_missed
    passed = (1.0) * env.command_manager.get_term(command_name).gate_passed
    return missed + passed


def lookat_next_gate(
    env: ManagerBasedRLEnv,
    std: float,
    command_name: str | None = None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Legacy reward for looking at the next gate."""

    asset: RigidObject = env.scene[asset_cfg.name]

    drone_pos = asset.data.root_pos_w
    drone_att = asset.data.root_quat_w
    next_gate_pos = env.command_manager.get_term(command_name).command[:, :3]

    vec_to_gate = math_utils.normalize(next_gate_pos - drone_pos)

    x_axis = torch.tensor([1.0, 0.0, 0.0], device=asset.device).expand(env.num_envs, 3)
    drone_x_axis = math_utils.normalize(math_utils.quat_apply(drone_att, x_axis))

    dot = (drone_x_axis * vec_to_gate).sum(dim=1).clamp(-1.0, 1.0)
    angle = torch.acos(dot)
    return torch.exp(-angle / std)


def lookat_truth_gate(
    env: ManagerBasedRLEnv,
    std: float,
    command_name: str | None = None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """GTShadow-only look-at reward using the truth mission gate."""
    asset: RigidObject = env.scene[asset_cfg.name]
    command = env.command_manager.get_term(command_name)

    if not hasattr(command, "gt_next_gate_idx"):
        raise AttributeError(
            "lookat_truth_gate requires a command exposing gt_next_gate_idx"
        )

    gate_indices = command.gt_next_gate_idx.to(dtype=torch.long)
    env_ids = torch.arange(env.num_envs, device=asset.device)
    next_gate_pos = command.track.data.object_com_pos_w[env_ids, gate_indices]

    drone_pos = asset.data.root_pos_w
    drone_att = asset.data.root_quat_w
    vec_to_gate = math_utils.normalize(next_gate_pos - drone_pos)

    x_axis = torch.tensor(
        [1.0, 0.0, 0.0],
        dtype=drone_pos.dtype,
        device=asset.device,
    ).expand(env.num_envs, 3)
    drone_x_axis = math_utils.normalize(math_utils.quat_apply(drone_att, x_axis))

    dot = (drone_x_axis * vec_to_gate).sum(dim=1).clamp(-1.0, 1.0)
    angle = torch.acos(dot)
    return torch.exp(-angle / std)


def swift_perception_awareness(
    env: ManagerBasedRLEnv,
    command_name: str,
    lambda_3: float = -10.0,
    camera_pos_b: tuple[float, float, float] = (0.14, 0.0, 0.05),
    camera_optical_axis_b: tuple[float, float, float] = (1.0, 0.0, 0.0),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Swift-2023 perception-aware reward without the external lambda_2 weight.

    Kaufmann et al. use

        r_perc = lambda_2 * exp(lambda_3 * delta_cam**4)

    where ``delta_cam`` is the angle between the camera optical axis and the
    center of the next gate. IsaacLab's RewardTerm applies ``lambda_2`` as the
    term weight, so this function returns only ``exp(lambda_3*delta_cam**4)``.

    The Stage2 calibrated camera optical +Z axis maps to body +X, hence the
    default body-frame optical axis is ``(1, 0, 0)``. The camera-center offset
    is included instead of measuring the angle from the vehicle origin.
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    device = asset.device

    body_pos_w = asset.data.root_pos_w
    body_quat_w = asset.data.root_quat_w
    gate_center_w = env.command_manager.get_term(command_name).command[:, :3]

    camera_offset_b = torch.tensor(camera_pos_b, dtype=body_pos_w.dtype, device=device).expand(env.num_envs, 3)
    optical_axis_b = torch.tensor(
        camera_optical_axis_b, dtype=body_pos_w.dtype, device=device
    ).expand(env.num_envs, 3)

    camera_pos_w = body_pos_w + math_utils.quat_apply(body_quat_w, camera_offset_b)
    optical_axis_w = math_utils.normalize(math_utils.quat_apply(body_quat_w, optical_axis_b))
    to_gate_w = math_utils.normalize(gate_center_w - camera_pos_w)

    dot = torch.sum(optical_axis_w * to_gate_w, dim=1).clamp(-1.0, 1.0)
    delta_cam = torch.acos(dot)
    return torch.exp(float(lambda_3) * torch.pow(delta_cam, 4))




def gt_next_gate_camera_angle_l2(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Dense squared angle from calibrated camera boresight to next gate center.

    Unlike a binary in-FOV test, this stays informative even when the gate is
    completely outside the image. It gives PPO a directional shaping signal for
    recovering camera observability during aggressive turns.
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    command = env.command_manager.get_term(command_name)
    device = asset.device
    dtype = asset.data.root_pos_w.dtype
    num_envs = env.num_envs

    gate_indices = command.next_gate_idx.to(dtype=torch.long)
    env_ids = torch.arange(num_envs, device=device)
    track_data = command.track.data
    gate_pos_all = getattr(track_data, "object_pos_w", None)
    gate_quat_all = getattr(track_data, "object_quat_w", None)
    if gate_pos_all is None or gate_quat_all is None:
        gate_pos_all = getattr(track_data, "object_link_pos_w", None)
        gate_quat_all = getattr(track_data, "object_link_quat_w", None)
    if gate_pos_all is None or gate_quat_all is None:
        raise RuntimeError(
            "GT camera-angle reward requires actor/link gate pose"
        )

    gate_pos_w = gate_pos_all[env_ids, gate_indices]
    gate_quat_w = gate_quat_all[env_ids, gate_indices]
    gate_R_wg = math_utils.matrix_from_quat(gate_quat_w)

    corners_g, camera_offset_b, R_bc = _gt_camera_reward_tensors(
        device,
        dtype,
    )
    gate_center_g = corners_g.mean(dim=0)
    gate_center_w = gate_pos_w + torch.einsum(
        "nij,j->ni",
        gate_R_wg,
        gate_center_g,
    )

    body_pos_w = asset.data.root_pos_w
    body_quat_w = asset.data.root_quat_w
    camera_pos_w = body_pos_w + math_utils.quat_apply(
        body_quat_w,
        camera_offset_b.view(1, 3).expand(num_envs, -1),
    )

    # Camera optical +Z expressed in body coordinates is column 2 of R_bc.
    optical_axis_b = R_bc[:, 2].view(1, 3).expand(num_envs, -1)
    optical_axis_w = math_utils.normalize(
        math_utils.quat_apply(body_quat_w, optical_axis_b)
    )
    to_gate_w = math_utils.normalize(gate_center_w - camera_pos_w)

    dot = torch.sum(optical_axis_w * to_gate_w, dim=1).clamp(-1.0, 1.0)
    angle = torch.acos(dot)
    return torch.square(angle)

def gt_next_gate_image_visibility(
    env: ManagerBasedRLEnv,
    command_name: str,
    margin_px: float = 24.0,
    center_sigma: float = 1.0,
    min_depth_m: float = 0.05,
    center_weight: float = 0.25,
    coverage_weight: float = 0.25,
    margin_weight: float = 0.20,
    usable_bonus_weight: float = 0.30,
    output_bias: float = 0.0,
    insufficient_visible_penalty: float = 0.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward keeping the active GT gate usable in the calibrated camera image.

    This is a training-only GT shaping term. It analytically projects the four
    calibrated gate-opening corners into the same 256x256 Stage2 pinhole camera
    used by the estimator. No rendered image or detector output is required, so
    the 4096-environment GT PPO training stage remains sensor-free and fast.

    The returned score is in [0, 1] and combines:
      * a smooth gate-center term that still gives shaping just outside the FOV;
      * the fraction of the four semantic corners actually inside the image;
      * image-border margin for visible corners;
      * an explicit bonus when >=2 corners are visible, matching the production
        direct-reprojection minimum measurement requirement.
    """
    if margin_px <= 0.0:
        raise ValueError("margin_px must be positive")
    if center_sigma <= 0.0:
        raise ValueError("center_sigma must be positive")

    weights = (
        float(center_weight),
        float(coverage_weight),
        float(margin_weight),
        float(usable_bonus_weight),
    )
    if any(value < 0.0 for value in weights):
        raise ValueError("image-visibility reward weights must be non-negative")
    weight_sum = sum(weights)
    if weight_sum <= 0.0:
        raise ValueError("image-visibility reward weights must sum to > 0")

    asset: RigidObject = env.scene[asset_cfg.name]
    command = env.command_manager.get_term(command_name)
    device = asset.device
    dtype = asset.data.root_pos_w.dtype
    num_envs = env.num_envs

    gate_indices = command.next_gate_idx.to(dtype=torch.long)
    env_ids = torch.arange(num_envs, device=device)

    # IMPORTANT: calibrated corners are in the gate actor frame, not the COM
    # frame. Match Stage2 truth labeling exactly.
    track_data = command.track.data
    gate_pos_all = getattr(track_data, "object_pos_w", None)
    gate_quat_all = getattr(track_data, "object_quat_w", None)
    if gate_pos_all is None or gate_quat_all is None:
        gate_pos_all = getattr(track_data, "object_link_pos_w", None)
        gate_quat_all = getattr(track_data, "object_link_quat_w", None)
    if gate_pos_all is None or gate_quat_all is None:
        raise RuntimeError(
            "GT camera-visibility reward requires actor/link gate pose"
        )

    gate_pos_w = gate_pos_all[env_ids, gate_indices]
    gate_quat_w = gate_quat_all[env_ids, gate_indices]
    gate_R_wg = math_utils.matrix_from_quat(gate_quat_w)

    corners_g_base, camera_offset_b, R_bc = _gt_camera_reward_tensors(
        device,
        dtype,
    )
    corners_g = corners_g_base.unsqueeze(0).expand(num_envs, -1, -1)
    corners_w = gate_pos_w[:, None, :] + torch.einsum(
        "nij,nkj->nki",
        gate_R_wg,
        corners_g,
    )

    body_pos_w = asset.data.root_pos_w
    body_R_wb = math_utils.matrix_from_quat(asset.data.root_quat_w)
    rel_w = corners_w - body_pos_w[:, None, :]
    corners_b = torch.einsum(
        "nji,nkj->nki",
        body_R_wb,
        rel_w,
    )

    # Row-vector form of p_c = R_cb * (p_b - t_bc): rel_b @ R_bc.
    corners_c = torch.matmul(
        corners_b - camera_offset_b.view(1, 1, 3),
        R_bc,
    )

    fx, fy, cx, cy = (float(v) for v in OPENVINS_CAMERA_INTRINSICS)
    image_width, image_height = (
        int(OPENVINS_CAMERA_RESOLUTION[0]),
        int(OPENVINS_CAMERA_RESOLUTION[1]),
    )

    z = corners_c[..., 2]
    safe_z = torch.where(
        z > float(min_depth_m),
        z,
        torch.ones_like(z),
    )
    u = fx * corners_c[..., 0] / safe_z + cx
    v = fy * corners_c[..., 1] / safe_z + cy

    forward = z > float(min_depth_m)
    visible = (
        forward
        & (u >= 0.0)
        & (u < float(image_width))
        & (v >= 0.0)
        & (v < float(image_height))
    )

    coverage = visible.to(dtype).mean(dim=1)
    usable_bonus = (visible.sum(dim=1) >= 2).to(dtype)

    edge_distance = torch.minimum(
        torch.minimum(u, float(image_width - 1) - u),
        torch.minimum(v, float(image_height - 1) - v),
    )
    per_corner_margin = torch.clamp(
        edge_distance / float(margin_px),
        min=0.0,
        max=1.0,
    )
    margin_score = (
        per_corner_margin * visible.to(dtype)
    ).mean(dim=1)

    # A smooth center term provides a recovery signal even when all four
    # corners are just outside the image but the gate remains in front.
    center_c = corners_c.mean(dim=1)
    center_z = center_c[:, 2]
    center_safe_z = torch.where(
        center_z > float(min_depth_m),
        center_z,
        torch.ones_like(center_z),
    )
    center_u = fx * center_c[:, 0] / center_safe_z + cx
    center_v = fy * center_c[:, 1] / center_safe_z + cy
    norm_u = (center_u - cx) / (0.5 * float(image_width))
    norm_v = (center_v - cy) / (0.5 * float(image_height))
    center_score = torch.exp(
        -0.5
        * (torch.square(norm_u) + torch.square(norm_v))
        / float(center_sigma * center_sigma)
    )
    center_score = center_score * (
        center_z > float(min_depth_m)
    ).to(dtype)

    score = (
        weights[0] * center_score
        + weights[1] * coverage
        + weights[2] * margin_score
        + weights[3] * usable_bonus
    ) / weight_sum

    # Optional signed shaping used by perception-aware V2. A negative output
    # bias turns the term into a penalty-only objective (perfect visibility
    # approaches zero rather than accumulating positive reward by hovering).
    # The extra hard penalty directly targets the production estimator
    # contract: fewer than two visible corners cannot produce a reprojection
    # update.
    visible_count = visible.sum(dim=1)
    score = score + float(output_bias)
    if float(insufficient_visible_penalty) != 0.0:
        score = score - float(insufficient_visible_penalty) * (
            visible_count < 2
        ).to(dtype)
    return score

def ang_vel_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize base angular velocity using L2 squared kernel."""
    asset: RigidObject = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.root_ang_vel_b), dim=1)


def swift_ctbr_body_rate_command_l2(
    env: ManagerBasedRLEnv,
    action_name: str = "control_action",
) -> torch.Tensor:
    """Squared physical body-rate command used by Swift's r_cmd term."""
    term = env.action_manager.get_term(action_name)
    command = getattr(term, "ctbr_command", None)
    if command is None:
        raise AttributeError(
            f"action term {action_name!r} does not expose ctbr_command"
        )
    return torch.sum(torch.square(command[:, 1:4]), dim=1)


def swift_ctbr_command_delta_l2(
    env: ManagerBasedRLEnv,
    action_name: str = "control_action",
) -> torch.Tensor:
    """Squared physical CTBR command increment ||a_t - a_{t-1}||^2."""
    term = env.action_manager.get_term(action_name)
    command = getattr(term, "ctbr_command", None)
    previous = getattr(term, "previous_ctbr_command", None)
    if command is None or previous is None:
        raise AttributeError(
            f"action term {action_name!r} does not expose CTBR command history"
        )
    return torch.sum(torch.square(command - previous), dim=1)

