"""Isaac Lab adapter for the Swift known-track map.

The real Swift system uses a surveyed track layout.  In simulation the same
concept is provided by the gate actor poses already present in the
``RigidObjectCollection``.  This adapter exposes those poses as an explicit
:class:`perception.track_layout.TrackLayout`.
"""

from __future__ import annotations

import numpy as np

from .rigid_transform import RigidTransform
from .track_layout import TrackLayout


def _to_numpy(value) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    return np.asarray(value)


def _gate_actor_pose_arrays(track):
    data = track.data
    positions = getattr(data, "object_pos_w", None)
    quaternions = getattr(data, "object_quat_w", None)
    if positions is None or quaternions is None:
        positions = getattr(data, "object_link_pos_w", None)
        quaternions = getattr(data, "object_link_quat_w", None)
    if positions is None or quaternions is None:
        raise RuntimeError(
            "Track collection does not expose actor/link gate poses. "
            "Do not substitute object_com_pos_w: Stage2 gate keypoints are "
            "calibrated in the gate actor frame."
        )
    return positions, quaternions


def track_layout_from_isaac(env, *, env_id: int = 0, track_name: str = "track") -> TrackLayout:
    """Snapshot every mapped gate actor pose ``T_wg`` for one Isaac environment."""

    track = env.scene[track_name]
    positions, quaternions = _gate_actor_pose_arrays(track)
    gate_poses = []
    for gate_index in range(track.num_objects):
        gate_poses.append(
            RigidTransform.from_pose_wxyz(
                _to_numpy(positions[env_id, gate_index]),
                _to_numpy(quaternions[env_id, gate_index]),
                to_frame="W",
                from_frame="G",
            )
        )
    return TrackLayout(tuple(gate_poses))


def active_gate_index_from_isaac(
    env,
    *,
    env_id: int = 0,
    command_name: str = "target",
) -> int:
    """Return the task's current next-gate index when its identity is known."""

    command = env.command_manager.get_term(command_name)
    return int(_to_numpy(command.next_gate_idx[env_id]).item())
