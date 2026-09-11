"""Known-track geometry used by the Swift-style Stage 2 estimator.

Swift 2023 treats racing gates as mapped landmarks.  The detector/IPPE path
produces a noisy relative transform ``T_cg`` (gate -> camera), while the map
provides ``T_wg`` (gate -> world).  These two transforms are sufficient to
recover a camera/body world-pose measurement without touching simulator truth.

Transform convention follows :mod:`perception.rigid_transform`:
``T_ab`` maps coordinates from frame ``b`` into frame ``a``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .rigid_transform import RigidTransform


@dataclass(frozen=True)
class GateAssociation:
    """Result of matching one relative gate observation to the known map."""

    gate_index: int
    T_wb: RigidTransform
    position_error_to_reference_m: float


@dataclass(frozen=True)
class TrackLayout:
    """Ordered set of mapped gate actor poses ``T_wg``."""

    gate_poses_wg: tuple[RigidTransform, ...]

    def __post_init__(self) -> None:
        poses = tuple(self.gate_poses_wg)
        if not poses:
            raise ValueError("TrackLayout requires at least one gate")
        for pose in poses:
            if pose.to_frame and pose.to_frame != "W":
                raise ValueError("Each gate pose must map into world frame W")
            if pose.from_frame and pose.from_frame != "G":
                raise ValueError("Each gate pose must map from gate frame G")
        object.__setattr__(self, "gate_poses_wg", poses)

    @property
    def num_gates(self) -> int:
        return len(self.gate_poses_wg)

    def gate_pose(self, gate_index: int) -> RigidTransform:
        if not 0 <= gate_index < self.num_gates:
            raise IndexError(f"gate_index {gate_index} outside [0, {self.num_gates})")
        return self.gate_poses_wg[gate_index]

    def camera_pose_from_gate(self, gate_index: int, T_cg: RigidTransform) -> RigidTransform:
        """Recover ``T_wc`` from mapped ``T_wg`` and measured ``T_cg``."""

        if T_cg.to_frame and T_cg.to_frame != "C":
            raise ValueError("T_cg.to_frame must be 'C'")
        if T_cg.from_frame and T_cg.from_frame != "G":
            raise ValueError("T_cg.from_frame must be 'G'")
        return self.gate_pose(gate_index) @ T_cg.inverse()

    def body_pose_from_gate(
        self,
        gate_index: int,
        T_cg: RigidTransform,
        T_bc: RigidTransform,
    ) -> RigidTransform:
        """Recover the gate-derived body world pose ``T_wb``.

        ``T_bc`` maps camera coordinates into body coordinates, hence
        ``T_cb = inverse(T_bc)`` and ``T_wb = T_wc @ T_cb``.
        """

        if T_bc.to_frame and T_bc.to_frame != "B":
            raise ValueError("T_bc.to_frame must be 'B'")
        if T_bc.from_frame and T_bc.from_frame != "C":
            raise ValueError("T_bc.from_frame must be 'C'")
        T_wc = self.camera_pose_from_gate(gate_index, T_cg)
        return T_wc @ T_bc.inverse()

    def associate_to_nearest_gate(
        self,
        T_cg: RigidTransform,
        T_bc: RigidTransform,
        reference_position_w_b,
    ) -> GateAssociation:
        """Associate an unlabeled detection using a reference VIO position.

        This mirrors the Swift paper's use of a known track layout: every
        candidate mapped gate implies a body world pose; the candidate closest
        to the current VIO position is selected.
        """

        reference = np.asarray(reference_position_w_b, dtype=np.float64).reshape(3)
        best: GateAssociation | None = None
        for gate_index in range(self.num_gates):
            T_wb = self.body_pose_from_gate(gate_index, T_cg, T_bc)
            distance = float(np.linalg.norm(T_wb.t - reference))
            candidate = GateAssociation(gate_index, T_wb, distance)
            if best is None or distance < best.position_error_to_reference_m:
                best = candidate
        assert best is not None
        return best
