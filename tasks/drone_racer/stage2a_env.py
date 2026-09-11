"""Stage2A Isaac integration helpers.

The OpenCV PnP backend is deliberately a calibration/reference path. Do not
loop it over thousands of RL environments in production; replace the PnPBackend
with a batched implementation while preserving the same perception contract.
"""

from __future__ import annotations

from dataclasses import dataclass

from perception.gate_geometry import GateGeometry
from perception.isaac_adapter import IsaacStage2TruthAdapter
from perception.rigid_transform import RigidTransform
from perception.stage2a_pipeline import Stage2APerceptionPipeline, Stage2AResult


@dataclass
class Stage2AOracleRuntime:
    """Bind the simulator truth adapter to the Stage2A reference pipeline."""

    env: object
    geometry: GateGeometry
    T_bc: RigidTransform
    robot_name: str = "robot"
    track_name: str = "track"
    camera_name: str = "tiled_camera"
    command_name: str = "target"

    def __post_init__(self) -> None:
        self.adapter = IsaacStage2TruthAdapter(
            self.env,
            robot_name=self.robot_name,
            track_name=self.track_name,
            camera_name=self.camera_name,
            command_name=self.command_name,
        )

    def evaluate(self, env_id: int = 0) -> Stage2AResult:
        snapshot = self.adapter.snapshot(env_id)
        pipeline = Stage2APerceptionPipeline(
            geometry=self.geometry,
            camera=snapshot.camera,
            T_bc=self.T_bc,
        )
        return pipeline.process_truth(snapshot.truth)


def stage2a_target_observation(
    env,
    *,
    geometry: GateGeometry,
    T_bc: RigidTransform,
    env_id: int = 0,
):
    """Reference replacement for the oracle ``target_pos_b`` observation.

    Returns only the PnP-derived gate-center position in body coordinates.
    The complete Stage2A result remains available through Stage2AOracleRuntime
    for calibration/error analysis.
    """
    return Stage2AOracleRuntime(env, geometry, T_bc).evaluate(env_id).solution.target_pos_b
