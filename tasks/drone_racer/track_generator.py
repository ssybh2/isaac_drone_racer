# Copyright (c) 2025, Kousheek Chakraborty
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# This project uses the IsaacLab framework (https://github.com/isaac-sim/IsaacLab),
# which is licensed under the BSD-3-Clause License.

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
import torch
from isaaclab.assets import RigidObjectCfg, RigidObjectCollectionCfg


# Easy curriculum used by the deployment-faithful learned-inertial PPO task.
#
# The original expert track remains defined unchanged in DroneRacerSceneCfg.
# This track is a near-regular heptagon (radius 8 m, centre (0, 8)) with gate 1
# kept at the validated fixed-start location. Consecutive path headings change
# by about 51.4 deg, avoiding the expert track's >100 deg hairpins and vertical
# stacked-gate manoeuvre while the policy first learns perception-aware racing.
# Gate yaw follows the inbound segment direction so each opening is naturally
# aligned with the nominal flight path.
EASY_7_GATE_TRACK_CONFIG = {
    "1": {"pos": (0.0, 0.0, 1.0), "yaw": -torch.pi / 7.0},
    "2": {"pos": (6.25465186, 3.01208159, 1.0), "yaw": torch.pi / 7.0},
    "3": {"pos": (7.79942330, 9.78016747, 1.0), "yaw": 3.0 * torch.pi / 7.0},
    "4": {"pos": (3.47106991, 15.20775094, 1.0), "yaw": 5.0 * torch.pi / 7.0},
    "5": {"pos": (-3.47106991, 15.20775094, 1.0), "yaw": torch.pi},
    "6": {"pos": (-7.79942330, 9.78016747, 1.0), "yaw": -5.0 * torch.pi / 7.0},
    "7": {"pos": (-6.25465186, 3.01208159, 1.0), "yaw": -3.0 * torch.pi / 7.0},
}


def generate_track(track_config: dict | None) -> RigidObjectCollectionCfg:
    return RigidObjectCollectionCfg(
        rigid_objects={
            f"gate_{gate_id}": RigidObjectCfg(
                prim_path=f"/World/envs/env_.*/Gate_{gate_id}",
                spawn=sim_utils.UsdFileCfg(
                    usd_path="assets/gate/gate.usd",
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(
                        kinematic_enabled=True,
                        disable_gravity=True,
                    ),
                    scale=(1.0, 1.0, 1.0),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=gate_config["pos"],
                    rot=math_utils.quat_from_euler_xyz(
                        torch.tensor(0.0), torch.tensor(0.0), torch.tensor(gate_config["yaw"])
                    ).tolist(),
                ),
            )
            for gate_id, gate_config in track_config.items()
        }
    )
