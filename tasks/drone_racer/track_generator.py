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

from .gate_texture_variants import ensure_circular12_gate_usd_variants


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


# Estimator-validation circular track.
#
# The previous Easy-7 loop used a radius of 8 m and seven gates.  Consecutive
# chord headings therefore changed by 360/7 ~= 51.4 deg, which is larger than
# the validated Stage2 camera's ~47.2 deg horizontal FOV.  That made the next
# gate geometrically difficult to keep observable during high-speed racing,
# independent of estimator quality.
#
# This 12-gate loop is deliberately designed for the estimator-replacement
# experiment:
#   * 12 gates on a radius-12 m circle, centre (0, 12)
#   * neighbouring gates are ~6.21 m apart, close to Easy-7's ~6.94 m spacing
#   * gate normals follow the circle tangent (yaw step = 30 deg)
#   * at a gate crossing, the next gate centre is only ~15 deg from the
#     current tangent direction, comfortably inside the camera half-FOV
#     (~23.6 deg)
#
# Keeping the spacing similar preserves the original high-speed character while
# reducing the artificial visibility bottleneck.  This track is intended to
# produce a strong GT upper bound first; the exact same geometry can then be
# used when the 31-D policy observation source is switched to the learned
# IMU/TCN/EKF/vision estimator.
CIRCULAR_12_GATE_TRACK_CONFIG = {
    "1":  {"pos": (0.0, 0.0, 1.0), "yaw": 0.0},
    "2":  {"pos": (6.0, 1.60769515, 1.0), "yaw": torch.pi / 6.0},
    "3":  {"pos": (10.39230485, 6.0, 1.0), "yaw": torch.pi / 3.0},
    "4":  {"pos": (12.0, 12.0, 1.0), "yaw": torch.pi / 2.0},
    "5":  {"pos": (10.39230485, 18.0, 1.0), "yaw": 2.0 * torch.pi / 3.0},
    "6":  {"pos": (6.0, 22.39230485, 1.0), "yaw": 5.0 * torch.pi / 6.0},
    "7":  {"pos": (0.0, 24.0, 1.0), "yaw": torch.pi},
    "8":  {"pos": (-6.0, 22.39230485, 1.0), "yaw": -5.0 * torch.pi / 6.0},
    "9":  {"pos": (-10.39230485, 18.0, 1.0), "yaw": -2.0 * torch.pi / 3.0},
    "10": {"pos": (-12.0, 12.0, 1.0), "yaw": -torch.pi / 2.0},
    "11": {"pos": (-10.39230485, 6.0, 1.0), "yaw": -torch.pi / 3.0},
    "12": {"pos": (-6.0, 1.60769515, 1.0), "yaw": -torch.pi / 6.0},
}


# Deterministic deployment start chosen from the exact support of the GT
# Circular-12 training reset distribution.  GT training with randomise_start=True
# places the vehicle 1 m after the predecessor gate, with zero-mean world-frame
# position jitter and an identity-centered attitude perturbation.  For mission
# gate 1, the predecessor is gate 12 at yaw=-30 deg, so the zero-jitter sample is:
#   p0 = p_gate12 + Rz(-30 deg) * [1, 0, 0]
# This avoids the four-metres-before-gate state used by older perception
# diagnostics, which the racing policy never saw during training.
CIRCULAR_12_KNOWN_START_POS_W = (
    -5.133974596215562,
    1.10769515,
    1.0,
)
CIRCULAR_12_KNOWN_START_ROT_WXYZ = (1.0, 0.0, 0.0, 0.0)


def generate_track(track_config: dict | None) -> RigidObjectCollectionCfg:
    # Circular-12 is the localization track: each physical gate has a stable
    # visual identity. Build one USD sibling per gate from the authoritative
    # gate.usd and bind its unique checked-in texture before Isaac spawns /
    # clones the rigid objects. Other tracks keep the original shared asset.
    gate_usd_paths = (
        ensure_circular12_gate_usd_variants()
        if track_config is CIRCULAR_12_GATE_TRACK_CONFIG
        else None
    )

    return RigidObjectCollectionCfg(
        rigid_objects={
            f"gate_{gate_id}": RigidObjectCfg(
                prim_path=f"/World/envs/env_.*/Gate_{gate_id}",
                spawn=sim_utils.UsdFileCfg(
                    usd_path=(
                        gate_usd_paths[str(gate_id)]
                        if gate_usd_paths is not None
                        else "assets/gate/gate.usd"
                    ),
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
