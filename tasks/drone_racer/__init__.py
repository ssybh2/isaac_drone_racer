# Copyright (c) 2025, Kousheek Chakraborty
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from . import agents


gym.register(
    id="Isaac-Drone-Racer-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.drone_racer_env_cfg:DroneRacerEnvCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Drone-Racer-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.drone_racer_env_cfg:DroneRacerEnvCfg_PLAY",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Drone-Racer-Stage1-v0",
    entry_point=f"{__name__}.stage1_env:Stage1DroneRacerEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.drone_racer_stage1_env_cfg:DroneRacerStage1EnvCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Drone-Racer-Stage1-Play-v0",
    entry_point=f"{__name__}.stage1_env:Stage1DroneRacerEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.drone_racer_stage1_env_cfg:DroneRacerStage1EnvCfg_PLAY",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Drone-Racer-Stage2-Data-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.drone_racer_stage2_env_cfg:DroneRacerStage2DataEnvCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Drone-Racer-Swift-OpenVINS-v0",
    entry_point=f"{__name__}.swift_openvins_env:SwiftOpenVinsDiagnosticEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.drone_racer_swift_perception_env_cfg:DroneRacerSwiftPerceptionEnvCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Drone-Racer-Swift-Hybrid-OpenVINS-v0",
    entry_point=f"{__name__}.hybrid_openvins_env:HybridSwiftOpenVinsDiagnosticEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.drone_racer_hybrid_openvins_env_cfg:DroneRacerHybridOpenVinsEnvCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_cfg.yaml",
    },
)


gym.register(
    id="Isaac-Drone-Racer-Learned-Inertial-v0",
    entry_point=f"{__name__}.learned_inertial_racing_env:LearnedInertialRacingEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.drone_racer_learned_inertial_env_cfg:DroneRacerLearnedInertialEnvCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_cfg.yaml",
    },
)


gym.register(
    id="Isaac-Drone-Racer-Learned-Inertial-RL-v0",
    entry_point=f"{__name__}.learned_inertial_racing_env:LearnedInertialRacingEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.drone_racer_learned_inertial_env_cfg:"
            "DroneRacerLearnedInertialRLCfg"
        ),
        "skrl_cfg_entry_point": (
            f"{agents.__name__}:skrl_learned_inertial_cfg.yaml"
        ),
    },
)

gym.register(
    id="Isaac-Drone-Racer-Swift-CTBR-Control-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.drone_racer_swift_ctbr_env_cfg:"
            "DroneRacerSwiftCTBRControlEnvCfg"
        ),
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_cfg.yaml",
    },
)



gym.register(
    id="Isaac-Drone-Racer-Swift-CTBR-Train-v0",
    entry_point=f"{__name__}.swift_ctbr_racing_env:SwiftCTBRRacingEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.drone_racer_swift_ctbr_env_cfg:"
            "DroneRacerSwiftCTBRTrainEnvCfg"
        ),
        "skrl_cfg_entry_point": (
            f"{agents.__name__}:skrl_swift_ctbr_cfg.yaml"
        ),
    },
)




gym.register(
    id="Isaac-Drone-Racer-Swift-CTBR-GT-Racing-v0",
    entry_point=f"{__name__}.swift_ctbr_racing_env:SwiftCTBRRacingEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.drone_racer_swift_ctbr_env_cfg:"
            "DroneRacerSwiftCTBRGTRacingEnvCfg"
        ),
        "skrl_cfg_entry_point": (
            f"{agents.__name__}:skrl_swift_ctbr_gt_racing_cfg.yaml"
        ),
    },
)
gym.register(
    id="Isaac-Drone-Racer-Swift-CTBR-GT-Circular12-v0",
    entry_point=f"{__name__}.swift_ctbr_racing_env:SwiftCTBRRacingEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.drone_racer_swift_ctbr_env_cfg:"
            "DroneRacerSwiftCTBRGTCircular12RacingEnvCfg"
        ),
        "skrl_cfg_entry_point": (
            f"{agents.__name__}:skrl_swift_ctbr_gt_circular12_cfg.yaml"
        ),
    },
)

gym.register(
    id="Isaac-Drone-Racer-Swift-CTBR-GT-Circular12-Stable-v0",
    entry_point=f"{__name__}.swift_ctbr_racing_env:SwiftCTBRRacingEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.drone_racer_swift_ctbr_env_cfg:"
            "DroneRacerSwiftCTBRGTCircular12StableRacingEnvCfg"
        ),
        "skrl_cfg_entry_point": (
            f"{agents.__name__}:skrl_swift_ctbr_gt_circular12_stable_cfg.yaml"
        ),
    },
)

gym.register(
    id="Isaac-Drone-Racer-Swift-CTBR-GT-Circular12-Stable-KnownStart-v0",
    entry_point=f"{__name__}.swift_ctbr_racing_env:SwiftCTBRRacingEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.drone_racer_swift_ctbr_env_cfg:"
            "DroneRacerSwiftCTBRGTCircular12StableKnownStartEnvCfg"
        ),
        "skrl_cfg_entry_point": (
            f"{agents.__name__}:skrl_swift_ctbr_gt_circular12_stable_cfg.yaml"
        ),
    },
)

gym.register(
    id="Isaac-Drone-Racer-Swift-CTBR-GT-Circular12-StableHeading-v0",
    entry_point=f"{__name__}.swift_ctbr_racing_env:SwiftCTBRRacingEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.drone_racer_swift_ctbr_env_cfg:"
            "DroneRacerSwiftCTBRGTCircular12StableHeadingEnvCfg"
        ),
        "skrl_cfg_entry_point": (
            f"{agents.__name__}:skrl_swift_ctbr_gt_circular12_stable_heading_cfg.yaml"
        ),
    },
)

gym.register(
    id="Isaac-Drone-Racer-Swift-CTBR-GT-Circular12-StableMultiGate-v0",
    entry_point=f"{__name__}.swift_ctbr_racing_env:SwiftCTBRRacingEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.drone_racer_swift_ctbr_env_cfg:"
            "DroneRacerSwiftCTBRGTCircular12StableMultiGateEnvCfg"
        ),
        "skrl_cfg_entry_point": (
            f"{agents.__name__}:skrl_swift_ctbr_gt_circular12_stable_multigate_cfg.yaml"
        ),
    },
)

gym.register(
    id="Isaac-Drone-Racer-Swift-CTBR-GT-PerceptionAware-v0",
    entry_point=f"{__name__}.swift_ctbr_racing_env:SwiftCTBRRacingEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.drone_racer_swift_ctbr_env_cfg:"
            "DroneRacerSwiftCTBRGTPerceptionAwareEnvCfg"
        ),
        "skrl_cfg_entry_point": (
            f"{agents.__name__}:skrl_swift_ctbr_gt_perception_cfg.yaml"
        ),
    },
)

gym.register(
    id="Isaac-Drone-Racer-Swift-CTBR-GT-PerceptionAwareV2-v0",
    entry_point=f"{__name__}.swift_ctbr_racing_env:SwiftCTBRRacingEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.drone_racer_swift_ctbr_env_cfg:"
            "DroneRacerSwiftCTBRGTPerceptionAwareV2EnvCfg"
        ),
        "skrl_cfg_entry_point": (
            f"{agents.__name__}:skrl_swift_ctbr_gt_perception_v2_cfg.yaml"
        ),
    },
)

gym.register(
    id="Isaac-Drone-Racer-Swift-CTBR-GT-PerceptionAwareV3-v0",
    entry_point=f"{__name__}.swift_ctbr_racing_env:SwiftCTBRRacingEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.drone_racer_swift_ctbr_env_cfg:"
            "DroneRacerSwiftCTBRGTPerceptionAwareV3EnvCfg"
        ),
        "skrl_cfg_entry_point": (
            f"{agents.__name__}:skrl_swift_ctbr_gt_perception_v3_cfg.yaml"
        ),
    },
)

gym.register(
    id="Isaac-Drone-Racer-Swift-CTBR-GT-Circular12-KnownStart-v0",
    entry_point=f"{__name__}.swift_ctbr_racing_env:SwiftCTBRRacingEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.drone_racer_swift_ctbr_env_cfg:"
            "DroneRacerSwiftCTBRGTCircular12KnownStartEnvCfg"
        ),
        "skrl_cfg_entry_point": (
            f"{agents.__name__}:skrl_swift_ctbr_gt_circular12_cfg.yaml"
        ),
    },
)

gym.register(
    id="Isaac-Drone-Racer-Swift-CTBR-GT-Circular12-FixedStart-v0",
    entry_point=f"{__name__}.swift_ctbr_racing_env:SwiftCTBRRacingEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.drone_racer_swift_ctbr_env_cfg:"
            "DroneRacerSwiftCTBRGTCircular12FixedStartEnvCfg"
        ),
        "skrl_cfg_entry_point": (
            f"{agents.__name__}:skrl_swift_ctbr_gt_circular12_cfg.yaml"
        ),
    },
)

gym.register(
    id="Isaac-Drone-Racer-Swift-CTBR-GT-FixedStart-v0",
    entry_point=f"{__name__}.swift_ctbr_racing_env:SwiftCTBRRacingEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.drone_racer_swift_ctbr_env_cfg:"
            "DroneRacerSwiftCTBRGTFixedStartEnvCfg"
        ),
        "skrl_cfg_entry_point": (
            f"{agents.__name__}:skrl_swift_ctbr_gt_racing_cfg.yaml"
        ),
    },
)

gym.register(
    id="Isaac-Drone-Racer-Swift-CTBR-Train-PassState-v0",
    entry_point=f"{__name__}.swift_ctbr_racing_env:SwiftCTBRRacingEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.drone_racer_swift_ctbr_env_cfg:"
            "DroneRacerSwiftCTBRPassStateTrainEnvCfg"
        ),
        "skrl_cfg_entry_point": (
            f"{agents.__name__}:skrl_swift_ctbr_passstate_cfg.yaml"
        ),
    },
)

gym.register(
    id="Isaac-Drone-Racer-Learned-Inertial-Swift-CTBR-v0",
    entry_point=f"{__name__}.learned_inertial_racing_env:LearnedInertialRacingEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.drone_racer_swift_ctbr_env_cfg:"
            "DroneRacerLearnedInertialSwiftCTBRRLCfg"
        ),
        "skrl_cfg_entry_point": (
            f"{agents.__name__}:skrl_swift_ctbr_cfg.yaml"
        ),
    },
)



gym.register(
    id="Isaac-Drone-Racer-Learned-Inertial-Swift-CTBR-GTPolicy-v0",
    entry_point=f"{__name__}.learned_inertial_racing_env:LearnedInertialRacingEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.drone_racer_swift_ctbr_env_cfg:"
            "DroneRacerLearnedInertialSwiftCTBRRLCfg"
        ),
        # Deliberately use the exact same 256x256x256 shared-model definition
        # and RunningStandardScaler modules as the successful GT racing
        # checkpoint. Only the 31D observation source changes from simulator
        # truth to the estimator-backed deployment observation.
        "skrl_cfg_entry_point": (
            f"{agents.__name__}:skrl_swift_ctbr_gt_racing_cfg.yaml"
        ),
    },
)


gym.register(
    id="Isaac-Drone-Racer-Learned-Inertial-Swift-CTBR-Circular12-KnownStart-GTPolicy-v0",
    entry_point=f"{__name__}.learned_inertial_racing_env:LearnedInertialRacingEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.drone_racer_swift_ctbr_env_cfg:"
            "DroneRacerLearnedInertialSwiftCTBRCircular12KnownStartGTPolicyCfg"
        ),
        "skrl_cfg_entry_point": (
            f"{agents.__name__}:skrl_swift_ctbr_gt_circular12_cfg.yaml"
        ),
    },
)


gym.register(
    id="Isaac-Drone-Racer-Learned-Inertial-Swift-CTBR-Circular12-GTPolicy-v0",
    entry_point=f"{__name__}.learned_inertial_racing_env:LearnedInertialRacingEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.drone_racer_swift_ctbr_env_cfg:"
            "DroneRacerLearnedInertialSwiftCTBRCircular12GTPolicyCfg"
        ),
        "skrl_cfg_entry_point": (
            f"{agents.__name__}:skrl_swift_ctbr_gt_circular12_cfg.yaml"
        ),
    },
)


gym.register(
    id="Isaac-Drone-Racer-Learned-Inertial-Swift-CTBR-Circular12-KnownStart-GTPolicy-V7-v0",
    entry_point=f"{__name__}.learned_inertial_racing_env:LearnedInertialRacingEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.drone_racer_swift_ctbr_env_cfg:"
            "DroneRacerLearnedInertialSwiftCTBRCircular12KnownStartGTPolicyV7Cfg"
        ),
        "skrl_cfg_entry_point": (
            f"{agents.__name__}:skrl_swift_ctbr_gt_circular12_cfg.yaml"
        ),
    },
)


gym.register(
    id="Isaac-Drone-Racer-Learned-Inertial-Swift-CTBR-Circular12-KnownStart-GTShadow-V7-v0",
    entry_point=f"{__name__}.learned_inertial_racing_env:LearnedInertialRacingEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.drone_racer_swift_ctbr_env_cfg:"
            "DroneRacerLearnedInertialSwiftCTBRCircular12KnownStartGTShadowV7Cfg"
        ),
        "skrl_cfg_entry_point": (
            f"{agents.__name__}:skrl_swift_ctbr_gt_circular12_cfg.yaml"
        ),
    },
)


gym.register(
    id="Isaac-Drone-Racer-Learned-Inertial-Swift-CTBR-Circular12-KnownStart-GTShadow-V7-NoVision-v0",
    entry_point=f"{__name__}.learned_inertial_racing_env:LearnedInertialRacingEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.drone_racer_swift_ctbr_env_cfg:"
            "DroneRacerLearnedInertialSwiftCTBRCircular12KnownStartGTShadowV7NoVisionCfg"
        ),
        "skrl_cfg_entry_point": (
            f"{agents.__name__}:skrl_swift_ctbr_gt_circular12_cfg.yaml"
        ),
    },
)


gym.register(
    id="Isaac-Drone-Racer-Learned-Inertial-Swift-CTBR-Circular12-KnownStart-GTShadow-V7-IMUOnly-v0",
    entry_point=f"{__name__}.learned_inertial_racing_env:LearnedInertialRacingEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.drone_racer_swift_ctbr_env_cfg:"
            "DroneRacerLearnedInertialSwiftCTBRCircular12KnownStartGTShadowV7ImuOnlyCfg"
        ),
        "skrl_cfg_entry_point": (
            f"{agents.__name__}:skrl_swift_ctbr_gt_circular12_cfg.yaml"
        ),
    },
)


gym.register(
    id="Isaac-Drone-Racer-Learned-Inertial-Swift-CTBR-Circular12-KnownStart-GTShadow-V7-OnlineAudit-v0",
    entry_point=f"{__name__}.learned_inertial_racing_env:LearnedInertialRacingEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.drone_racer_swift_ctbr_env_cfg:"
            "DroneRacerLearnedInertialSwiftCTBRCircular12KnownStartGTShadowV7OnlineAuditCfg"
        ),
        "skrl_cfg_entry_point": (
            f"{agents.__name__}:skrl_swift_ctbr_gt_circular12_cfg.yaml"
        ),
    },
)


gym.register(
    id="Isaac-Drone-Racer-Learned-Inertial-Swift-CTBR-Circular12-KnownStart-GTShadow-V7-OracleDV-v0",
    entry_point=f"{__name__}.learned_inertial_racing_env:LearnedInertialRacingEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.drone_racer_swift_ctbr_env_cfg:"
            "DroneRacerLearnedInertialSwiftCTBRCircular12KnownStartGTShadowV7OracleDVCfg"
        ),
        "skrl_cfg_entry_point": (
            f"{agents.__name__}:skrl_swift_ctbr_gt_circular12_cfg.yaml"
        ),
    },
)


gym.register(
    id="Isaac-Drone-Racer-Learned-Inertial-Swift-CTBR-Circular12-KnownStart-GTShadow-V7-OracleWorldDV-v0",
    entry_point=f"{__name__}.learned_inertial_racing_env:LearnedInertialRacingEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.drone_racer_swift_ctbr_env_cfg:"
            "DroneRacerLearnedInertialSwiftCTBRCircular12KnownStartGTShadowV7OracleWorldDVCfg"
        ),
        "skrl_cfg_entry_point": (
            f"{agents.__name__}:skrl_swift_ctbr_gt_circular12_cfg.yaml"
        ),
    },
)


gym.register(
    id="Isaac-Drone-Racer-Learned-Inertial-Swift-CTBR-Circular12-KnownStart-GTShadow-V7-WorldProjected-v0",
    entry_point=f"{__name__}.learned_inertial_racing_env:LearnedInertialRacingEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.drone_racer_swift_ctbr_env_cfg:"
            "DroneRacerLearnedInertialSwiftCTBRCircular12KnownStartGTShadowV7WorldProjectedCfg"
        ),
        "skrl_cfg_entry_point": (
            f"{agents.__name__}:skrl_swift_ctbr_gt_circular12_cfg.yaml"
        ),
    },
)


gym.register(
    id="Isaac-Drone-Racer-Learned-Inertial-Swift-CTBR-Circular12-KnownStart-GTShadow-V7-WorldProjected-FreezeAttBias-v0",
    entry_point=f"{__name__}.learned_inertial_racing_env:LearnedInertialRacingEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.drone_racer_swift_ctbr_env_cfg:"
            "DroneRacerLearnedInertialSwiftCTBRCircular12KnownStartGTShadowV7WorldProjectedFreezeAttBiasCfg"
        ),
        "skrl_cfg_entry_point": (
            f"{agents.__name__}:skrl_swift_ctbr_gt_circular12_cfg.yaml"
        ),
    },
)


gym.register(
    id="Isaac-Drone-Racer-Learned-Inertial-Swift-CTBR-Circular12-KnownStart-GTShadow-V7-CalibratedFreezeAttBias-v0",
    entry_point=f"{__name__}.learned_inertial_racing_env:LearnedInertialRacingEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.drone_racer_swift_ctbr_env_cfg:"
            "DroneRacerLearnedInertialSwiftCTBRCircular12KnownStartGTShadowV7CalibratedFreezeAttBiasCfg"
        ),
        "skrl_cfg_entry_point": (
            f"{agents.__name__}:skrl_swift_ctbr_gt_circular12_cfg.yaml"
        ),
    },
)


gym.register(
    id="Isaac-Drone-Racer-Learned-Inertial-Swift-CTBR-Circular12-KnownStart-GTShadow-V7-CalibratedVelocityOnly-v0",
    entry_point=f"{__name__}.learned_inertial_racing_env:LearnedInertialRacingEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.drone_racer_swift_ctbr_env_cfg:"
            "DroneRacerLearnedInertialSwiftCTBRCircular12KnownStartGTShadowV7CalibratedVelocityOnlyCfg"
        ),
        "skrl_cfg_entry_point": (
            f"{agents.__name__}:skrl_swift_ctbr_gt_circular12_cfg.yaml"
        ),
    },
)


gym.register(
    id="Isaac-Drone-Racer-Learned-Inertial-Swift-CTBR-Circular12-KnownStart-GTShadow-MultiGateVision-v0",
    entry_point=f"{__name__}.learned_inertial_racing_env:LearnedInertialRacingEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.drone_racer_swift_ctbr_env_cfg:"
            "DroneRacerLearnedInertialSwiftCTBRCircular12KnownStartGTShadowMultiGateVisionCfg"
        ),
        "skrl_cfg_entry_point": (
            f"{agents.__name__}:skrl_swift_ctbr_gt_circular12_cfg.yaml"
        ),
    },
)


gym.register(
    id="Isaac-Drone-Racer-Learned-Inertial-Swift-CTBR-Circular12-KnownStart-GTPolicy-MultiGateVision-v0",
    entry_point=f"{__name__}.learned_inertial_racing_env:LearnedInertialRacingEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.drone_racer_swift_ctbr_env_cfg:"
            "DroneRacerLearnedInertialSwiftCTBRCircular12KnownStartGTPolicyMultiGateVisionCfg"
        ),
        "skrl_cfg_entry_point": (
            f"{agents.__name__}:skrl_swift_ctbr_gt_circular12_cfg.yaml"
        ),
    },
)


gym.register(
    id="Isaac-Drone-Racer-Learned-Inertial-Swift-CTBR-GTShadow-v0",
    entry_point=f"{__name__}.learned_inertial_racing_env:LearnedInertialRacingEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.drone_racer_swift_ctbr_env_cfg:"
            "DroneRacerLearnedInertialSwiftCTBRGTShadowCfg"
        ),
        "skrl_cfg_entry_point": (
            f"{agents.__name__}:skrl_swift_ctbr_gt_racing_cfg.yaml"
        ),
    },
)
