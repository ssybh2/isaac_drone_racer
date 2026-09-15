"""Hardware-portable state-estimation interfaces for the drone racer.

Exports are resolved lazily so NumPy-only OpenVINS/learned-motion tooling does
not import PyTorch-backed fake sensor modules unless those symbols are actually
requested.
"""

from __future__ import annotations

from importlib import import_module


_EXPORTS = {
    "FakeImu": (".fake_imu", "FakeImu"),
    "FakeImuCfg": (".fake_sensor_cfg", "FakeImuCfg"),
    "FakeSensorPipelineCfg": (".fake_sensor_cfg", "FakeSensorPipelineCfg"),
    "FakeVio": (".fake_vio", "FakeVio"),
    "FakeVioCfg": (".fake_sensor_cfg", "FakeVioCfg"),
    "UniformRange": (".fake_sensor_cfg", "UniformRange"),
    "compose_transform_w_b": (".frame_math", "compose_transform_w_b"),
    "rotate_world_to_body": (".frame_math", "rotate_world_to_body"),
    "OpenVinsFrameAlignment": (".openvins_bridge", "OpenVinsFrameAlignment"),
    "OpenVinsOdomSample": (".openvins_bridge", "OpenVinsOdomSample"),
    "OpenVinsRos2Bridge": (".openvins_bridge", "OpenVinsRos2Bridge"),
    "OpenVinsSensorRateGate": (".openvins_bridge", "OpenVinsSensorRateGate"),
    "Stage1StatePipeline": (".pipeline", "Stage1StatePipeline"),
    "policy_drone_state": (".policy_adapter", "policy_drone_state"),
    "GroundTruthState": (".state_estimate", "GroundTruthState"),
    "ImuEstimate": (".state_estimate", "ImuEstimate"),
    "SourceStatus": (".state_estimate", "SourceStatus"),
    "StateEstimate": (".state_estimate", "StateEstimate"),
    "VioEstimate": (".state_estimate", "VioEstimate"),
    "StateEstimateAssembler": (".state_estimate_assembler", "StateEstimateAssembler"),
    "SwiftFusionResult": (".swift_fusion", "SwiftFusionResult"),
    "SwiftPerceptionFusion": (".swift_fusion", "SwiftPerceptionFusion"),
    "FusedWorldEstimate": (".swift_vio_drift", "FusedWorldEstimate"),
    "KalmanUpdateDiagnostics": (".swift_vio_drift", "KalmanUpdateDiagnostics"),
    "VioDriftKalmanFilter": (".swift_vio_drift", "VioDriftKalmanFilter"),
    "VioWorldEstimate": (".swift_vio_drift", "VioWorldEstimate"),
    "VioWorldEstimateBuffer": (".vio_time_buffer", "VioWorldEstimateBuffer"),
    "LearnedDisplacementMeasurement": (".learned_vio_drift", "LearnedDisplacementMeasurement"),
    "LearnedDriftUpdateDiagnostics": (".learned_vio_drift", "LearnedDriftUpdateDiagnostics"),
    "LearnedVioDriftFilter": (".learned_vio_drift", "LearnedVioDriftFilter"),
    "DisplacementPrediction": (".learned_motion", "DisplacementPrediction"),
    "LearnedMotionBuffer": (".learned_motion", "LearnedMotionBuffer"),
    "MotionWindow": (".learned_motion", "MotionWindow"),
    "TorchTcnDisplacementPredictor": (".learned_motion", "TorchTcnDisplacementPredictor"),
    "HybridLearnedVioCorrector": (".hybrid_vio_corrector", "HybridLearnedVioCorrector"),
    "HybridVioCorrectionResult": (".hybrid_vio_corrector", "HybridVioCorrectionResult"),
    "body_motion_to_world": (".hybrid_vio_corrector", "body_motion_to_world"),
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str):
    try:
        module_name, attribute_name = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    value = getattr(import_module(module_name, __name__), attribute_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
