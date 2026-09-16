import re
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OPENVINS_CONFIG = PROJECT_ROOT / "config" / "openvins" / "swift_sim" / "estimator_config.yaml"
TRUE_STATIC_CONFIG = PROJECT_ROOT / "config" / "openvins" / "swift_sim" / "estimator_config_debug_true_static.yaml"
IMU_CONFIG = PROJECT_ROOT / "config" / "openvins" / "swift_sim" / "kalibr_imu_chain.yaml"
CAMERA_CONFIG = PROJECT_ROOT / "config" / "openvins" / "swift_sim" / "kalibr_imucam_chain.yaml"


def _scalar(text: str, key: str) -> str:
    match = re.search(rf"^\s*{re.escape(key)}:\s*([^#\n]+)", text, flags=re.MULTILINE)
    assert match is not None, f"Missing OpenVINS config key: {key}"
    return match.group(1).strip()


def test_stationary_diagnostic_can_use_static_initializer_without_waiting_for_jerk():
    text = OPENVINS_CONFIG.read_text(encoding="utf-8")

    # OpenVINS sets wait_for_jerk=False when the ZUPT updater exists. Keeping
    # ZUPT beginning-only allows a stationary simulator start to initialize,
    # while preventing zero-velocity updates from remaining active in flight.
    assert _scalar(text, "try_zupt") == "true"
    assert _scalar(text, "zupt_only_at_beginning") == "true"
    assert float(_scalar(text, "init_max_disparity")) == 15.0


def test_true_static_ablation_disables_dynamic_initializer_but_keeps_startup_zupt():
    text = TRUE_STATIC_CONFIG.read_text(encoding="utf-8")

    assert _scalar(text, "init_dyn_use") == "false"
    assert _scalar(text, "try_zupt") == "true"
    assert _scalar(text, "zupt_only_at_beginning") == "true"
    assert _scalar(text, "use_fej") == "true"
    assert _scalar(text, "integration") == '"rk4"'
    assert int(_scalar(text, "max_cameras")) == 1
    assert _scalar(text, "relative_config_imu") == '"kalibr_imu_chain.yaml"'
    assert _scalar(text, "relative_config_imucam") == '"kalibr_imucam_chain.yaml"'


def test_external_30hz_camera_is_not_throttled_again_inside_openvins():
    text = OPENVINS_CONFIG.read_text(encoding="utf-8")
    assert float(_scalar(text, "track_frequency")) >= 60.0


def test_openvins_topics_match_repository_ros_bridge_contract():
    imu_text = IMU_CONFIG.read_text(encoding="utf-8")
    camera_text = CAMERA_CONFIG.read_text(encoding="utf-8")

    assert _scalar(imu_text, "rostopic") == "/swift/imu"
    assert _scalar(camera_text, "rostopic") == "/swift/camera/image_raw"
    assert float(_scalar(imu_text, "update_rate")) == 200.0
