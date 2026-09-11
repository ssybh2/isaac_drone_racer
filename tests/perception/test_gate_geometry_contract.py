import json
from pathlib import Path

import numpy as np
import pytest

from perception.gate_geometry import CORNER_NAMES, GateGeometry
from perception.gate_usd_config import load_gate_keypoint_calibration


def test_rectangular_helper_uses_actor_x_normal_and_semantic_corner_order():
    geometry = GateGeometry.rectangular_x_normal(2.0, 1.0)
    assert geometry.corner_names == CORNER_NAMES
    assert np.allclose(
        geometry.object_points_g,
        [
            [0.0, 1.0, -0.5],
            [0.0, -1.0, -0.5],
            [0.0, -1.0, 0.5],
            [0.0, 1.0, 0.5],
        ],
    )
    assert geometry.opening_width_m == pytest.approx(2.0)
    assert geometry.opening_height_m == pytest.approx(1.0)


def test_calibration_loader_has_no_guessed_numeric_fallback(tmp_path):
    missing = tmp_path / "gate_keypoints.json"
    with pytest.raises(FileNotFoundError):
        load_gate_keypoint_calibration(missing)

    payload = {
        "frame": "gate_actor",
        "corner_order": list(CORNER_NAMES),
        "object_points_gate_actor_m": [
            [0.0, 1.0, -0.5],
            [0.0, -1.0, -0.5],
            [0.0, -1.0, 0.5],
            [0.0, 1.0, 0.5],
        ],
    }
    missing.write_text(json.dumps(payload), encoding="utf-8")
    geometry = load_gate_keypoint_calibration(missing)
    assert geometry.opening_width_m == pytest.approx(2.0)


def test_repository_gate_calibration_matches_measured_opening():
    calibration = Path(__file__).parents[2] / "assets" / "gate" / "gate_keypoints.json"
    geometry = load_gate_keypoint_calibration(calibration)

    assert geometry.opening_width_m == pytest.approx(1.524, abs=1e-6)
    assert geometry.opening_height_m == pytest.approx(1.524, abs=1e-6)
    assert geometry.center_g == pytest.approx([-0.0661411, 0.0, 1.0668], abs=1e-6)


def test_calibration_rejects_non_actor_frame(tmp_path):
    path = tmp_path / "wrong_frame.json"
    path.write_text(
        json.dumps(
            {
                "frame": "gate_com",
                "object_points_gate_actor_m": [
                    [0, 1, -1],
                    [0, -1, -1],
                    [0, -1, 1],
                    [0, 1, 1],
                ],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="gate_actor"):
        load_gate_keypoint_calibration(path)
