import json

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
