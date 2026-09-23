import json
from pathlib import Path

import numpy as np
import pytest

from perception.corner_detection import CornerObservation
from perception.gate_identity import (
    CIRCULAR12_GATE_ID_COUNT,
    gate_index_from_id,
    identity_for_gate_id,
    identity_for_gate_index,
)


def test_circular12_gate_identity_round_trip():
    assert CIRCULAR12_GATE_ID_COUNT == 12
    for gate_index in range(12):
        identity = identity_for_gate_index(gate_index)
        assert identity.gate_id == gate_index + 1
        assert gate_index_from_id(identity.gate_id) == gate_index
        assert identity_for_gate_id(identity.gate_id) == identity


def test_corner_observation_can_carry_gate_identity():
    obs = CornerObservation(
        corners_uv=np.zeros((4, 2)),
        gate_id=7,
        gate_id_confidence=0.91,
    )
    assert obs.gate_id == 7
    assert obs.gate_id_confidence == pytest.approx(0.91)


def test_corner_observation_rejects_invalid_identity_confidence():
    with pytest.raises(ValueError, match="gate_id_confidence"):
        CornerObservation(
            corners_uv=np.zeros((4, 2)),
            gate_id=1,
            gate_id_confidence=1.1,
        )


def test_ideal_bank_experiment_exports_identity_labels():
    root = Path(__file__).resolve().parents[2]
    text = (
        root
        / "artifacts/swift_ctbr/circular12_constant_bank_3lap/experiment.py"
    ).read_text(encoding="utf-8")
    assert "--capture-dataset" in text
    assert '"gate_id": int(identity.gate_id)' in text
    assert '"color_name": identity.color_name' in text
    assert "vision_smoke" in text
