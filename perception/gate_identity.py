"""Stable visual identities for the color-coded Circular-12 gate map."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
GATE_IDENTITY_MAP_PATH = (
    REPO_ROOT / "assets/gate/textures/circular12_gate_texture_map.json"
)


@dataclass(frozen=True)
class GateIdentity:
    gate_index: int
    gate_id: int
    color_name: str
    rgb: tuple[int, int, int]
    hex: str
    position_m: tuple[float, float, float]
    yaw_deg: float


def load_circular12_gate_identities() -> tuple[GateIdentity, ...]:
    data = json.loads(GATE_IDENTITY_MAP_PATH.read_text(encoding="utf-8"))
    gates = data.get("gates", [])
    if len(gates) != 12:
        raise RuntimeError(
            f"expected 12 Circular-12 gate identities, got {len(gates)}"
        )

    identities: list[GateIdentity] = []
    for gate_index, gate in enumerate(gates):
        gate_id = int(gate["gate_id"])
        if gate_id != gate_index + 1:
            raise RuntimeError(
                f"gate map order mismatch: index={gate_index} gate_id={gate_id}"
            )
        identities.append(
            GateIdentity(
                gate_index=gate_index,
                gate_id=gate_id,
                color_name=str(gate["color_name"]),
                rgb=tuple(int(v) for v in gate["rgb"]),
                hex=str(gate["hex"]),
                position_m=tuple(float(v) for v in gate["position_m"]),
                yaw_deg=float(gate["yaw_deg"]),
            )
        )
    return tuple(identities)


CIRCULAR12_GATE_IDENTITIES = load_circular12_gate_identities()
CIRCULAR12_GATE_ID_COUNT = len(CIRCULAR12_GATE_IDENTITIES)


def identity_for_gate_index(gate_index: int) -> GateIdentity:
    index = int(gate_index)
    if not 0 <= index < CIRCULAR12_GATE_ID_COUNT:
        raise ValueError(f"gate_index out of range: {gate_index}")
    return CIRCULAR12_GATE_IDENTITIES[index]


def identity_for_gate_id(gate_id: int) -> GateIdentity:
    gate_id_int = int(gate_id)
    if not 1 <= gate_id_int <= CIRCULAR12_GATE_ID_COUNT:
        raise ValueError(f"gate_id out of range: {gate_id}")
    return CIRCULAR12_GATE_IDENTITIES[gate_id_int - 1]


def gate_index_from_id(gate_id: int) -> int:
    return identity_for_gate_id(gate_id).gate_index
