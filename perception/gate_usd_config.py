"""Gate USD calibration placeholder for Stage2A.

The actual values should be calibrated from assets/gate/gate.usd.
Keeping this separated avoids hard-coding dimensions inside projection/PnP code.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class GateUsdConfig:
    usd_path: str = "assets/gate/gate.usd"
    opening_width_m: float = 1.0
    opening_height_m: float = 1.0
    frame_origin: str = "opening_center"


DEFAULT_GATE_USD = GateUsdConfig()
