from dataclasses import dataclass
import torch


@dataclass
class GateGeometry:
    """Standard Gate frame definition.

    Gate frame G:
      X: right direction when looking through gate
      Y: upward direction
      Z: gate normal direction

    Origin is the geometric center of the opening.
    """

    width: float = 1.0
    height: float = 1.0

    def corners_g(self, device="cpu"):
        w = self.width / 2.0
        h = self.height / 2.0
        return torch.tensor([
            [-w, -h, 0.0],
            [ w, -h, 0.0],
            [ w,  h, 0.0],
            [-w,  h, 0.0],
        ], dtype=torch.float32, device=device)


DEFAULT_GATE_GEOMETRY = GateGeometry(
    width=1.0,
    height=1.0,
)
