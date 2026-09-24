"""Behavior-cloning policy matching the CTBR PPO actor topology."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import nn


@dataclass(frozen=True)
class Circular12BCConfig:
    observation_dim: int = 31
    action_dim: int = 4
    hidden_dims: tuple[int, ...] = (256, 256, 256)
    standardized_observation_clip: float | None = None

    def __post_init__(self) -> None:
        if (
            self.standardized_observation_clip is not None
            and self.standardized_observation_clip <= 0.0
        ):
            raise ValueError(
                "standardized_observation_clip must be positive when set"
            )


class Circular12BCPolicy(nn.Module):
    """31D -> 256x256x256 ELU -> tanh(4D) student."""

    def __init__(
        self,
        cfg: Circular12BCConfig | None = None,
        *,
        observation_mean: torch.Tensor | None = None,
        observation_std: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.cfg = cfg or Circular12BCConfig()
        dims = (
            self.cfg.observation_dim,
            *self.cfg.hidden_dims,
            self.cfg.action_dim,
        )
        layers: list[nn.Module] = []
        for index in range(len(dims) - 1):
            layers.append(nn.Linear(dims[index], dims[index + 1]))
            if index < len(dims) - 2:
                layers.append(nn.ELU())
        self.net = nn.Sequential(*layers)

        mean = (
            torch.zeros(self.cfg.observation_dim)
            if observation_mean is None
            else observation_mean.detach().float().reshape(-1)
        )
        std = (
            torch.ones(self.cfg.observation_dim)
            if observation_std is None
            else observation_std.detach().float().reshape(-1)
        )
        if mean.numel() != self.cfg.observation_dim:
            raise ValueError("observation_mean has the wrong dimension")
        if std.numel() != self.cfg.observation_dim:
            raise ValueError("observation_std has the wrong dimension")
        self.register_buffer("observation_mean", mean)
        self.register_buffer("observation_std", std.clamp_min(1.0e-6))

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        standardized = (
            observation - self.observation_mean
        ) / self.observation_std
        if self.cfg.standardized_observation_clip is not None:
            clip = float(self.cfg.standardized_observation_clip)
            standardized = standardized.clamp(-clip, clip)
        return torch.tanh(self.net(standardized))

    def linear_layers(self) -> list[nn.Linear]:
        return [
            module for module in self.net if isinstance(module, nn.Linear)
        ]

    def checkpoint_payload(self, metadata: dict | None = None) -> dict:
        return {
            "schema": "isaac_drone_racer.circular12_bc.v1",
            "config": asdict(self.cfg),
            "state_dict": self.state_dict(),
            "metadata": dict(metadata or {}),
        }

    def save(
        self, path: str | Path, metadata: dict | None = None
    ) -> None:
        output = Path(path).expanduser()
        output.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.checkpoint_payload(metadata), output)

    @classmethod
    def load(
        cls,
        path: str | Path,
        map_location: str | torch.device = "cpu",
    ) -> tuple["Circular12BCPolicy", dict]:
        payload = torch.load(
            Path(path).expanduser(), map_location=map_location
        )
        if payload.get("schema") != "isaac_drone_racer.circular12_bc.v1":
            raise ValueError("unsupported Circular12 BC checkpoint schema")
        cfg_dict = dict(payload["config"])
        cfg_dict["hidden_dims"] = tuple(cfg_dict["hidden_dims"])
        model = cls(Circular12BCConfig(**cfg_dict))
        model.load_state_dict(payload["state_dict"])
        return model, dict(payload.get("metadata", {}))
