"""Manager-based racing environment with an explicit bounded CTBR action space."""

from __future__ import annotations

import gymnasium as gym
import numpy as np
from isaaclab.envs import ManagerBasedRLEnv


class SwiftCTBRRacingEnv(ManagerBasedRLEnv):
    """Expose the normalized [-1, 1]^4 CTBR contract to Gym/skrl."""

    def __init__(self, cfg, render_mode: str | None = None, **kwargs) -> None:
        super().__init__(cfg=cfg, render_mode=render_mode, **kwargs)

        action_dim = int(self.action_manager.total_action_dim)
        if action_dim != 4:
            raise ValueError(
                f"SwiftCTBRRacingEnv expects 4 CTBR actions, got {action_dim}"
            )

        self.single_action_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(action_dim,),
            dtype=np.float32,
        )
        self.action_space = gym.vector.utils.batch_space(
            self.single_action_space,
            self.num_envs,
        )
