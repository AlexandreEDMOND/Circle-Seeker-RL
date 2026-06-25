from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from src.env import CircleSeekEnv


class CircleSeekGymEnv(gym.Env):
    """Gymnasium adapter around the existing CircleSeekEnv."""

    metadata = {"render_modes": []}

    def __init__(self, **env_kwargs: Any) -> None:
        super().__init__()
        self.env = CircleSeekEnv(**env_kwargs)

        observation_size = self.env.get_observation().shape[0]
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(observation_size,),
            dtype=np.float32,
        )
        self.action_space = spaces.MultiBinary(CircleSeekEnv.ACTION_SIZE)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        observation = self.env.reset(seed=seed)
        return observation, self.env._info()

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        observation, reward, terminated, truncated, info = self.env.step(action)
        return observation, reward, terminated, truncated, info
