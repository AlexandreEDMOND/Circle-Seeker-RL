from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
import pygame
from gymnasium import spaces

from src.env import CircleSeekEnv
from src.renderer import CircleSeekRenderer


class CircleSeekGymEnv(gym.Env):
    """Gymnasium adapter around the existing CircleSeekEnv."""

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(
        self,
        render_mode: str | None = None,
        render_fps: int | None = None,
        controls_text: str = "arrows: move | R: reset | ESC: quit",
        **env_kwargs: Any,
    ) -> None:
        super().__init__()
        if render_mode not in {None, *self.metadata["render_modes"]}:
            raise ValueError(f"Unsupported render_mode: {render_mode!r}.")

        self.env = CircleSeekEnv(**env_kwargs)
        self.render_mode = render_mode
        self.render_fps = render_fps or int(self.metadata["render_fps"])
        self.controls_text = controls_text
        self._renderer: CircleSeekRenderer | None = None
        self._rgb_surface: pygame.Surface | None = None
        self._last_reward = 0.0
        self._total_reward = 0.0

        observation_size = self.env.get_observation().shape[0]
        low, high = self._observation_bounds()
        self.observation_space = spaces.Box(
            low=low,
            high=high,
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
        self._last_reward = 0.0
        self._total_reward = 0.0
        return observation, self.env._info()

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        observation, reward, terminated, truncated, info = self.env.step(action)
        self._last_reward = float(reward)
        self._total_reward += float(reward)
        return observation, reward, terminated, truncated, info

    def render(self) -> np.ndarray | None:
        if self.render_mode is None:
            return None

        if self.render_mode == "human":
            if self._renderer is None:
                self._renderer = CircleSeekRenderer(
                    self.env,
                    fps=self.render_fps,
                    controls_text=self.controls_text,
                )
            self._renderer.render(self._last_reward, self._total_reward)
            return None

        if self._rgb_surface is None:
            self._rgb_surface = pygame.Surface((self.env.width, self.env.height))
            self._renderer = CircleSeekRenderer(
                self.env,
                fps=self.render_fps,
                controls_text=self.controls_text,
                surface=self._rgb_surface,
            )
        self._renderer.render(self._last_reward, self._total_reward)
        return np.transpose(pygame.surfarray.array3d(self._rgb_surface), (1, 0, 2))

    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
        self._rgb_surface = None

    def _observation_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        ray_low = np.zeros(self.env.ray_count, dtype=np.float32)
        ray_high = np.ones(self.env.ray_count, dtype=np.float32)
        low = np.array([0.0, -1.0, 0.0, -1.0, -1.0], dtype=np.float32)
        high = np.array([1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32)
        return np.concatenate((ray_low, low)), np.concatenate((ray_high, high))
