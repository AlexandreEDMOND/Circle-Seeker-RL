from __future__ import annotations

import pygame
import numpy as np

from env import CircleSeekEnv
from renderer import CircleSeekRenderer


def main() -> None:
    env = CircleSeekEnv()
    renderer = CircleSeekRenderer(env)
    reward = 0.0
    total_reward = 0.0
    running = True

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_r:
                    env.reset()
                    reward = 0.0
                    total_reward = 0.0

        if env.status == "running":
            action = env.rng.integers(0, 2, size=CircleSeekEnv.ACTION_SIZE, dtype=np.int8)
            _, reward, _, _, _ = env.step(action)
            total_reward += reward

        renderer.render(reward, total_reward)

    renderer.close()


if __name__ == "__main__":
    main()
