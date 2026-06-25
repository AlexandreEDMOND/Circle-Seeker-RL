from __future__ import annotations

import pygame
import numpy as np

from env import CircleSeekEnv
from renderer import CircleSeekRenderer


KEY_TO_ACTION = {
    pygame.K_UP: CircleSeekEnv.ACTION_UP,
    pygame.K_DOWN: CircleSeekEnv.ACTION_DOWN,
    pygame.K_LEFT: CircleSeekEnv.ACTION_LEFT,
    pygame.K_RIGHT: CircleSeekEnv.ACTION_RIGHT,
    pygame.K_a: CircleSeekEnv.ACTION_TURN_LEFT,
    pygame.K_d: CircleSeekEnv.ACTION_TURN_RIGHT,
}


def pressed_action() -> np.ndarray:
    action = np.zeros(CircleSeekEnv.ACTION_SIZE, dtype=np.int8)
    keys = pygame.key.get_pressed()
    for key, action_index in KEY_TO_ACTION.items():
        if keys[key]:
            action[action_index] = 1
    return action


def main() -> None:
    env = CircleSeekEnv()
    renderer = CircleSeekRenderer(
        env,
        controls_text="arrows: move | A/D: turn vision | R: reset | ESC: quit",
    )
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
            _, reward, _, _, _ = env.step(pressed_action())
            total_reward += reward

        renderer.render(reward, total_reward)

    renderer.close()


if __name__ == "__main__":
    main()
