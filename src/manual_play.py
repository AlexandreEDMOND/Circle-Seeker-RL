from __future__ import annotations

import pygame

from env import CircleSeekEnv
from renderer import CircleSeekRenderer


KEY_TO_ACTION = {
    pygame.K_UP: CircleSeekEnv.ACTION_UP,
    pygame.K_DOWN: CircleSeekEnv.ACTION_DOWN,
    pygame.K_LEFT: CircleSeekEnv.ACTION_LEFT,
    pygame.K_RIGHT: CircleSeekEnv.ACTION_RIGHT,
}


def pressed_action() -> int:
    keys = pygame.key.get_pressed()
    for key, action in KEY_TO_ACTION.items():
        if keys[key]:
            return action
    return CircleSeekEnv.ACTION_NOOP


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
            _, reward, _, _, _ = env.step(pressed_action())
            total_reward += reward

        renderer.render(reward, total_reward)

    renderer.close()


if __name__ == "__main__":
    main()
