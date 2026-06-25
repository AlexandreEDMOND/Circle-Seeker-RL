from __future__ import annotations

import math

import numpy as np
import pygame

try:
    from .env import CircleSeekEnv
except ImportError:
    from env import CircleSeekEnv


class CircleSeekRenderer:
    def __init__(
        self,
        env: CircleSeekEnv,
        fps: int = 60,
        controls_text: str = "arrows: move | R: reset | ESC: quit",
    ) -> None:
        pygame.init()
        self.env = env
        self.fps = fps
        self.controls_text = controls_text
        self.screen = pygame.display.set_mode((env.width, env.height))
        pygame.display.set_caption("Circle Seeker RL")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("Arial", 20)

    def render(self, reward: float = 0.0, total_reward: float = 0.0) -> None:
        self.screen.fill((18, 22, 28))
        pygame.draw.rect(
            self.screen,
            (215, 220, 230),
            pygame.Rect(0, 0, self.env.width, self.env.height),
            width=2,
        )

        self._draw_vision()

        pygame.draw.circle(
            self.screen,
            (65, 205, 95),
            self.env.target_position.astype(int),
            int(self.env.target_radius),
        )

        for obstacle in self.env.obstacles:
            pygame.draw.polygon(
                self.screen,
                (220, 70, 70),
                obstacle.vertices().astype(int),
            )

        pygame.draw.circle(
            self.screen,
            (90, 165, 255),
            self.env.agent_position.astype(int),
            int(self.env.agent_radius),
        )
        heading = self.env.agent_position + self.env.agent_radius * 1.8 * np.array(
            [
                math.cos(self.env.agent_orientation),
                math.sin(self.env.agent_orientation),
            ],
            dtype=np.float32,
        )
        pygame.draw.line(
            self.screen,
            (235, 245, 255),
            self.env.agent_position.astype(int),
            tuple(heading.astype(int)),
            width=3,
        )

        self._draw_hud(reward, total_reward)
        pygame.display.flip()
        self.clock.tick(self.fps)

    def close(self) -> None:
        pygame.quit()

    def _draw_vision(self) -> None:
        polygon = self.env.vision_polygon()
        if len(polygon) < 3:
            return

        overlay = pygame.Surface((self.env.width, self.env.height), pygame.SRCALPHA)
        pygame.draw.polygon(
            overlay,
            (90, 165, 255, 45),
            polygon.astype(int),
        )
        pygame.draw.lines(
            overlay,
            (130, 190, 255, 120),
            False,
            polygon.astype(int),
            width=1,
        )
        self.screen.blit(overlay, (0, 0))

    def _draw_hud(self, reward: float, total_reward: float) -> None:
        lines = [
            f"step: {self.env.step_count}/{self.env.max_steps}",
            f"reward: {reward:.3f}",
            f"total reward: {total_reward:.3f}",
            f"distance: {self.env.distance_to_target():.1f}",
            f"status: {self.env.status}",
            self.controls_text,
        ]

        x = 12
        y = 10
        line_height = 24
        for index, line in enumerate(lines):
            surface = self.font.render(line, True, (235, 238, 245))
            self.screen.blit(surface, (x, y + index * line_height))
