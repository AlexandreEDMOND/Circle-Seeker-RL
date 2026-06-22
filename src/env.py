from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Obstacle:
    position: np.ndarray
    velocity: np.ndarray
    radius: float


class CircleSeekEnv:
    """Simple 2D circle-seeking environment with a Gymnasium-like API."""

    ACTION_NOOP = 0
    ACTION_UP = 1
    ACTION_DOWN = 2
    ACTION_LEFT = 3
    ACTION_RIGHT = 4

    ACTIONS = {
        ACTION_NOOP: np.array([0.0, 0.0], dtype=np.float32),
        ACTION_UP: np.array([0.0, -1.0], dtype=np.float32),
        ACTION_DOWN: np.array([0.0, 1.0], dtype=np.float32),
        ACTION_LEFT: np.array([-1.0, 0.0], dtype=np.float32),
        ACTION_RIGHT: np.array([1.0, 0.0], dtype=np.float32),
    }

    def __init__(
        self,
        width: int = 800,
        height: int = 600,
        obstacle_count: int = 4,
        max_steps: int = 600,
        agent_speed: float = 5.0,
        obstacle_speed: float = 2.5,
        agent_radius: float = 12.0,
        target_radius: float = 18.0,
        obstacle_radius: float = 16.0,
        approach_bonus: bool = True,
    ) -> None:
        self.width = width
        self.height = height
        self.obstacle_count = obstacle_count
        self.max_steps = max_steps
        self.agent_speed = agent_speed
        self.obstacle_speed = obstacle_speed
        self.agent_radius = agent_radius
        self.target_radius = target_radius
        self.obstacle_radius = obstacle_radius
        self.approach_bonus = approach_bonus
        self.diagonal = float(np.hypot(width, height))

        self.rng = np.random.default_rng()
        self.agent_position = np.zeros(2, dtype=np.float32)
        self.target_position = np.zeros(2, dtype=np.float32)
        self.obstacles: list[Obstacle] = []
        self.step_count = 0
        self.status = "running"
        self.previous_distance = 0.0

        self.reset()

    def reset(self, seed: int | None = None) -> np.ndarray:
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self.step_count = 0
        self.status = "running"
        self.agent_position = np.array(
            [self.width * 0.5, self.height * 0.5], dtype=np.float32
        )

        self.obstacles = []
        self.target_position = np.zeros(2, dtype=np.float32)
        self.target_position = self._sample_free_position(self.target_radius)
        for _ in range(self.obstacle_count):
            position = self._sample_free_position(self.obstacle_radius)
            velocity = self._sample_obstacle_velocity()
            self.obstacles.append(Obstacle(position, velocity, self.obstacle_radius))

        self.previous_distance = self.distance_to_target()
        return self.get_observation()

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        if action not in self.ACTIONS:
            raise ValueError(f"Unknown action {action}. Expected an int from 0 to 4.")

        if self.status != "running":
            return (
                self.get_observation(),
                0.0,
                self.status in {"success", "collision"},
                self.status == "timeout",
                self._info(),
            )

        self.step_count += 1
        self._move_agent(action)
        self._update_obstacles()

        reward = -0.01
        terminated = False
        truncated = False

        distance = self.distance_to_target()
        if self.approach_bonus:
            reward += 0.1 * (self.previous_distance - distance) / self.diagonal
        self.previous_distance = distance

        if self._agent_touches_target():
            reward = 10.0
            terminated = True
            self.status = "success"
        elif self._agent_touches_obstacle():
            reward = -10.0
            terminated = True
            self.status = "collision"
        elif self.step_count >= self.max_steps:
            truncated = True
            self.status = "timeout"

        return self.get_observation(), float(reward), terminated, truncated, self._info()

    def get_observation(self) -> np.ndarray:
        values: list[float] = []

        world_size = np.array([self.width, self.height], dtype=np.float32)
        agent_norm = self.agent_position / world_size
        target_relative = (self.target_position - self.agent_position) / np.array(
            [self.width, self.height], dtype=np.float32
        )
        values.extend(agent_norm.tolist())
        values.extend(target_relative.tolist())

        for obstacle in self.obstacles:
            relative_position = (obstacle.position - self.agent_position) / np.array(
                [self.width, self.height], dtype=np.float32
            )
            relative_velocity = obstacle.velocity / max(self.obstacle_speed, 1.0)
            values.extend(relative_position.tolist())
            values.extend(relative_velocity.tolist())

        values.append(self.distance_to_target() / self.diagonal)
        return np.array(values, dtype=np.float32)

    def distance_to_target(self) -> float:
        return self._distance(self.agent_position, self.target_position)

    def _move_agent(self, action: int) -> None:
        direction = self.ACTIONS[action]
        self.agent_position = self.agent_position + direction * self.agent_speed
        self.agent_position[0] = np.clip(
            self.agent_position[0], self.agent_radius, self.width - self.agent_radius
        )
        self.agent_position[1] = np.clip(
            self.agent_position[1], self.agent_radius, self.height - self.agent_radius
        )

    def _update_obstacles(self) -> None:
        for obstacle in self.obstacles:
            obstacle.position += obstacle.velocity

            if obstacle.position[0] <= obstacle.radius:
                obstacle.position[0] = obstacle.radius
                obstacle.velocity[0] *= -1.0
            elif obstacle.position[0] >= self.width - obstacle.radius:
                obstacle.position[0] = self.width - obstacle.radius
                obstacle.velocity[0] *= -1.0

            if obstacle.position[1] <= obstacle.radius:
                obstacle.position[1] = obstacle.radius
                obstacle.velocity[1] *= -1.0
            elif obstacle.position[1] >= self.height - obstacle.radius:
                obstacle.position[1] = self.height - obstacle.radius
                obstacle.velocity[1] *= -1.0

    def _agent_touches_target(self) -> bool:
        return self._circles_overlap(
            self.agent_position,
            self.agent_radius,
            self.target_position,
            self.target_radius,
        )

    def _agent_touches_obstacle(self) -> bool:
        return any(
            self._circles_overlap(
                self.agent_position,
                self.agent_radius,
                obstacle.position,
                obstacle.radius,
            )
            for obstacle in self.obstacles
        )

    def _sample_obstacle_velocity(self) -> np.ndarray:
        angle = self.rng.uniform(0.0, 2.0 * np.pi)
        speed = self.rng.uniform(self.obstacle_speed * 0.6, self.obstacle_speed)
        return np.array([np.cos(angle) * speed, np.sin(angle) * speed], dtype=np.float32)

    def _sample_free_position(self, radius: float) -> np.ndarray:
        for _ in range(500):
            position = np.array(
                [
                    self.rng.uniform(radius, self.width - radius),
                    self.rng.uniform(radius, self.height - radius),
                ],
                dtype=np.float32,
            )
            if self._position_is_free(position, radius):
                return position
        raise RuntimeError("Could not sample a free position. Try fewer obstacles.")

    def _position_is_free(self, position: np.ndarray, radius: float) -> bool:
        margin = 20.0
        if self._circles_overlap(
            position, radius + margin, self.agent_position, self.agent_radius
        ):
            return False
        if self.target_position.any() and self._circles_overlap(
            position, radius + margin, self.target_position, self.target_radius
        ):
            return False
        return not any(
            self._circles_overlap(position, radius + margin, obstacle.position, obstacle.radius)
            for obstacle in self.obstacles
        )

    def _info(self) -> dict:
        return {
            "step": self.step_count,
            "status": self.status,
            "distance_to_target": self.distance_to_target(),
        }

    @staticmethod
    def _distance(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.linalg.norm(a - b))

    @staticmethod
    def _circles_overlap(
        a_position: np.ndarray,
        a_radius: float,
        b_position: np.ndarray,
        b_radius: float,
    ) -> bool:
        return CircleSeekEnv._distance(a_position, b_position) <= a_radius + b_radius
