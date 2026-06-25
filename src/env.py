from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass
class Obstacle:
    position: np.ndarray
    velocity: np.ndarray
    radius: float
    sides: int
    angle: float

    def vertices(self) -> np.ndarray:
        angles = self.angle + np.linspace(0.0, 2.0 * np.pi, self.sides, endpoint=False)
        unit = np.column_stack((np.cos(angles), np.sin(angles))).astype(np.float32)
        return self.position + unit * self.radius


class CircleSeekEnv:
    """2D circle-seeking environment with polygon obstacles and partial vision."""

    ACTION_SIZE = 6
    ACTION_UP = 0
    ACTION_DOWN = 1
    ACTION_LEFT = 2
    ACTION_RIGHT = 3
    ACTION_TURN_LEFT = 4
    ACTION_TURN_RIGHT = 5
    ACTION_NOOP = np.zeros(ACTION_SIZE, dtype=np.int8)

    def __init__(
        self,
        width: int = 800,
        height: int = 600,
        obstacle_count: int = 4,
        max_steps: int = 600,
        agent_speed: float = 5.0,
        agent_turn_speed: float = 0.12,
        obstacle_speed: float = 2.5,
        agent_radius: float = 12.0,
        target_radius: float = 18.0,
        obstacle_radius: float = 36.0,
        min_target_distance: float = 200.0,
        fov_degrees: float = 90.0,
        ray_count: int = 31,
        approach_bonus: bool = True,
    ) -> None:
        self.width = width
        self.height = height
        self.obstacle_count = obstacle_count
        self.max_steps = max_steps
        self.agent_speed = agent_speed
        self.agent_turn_speed = agent_turn_speed
        self.obstacle_speed = obstacle_speed
        self.agent_radius = agent_radius
        self.target_radius = target_radius
        self.obstacle_radius = obstacle_radius
        self.min_target_distance = min_target_distance
        self.fov_radians = np.deg2rad(fov_degrees)
        self.ray_count = ray_count
        self.approach_bonus = approach_bonus
        self.diagonal = float(np.hypot(width, height))

        self.rng = np.random.default_rng()
        self.agent_position = np.zeros(2, dtype=np.float32)
        self.agent_orientation = -np.pi / 2.0
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
        self.agent_orientation = -np.pi / 2.0

        self.obstacles = []
        self.target_position = np.zeros(2, dtype=np.float32)
        self.target_position = self._sample_free_position(
            self.target_radius,
            min_agent_distance=self.min_target_distance,
        )
        for _ in range(self.obstacle_count):
            position = self._sample_free_position(self.obstacle_radius)
            velocity = self._sample_obstacle_velocity()
            sides = int(self.rng.integers(3, 7))
            angle = float(self.rng.uniform(0.0, 2.0 * np.pi))
            self.obstacles.append(
                Obstacle(position, velocity, self.obstacle_radius, sides, angle)
            )

        self.previous_distance = self.distance_to_target()
        return self.get_observation()

    def step(
        self,
        action: Iterable[int | bool] | np.ndarray,
    ) -> tuple[np.ndarray, float, bool, bool, dict]:
        action_vector = self._validate_action(action)

        if self.status != "running":
            return (
                self.get_observation(),
                0.0,
                self.status in {"success", "collision"},
                self.status == "timeout",
                self._info(),
            )

        self.step_count += 1
        self._rotate_agent(action_vector)
        self._move_agent(action_vector)
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
        ray_distances = self.cast_vision_rays() / self.diagonal
        target_is_visible = self.target_is_visible()
        target_visible = 1.0 if target_is_visible else 0.0
        if target_is_visible:
            target_angle = self._angle_to(self.target_position)
            target_angle_norm = np.clip(target_angle / (self.fov_radians * 0.5), -1.0, 1.0)
            target_distance_norm = self.distance_to_target() / self.diagonal
        else:
            target_angle_norm = 0.0
            target_distance_norm = 0.0

        values = [
            *ray_distances.tolist(),
            target_visible,
            float(target_angle_norm),
            target_distance_norm,
            float(np.cos(self.agent_orientation)),
            float(np.sin(self.agent_orientation)),
        ]
        return np.array(values, dtype=np.float32)

    def observation_size(self) -> int:
        return self.ray_count + 5

    def distance_to_target(self) -> float:
        return self._distance(self.agent_position, self.target_position)

    def cast_vision_rays(self) -> np.ndarray:
        distances = [
            self._ray_distance(self.agent_position, angle)
            for angle in self._vision_ray_angles()
        ]
        return np.array(distances, dtype=np.float32)

    def vision_polygon(self) -> np.ndarray:
        points = [self.agent_position.copy()]
        for angle, distance in zip(self._vision_ray_angles(), self.cast_vision_rays()):
            direction = np.array([np.cos(angle), np.sin(angle)], dtype=np.float32)
            points.append(self.agent_position + direction * distance)
        return np.array(points, dtype=np.float32)

    def target_is_visible(self) -> bool:
        relative_angle = self._angle_to(self.target_position)
        if abs(relative_angle) > self.fov_radians * 0.5:
            return False

        distance = self.distance_to_target()
        ray_distance = self._ray_distance(self.agent_position, self.agent_orientation + relative_angle)
        return distance <= ray_distance + self.target_radius

    def _validate_action(self, action: Iterable[int | bool] | np.ndarray) -> np.ndarray:
        action_vector = np.asarray(action, dtype=np.int8)
        if action_vector.shape != (self.ACTION_SIZE,):
            raise ValueError(
                f"Expected action shape {(self.ACTION_SIZE,)}, got {action_vector.shape}."
            )
        if not np.isin(action_vector, [0, 1]).all():
            raise ValueError("Action entries must be 0 or 1.")
        return action_vector

    def _rotate_agent(self, action: np.ndarray) -> None:
        turn = float(action[self.ACTION_TURN_RIGHT] - action[self.ACTION_TURN_LEFT])
        self.agent_orientation = self._wrap_angle(
            self.agent_orientation + turn * self.agent_turn_speed
        )

    def _move_agent(self, action: np.ndarray) -> None:
        direction = np.array(
            [
                float(action[self.ACTION_RIGHT] - action[self.ACTION_LEFT]),
                float(action[self.ACTION_DOWN] - action[self.ACTION_UP]),
            ],
            dtype=np.float32,
        )
        norm = float(np.linalg.norm(direction))
        if norm > 0.0:
            direction /= norm

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
            self._bounce_obstacle_on_walls(obstacle)

        for left_index in range(len(self.obstacles)):
            for right_index in range(left_index + 1, len(self.obstacles)):
                self._bounce_obstacles(self.obstacles[left_index], self.obstacles[right_index])

    def _bounce_obstacle_on_walls(self, obstacle: Obstacle) -> None:
        if obstacle.position[0] <= obstacle.radius:
            obstacle.position[0] = obstacle.radius
            obstacle.velocity[0] = abs(obstacle.velocity[0])
        elif obstacle.position[0] >= self.width - obstacle.radius:
            obstacle.position[0] = self.width - obstacle.radius
            obstacle.velocity[0] = -abs(obstacle.velocity[0])

        if obstacle.position[1] <= obstacle.radius:
            obstacle.position[1] = obstacle.radius
            obstacle.velocity[1] = abs(obstacle.velocity[1])
        elif obstacle.position[1] >= self.height - obstacle.radius:
            obstacle.position[1] = self.height - obstacle.radius
            obstacle.velocity[1] = -abs(obstacle.velocity[1])

    def _bounce_obstacles(self, first: Obstacle, second: Obstacle) -> None:
        delta = second.position - first.position
        distance = float(np.linalg.norm(delta))
        min_distance = first.radius + second.radius
        if distance == 0.0 or distance >= min_distance:
            return

        normal = delta / distance
        overlap = min_distance - distance
        first.position -= normal * (overlap * 0.5)
        second.position += normal * (overlap * 0.5)

        relative_velocity = first.velocity - second.velocity
        speed_along_normal = float(np.dot(relative_velocity, normal))
        if speed_along_normal <= 0.0:
            return

        first.velocity -= speed_along_normal * normal
        second.velocity += speed_along_normal * normal

    def _agent_touches_target(self) -> bool:
        return self._circles_overlap(
            self.agent_position,
            self.agent_radius,
            self.target_position,
            self.target_radius,
        )

    def _agent_touches_obstacle(self) -> bool:
        return any(
            self._circle_intersects_polygon(
                self.agent_position,
                self.agent_radius,
                obstacle.vertices(),
            )
            for obstacle in self.obstacles
        )

    def _sample_obstacle_velocity(self) -> np.ndarray:
        angle = self.rng.uniform(0.0, 2.0 * np.pi)
        speed = self.rng.uniform(self.obstacle_speed * 0.6, self.obstacle_speed)
        return np.array([np.cos(angle) * speed, np.sin(angle) * speed], dtype=np.float32)

    def _sample_free_position(
        self,
        radius: float,
        min_agent_distance: float = 0.0,
    ) -> np.ndarray:
        for _ in range(500):
            position = np.array(
                [
                    self.rng.uniform(radius, self.width - radius),
                    self.rng.uniform(radius, self.height - radius),
                ],
                dtype=np.float32,
            )
            if self._position_is_free(position, radius, min_agent_distance):
                return position
        raise RuntimeError("Could not sample a free position. Try fewer obstacles.")

    def _position_is_free(
        self,
        position: np.ndarray,
        radius: float,
        min_agent_distance: float = 0.0,
    ) -> bool:
        margin = 20.0
        agent_clearance = max(self.agent_radius + radius + margin, min_agent_distance)
        if self._distance(position, self.agent_position) < agent_clearance:
            return False
        if self.target_position.any() and self._circles_overlap(
            position,
            radius + margin,
            self.target_position,
            self.target_radius,
        ):
            return False
        return not any(
            self._circles_overlap(
                position,
                radius + margin,
                obstacle.position,
                obstacle.radius,
            )
            for obstacle in self.obstacles
        )

    def _vision_ray_angles(self) -> np.ndarray:
        offsets = np.linspace(
            -self.fov_radians * 0.5,
            self.fov_radians * 0.5,
            self.ray_count,
            dtype=np.float32,
        )
        return self.agent_orientation + offsets

    def _ray_distance(self, origin: np.ndarray, angle: float) -> float:
        direction = np.array([np.cos(angle), np.sin(angle)], dtype=np.float32)
        distances = [
            self._ray_segment_distance(origin, direction, start, end)
            for start, end in self._world_edges()
        ]

        for obstacle in self.obstacles:
            vertices = obstacle.vertices()
            for start, end in self._polygon_edges(vertices):
                distances.append(self._ray_segment_distance(origin, direction, start, end))

        finite_distances = [distance for distance in distances if distance is not None]
        return float(min(finite_distances)) if finite_distances else self.diagonal

    def _angle_to(self, position: np.ndarray) -> float:
        delta = position - self.agent_position
        angle = float(np.arctan2(delta[1], delta[0]))
        return self._wrap_angle(angle - self.agent_orientation)

    def _info(self) -> dict:
        return {
            "step": self.step_count,
            "status": self.status,
            "distance_to_target": self.distance_to_target(),
            "target_visible": self.target_is_visible(),
        }

    def _world_edges(self) -> list[tuple[np.ndarray, np.ndarray]]:
        top_left = np.array([0.0, 0.0], dtype=np.float32)
        top_right = np.array([self.width, 0.0], dtype=np.float32)
        bottom_right = np.array([self.width, self.height], dtype=np.float32)
        bottom_left = np.array([0.0, self.height], dtype=np.float32)
        return [
            (top_left, top_right),
            (top_right, bottom_right),
            (bottom_right, bottom_left),
            (bottom_left, top_left),
        ]

    @staticmethod
    def _polygon_edges(vertices: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
        return [
            (vertices[index], vertices[(index + 1) % len(vertices)])
            for index in range(len(vertices))
        ]

    @staticmethod
    def _ray_segment_distance(
        origin: np.ndarray,
        direction: np.ndarray,
        start: np.ndarray,
        end: np.ndarray,
    ) -> float | None:
        segment = end - start
        denominator = CircleSeekEnv._cross(direction, segment)
        if abs(denominator) < 1e-8:
            return None

        offset = start - origin
        ray_distance = CircleSeekEnv._cross(offset, segment) / denominator
        segment_fraction = CircleSeekEnv._cross(offset, direction) / denominator
        if ray_distance >= 0.0 and 0.0 <= segment_fraction <= 1.0:
            return float(ray_distance)
        return None

    @staticmethod
    def _circle_intersects_polygon(
        center: np.ndarray,
        radius: float,
        vertices: np.ndarray,
    ) -> bool:
        if CircleSeekEnv._point_in_polygon(center, vertices):
            return True
        return any(
            CircleSeekEnv._point_segment_distance(center, start, end) <= radius
            for start, end in CircleSeekEnv._polygon_edges(vertices)
        )

    @staticmethod
    def _point_in_polygon(point: np.ndarray, vertices: np.ndarray) -> bool:
        inside = False
        j = len(vertices) - 1
        for i in range(len(vertices)):
            yi = vertices[i][1]
            yj = vertices[j][1]
            xi = vertices[i][0]
            xj = vertices[j][0]
            intersects = (yi > point[1]) != (yj > point[1])
            if intersects:
                x_intersection = (xj - xi) * (point[1] - yi) / (yj - yi) + xi
                if point[0] < x_intersection:
                    inside = not inside
            j = i
        return inside

    @staticmethod
    def _point_segment_distance(point: np.ndarray, start: np.ndarray, end: np.ndarray) -> float:
        segment = end - start
        length_squared = float(np.dot(segment, segment))
        if length_squared == 0.0:
            return CircleSeekEnv._distance(point, start)
        projection = np.clip(float(np.dot(point - start, segment) / length_squared), 0.0, 1.0)
        closest = start + projection * segment
        return CircleSeekEnv._distance(point, closest)

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

    @staticmethod
    def _cross(a: np.ndarray, b: np.ndarray) -> float:
        return float(a[0] * b[1] - a[1] * b[0])

    @staticmethod
    def _wrap_angle(angle: float) -> float:
        return float((angle + np.pi) % (2.0 * np.pi) - np.pi)
