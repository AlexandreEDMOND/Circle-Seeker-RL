import numpy as np
import pytest

from src.env import CircleSeekEnv, Obstacle


def action_with(*indices: int) -> np.ndarray:
    action = np.zeros(CircleSeekEnv.ACTION_SIZE, dtype=np.int8)
    for index in indices:
        action[index] = 1
    return action


def test_reset_returns_stable_partial_observation_shape() -> None:
    env = CircleSeekEnv(obstacle_count=4, ray_count=31)

    observation = env.reset(seed=123)

    assert observation.dtype == np.float32
    assert observation.shape == (36,)
    assert env.step_count == 0
    assert env.status == "running"


def test_reset_is_deterministic_for_same_seed() -> None:
    env = CircleSeekEnv(obstacle_count=2)

    observation_a = env.reset(seed=123)
    observation_b = env.reset(seed=123)

    np.testing.assert_array_equal(observation_a, observation_b)


def test_target_spawns_at_minimum_distance_from_agent() -> None:
    env = CircleSeekEnv(obstacle_count=0, min_target_distance=250.0)

    env.reset(seed=123)

    assert env.distance_to_target() >= 250.0


def test_step_allows_simultaneous_movement_and_independent_rotation() -> None:
    env = CircleSeekEnv(obstacle_count=0, approach_bonus=False)
    env.reset(seed=123)
    start_position = env.agent_position.copy()
    start_orientation = env.agent_orientation

    action = action_with(
        CircleSeekEnv.ACTION_UP,
        CircleSeekEnv.ACTION_LEFT,
        CircleSeekEnv.ACTION_TURN_RIGHT,
    )
    observation, reward, terminated, truncated, info = env.step(action)

    assert observation.shape == (36,)
    assert reward == pytest.approx(-0.01)
    assert terminated is False
    assert truncated is False
    assert info["step"] == 1
    assert env.agent_position[0] < start_position[0]
    assert env.agent_position[1] < start_position[1]
    assert np.linalg.norm(env.agent_position - start_position) == pytest.approx(
        env.agent_speed,
        abs=1e-4,
    )
    assert env.agent_orientation > start_orientation


def test_reaching_target_terminates_with_success_reward() -> None:
    env = CircleSeekEnv(obstacle_count=0, approach_bonus=False)
    env.reset(seed=123)
    env.target_position = env.agent_position.copy()

    _, reward, terminated, truncated, info = env.step(CircleSeekEnv.ACTION_NOOP)

    assert reward == 10.0
    assert terminated is True
    assert truncated is False
    assert info["status"] == "success"


def test_touching_polygon_obstacle_terminates_with_collision_penalty() -> None:
    env = CircleSeekEnv(obstacle_count=0, approach_bonus=False)
    env.reset(seed=123)
    env.obstacles = [
        Obstacle(
            position=env.agent_position.copy(),
            velocity=np.zeros(2, dtype=np.float32),
            radius=28.0,
            sides=4,
            angle=np.pi / 4.0,
        )
    ]

    _, reward, terminated, truncated, info = env.step(CircleSeekEnv.ACTION_NOOP)

    assert reward == -10.0
    assert terminated is True
    assert truncated is False
    assert info["status"] == "collision"


def test_polygon_obstacle_blocks_target_visibility() -> None:
    env = CircleSeekEnv(
        width=200,
        height=200,
        obstacle_count=0,
        ray_count=5,
        min_target_distance=40.0,
    )
    env.reset(seed=123)
    env.agent_position = np.array([100.0, 100.0], dtype=np.float32)
    env.agent_orientation = 0.0
    env.target_position = np.array([180.0, 100.0], dtype=np.float32)
    env.obstacles = [
        Obstacle(
            position=np.array([140.0, 100.0], dtype=np.float32),
            velocity=np.zeros(2, dtype=np.float32),
            radius=20.0,
            sides=4,
            angle=np.pi / 4.0,
        )
    ]

    observation = env.get_observation()

    assert env.target_is_visible() is False
    assert observation[env.ray_count] == 0.0
    assert env.cast_vision_rays()[env.ray_count // 2] < env.distance_to_target()


def test_hidden_target_does_not_leak_angle_or_distance_in_observation() -> None:
    env = CircleSeekEnv(
        width=200,
        height=200,
        obstacle_count=0,
        ray_count=5,
        min_target_distance=40.0,
    )
    env.reset(seed=123)
    env.agent_position = np.array([100.0, 100.0], dtype=np.float32)
    env.agent_orientation = 0.0
    env.target_position = np.array([100.0, 180.0], dtype=np.float32)

    observation = env.get_observation()

    assert env.target_is_visible() is False
    assert observation[env.ray_count] == 0.0
    assert observation[env.ray_count + 1] == 0.0
    assert observation[env.ray_count + 2] == 0.0


def test_visible_target_includes_relative_angle_and_distance_in_observation() -> None:
    env = CircleSeekEnv(
        width=200,
        height=200,
        obstacle_count=0,
        ray_count=5,
        min_target_distance=40.0,
    )
    env.reset(seed=123)
    env.agent_position = np.array([100.0, 100.0], dtype=np.float32)
    env.agent_orientation = 0.0
    env.target_position = np.array([180.0, 100.0], dtype=np.float32)

    observation = env.get_observation()

    assert env.target_is_visible() is True
    assert observation[env.ray_count] == 1.0
    assert observation[env.ray_count + 1] == pytest.approx(0.0)
    assert observation[env.ray_count + 2] == pytest.approx(80.0 / env.diagonal)


def test_polygon_obstacles_bounce_off_each_other() -> None:
    env = CircleSeekEnv(obstacle_count=0)
    first = Obstacle(
        position=np.array([100.0, 100.0], dtype=np.float32),
        velocity=np.array([1.0, 0.0], dtype=np.float32),
        radius=30.0,
        sides=4,
        angle=0.0,
    )
    second = Obstacle(
        position=np.array([140.0, 100.0], dtype=np.float32),
        velocity=np.array([-1.0, 0.0], dtype=np.float32),
        radius=30.0,
        sides=4,
        angle=0.0,
    )

    env._bounce_obstacles(first, second)

    assert first.velocity[0] < 0.0
    assert second.velocity[0] > 0.0


def test_max_steps_truncates_episode() -> None:
    env = CircleSeekEnv(obstacle_count=0, max_steps=1, approach_bonus=False)
    env.reset(seed=123)

    _, reward, terminated, truncated, info = env.step(CircleSeekEnv.ACTION_NOOP)

    assert reward == pytest.approx(-0.01)
    assert terminated is False
    assert truncated is True
    assert info["status"] == "timeout"


def test_invalid_action_shape_raises_clear_error() -> None:
    env = CircleSeekEnv()

    with pytest.raises(ValueError, match="Expected action shape"):
        env.step(np.zeros(5, dtype=np.int8))


def test_invalid_action_value_raises_clear_error() -> None:
    env = CircleSeekEnv()

    with pytest.raises(ValueError, match="0 or 1"):
        env.step(np.array([0, 0, 0, 0, 0, 2], dtype=np.int8))
