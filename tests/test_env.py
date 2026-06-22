import numpy as np
import pytest

from src.env import CircleSeekEnv


def test_reset_returns_stable_observation_shape() -> None:
    env = CircleSeekEnv(obstacle_count=4)

    observation = env.reset(seed=123)

    assert observation.dtype == np.float32
    assert observation.shape == (21,)
    assert env.step_count == 0
    assert env.status == "running"


def test_reset_is_deterministic_for_same_seed() -> None:
    env = CircleSeekEnv(obstacle_count=2)

    observation_a = env.reset(seed=123)
    observation_b = env.reset(seed=123)

    np.testing.assert_array_equal(observation_a, observation_b)


def test_step_moves_agent_and_returns_gymnasium_like_tuple() -> None:
    env = CircleSeekEnv(obstacle_count=0, approach_bonus=False)
    env.reset(seed=123)
    start_position = env.agent_position.copy()

    observation, reward, terminated, truncated, info = env.step(CircleSeekEnv.ACTION_RIGHT)

    assert observation.shape == (5,)
    assert reward == pytest.approx(-0.01)
    assert terminated is False
    assert truncated is False
    assert info["step"] == 1
    assert env.agent_position[0] > start_position[0]


def test_reaching_target_terminates_with_success_reward() -> None:
    env = CircleSeekEnv(obstacle_count=0, approach_bonus=False)
    env.reset(seed=123)
    env.target_position = env.agent_position.copy()

    _, reward, terminated, truncated, info = env.step(CircleSeekEnv.ACTION_NOOP)

    assert reward == 10.0
    assert terminated is True
    assert truncated is False
    assert info["status"] == "success"


def test_touching_obstacle_terminates_with_collision_penalty() -> None:
    env = CircleSeekEnv(obstacle_count=1, approach_bonus=False)
    env.reset(seed=123)
    env.obstacles[0].position = env.agent_position.copy()

    _, reward, terminated, truncated, info = env.step(CircleSeekEnv.ACTION_NOOP)

    assert reward == -10.0
    assert terminated is True
    assert truncated is False
    assert info["status"] == "collision"


def test_max_steps_truncates_episode() -> None:
    env = CircleSeekEnv(obstacle_count=0, max_steps=1, approach_bonus=False)
    env.reset(seed=123)

    _, reward, terminated, truncated, info = env.step(CircleSeekEnv.ACTION_NOOP)

    assert reward == pytest.approx(-0.01)
    assert terminated is False
    assert truncated is True
    assert info["status"] == "timeout"


def test_invalid_action_raises_clear_error() -> None:
    env = CircleSeekEnv()

    with pytest.raises(ValueError, match="Unknown action"):
        env.step(99)
