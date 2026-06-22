import numpy as np
from gymnasium import spaces

from src.gym_env import CircleSeekGymEnv


def test_gym_env_defines_spaces_from_core_environment() -> None:
    env = CircleSeekGymEnv(obstacle_count=4)

    assert isinstance(env.action_space, spaces.Discrete)
    assert env.action_space.n == 5
    assert isinstance(env.observation_space, spaces.Box)
    assert env.observation_space.shape == (21,)
    assert env.observation_space.dtype == np.float32


def test_gym_reset_returns_observation_and_info() -> None:
    env = CircleSeekGymEnv(obstacle_count=0)

    observation, info = env.reset(seed=123)

    assert observation.shape == (5,)
    assert observation.dtype == np.float32
    assert env.observation_space.contains(observation)
    assert info["step"] == 0
    assert info["status"] == "running"


def test_gym_step_returns_standard_tuple() -> None:
    env = CircleSeekGymEnv(obstacle_count=0, approach_bonus=False)
    env.reset(seed=123)

    observation, reward, terminated, truncated, info = env.step(0)

    assert env.observation_space.contains(observation)
    assert reward == -0.01
    assert terminated is False
    assert truncated is False
    assert info["step"] == 1


def test_gym_reset_is_deterministic_for_same_seed() -> None:
    env = CircleSeekGymEnv(obstacle_count=2)

    observation_a, _ = env.reset(seed=123)
    observation_b, _ = env.reset(seed=123)

    np.testing.assert_array_equal(observation_a, observation_b)
