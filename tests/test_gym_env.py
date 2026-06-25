import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import numpy as np
from gymnasium import spaces

from src.env import CircleSeekEnv
from src.gym_env import CircleSeekGymEnv


def test_gym_env_defines_spaces_from_core_environment() -> None:
    env = CircleSeekGymEnv(obstacle_count=4)

    assert isinstance(env.action_space, spaces.MultiBinary)
    assert env.action_space.n == CircleSeekEnv.ACTION_SIZE
    assert isinstance(env.observation_space, spaces.Box)
    assert env.observation_space.shape == (36,)
    assert env.observation_space.dtype == np.float32
    assert np.isfinite(env.observation_space.low).all()
    assert np.isfinite(env.observation_space.high).all()
    np.testing.assert_array_equal(env.observation_space.low[: env.env.ray_count], 0.0)
    np.testing.assert_array_equal(env.observation_space.high[: env.env.ray_count], 1.0)
    np.testing.assert_array_equal(
        env.observation_space.low[env.env.ray_count :],
        np.array([0.0, -1.0, 0.0, -1.0, -1.0], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        env.observation_space.high[env.env.ray_count :],
        np.array([1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32),
    )


def test_gym_reset_returns_observation_and_info() -> None:
    env = CircleSeekGymEnv(obstacle_count=0, ray_count=7)

    observation, info = env.reset(seed=123)

    assert observation.shape == (12,)
    assert observation.dtype == np.float32
    assert env.observation_space.contains(observation)
    assert info["step"] == 0
    assert info["status"] == "running"


def test_gym_step_returns_standard_tuple() -> None:
    env = CircleSeekGymEnv(obstacle_count=0, approach_bonus=False)
    env.reset(seed=123)

    observation, reward, terminated, truncated, info = env.step(CircleSeekEnv.ACTION_NOOP)

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


def test_gym_rgb_array_render_returns_frame() -> None:
    env = CircleSeekGymEnv(
        render_mode="rgb_array",
        width=160,
        height=120,
        obstacle_count=0,
        ray_count=7,
        min_target_distance=40.0,
    )
    env.reset(seed=123)

    frame = env.render()

    assert frame is not None
    assert frame.shape == (120, 160, 3)
    assert frame.dtype == np.uint8
    assert frame.max() > 0
    env.close()


def test_gym_rejects_unknown_render_mode() -> None:
    try:
        CircleSeekGymEnv(render_mode="ansi")
    except ValueError as exc:
        assert "Unsupported render_mode" in str(exc)
    else:
        raise AssertionError("Expected unsupported render_mode to raise ValueError.")
