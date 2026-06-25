import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

from src.env import CircleSeekEnv
from src.renderer import CircleSeekRenderer


def test_renderer_draws_one_frame_without_crashing() -> None:
    env = CircleSeekEnv(obstacle_count=1)
    env.reset(seed=123)
    renderer = CircleSeekRenderer(env)

    try:
        renderer.render(reward=0.0, total_reward=0.0)
    finally:
        renderer.close()
