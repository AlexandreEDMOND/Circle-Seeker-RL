from __future__ import annotations

import argparse

import numpy as np
import pygame

from src.evaluate_baselines import POLICIES
from src.gym_env import CircleSeekGymEnv
from src.renderer import CircleSeekRenderer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Watch a baseline policy in pygame.")
    parser.add_argument("--policy", choices=sorted(POLICIES), default="heuristic")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--obstacle-count", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=600)
    parser.add_argument(
        "--auto-reset",
        action="store_true",
        help="Automatically start the next episode after a terminal state.",
    )
    parser.add_argument(
        "--reset-delay-frames",
        type=int,
        default=30,
        help="Frames to keep the terminal state visible before auto-reset.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    env = CircleSeekGymEnv(obstacle_count=args.obstacle_count, max_steps=args.max_steps)
    policy = POLICIES[args.policy]
    rng = np.random.default_rng(args.seed)
    renderer = CircleSeekRenderer(
        env.env,
        fps=args.fps,
        controls_text=(
            f"policy: {args.policy} | SPACE: pause | R: reset | ESC: quit"
        ),
    )

    episode_idx = 0
    reward = 0.0
    total_reward = 0.0
    paused = False
    done_frames = 0
    running = True

    env.reset(seed=args.seed + episode_idx)

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_SPACE:
                    paused = not paused
                elif event.key == pygame.K_r:
                    episode_idx += 1
                    env.reset(seed=args.seed + episode_idx)
                    reward = 0.0
                    total_reward = 0.0
                    done_frames = 0

        if not paused and env.env.status == "running":
            action = policy(env, rng)
            _, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            if terminated or truncated:
                done_frames = 0
        elif args.auto_reset and env.env.status != "running":
            done_frames += 1
            if done_frames >= args.reset_delay_frames:
                episode_idx += 1
                env.reset(seed=args.seed + episode_idx)
                reward = 0.0
                total_reward = 0.0
                done_frames = 0

        renderer.render(reward, total_reward)

    renderer.close()


if __name__ == "__main__":
    main()
