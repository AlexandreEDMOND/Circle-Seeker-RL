from __future__ import annotations

import argparse
from pathlib import Path

import pygame

from src.ppo import build_env, load_checkpoint, select_action
from src.renderer import CircleSeekRenderer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Watch a trained PPO checkpoint in pygame.")
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument(
        "--sample",
        action="store_true",
        help="Sample from the policy instead of using deterministic argmax actions.",
    )
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
    model, config, _ = load_checkpoint(args.checkpoint)
    env = build_env(config)
    renderer = CircleSeekRenderer(
        env.env,
        fps=args.fps,
        controls_text="policy: ppo | SPACE: pause | R: reset | ESC: quit",
    )

    episode_idx = 0
    reward = 0.0
    total_reward = 0.0
    paused = False
    done_frames = 0
    running = True
    observation, _ = env.reset(seed=args.seed + episode_idx)

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
                    observation, _ = env.reset(seed=args.seed + episode_idx)
                    reward = 0.0
                    total_reward = 0.0
                    done_frames = 0

        if not paused and env.env.status == "running":
            action = select_action(model, observation, deterministic=not args.sample)
            observation, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            if terminated or truncated:
                done_frames = 0
        elif args.auto_reset and env.env.status != "running":
            done_frames += 1
            if done_frames >= args.reset_delay_frames:
                episode_idx += 1
                observation, _ = env.reset(seed=args.seed + episode_idx)
                reward = 0.0
                total_reward = 0.0
                done_frames = 0

        renderer.render(reward, total_reward)

    renderer.close()


if __name__ == "__main__":
    main()
