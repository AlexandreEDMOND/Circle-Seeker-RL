from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pygame

from src.evaluate_baselines import random_policy, target_seeking_policy
from src.gym_env import CircleSeekGymEnv
from src.ppo import build_env, load_checkpoint, select_action


BG = (18, 22, 28)
BORDER = (215, 220, 230)
TEXT = (235, 238, 245)
TARGET = (65, 205, 95)
OBSTACLE = (220, 70, 70)
AGENT = (90, 165, 255)
RANDOM_PATH = (255, 196, 70)
PPO_PATH = (90, 165, 255)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate README screenshots and GIFs.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("docs/media"))
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--gif-frames", type=int, default=140)
    parser.add_argument("--fps", type=int, default=24)
    return parser.parse_args()


def draw_env(
    surface: pygame.Surface,
    env: CircleSeekGymEnv,
    *,
    reward: float = 0.0,
    total_reward: float = 0.0,
    label: str = "",
) -> None:
    core = env.env
    surface.fill(BG)
    pygame.draw.rect(surface, BORDER, pygame.Rect(0, 0, core.width, core.height), width=2)
    draw_vision(surface, env, offset_x=0)
    pygame.draw.circle(surface, TARGET, core.target_position.astype(int), int(core.target_radius))

    for obstacle in core.obstacles:
        pygame.draw.polygon(surface, OBSTACLE, obstacle.vertices().astype(int))

    pygame.draw.circle(surface, AGENT, core.agent_position.astype(int), int(core.agent_radius))
    draw_heading(surface, env, offset_x=0)
    draw_hud(surface, env, reward=reward, total_reward=total_reward, label=label)


def draw_vision(surface: pygame.Surface, env: CircleSeekGymEnv, *, offset_x: int) -> None:
    polygon = env.env.vision_polygon()
    if len(polygon) < 3:
        return
    shifted = polygon.copy()
    shifted[:, 0] += offset_x
    overlay = pygame.Surface(surface.get_size(), pygame.SRCALPHA)
    pygame.draw.polygon(overlay, (90, 165, 255, 45), shifted.astype(int))
    pygame.draw.lines(overlay, (130, 190, 255, 120), False, shifted.astype(int), width=1)
    surface.blit(overlay, (0, 0))


def draw_heading(surface: pygame.Surface, env: CircleSeekGymEnv, *, offset_x: int) -> None:
    core = env.env
    start = core.agent_position.copy()
    start[0] += offset_x
    heading = start + core.agent_radius * 1.8 * np.array(
        [np.cos(core.agent_orientation), np.sin(core.agent_orientation)],
        dtype=np.float32,
    )
    pygame.draw.line(surface, TEXT, start.astype(int), heading.astype(int), width=3)


def draw_hud(
    surface: pygame.Surface,
    env: CircleSeekGymEnv,
    *,
    reward: float,
    total_reward: float,
    label: str,
) -> None:
    font = pygame.font.SysFont("Arial", 20)
    lines = [
        label,
        f"step: {env.env.step_count}/{env.env.max_steps}",
        f"reward: {reward:.3f}",
        f"total reward: {total_reward:.3f}",
        f"distance: {env.env.distance_to_target():.1f}",
        f"status: {env.env.status}",
    ]
    y = 10
    for line in [line for line in lines if line]:
        rendered = font.render(line, True, TEXT)
        surface.blit(rendered, (12, y))
        y += 24


def save_environment_png(output_dir: Path, seed: int) -> None:
    env = CircleSeekGymEnv(obstacle_count=4, max_steps=600)
    env.reset(seed=seed)
    surface = pygame.Surface((env.env.width, env.env.height))
    draw_env(surface, env, label="Circle Seeker RL environment")
    pygame.image.save(surface, output_dir / "environment.png")


def save_heuristic_gif(output_dir: Path, seed: int, frame_count: int, fps: int) -> None:
    rng = np.random.default_rng(seed)
    env = CircleSeekGymEnv(obstacle_count=4, max_steps=600)
    env.reset(seed=seed)
    surface = pygame.Surface((env.env.width, env.env.height))

    with tempfile.TemporaryDirectory() as tmp:
        frames_dir = Path(tmp)
        total_reward = 0.0
        reward = 0.0
        for frame_idx in range(frame_count):
            if env.env.status == "running":
                action = target_seeking_policy(env, rng)
                _, reward, terminated, truncated, _ = env.step(action)
                total_reward += reward
                if terminated or truncated:
                    draw_env(
                        surface,
                        env,
                        reward=reward,
                        total_reward=total_reward,
                        label="target-seeking baseline",
                    )
                else:
                    draw_env(
                        surface,
                        env,
                        reward=reward,
                        total_reward=total_reward,
                        label="target-seeking baseline",
                    )
            else:
                draw_env(
                    surface,
                    env,
                    reward=reward,
                    total_reward=total_reward,
                    label="target-seeking baseline",
                )
            pygame.image.save(surface, frames_dir / f"frame_{frame_idx:04d}.png")

        encode_gif(frames_dir, output_dir / "agent_movement.gif", fps)


def save_ppo_gif(
    output_dir: Path,
    checkpoint: Path,
    seed: int,
    frame_count: int,
    fps: int,
) -> None:
    model, config, _ = load_checkpoint(checkpoint)
    env = build_env(config)
    observation, _ = env.reset(seed=seed)
    surface = pygame.Surface((env.env.width, env.env.height))

    with tempfile.TemporaryDirectory() as tmp:
        frames_dir = Path(tmp)
        total_reward = 0.0
        reward = 0.0
        for frame_idx in range(frame_count):
            if env.env.status == "running":
                action = select_action(model, observation, deterministic=True)
                observation, reward, terminated, truncated, _ = env.step(action)
                total_reward += reward
                if terminated or truncated:
                    draw_env(
                        surface,
                        env,
                        reward=reward,
                        total_reward=total_reward,
                        label="trained PPO policy",
                    )
                else:
                    draw_env(
                        surface,
                        env,
                        reward=reward,
                        total_reward=total_reward,
                        label="trained PPO policy",
                    )
            else:
                draw_env(
                    surface,
                    env,
                    reward=reward,
                    total_reward=total_reward,
                    label="trained PPO policy",
                )
            pygame.image.save(surface, frames_dir / f"frame_{frame_idx:04d}.png")

        encode_gif(frames_dir, output_dir / "ppo_trained.gif", fps)


def encode_gif(frames_dir: Path, output_path: Path, fps: int) -> None:
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg is required to generate animated GIFs.")

    palette = frames_dir / "palette.png"
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-framerate",
            str(fps),
            "-i",
            str(frames_dir / "frame_%04d.png"),
            "-vf",
            "fps=12,scale=640:-1:flags=lanczos,palettegen",
            str(palette),
        ],
        check=True,
    )
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-framerate",
            str(fps),
            "-i",
            str(frames_dir / "frame_%04d.png"),
            "-i",
            str(palette),
            "-lavfi",
            "fps=12,scale=640:-1:flags=lanczos[x];[x][1:v]paletteuse",
            str(output_path),
        ],
        check=True,
    )


def rollout_positions(
    env: CircleSeekGymEnv,
    *,
    seed: int,
    policy: str,
    checkpoint: Path | None = None,
    max_steps: int = 220,
) -> tuple[list[tuple[int, int]], np.ndarray]:
    observation, _ = env.reset(seed=seed)
    target_position = env.env.target_position.copy()
    positions = [tuple(env.env.agent_position.astype(int))]
    rng = np.random.default_rng(seed)
    model = None
    if policy == "ppo":
        if checkpoint is None:
            raise ValueError("checkpoint is required for PPO rollout")
        model, _, _ = load_checkpoint(checkpoint)

    for _ in range(max_steps):
        if env.env.status != "running":
            break
        if policy == "random":
            action = random_policy(env, rng)
        elif policy == "ppo":
            if model is None:
                raise RuntimeError("PPO model was not loaded")
            action = select_action(model, observation, deterministic=True)
        else:
            raise ValueError(f"Unknown policy {policy}")
        observation, _, _, _, _ = env.step(action)
        positions.append(tuple(env.env.agent_position.astype(int)))

    return positions, target_position


def save_trajectory_comparison(output_dir: Path, checkpoint: Path, seed: int) -> None:
    _, config, _ = load_checkpoint(checkpoint)
    random_env = build_env(config)
    ppo_env = build_env(config)
    random_path, target_position = rollout_positions(
        random_env,
        seed=seed,
        policy="random",
        max_steps=320,
    )
    ppo_path, _ = rollout_positions(
        ppo_env,
        seed=seed,
        policy="ppo",
        checkpoint=checkpoint,
        max_steps=320,
    )

    width = random_env.env.width
    height = random_env.env.height
    panel_gap = 24
    surface = pygame.Surface((width * 2 + panel_gap, height))
    surface.fill((10, 12, 16))
    font = pygame.font.SysFont("Arial", 24)

    draw_trajectory_panel(
        surface,
        offset_x=0,
        env=random_env,
        path=random_path,
        path_color=RANDOM_PATH,
        target_position=target_position,
        title="Random policy trajectory",
        font=font,
    )
    draw_trajectory_panel(
        surface,
        offset_x=width + panel_gap,
        env=ppo_env,
        path=ppo_path,
        path_color=PPO_PATH,
        target_position=target_position,
        title="Trained PPO trajectory",
        font=font,
    )
    pygame.image.save(surface, output_dir / "trajectory_comparison.png")


def draw_trajectory_panel(
    surface: pygame.Surface,
    *,
    offset_x: int,
    env: CircleSeekGymEnv,
    path: list[tuple[int, int]],
    path_color: tuple[int, int, int],
    target_position: np.ndarray,
    title: str,
    font: pygame.font.Font,
) -> None:
    rect = pygame.Rect(offset_x, 0, env.env.width, env.env.height)
    pygame.draw.rect(surface, BG, rect)
    pygame.draw.rect(surface, BORDER, rect, width=2)
    target = (int(target_position[0]) + offset_x, int(target_position[1]))
    pygame.draw.circle(surface, TARGET, target, int(env.env.target_radius))

    for obstacle in env.env.obstacles:
        vertices = obstacle.vertices().copy()
        vertices[:, 0] += offset_x
        pygame.draw.polygon(surface, OBSTACLE, vertices.astype(int))

    shifted_path = [(x + offset_x, y) for x, y in path]
    if len(shifted_path) > 1:
        pygame.draw.lines(surface, path_color, False, shifted_path, width=4)

    start = shifted_path[0]
    end = shifted_path[-1]
    pygame.draw.circle(surface, (245, 245, 245), start, int(env.env.agent_radius), width=3)
    pygame.draw.circle(surface, path_color, end, int(env.env.agent_radius))

    label = font.render(title, True, TEXT)
    surface.blit(label, (offset_x + 12, 12))


def main() -> None:
    args = parse_args()
    pygame.init()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_environment_png(args.output_dir, args.seed)
    save_heuristic_gif(args.output_dir, args.seed, args.gif_frames, args.fps)
    save_ppo_gif(args.output_dir, args.checkpoint, args.seed, args.gif_frames, args.fps)
    save_trajectory_comparison(args.output_dir, args.checkpoint, args.seed)
    pygame.quit()


if __name__ == "__main__":
    main()
