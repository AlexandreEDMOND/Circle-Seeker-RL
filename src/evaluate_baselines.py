from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Callable

import numpy as np

from src.env import CircleSeekEnv
from src.gym_env import CircleSeekGymEnv


PolicyFn = Callable[[CircleSeekGymEnv, np.random.Generator], np.ndarray]


def random_policy(env: CircleSeekGymEnv, rng: np.random.Generator) -> np.ndarray:
    return rng.integers(0, 2, size=CircleSeekEnv.ACTION_SIZE, dtype=np.int8)


def target_seeking_policy(env: CircleSeekGymEnv, rng: np.random.Generator) -> np.ndarray:
    action = np.zeros(CircleSeekEnv.ACTION_SIZE, dtype=np.int8)
    delta = env.env.target_position - env.env.agent_position
    if abs(delta[0]) > 1.0:
        action[CircleSeekEnv.ACTION_RIGHT if delta[0] > 0 else CircleSeekEnv.ACTION_LEFT] = 1
    if abs(delta[1]) > 1.0:
        action[CircleSeekEnv.ACTION_DOWN if delta[1] > 0 else CircleSeekEnv.ACTION_UP] = 1

    angle_to_target = env.env._angle_to(env.env.target_position)
    if angle_to_target > 0.05:
        action[CircleSeekEnv.ACTION_TURN_RIGHT] = 1
    elif angle_to_target < -0.05:
        action[CircleSeekEnv.ACTION_TURN_LEFT] = 1
    return action


POLICIES: dict[str, PolicyFn] = {
    "random": random_policy,
    "heuristic": target_seeking_policy,
}


def evaluate_policy(
    policy: PolicyFn,
    *,
    episodes: int = 100,
    seed: int = 123,
    env_kwargs: dict[str, Any] | None = None,
) -> dict[str, float | int]:
    env = CircleSeekGymEnv(**(env_kwargs or {}))
    rng = np.random.default_rng(seed)

    returns: list[float] = []
    lengths: list[int] = []
    status_counts = {"success": 0, "collision": 0, "timeout": 0}

    for episode_idx in range(episodes):
        env.reset(seed=seed + episode_idx)
        total_reward = 0.0
        length = 0

        while True:
            action = policy(env, rng)
            _, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            length += 1

            if terminated or truncated:
                status = str(info["status"])
                status_counts[status] = status_counts.get(status, 0) + 1
                returns.append(total_reward)
                lengths.append(length)
                break

    return {
        "episodes": episodes,
        "success_rate": status_counts["success"] / episodes,
        "collision_rate": status_counts["collision"] / episodes,
        "timeout_rate": status_counts["timeout"] / episodes,
        "mean_return": float(np.mean(returns)),
        "mean_episode_length": float(np.mean(lengths)),
        "successes": status_counts["success"],
        "collisions": status_counts["collision"],
        "timeouts": status_counts["timeout"],
    }


def evaluate_baselines(
    *,
    policies: list[str],
    episodes: int,
    seed: int,
    env_kwargs: dict[str, Any] | None = None,
) -> dict[str, dict[str, float | int]]:
    return {
        policy_name: evaluate_policy(
            POLICIES[policy_name],
            episodes=episodes,
            seed=seed,
            env_kwargs=env_kwargs,
        )
        for policy_name in policies
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate non-learning baseline policies.")
    parser.add_argument(
        "--policy",
        choices=["random", "heuristic", "both"],
        default="both",
        help="Baseline policy to evaluate.",
    )
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--obstacle-count", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=600)
    parser.add_argument("--min-target-distance", type=float, default=200.0)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    policies = ["random", "heuristic"] if args.policy == "both" else [args.policy]
    results = evaluate_baselines(
        policies=policies,
        episodes=args.episodes,
        seed=args.seed,
        env_kwargs={
            "obstacle_count": args.obstacle_count,
            "max_steps": args.max_steps,
            "min_target_distance": args.min_target_distance,
        },
    )

    output = json.dumps(results, indent=2, sort_keys=True)
    print(output)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
