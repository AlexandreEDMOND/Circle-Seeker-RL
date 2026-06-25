from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.ppo import PPOConfig, train_ppo


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a simple PPO agent.")
    parser.add_argument("--total-timesteps", type=int, default=PPOConfig.total_timesteps)
    parser.add_argument("--rollout-steps", type=int, default=PPOConfig.rollout_steps)
    parser.add_argument("--update-epochs", type=int, default=PPOConfig.update_epochs)
    parser.add_argument("--minibatch-size", type=int, default=PPOConfig.minibatch_size)
    parser.add_argument("--learning-rate", type=float, default=PPOConfig.learning_rate)
    parser.add_argument(
        "--distance-reward-coef",
        type=float,
        default=PPOConfig.distance_reward_coef,
        help="Extra training-only reward for reducing target distance.",
    )
    parser.add_argument(
        "--action-conflict-penalty",
        type=float,
        default=PPOConfig.action_conflict_penalty,
        help="Training-only penalty for each contradictory action pair.",
    )
    parser.add_argument(
        "--target-visible-reward-coef",
        type=float,
        default=PPOConfig.target_visible_reward_coef,
        help="Training-only reward when the target is visible in the vision cone.",
    )
    parser.add_argument(
        "--target-found-reward-coef",
        type=float,
        default=PPOConfig.target_found_reward_coef,
        help="Training-only reward when the target becomes visible after being hidden.",
    )
    parser.add_argument(
        "--no-vision-no-turn-penalty",
        type=float,
        default=PPOConfig.no_vision_no_turn_penalty,
        help="Training-only penalty when the target is hidden and the agent does not turn.",
    )
    parser.add_argument("--hidden-size", type=int, default=PPOConfig.hidden_size)
    parser.add_argument("--seed", type=int, default=PPOConfig.seed)
    parser.add_argument("--obstacle-count", type=int, default=PPOConfig.obstacle_count)
    parser.add_argument("--obstacle-speed", type=float, default=PPOConfig.obstacle_speed)
    parser.add_argument("--obstacle-radius", type=float, default=PPOConfig.obstacle_radius)
    parser.add_argument(
        "--min-target-distance",
        type=float,
        default=PPOConfig.min_target_distance,
        help="Minimum spawn distance between the agent and target.",
    )
    parser.add_argument("--max-steps", type=int, default=PPOConfig.max_steps)
    parser.add_argument(
        "--curriculum",
        action="store_true",
        help=(
            "Train with fixed obstacle observation slots while progressively "
            "increasing obstacle speed and radius."
        ),
    )
    parser.add_argument("--curriculum-stages", type=int, default=PPOConfig.curriculum_stages)
    parser.add_argument(
        "--curriculum-start-obstacle-speed",
        type=float,
        default=PPOConfig.curriculum_start_obstacle_speed,
    )
    parser.add_argument(
        "--curriculum-start-obstacle-radius",
        type=float,
        default=PPOConfig.curriculum_start_obstacle_radius,
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("checkpoints/ppo.pt"),
        help="Where to save the trained checkpoint.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = PPOConfig(
        total_timesteps=args.total_timesteps,
        rollout_steps=args.rollout_steps,
        update_epochs=args.update_epochs,
        minibatch_size=args.minibatch_size,
        learning_rate=args.learning_rate,
        distance_reward_coef=args.distance_reward_coef,
        action_conflict_penalty=args.action_conflict_penalty,
        target_visible_reward_coef=args.target_visible_reward_coef,
        target_found_reward_coef=args.target_found_reward_coef,
        no_vision_no_turn_penalty=args.no_vision_no_turn_penalty,
        hidden_size=args.hidden_size,
        seed=args.seed,
        obstacle_count=args.obstacle_count,
        obstacle_speed=args.obstacle_speed,
        obstacle_radius=args.obstacle_radius,
        min_target_distance=args.min_target_distance,
        max_steps=args.max_steps,
        curriculum=args.curriculum,
        curriculum_stages=args.curriculum_stages,
        curriculum_start_obstacle_speed=args.curriculum_start_obstacle_speed,
        curriculum_start_obstacle_radius=args.curriculum_start_obstacle_radius,
    )
    metrics = train_ppo(config, args.checkpoint)
    print(json.dumps(metrics, indent=2, sort_keys=True))
    print(f"Saved checkpoint: {args.checkpoint}")


if __name__ == "__main__":
    main()
