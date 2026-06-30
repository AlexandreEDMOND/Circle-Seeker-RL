from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.ppo import evaluate_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained PPO checkpoint.")
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument(
        "--sample",
        action="store_true",
        help="Sample from the policy instead of using deterministic argmax actions.",
    )
    parser.add_argument(
        "--include-training-metrics",
        action="store_true",
        help="Include the full checkpoint training history in the JSON output.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics = evaluate_checkpoint(
        args.checkpoint,
        episodes=args.episodes,
        seed=args.seed,
        deterministic=not args.sample,
        include_training_metrics=args.include_training_metrics,
    )
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
