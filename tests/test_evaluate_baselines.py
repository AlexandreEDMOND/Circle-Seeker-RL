from src.evaluate_baselines import (
    evaluate_baselines,
    evaluate_policy,
    random_policy,
    target_seeking_policy,
)


def test_evaluate_random_policy_returns_expected_metrics() -> None:
    metrics = evaluate_policy(
        random_policy,
        episodes=3,
        seed=123,
        env_kwargs={"obstacle_count": 0, "max_steps": 5},
    )

    assert metrics["episodes"] == 3
    assert metrics["successes"] + metrics["collisions"] + metrics["timeouts"] == 3
    assert 0.0 <= metrics["success_rate"] <= 1.0
    assert 0.0 <= metrics["collision_rate"] <= 1.0
    assert 0.0 <= metrics["timeout_rate"] <= 1.0
    assert metrics["mean_episode_length"] > 0.0


def test_evaluate_baselines_runs_random_and_heuristic() -> None:
    results = evaluate_baselines(
        policies=["random", "heuristic"],
        episodes=2,
        seed=123,
        env_kwargs={"obstacle_count": 0, "max_steps": 10},
    )

    assert set(results) == {"random", "heuristic"}
    assert results["random"]["episodes"] == 2
    assert results["heuristic"]["episodes"] == 2


def test_target_seeking_policy_moves_toward_largest_axis_delta() -> None:
    metrics = evaluate_policy(
        target_seeking_policy,
        episodes=1,
        seed=123,
        env_kwargs={"obstacle_count": 0, "max_steps": 100},
    )

    assert metrics["episodes"] == 1
    assert metrics["successes"] == 1
