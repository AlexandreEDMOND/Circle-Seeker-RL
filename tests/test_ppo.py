from pathlib import Path

from src.ppo import PPOConfig, evaluate_checkpoint, train_ppo


def test_train_ppo_saves_checkpoint_and_evaluates(tmp_path: Path) -> None:
    checkpoint = tmp_path / "ppo.pt"
    config = PPOConfig(
        total_timesteps=32,
        rollout_steps=16,
        update_epochs=1,
        minibatch_size=8,
        hidden_size=16,
        seed=123,
        obstacle_count=0,
        max_steps=20,
    )

    train_metrics = train_ppo(config, checkpoint)
    eval_metrics = evaluate_checkpoint(checkpoint, episodes=2, seed=456)

    assert checkpoint.exists()
    assert train_metrics["timesteps"] == 32
    assert eval_metrics["episodes"] == 2
    assert (
        eval_metrics["successes"]
        + eval_metrics["collisions"]
        + eval_metrics["timeouts"]
        == 2
    )
    assert eval_metrics["mean_moving_steps"] >= 0.0


def test_train_ppo_supports_curriculum(tmp_path: Path) -> None:
    checkpoint = tmp_path / "ppo_curriculum.pt"
    config = PPOConfig(
        total_timesteps=32,
        rollout_steps=16,
        update_epochs=1,
        minibatch_size=8,
        hidden_size=16,
        seed=123,
        obstacle_count=2,
        max_steps=20,
        curriculum=True,
        curriculum_stages=2,
    )

    train_metrics = train_ppo(config, checkpoint)
    eval_metrics = evaluate_checkpoint(checkpoint, episodes=2, seed=456)

    assert checkpoint.exists()
    assert train_metrics["curriculum"]["enabled"] is True
    assert train_metrics["curriculum"]["stages"] == 2
    assert len(train_metrics["curriculum"]["stage_counts"]) == 2
    assert eval_metrics["episodes"] == 2
