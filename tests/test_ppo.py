import torch
import pytest

from src.ppo import (
    ActorCritic,
    PPOConfig,
    RolloutBuffer,
    action_conflict_count,
    action_turns,
    build_env,
    collect_rollout,
    compute_ppo_loss,
    evaluate_checkpoint,
    target_visibility_training_reward,
    train_ppo,
)


def test_actor_critic_forward_returns_policy_logits_and_values() -> None:
    model = ActorCritic(observation_size=36, action_count=6)
    observations = torch.zeros((3, 36), dtype=torch.float32)

    logits, values = model(observations)

    assert logits.shape == (3, 6)
    assert values.shape == (3,)


def test_actor_critic_samples_actions_and_returns_policy_terms() -> None:
    torch.manual_seed(123)
    model = ActorCritic(observation_size=36, action_count=6)
    observations = torch.zeros((4, 36), dtype=torch.float32)

    actions, log_probs, entropy, values = model.get_action_and_value(observations)

    assert actions.shape == (4, 6)
    assert actions.dtype == torch.float32
    assert torch.all(actions >= 0)
    assert torch.all(actions <= 1)
    assert log_probs.shape == (4,)
    assert entropy.shape == (4,)
    assert values.shape == (4,)


def test_actor_critic_evaluates_provided_actions() -> None:
    model = ActorCritic(observation_size=36, action_count=6)
    observations = torch.zeros((2, 36), dtype=torch.float32)
    actions = torch.tensor(
        [
            [1, 0, 0, 1, 0, 0],
            [0, 1, 1, 0, 0, 1],
        ],
        dtype=torch.float32,
    )

    evaluated_actions, log_probs, entropy, values = model.get_action_and_value(
        observations,
        actions,
    )

    torch.testing.assert_close(evaluated_actions, actions)
    assert log_probs.shape == (2,)
    assert entropy.shape == (2,)
    assert values.shape == (2,)


def test_rollout_buffer_stores_transition_tensors() -> None:
    buffer = RolloutBuffer(rollout_length=2, observation_size=3, action_size=6)

    buffer.add(
        observation=torch.tensor([1.0, 2.0, 3.0]),
        action=torch.tensor([1, 0, 0, 1, 0, 0]),
        reward=1.5,
        done=False,
        value=0.25,
        log_prob=-0.75,
    )
    buffer.add(
        observation=torch.tensor([4.0, 5.0, 6.0]),
        action=torch.tensor([0, 1, 1, 0, 0, 1]),
        reward=-1.0,
        done=True,
        value=0.5,
        log_prob=-1.25,
    )

    assert buffer.is_full is True
    assert buffer.observations.shape == (2, 3)
    torch.testing.assert_close(
        buffer.actions,
        torch.tensor(
            [
                [1, 0, 0, 1, 0, 0],
                [0, 1, 1, 0, 0, 1],
            ],
            dtype=torch.float32,
        ),
    )
    torch.testing.assert_close(buffer.rewards, torch.tensor([1.5, -1.0]))
    torch.testing.assert_close(buffer.dones, torch.tensor([0.0, 1.0]))
    torch.testing.assert_close(buffer.values, torch.tensor([0.25, 0.5]))
    torch.testing.assert_close(buffer.log_probs, torch.tensor([-0.75, -1.25]))


def test_rollout_buffer_rejects_overflow() -> None:
    buffer = RolloutBuffer(rollout_length=1, observation_size=2)
    buffer.add(torch.zeros(2), action=0, reward=0.0, done=False, value=0.0, log_prob=0.0)

    with pytest.raises(RuntimeError, match="full"):
        buffer.add(torch.zeros(2), action=0, reward=0.0, done=False, value=0.0, log_prob=0.0)


def test_rollout_buffer_requires_full_rollout_before_gae() -> None:
    buffer = RolloutBuffer(rollout_length=2, observation_size=2)
    buffer.add(torch.zeros(2), action=0, reward=0.0, done=False, value=0.0, log_prob=0.0)

    with pytest.raises(RuntimeError, match="must be full"):
        buffer.compute_returns_and_advantages(last_value=0.0)


def test_rollout_buffer_computes_gae_and_returns_without_terminal_bootstrap() -> None:
    buffer = RolloutBuffer(rollout_length=3, observation_size=2)
    rewards = [1.0, 1.0, 1.0]
    values = [0.5, 0.4, 0.3]
    dones = [False, False, True]

    for reward, value, done in zip(rewards, values, dones):
        buffer.add(
            observation=torch.zeros(2),
            action=0,
            reward=reward,
            done=done,
            value=value,
            log_prob=0.0,
        )

    returns, advantages = buffer.compute_returns_and_advantages(
        last_value=0.9,
        gamma=0.9,
        gae_lambda=0.8,
    )

    torch.testing.assert_close(
        advantages,
        torch.tensor([1.84928, 1.37400, 0.70000]),
        atol=1e-5,
        rtol=0.0,
    )
    torch.testing.assert_close(
        returns,
        torch.tensor([2.34928, 1.77400, 1.00000]),
        atol=1e-5,
        rtol=0.0,
    )


def test_rollout_buffer_computes_gae_with_final_bootstrap() -> None:
    buffer = RolloutBuffer(rollout_length=2, observation_size=2)

    buffer.add(torch.zeros(2), action=0, reward=1.0, done=False, value=0.5, log_prob=0.0)
    buffer.add(torch.zeros(2), action=0, reward=2.0, done=False, value=0.25, log_prob=0.0)

    returns, advantages = buffer.compute_returns_and_advantages(
        last_value=0.75,
        gamma=0.9,
        gae_lambda=1.0,
    )

    torch.testing.assert_close(
        advantages,
        torch.tensor([2.90750, 2.42500]),
        atol=1e-5,
        rtol=0.0,
    )
    torch.testing.assert_close(
        returns,
        torch.tensor([3.40750, 2.67500]),
        atol=1e-5,
        rtol=0.0,
    )


def test_action_conflict_count_counts_opposing_action_pairs() -> None:
    assert action_conflict_count([1, 1]) == 1
    assert action_conflict_count([0, 0, 1, 1, 0, 0]) == 1
    assert action_conflict_count([0, 0, 0, 0, 1, 1]) == 1
    assert action_conflict_count([1, 0, 0, 1, 1, 0]) == 0


def test_action_turns_detects_turn_actions() -> None:
    assert action_turns([0, 0, 0, 0, 1, 0]) is True
    assert action_turns([0, 0, 0, 0, 0, 1]) is True
    assert action_turns([1, 0, 0, 1, 0, 0]) is False


def test_target_visibility_training_reward_shapes_search_behavior() -> None:
    assert target_visibility_training_reward(
        target_visible=True,
        previous_target_visible=False,
        turning=False,
        visible_reward_coef=0.02,
        found_reward_coef=0.2,
        no_vision_no_turn_penalty=0.005,
    ) == pytest.approx(0.22)
    assert target_visibility_training_reward(
        target_visible=True,
        previous_target_visible=True,
        turning=False,
        visible_reward_coef=0.02,
        found_reward_coef=0.2,
        no_vision_no_turn_penalty=0.005,
    ) == pytest.approx(0.02)
    assert target_visibility_training_reward(
        target_visible=False,
        previous_target_visible=False,
        turning=True,
        visible_reward_coef=0.02,
        found_reward_coef=0.2,
        no_vision_no_turn_penalty=0.005,
    ) == pytest.approx(0.0)
    assert target_visibility_training_reward(
        target_visible=False,
        previous_target_visible=False,
        turning=False,
        visible_reward_coef=0.02,
        found_reward_coef=0.2,
        no_vision_no_turn_penalty=0.005,
    ) == pytest.approx(-0.005)


def test_collect_rollout_fills_buffer_and_computes_returns() -> None:
    rollout_length = 8
    observation_size = 12
    model = ActorCritic(observation_size=observation_size, action_count=6)

    buffer, info = collect_rollout(
        model,
        rollout_length=rollout_length,
        seed=123,
        env_kwargs={"obstacle_count": 0, "ray_count": 7},
    )

    assert buffer.is_full is True
    assert info["steps"] == rollout_length
    assert buffer.observations.shape == (rollout_length, observation_size)
    assert buffer.actions.shape == (rollout_length, 6)
    assert buffer.rewards.shape == (rollout_length,)
    assert buffer.dones.shape == (rollout_length,)
    assert buffer.values.shape == (rollout_length,)
    assert buffer.log_probs.shape == (rollout_length,)
    assert buffer.returns.shape == (rollout_length,)
    assert buffer.advantages.shape == (rollout_length,)
    assert "action_conflicts" in info
    assert "action_conflict_steps" in info
    assert "target_visible_steps" in info
    assert "target_found_steps" in info
    assert "turn_steps" in info
    assert "no_vision_no_turn_steps" in info
    assert torch.all(buffer.actions >= 0)
    assert torch.all(buffer.actions <= 1)
    assert torch.isfinite(buffer.returns).all()
    assert torch.isfinite(buffer.advantages).all()
    assert torch.any(buffer.returns != 0.0)
    assert torch.any(buffer.advantages != 0.0)


def test_collect_rollout_resets_when_episode_ends_mid_rollout() -> None:
    rollout_length = 5
    observation_size = 8
    model = ActorCritic(observation_size=observation_size, action_count=6)

    buffer, info = collect_rollout(
        model,
        rollout_length=rollout_length,
        seed=123,
        env_kwargs={
            "obstacle_count": 0,
            "ray_count": 3,
            "max_steps": 1,
            "approach_bonus": False,
        },
    )

    assert buffer.is_full is True
    assert torch.count_nonzero(buffer.dones).item() == rollout_length
    assert info["episodes"] == rollout_length
    assert info["episode_lengths"] == [1] * rollout_length
    assert torch.isfinite(buffer.returns).all()
    assert torch.isfinite(buffer.advantages).all()


def test_build_env_passes_min_target_distance_from_config() -> None:
    env = build_env(PPOConfig(obstacle_count=0, min_target_distance=80.0))

    assert env.env.min_target_distance == 80.0


def test_compute_ppo_loss_returns_scalar_diagnostics() -> None:
    model = ActorCritic(observation_size=3, action_count=6)
    observations = torch.zeros((4, 3), dtype=torch.float32)
    actions = torch.tensor(
        [
            [1, 0, 0, 1, 0, 0],
            [0, 1, 1, 0, 0, 1],
            [1, 1, 0, 0, 1, 0],
            [0, 0, 1, 1, 0, 1],
        ],
        dtype=torch.float32,
    )
    with torch.no_grad():
        _, old_log_probs, _, _ = model.get_action_and_value(observations, actions)
    advantages = torch.tensor([1.0, -0.5, 0.25, -1.0])
    returns = torch.tensor([1.0, 0.5, -0.25, 0.0])

    loss_terms = compute_ppo_loss(
        model,
        observations,
        actions,
        old_log_probs,
        advantages,
        returns,
    )

    assert set(loss_terms) == {
        "loss",
        "policy_loss",
        "value_loss",
        "entropy",
        "approx_kl",
        "clip_fraction",
    }
    for value in loss_terms.values():
        assert value.shape == ()
        assert torch.isfinite(value)
    assert loss_terms["loss"].requires_grad is True


def test_train_ppo_saves_checkpoint_and_evaluates(tmp_path) -> None:
    checkpoint = tmp_path / "ppo.pt"
    config = PPOConfig(
        total_timesteps=16,
        rollout_steps=8,
        update_epochs=1,
        minibatch_size=4,
        hidden_size=16,
        seed=123,
        obstacle_count=0,
        max_steps=5,
    )

    train_metrics = train_ppo(config, checkpoint)
    eval_metrics = evaluate_checkpoint(checkpoint, episodes=2, seed=456)

    assert checkpoint.exists()
    assert train_metrics["timesteps"] == 16
    assert "policy_loss" in train_metrics
    assert "value_loss" in train_metrics
    assert "entropy" in train_metrics
    assert "approx_kl" in train_metrics
    assert "clip_fraction" in train_metrics
    assert "mean_initial_distance" in train_metrics
    assert "mean_final_distance" in train_metrics
    assert "mean_distance_delta" in train_metrics
    assert "mean_training_return" in train_metrics
    assert "action_conflict_rate" in train_metrics
    assert "mean_action_conflicts" in train_metrics
    assert "target_visible_rate" in train_metrics
    assert "target_found_rate" in train_metrics
    assert "turn_action_rate" in train_metrics
    assert "no_vision_no_turn_rate" in train_metrics
    assert eval_metrics["episodes"] == 2
    assert (
        eval_metrics["successes"]
        + eval_metrics["collisions"]
        + eval_metrics["timeouts"]
        == 2
    )
