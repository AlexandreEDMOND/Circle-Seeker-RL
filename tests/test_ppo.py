import torch
import pytest
import src.ppo as ppo_module

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
    nearest_obstacle_clearance,
    obstacle_proximity_penalty,
    select_action,
    should_stop_for_kl,
    structured_action_to_multibinary,
    target_visibility_training_reward,
    train_ppo,
)


def test_structured_action_to_multibinary_maps_valid_actions() -> None:
    assert structured_action_to_multibinary([0, 0]).tolist() == [0, 0, 0, 0, 0, 0]
    assert structured_action_to_multibinary([1, 1]).tolist() == [1, 0, 0, 0, 1, 0]
    assert structured_action_to_multibinary([6, 2]).tolist() == [1, 0, 0, 1, 0, 1]
    assert structured_action_to_multibinary([8, 0]).tolist() == [0, 1, 0, 1, 0, 0]


def test_structured_action_to_multibinary_rejects_invalid_actions() -> None:
    with pytest.raises(ValueError, match="movement action index"):
        structured_action_to_multibinary([9, 0])
    with pytest.raises(ValueError, match="turn action index"):
        structured_action_to_multibinary([0, 3])


def test_actor_critic_forward_returns_policy_logits_and_values() -> None:
    model = ActorCritic(observation_size=36, action_count=6)
    observations = torch.zeros((3, 36), dtype=torch.float32)

    (movement_logits, turn_logits), values = model(observations)

    assert movement_logits.shape == (3, 9)
    assert turn_logits.shape == (3, 3)
    assert values.shape == (3,)


def test_actor_critic_samples_actions_and_returns_policy_terms() -> None:
    torch.manual_seed(123)
    model = ActorCritic(observation_size=36, action_count=6)
    observations = torch.zeros((4, 36), dtype=torch.float32)

    actions, log_probs, entropy, values = model.get_action_and_value(observations)

    assert actions.shape == (4, 2)
    assert actions.dtype == torch.int64
    assert torch.all(actions[:, 0] >= 0)
    assert torch.all(actions[:, 0] < 9)
    assert torch.all(actions[:, 1] >= 0)
    assert torch.all(actions[:, 1] < 3)
    assert log_probs.shape == (4,)
    assert entropy.shape == (4,)
    assert values.shape == (4,)


def test_actor_critic_evaluates_provided_actions() -> None:
    model = ActorCritic(observation_size=36, action_count=6)
    observations = torch.zeros((2, 36), dtype=torch.float32)
    actions = torch.tensor(
        [
            [1, 0],
            [8, 2],
        ],
        dtype=torch.long,
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
    buffer = RolloutBuffer(rollout_length=2, observation_size=3, action_size=2)

    buffer.add(
        observation=torch.tensor([1.0, 2.0, 3.0]),
        action=torch.tensor([1, 0]),
        reward=1.5,
        done=False,
        value=0.25,
        log_prob=-0.75,
    )
    buffer.add(
        observation=torch.tensor([4.0, 5.0, 6.0]),
        action=torch.tensor([8, 2]),
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
                [1, 0],
                [8, 2],
            ],
            dtype=torch.long,
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


def test_obstacle_proximity_penalty_scales_with_clearance() -> None:
    assert obstacle_proximity_penalty(
        None,
        threshold=60.0,
        penalty_coef=0.05,
    ) == pytest.approx(0.0)
    assert obstacle_proximity_penalty(
        90.0,
        threshold=60.0,
        penalty_coef=0.05,
    ) == pytest.approx(0.0)
    assert obstacle_proximity_penalty(
        30.0,
        threshold=60.0,
        penalty_coef=0.05,
    ) == pytest.approx(0.025)
    assert obstacle_proximity_penalty(
        -5.0,
        threshold=60.0,
        penalty_coef=0.05,
    ) == pytest.approx(0.05)


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
    assert buffer.actions.shape == (rollout_length, 2)
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
    assert "near_obstacle_steps" in info
    assert "obstacle_proximity_penalty_sum" in info
    assert "mean_nearest_obstacle_clearance" in info
    assert "min_nearest_obstacle_clearance" in info
    assert torch.all(buffer.actions[:, 0] >= 0)
    assert torch.all(buffer.actions[:, 0] < 9)
    assert torch.all(buffer.actions[:, 1] >= 0)
    assert torch.all(buffer.actions[:, 1] < 3)
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


def test_select_action_deterministic_returns_non_conflicting_multibinary_action() -> None:
    model = ActorCritic(observation_size=36, action_count=6)
    observation = torch.zeros(36, dtype=torch.float32).numpy()

    action = select_action(model, observation, deterministic=True)

    assert action.shape == (6,)
    assert set(action.tolist()) <= {0, 1}
    assert action_conflict_count(action) == 0


def test_build_env_passes_min_target_distance_from_config() -> None:
    env = build_env(PPOConfig(obstacle_count=0, min_target_distance=80.0))

    assert env.env.min_target_distance == 80.0
    assert nearest_obstacle_clearance(env) is None


def test_compute_ppo_loss_returns_scalar_diagnostics() -> None:
    model = ActorCritic(observation_size=3, action_count=6)
    observations = torch.zeros((4, 3), dtype=torch.float32)
    actions = torch.tensor(
        [
            [1, 0],
            [8, 2],
            [5, 1],
            [0, 0],
        ],
        dtype=torch.long,
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


def test_should_stop_for_kl_uses_optional_threshold() -> None:
    assert should_stop_for_kl(0.02, None) is False
    assert should_stop_for_kl(0.02, 0.03) is False
    assert should_stop_for_kl(0.02, 0.02) is False
    assert should_stop_for_kl(0.02, 0.01) is True


def test_train_ppo_stops_update_early_when_target_kl_is_exceeded(
    tmp_path,
    monkeypatch,
) -> None:
    calls = 0

    def high_kl_loss(*args, **kwargs):
        nonlocal calls
        calls += 1
        model = args[0]
        loss = next(model.parameters()).sum() * 0.0
        return {
            "loss": loss,
            "policy_loss": torch.tensor(0.0),
            "value_loss": torch.tensor(0.0),
            "entropy": torch.tensor(0.0),
            "approx_kl": torch.tensor(0.02),
            "clip_fraction": torch.tensor(0.0),
        }

    monkeypatch.setattr(ppo_module, "compute_ppo_loss", high_kl_loss)

    metrics = train_ppo(
        PPOConfig(
            total_timesteps=8,
            rollout_steps=8,
            update_epochs=5,
            minibatch_size=4,
            hidden_size=16,
            seed=123,
            obstacle_count=0,
            max_steps=5,
            target_kl=0.01,
        ),
        tmp_path / "ppo.pt",
    )

    assert calls == 1
    assert metrics["target_kl"] == 0.01
    assert metrics["kl_early_stops"] == 1
    assert metrics["updates"][0]["kl_early_stopped"] is True
    assert metrics["updates"][0]["completed_update_epochs"] == 1
    assert metrics["updates"][0]["optimized_minibatches"] == 1


def test_train_ppo_saves_checkpoint_and_evaluates(tmp_path) -> None:
    checkpoint = tmp_path / "ppo.pt"
    best_checkpoint = tmp_path / "ppo_best.pt"
    config = PPOConfig(
        total_timesteps=16,
        rollout_steps=8,
        update_epochs=1,
        minibatch_size=4,
        hidden_size=16,
        seed=123,
        obstacle_count=0,
        max_steps=5,
        eval_every=8,
        eval_episodes=1,
        eval_seed=456,
    )

    train_metrics = train_ppo(config, checkpoint, best_checkpoint_path=best_checkpoint)
    eval_metrics = evaluate_checkpoint(checkpoint, episodes=2, seed=456)
    full_eval_metrics = evaluate_checkpoint(
        checkpoint,
        episodes=2,
        seed=456,
        include_training_metrics=True,
    )
    checkpoint_payload = torch.load(checkpoint, map_location="cpu", weights_only=False)

    assert checkpoint.exists()
    assert best_checkpoint.exists()
    assert checkpoint_payload["action_format"] == "structured-categorical-v1"
    assert checkpoint_payload["movement_action_count"] == 9
    assert checkpoint_payload["turn_action_count"] == 3
    assert train_metrics["timesteps"] == 16
    assert "policy_loss" in train_metrics
    assert "value_loss" in train_metrics
    assert "entropy" in train_metrics
    assert "approx_kl" in train_metrics
    assert "clip_fraction" in train_metrics
    assert "target_kl" in train_metrics
    assert "kl_early_stops" in train_metrics
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
    assert "near_obstacle_rate" in train_metrics
    assert "mean_obstacle_proximity_penalty" in train_metrics
    assert "obstacle_proximity_penalty_coef" in train_metrics
    assert "obstacle_proximity_threshold" in train_metrics
    assert "best_eval_score" in train_metrics
    assert "best_eval" in train_metrics
    assert "eval_history" in train_metrics
    assert len(train_metrics["eval_history"]) == 2
    assert len(train_metrics["updates"]) == 2
    assert train_metrics["updates"][0]["global_step"] == 8
    assert "mean_return" in train_metrics["updates"][0]
    assert "success_rate" in train_metrics["updates"][0]
    assert "approx_kl" in train_metrics["updates"][0]
    assert "kl_early_stopped" in train_metrics["updates"][0]
    assert "optimized_minibatches" in train_metrics["updates"][0]
    assert eval_metrics["episodes"] == 2
    assert "training_metrics" not in eval_metrics
    assert "turn_action_rate" in eval_metrics
    assert "target_visible_rate" in eval_metrics
    assert "near_obstacle_rate" in eval_metrics
    assert "mean_nearest_obstacle_clearance" in eval_metrics
    assert "min_nearest_obstacle_clearance" in eval_metrics
    assert "training_metrics" in full_eval_metrics
    assert full_eval_metrics["training_metrics"]["timesteps"] == 16
    assert (
        eval_metrics["successes"]
        + eval_metrics["collisions"]
        + eval_metrics["timeouts"]
        == 2
    )
