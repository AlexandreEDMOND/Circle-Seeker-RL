from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

from src.gym_env import CircleSeekGymEnv


@dataclass(frozen=True)
class PPOConfig:
    total_timesteps: int = 100_000
    rollout_steps: int = 2048
    update_epochs: int = 8
    minibatch_size: int = 256
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    distance_reward_coef: float = 0.05
    learning_rate: float = 3e-4
    max_grad_norm: float = 0.5
    hidden_size: int = 64
    seed: int = 123
    obstacle_count: int = 4
    obstacle_speed: float = 2.5
    obstacle_radius: float = 16.0
    max_steps: int = 600
    curriculum: bool = False
    curriculum_stages: int = 4
    curriculum_start_obstacle_speed: float = 0.0
    curriculum_start_obstacle_radius: float = 0.0


class ActorCritic(nn.Module):
    def __init__(self, observation_dim: int, action_dim: int, hidden_size: int = 64) -> None:
        super().__init__()
        self.policy_net = nn.Sequential(
            nn.Linear(observation_dim, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, action_dim),
        )
        self.value_net = nn.Sequential(
            nn.Linear(observation_dim, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
        )

    def distribution(self, observations: torch.Tensor) -> Categorical:
        return Categorical(logits=self.policy_net(observations))

    def value(self, observations: torch.Tensor) -> torch.Tensor:
        return self.value_net(observations).squeeze(-1)

    def act(self, observations: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        distribution = self.distribution(observations)
        actions = distribution.sample()
        return actions, distribution.log_prob(actions), self.value(observations)


class RolloutBuffer:
    def __init__(self, steps: int, observation_dim: int) -> None:
        self.observations = torch.zeros((steps, observation_dim), dtype=torch.float32)
        self.actions = torch.zeros(steps, dtype=torch.long)
        self.log_probs = torch.zeros(steps, dtype=torch.float32)
        self.rewards = torch.zeros(steps, dtype=torch.float32)
        self.dones = torch.zeros(steps, dtype=torch.float32)
        self.values = torch.zeros(steps, dtype=torch.float32)
        self.advantages = torch.zeros(steps, dtype=torch.float32)
        self.returns = torch.zeros(steps, dtype=torch.float32)

    def compute_returns_and_advantages(
        self,
        *,
        last_value: torch.Tensor,
        last_done: bool,
        gamma: float,
        gae_lambda: float,
    ) -> None:
        last_gae = torch.tensor(0.0)
        next_value = last_value.detach().reshape(())
        next_non_terminal = torch.tensor(0.0 if last_done else 1.0)

        for step in reversed(range(len(self.rewards))):
            if step < len(self.rewards) - 1:
                next_value = self.values[step + 1]
                next_non_terminal = 1.0 - self.dones[step + 1]

            delta = (
                self.rewards[step]
                + gamma * next_value * next_non_terminal
                - self.values[step]
            )
            last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae
            self.advantages[step] = last_gae

        self.returns = self.advantages + self.values


def set_seed(seed: int) -> np.random.Generator:
    np.random.seed(seed)
    torch.manual_seed(seed)
    return np.random.default_rng(seed)


def build_env(config: PPOConfig, stage_progress: float | None = None) -> CircleSeekGymEnv:
    obstacle_speed = config.obstacle_speed
    obstacle_radius = config.obstacle_radius
    if config.curriculum and stage_progress is not None:
        progress = float(np.clip(stage_progress, 0.0, 1.0))
        obstacle_speed = (
            config.curriculum_start_obstacle_speed
            + (config.obstacle_speed - config.curriculum_start_obstacle_speed) * progress
        )
        obstacle_radius = (
            config.curriculum_start_obstacle_radius
            + (config.obstacle_radius - config.curriculum_start_obstacle_radius) * progress
        )

    return CircleSeekGymEnv(
        obstacle_count=config.obstacle_count,
        obstacle_speed=obstacle_speed,
        obstacle_radius=obstacle_radius,
        max_steps=config.max_steps,
    )


def curriculum_stage(config: PPOConfig, global_step: int) -> tuple[int, float]:
    if not config.curriculum:
        return 0, 1.0

    stage_count = max(config.curriculum_stages, 1)
    stage_index = min(
        int(global_step / max(config.total_timesteps, 1) * stage_count),
        stage_count - 1,
    )
    if stage_count == 1:
        return stage_index, 1.0
    return stage_index, stage_index / (stage_count - 1)


def select_action(
    model: ActorCritic,
    observation: np.ndarray,
    *,
    deterministic: bool,
) -> int:
    observation_tensor = torch.as_tensor(observation, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        distribution = model.distribution(observation_tensor)
        if deterministic:
            action = torch.argmax(distribution.logits, dim=-1)
        else:
            action = distribution.sample()
    return int(action.item())


def train_ppo(config: PPOConfig, checkpoint_path: Path | None = None) -> dict[str, Any]:
    rng = set_seed(config.seed)
    current_stage, stage_progress = curriculum_stage(config, 0)
    env = build_env(config, stage_progress)
    observation_dim = int(env.observation_space.shape[0])
    action_dim = int(env.action_space.n)
    model = ActorCritic(observation_dim, action_dim, config.hidden_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    observation, _ = env.reset(seed=config.seed)
    previous_distance = env.env.distance_to_target()
    done = False
    global_step = 0
    episode_return = 0.0
    episode_training_return = 0.0
    episode_length = 0
    completed_returns: list[float] = []
    completed_training_returns: list[float] = []
    completed_lengths: list[int] = []
    status_counts = {"success": 0, "collision": 0, "timeout": 0}
    stage_counts: list[dict[str, int]] = [
        {"stage": idx, "success": 0, "collision": 0, "timeout": 0}
        for idx in range(max(config.curriculum_stages if config.curriculum else 1, 1))
    ]

    while global_step < config.total_timesteps:
        next_stage, stage_progress = curriculum_stage(config, global_step)
        if next_stage != current_stage:
            current_stage = next_stage
            env = build_env(config, stage_progress)
            observation, _ = env.reset(seed=config.seed + global_step)
            previous_distance = env.env.distance_to_target()
            done = False
            episode_return = 0.0
            episode_training_return = 0.0
            episode_length = 0

        rollout_size = min(config.rollout_steps, config.total_timesteps - global_step)
        buffer = RolloutBuffer(rollout_size, observation_dim)

        for step in range(rollout_size):
            buffer.observations[step] = torch.as_tensor(observation, dtype=torch.float32)
            buffer.dones[step] = float(done)

            with torch.no_grad():
                action, log_prob, value = model.act(buffer.observations[step].unsqueeze(0))

            action_int = int(action.item())
            next_observation, reward, terminated, truncated, info = env.step(action_int)
            next_done = bool(terminated or truncated)
            current_distance = float(info["distance_to_target"])
            distance_delta = previous_distance - current_distance
            training_reward = float(reward) + config.distance_reward_coef * (
                distance_delta / max(env.env.agent_speed, 1.0)
            )

            buffer.actions[step] = action_int
            buffer.log_probs[step] = log_prob.squeeze(0)
            buffer.values[step] = value.squeeze(0)
            buffer.rewards[step] = training_reward

            episode_return += float(reward)
            episode_training_return += training_reward
            episode_length += 1
            global_step += 1

            if next_done:
                status = str(info["status"])
                status_counts[status] = status_counts.get(status, 0) + 1
                if status in stage_counts[current_stage]:
                    stage_counts[current_stage][status] += 1
                completed_returns.append(episode_return)
                completed_training_returns.append(episode_training_return)
                completed_lengths.append(episode_length)
                observation, _ = env.reset(seed=config.seed + global_step + int(rng.integers(10_000)))
                previous_distance = env.env.distance_to_target()
                episode_return = 0.0
                episode_training_return = 0.0
                episode_length = 0
            else:
                observation = next_observation
                previous_distance = current_distance

            done = next_done

        with torch.no_grad():
            last_observation = torch.as_tensor(observation, dtype=torch.float32).unsqueeze(0)
            last_value = model.value(last_observation).squeeze(0)

        buffer.compute_returns_and_advantages(
            last_value=last_value,
            last_done=done,
            gamma=config.gamma,
            gae_lambda=config.gae_lambda,
        )
        advantages = buffer.advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        batch_indices = np.arange(rollout_size)
        for _ in range(config.update_epochs):
            rng.shuffle(batch_indices)
            for start in range(0, rollout_size, config.minibatch_size):
                indices = batch_indices[start : start + config.minibatch_size]
                mb_obs = buffer.observations[indices]
                mb_actions = buffer.actions[indices]
                mb_old_log_probs = buffer.log_probs[indices]
                mb_advantages = advantages[indices]
                mb_returns = buffer.returns[indices]

                distribution = model.distribution(mb_obs)
                new_log_probs = distribution.log_prob(mb_actions)
                entropy = distribution.entropy().mean()
                ratio = (new_log_probs - mb_old_log_probs).exp()

                unclipped_policy_loss = -mb_advantages * ratio
                clipped_policy_loss = -mb_advantages * torch.clamp(
                    ratio,
                    1.0 - config.clip_coef,
                    1.0 + config.clip_coef,
                )
                policy_loss = torch.max(unclipped_policy_loss, clipped_policy_loss).mean()
                value_loss = 0.5 * (model.value(mb_obs) - mb_returns).pow(2).mean()
                loss = (
                    policy_loss
                    + config.value_coef * value_loss
                    - config.entropy_coef * entropy
                )

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                optimizer.step()

    metrics = {
        "timesteps": global_step,
        "episodes": len(completed_returns),
        "successes": status_counts["success"],
        "collisions": status_counts["collision"],
        "timeouts": status_counts["timeout"],
        "success_rate": status_counts["success"] / max(len(completed_returns), 1),
        "collision_rate": status_counts["collision"] / max(len(completed_returns), 1),
        "timeout_rate": status_counts["timeout"] / max(len(completed_returns), 1),
        "mean_return": float(np.mean(completed_returns)) if completed_returns else 0.0,
        "mean_training_return": (
            float(np.mean(completed_training_returns))
            if completed_training_returns
            else 0.0
        ),
        "mean_episode_length": (
            float(np.mean(completed_lengths)) if completed_lengths else 0.0
        ),
        "curriculum": {
            "enabled": config.curriculum,
            "stages": max(config.curriculum_stages if config.curriculum else 1, 1),
            "stage_counts": stage_counts,
            "final_obstacle_speed": config.obstacle_speed,
            "final_obstacle_radius": config.obstacle_radius,
        },
    }

    if checkpoint_path is not None:
        save_checkpoint(checkpoint_path, model, config, observation_dim, action_dim, metrics)

    return metrics


def save_checkpoint(
    path: Path,
    model: ActorCritic,
    config: PPOConfig,
    observation_dim: int,
    action_dim: int,
    metrics: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": asdict(config),
            "observation_dim": observation_dim,
            "action_dim": action_dim,
            "training_metrics": metrics,
        },
        path,
    )


def load_checkpoint(path: Path) -> tuple[ActorCritic, PPOConfig, dict[str, Any]]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    config = PPOConfig(**checkpoint["config"])
    model = ActorCritic(
        int(checkpoint["observation_dim"]),
        int(checkpoint["action_dim"]),
        config.hidden_size,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, config, dict(checkpoint.get("training_metrics", {}))


def evaluate_checkpoint(
    checkpoint_path: Path,
    *,
    episodes: int = 20,
    seed: int = 123,
    deterministic: bool = True,
) -> dict[str, Any]:
    model, config, training_metrics = load_checkpoint(checkpoint_path)
    env = build_env(config)
    status_counts = {"success": 0, "collision": 0, "timeout": 0}
    returns: list[float] = []
    lengths: list[int] = []
    moving_steps: list[int] = []
    initial_distances: list[float] = []
    final_distances: list[float] = []

    for episode_idx in range(episodes):
        observation, info = env.reset(seed=seed + episode_idx)
        total_reward = 0.0
        length = 0
        moved = 0
        initial_distances.append(float(info["distance_to_target"]))

        while True:
            before = env.env.agent_position.copy()
            action = select_action(model, observation, deterministic=deterministic)
            observation, reward, terminated, truncated, info = env.step(action)
            after = env.env.agent_position.copy()
            total_reward += float(reward)
            length += 1
            if np.linalg.norm(after - before) > 1e-6:
                moved += 1

            if terminated or truncated:
                status = str(info["status"])
                status_counts[status] = status_counts.get(status, 0) + 1
                returns.append(total_reward)
                lengths.append(length)
                moving_steps.append(moved)
                final_distances.append(float(info["distance_to_target"]))
                break

    return {
        "episodes": episodes,
        "success_rate": status_counts["success"] / episodes,
        "collision_rate": status_counts["collision"] / episodes,
        "timeout_rate": status_counts["timeout"] / episodes,
        "mean_return": float(np.mean(returns)),
        "mean_episode_length": float(np.mean(lengths)),
        "mean_moving_steps": float(np.mean(moving_steps)),
        "mean_initial_distance": float(np.mean(initial_distances)),
        "mean_final_distance": float(np.mean(final_distances)),
        "successes": status_counts["success"],
        "collisions": status_counts["collision"],
        "timeouts": status_counts["timeout"],
        "training_metrics": training_metrics,
    }
