from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical
from tqdm.auto import tqdm

from src.gym_env import CircleSeekGymEnv

STRUCTURED_ACTION_FORMAT = "structured-categorical-v1"
STRUCTURED_ACTION_SIZE = 2
MOVEMENT_ACTIONS = np.asarray(
    [
        [0, 0, 0, 0],  # noop
        [1, 0, 0, 0],  # up
        [0, 1, 0, 0],  # down
        [0, 0, 1, 0],  # left
        [0, 0, 0, 1],  # right
        [1, 0, 1, 0],  # up-left
        [1, 0, 0, 1],  # up-right
        [0, 1, 1, 0],  # down-left
        [0, 1, 0, 1],  # down-right
    ],
    dtype=np.int8,
)
TURN_ACTIONS = np.asarray(
    [
        [0, 0],  # none
        [1, 0],  # turn left
        [0, 1],  # turn right
    ],
    dtype=np.int8,
)
MOVEMENT_ACTION_COUNT = int(MOVEMENT_ACTIONS.shape[0])
TURN_ACTION_COUNT = int(TURN_ACTIONS.shape[0])


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
    action_conflict_penalty: float = 0.02
    target_visible_reward_coef: float = 0.02
    target_found_reward_coef: float = 0.2
    no_vision_no_turn_penalty: float = 0.005
    learning_rate: float = 3e-4
    max_grad_norm: float = 0.5
    target_kl: float | None = None
    hidden_size: int = 64
    seed: int = 123
    obstacle_count: int = 4
    obstacle_speed: float = 2.5
    obstacle_radius: float = 36.0
    min_target_distance: float = 200.0
    max_steps: int = 600
    curriculum: bool = False
    curriculum_stages: int = 4
    curriculum_start_obstacle_speed: float = 0.0
    curriculum_start_obstacle_radius: float = 0.0


class ActorCritic(nn.Module):
    """Shared MLP actor-critic for structured movement and turn actions."""

    def __init__(
        self,
        observation_size: int,
        action_count: int = 6,
        hidden_size: int = 64,
    ) -> None:
        super().__init__()
        self.action_count = action_count
        self.shared = nn.Sequential(
            nn.Linear(observation_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
        )
        self.movement_policy_head = nn.Linear(hidden_size, MOVEMENT_ACTION_COUNT)
        self.turn_policy_head = nn.Linear(hidden_size, TURN_ACTION_COUNT)
        self.value_head = nn.Linear(hidden_size, 1)

    def forward(
        self,
        observations: torch.Tensor,
    ) -> tuple[tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        features = self.shared(observations)
        movement_logits = self.movement_policy_head(features)
        turn_logits = self.turn_policy_head(features)
        values = self.value_head(features).squeeze(-1)
        return (movement_logits, turn_logits), values

    def get_action_and_value(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        (movement_logits, turn_logits), values = self(observations)
        movement_distribution = Categorical(logits=movement_logits)
        turn_distribution = Categorical(logits=turn_logits)

        if actions is None:
            movement_actions = movement_distribution.sample()
            turn_actions = turn_distribution.sample()
            actions = torch.stack((movement_actions, turn_actions), dim=-1)
        else:
            actions = actions.to(dtype=torch.long)
            if actions.ndim == 1:
                actions = actions.reshape(-1, STRUCTURED_ACTION_SIZE)
            movement_actions = actions[:, 0]
            turn_actions = actions[:, 1]

        log_probs = (
            movement_distribution.log_prob(movement_actions)
            + turn_distribution.log_prob(turn_actions)
        )
        entropy = movement_distribution.entropy() + turn_distribution.entropy()
        return actions, log_probs, entropy, values


class RolloutBuffer:
    """Fixed-size rollout storage for one environment."""

    def __init__(
        self,
        rollout_length: int,
        observation_size: int,
        action_size: int = 1,
        device: torch.device | str = "cpu",
    ) -> None:
        if rollout_length <= 0:
            raise ValueError("rollout_length must be positive.")

        self.rollout_length = rollout_length
        self.observation_size = observation_size
        self.action_size = action_size
        self.device = torch.device(device)
        self.observations = torch.zeros(
            (rollout_length, observation_size),
            dtype=torch.float32,
            device=self.device,
        )
        self.actions = torch.zeros(
            (rollout_length, action_size),
            dtype=torch.long,
            device=self.device,
        )
        self.rewards = torch.zeros(rollout_length, dtype=torch.float32, device=self.device)
        self.dones = torch.zeros(rollout_length, dtype=torch.float32, device=self.device)
        self.values = torch.zeros(rollout_length, dtype=torch.float32, device=self.device)
        self.log_probs = torch.zeros(rollout_length, dtype=torch.float32, device=self.device)
        self.advantages = torch.zeros(rollout_length, dtype=torch.float32, device=self.device)
        self.returns = torch.zeros(rollout_length, dtype=torch.float32, device=self.device)
        self.step = 0

    @property
    def is_full(self) -> bool:
        return self.step == self.rollout_length

    def add(
        self,
        observation: torch.Tensor,
        action: Iterable[int | float | bool] | torch.Tensor,
        reward: float | torch.Tensor,
        done: bool | torch.Tensor,
        value: float | torch.Tensor,
        log_prob: float | torch.Tensor,
    ) -> None:
        if self.is_full:
            raise RuntimeError("RolloutBuffer is full.")

        observation = torch.as_tensor(
            observation,
            dtype=torch.float32,
            device=self.device,
        )
        if observation.shape != (self.observation_size,):
            raise ValueError(
                f"Expected observation shape {(self.observation_size,)}, "
                f"got {tuple(observation.shape)}."
            )

        action = torch.as_tensor(action, dtype=torch.long, device=self.device)
        if action.shape == ():
            action = action.reshape(1)
        if action.shape != (self.action_size,):
            raise ValueError(
                f"Expected action shape {(self.action_size,)}, got {tuple(action.shape)}."
            )

        self.observations[self.step].copy_(observation)
        self.actions[self.step].copy_(action)
        self.rewards[self.step] = self._scalar_tensor(reward, torch.float32)
        self.dones[self.step] = self._scalar_tensor(done, torch.float32)
        self.values[self.step] = self._scalar_tensor(value, torch.float32)
        self.log_probs[self.step] = self._scalar_tensor(log_prob, torch.float32)
        self.step += 1

    def reset(self) -> None:
        self.step = 0

    def compute_returns_and_advantages(
        self,
        last_value: float | torch.Tensor,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.is_full:
            raise RuntimeError("RolloutBuffer must be full before computing returns.")

        last_value = torch.as_tensor(last_value, dtype=torch.float32, device=self.device)
        last_gae = torch.zeros((), dtype=torch.float32, device=self.device)

        for index in reversed(range(self.rollout_length)):
            if index == self.rollout_length - 1:
                next_value = last_value
            else:
                next_value = self.values[index + 1]

            next_non_terminal = 1.0 - self.dones[index]
            delta = (
                self.rewards[index]
                + gamma * next_value * next_non_terminal
                - self.values[index]
            )
            last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae
            self.advantages[index] = last_gae

        self.returns.copy_(self.advantages + self.values)
        return self.returns, self.advantages

    def _scalar_tensor(
        self,
        value: int | float | bool | torch.Tensor,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        return torch.as_tensor(value, dtype=dtype, device=self.device).reshape(())


def structured_action_to_multibinary(
    action: Iterable[int] | np.ndarray | torch.Tensor,
) -> np.ndarray:
    if isinstance(action, torch.Tensor):
        structured_action = action.detach().cpu().numpy().astype(np.int64).reshape(-1)
    else:
        structured_action = np.asarray(action, dtype=np.int64).reshape(-1)
    if structured_action.shape != (STRUCTURED_ACTION_SIZE,):
        raise ValueError(
            f"Expected structured action shape {(STRUCTURED_ACTION_SIZE,)}, "
            f"got {tuple(structured_action.shape)}."
        )

    movement_index = int(structured_action[0])
    turn_index = int(structured_action[1])
    if not 0 <= movement_index < MOVEMENT_ACTION_COUNT:
        raise ValueError(f"Invalid movement action index: {movement_index}.")
    if not 0 <= turn_index < TURN_ACTION_COUNT:
        raise ValueError(f"Invalid turn action index: {turn_index}.")

    return np.concatenate(
        (MOVEMENT_ACTIONS[movement_index], TURN_ACTIONS[turn_index])
    ).astype(np.int8)


def action_conflict_count(action: Iterable[int | float | bool] | np.ndarray) -> int:
    action_vector = np.asarray(action, dtype=np.int8).reshape(-1)
    if action_vector.size < 6:
        action_vector = np.pad(action_vector, (0, 6 - action_vector.size))
    conflicts = 0
    conflicts += int(action_vector[0] and action_vector[1])
    conflicts += int(action_vector[2] and action_vector[3])
    conflicts += int(action_vector[4] and action_vector[5])
    return conflicts


def action_turns(action: Iterable[int | float | bool] | np.ndarray) -> bool:
    action_vector = np.asarray(action, dtype=np.int8).reshape(-1)
    if action_vector.size < 6:
        action_vector = np.pad(action_vector, (0, 6 - action_vector.size))
    return bool(action_vector[4] or action_vector[5])


def target_visibility_training_reward(
    *,
    target_visible: bool,
    previous_target_visible: bool,
    turning: bool,
    visible_reward_coef: float,
    found_reward_coef: float,
    no_vision_no_turn_penalty: float,
) -> float:
    reward = visible_reward_coef if target_visible else 0.0
    if target_visible and not previous_target_visible:
        reward += found_reward_coef
    if not target_visible and not turning:
        reward -= no_vision_no_turn_penalty
    return reward


def collect_rollout(
    model: ActorCritic,
    rollout_length: int,
    env: CircleSeekGymEnv | None = None,
    *,
    seed: int | None = None,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    device: torch.device | str = "cpu",
    env_kwargs: dict | None = None,
    distance_reward_coef: float = 0.0,
    action_conflict_penalty: float = 0.0,
    target_visible_reward_coef: float = 0.0,
    target_found_reward_coef: float = 0.0,
    no_vision_no_turn_penalty: float = 0.0,
) -> tuple[RolloutBuffer, dict]:
    """Collect one fixed-length rollout from CircleSeekGymEnv."""

    if env is None:
        env = CircleSeekGymEnv(**(env_kwargs or {}))

    device = torch.device(device)
    observation, reset_info = env.reset(seed=seed)
    observation_size = int(env.observation_space.shape[0])
    buffer = RolloutBuffer(
        rollout_length=rollout_length,
        observation_size=observation_size,
        action_size=STRUCTURED_ACTION_SIZE,
        device=device,
    )

    previous_distance = float(reset_info["distance_to_target"])
    previous_target_visible = bool(reset_info["target_visible"])
    episode_initial_distance = previous_distance
    episode_env_return = 0.0
    episode_training_return = 0.0
    episode_length = 0
    episode_returns: list[float] = []
    episode_training_returns: list[float] = []
    episode_lengths: list[int] = []
    episode_statuses: list[str] = []
    initial_distances: list[float] = []
    final_distances: list[float] = []
    conflict_counts: list[int] = []
    target_visible_steps = 0
    target_found_steps = 0
    turn_steps = 0
    no_vision_no_turn_steps = 0

    for _ in range(rollout_length):
        observation_tensor = torch.as_tensor(
            observation,
            dtype=torch.float32,
            device=device,
        ).unsqueeze(0)

        with torch.no_grad():
            action_tensor, log_prob, _, value = model.get_action_and_value(observation_tensor)

        structured_action = action_tensor.squeeze(0)
        env_action = structured_action_to_multibinary(structured_action)
        next_observation, reward, terminated, truncated, step_info = env.step(env_action)
        done = terminated or truncated
        current_distance = float(step_info["distance_to_target"])
        distance_delta = previous_distance - current_distance
        conflict_count = action_conflict_count(env_action)
        turning = action_turns(env_action)
        target_visible = bool(step_info["target_visible"])
        target_found = target_visible and not previous_target_visible
        no_vision_no_turn = not target_visible and not turning
        visibility_reward = target_visibility_training_reward(
            target_visible=target_visible,
            previous_target_visible=previous_target_visible,
            turning=turning,
            visible_reward_coef=target_visible_reward_coef,
            found_reward_coef=target_found_reward_coef,
            no_vision_no_turn_penalty=no_vision_no_turn_penalty,
        )
        training_reward = (
            float(reward)
            + distance_reward_coef * distance_delta / max(float(env.env.agent_speed), 1.0)
            + visibility_reward
            - action_conflict_penalty * conflict_count
        )

        buffer.add(
            observation=observation,
            action=structured_action,
            reward=training_reward,
            done=done,
            value=value.squeeze(0),
            log_prob=log_prob.squeeze(0),
        )

        episode_env_return += float(reward)
        episode_training_return += training_reward
        episode_length += 1
        conflict_counts.append(conflict_count)
        target_visible_steps += int(target_visible)
        target_found_steps += int(target_found)
        turn_steps += int(turning)
        no_vision_no_turn_steps += int(no_vision_no_turn)

        if done:
            episode_returns.append(float(episode_env_return))
            episode_training_returns.append(float(episode_training_return))
            episode_lengths.append(episode_length)
            episode_statuses.append(str(step_info["status"]))
            initial_distances.append(episode_initial_distance)
            final_distances.append(current_distance)
            episode_env_return = 0.0
            episode_training_return = 0.0
            episode_length = 0
            next_observation, reset_info = env.reset()
            previous_distance = float(reset_info["distance_to_target"])
            previous_target_visible = bool(reset_info["target_visible"])
            episode_initial_distance = previous_distance
        else:
            previous_distance = current_distance
            previous_target_visible = target_visible

        observation = next_observation

    last_observation = torch.as_tensor(
        observation,
        dtype=torch.float32,
        device=device,
    ).unsqueeze(0)
    with torch.no_grad():
        _, last_value = model(last_observation)

    buffer.compute_returns_and_advantages(
        last_value=last_value.squeeze(0),
        gamma=gamma,
        gae_lambda=gae_lambda,
    )

    info = {
        "steps": rollout_length,
        "episodes": len(episode_returns),
        "episode_returns": episode_returns,
        "episode_training_returns": episode_training_returns,
        "episode_lengths": episode_lengths,
        "episode_statuses": episode_statuses,
        "initial_distances": initial_distances,
        "final_distances": final_distances,
        "action_conflict_steps": sum(1 for count in conflict_counts if count > 0),
        "action_conflicts": int(sum(conflict_counts)),
        "target_visible_steps": target_visible_steps,
        "target_found_steps": target_found_steps,
        "turn_steps": turn_steps,
        "no_vision_no_turn_steps": no_vision_no_turn_steps,
        "last_value": float(last_value.squeeze(0).item()),
    }
    return buffer, info


def compute_ppo_loss(
    model: ActorCritic,
    observations: torch.Tensor,
    actions: torch.Tensor,
    old_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    returns: torch.Tensor,
    *,
    clip_coef: float = 0.2,
    value_coef: float = 0.5,
    entropy_coef: float = 0.01,
) -> dict[str, torch.Tensor]:
    """Compute PPO clipped objective terms for the structured categorical policy."""

    _, new_log_probs, entropy, values = model.get_action_and_value(observations, actions)
    ratio = (new_log_probs - old_log_probs).exp()

    unclipped_policy_loss = ratio * advantages
    clipped_policy_loss = torch.clamp(
        ratio,
        1.0 - clip_coef,
        1.0 + clip_coef,
    ) * advantages
    policy_loss = -torch.min(unclipped_policy_loss, clipped_policy_loss).mean()
    value_loss = 0.5 * (returns - values).pow(2).mean()
    entropy_bonus = entropy.mean()
    loss = policy_loss + value_coef * value_loss - entropy_coef * entropy_bonus

    with torch.no_grad():
        approx_kl = (old_log_probs - new_log_probs).mean()
        clip_fraction = ((ratio - 1.0).abs() > clip_coef).float().mean()

    return {
        "loss": loss,
        "policy_loss": policy_loss,
        "value_loss": value_loss,
        "entropy": entropy_bonus,
        "approx_kl": approx_kl,
        "clip_fraction": clip_fraction,
    }


def should_stop_for_kl(approx_kl: float, target_kl: float | None) -> bool:
    return target_kl is not None and approx_kl > target_kl


def set_seed(seed: int) -> np.random.Generator:
    np.random.seed(seed)
    torch.manual_seed(seed)
    return np.random.default_rng(seed)


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


def build_env(
    config: PPOConfig,
    stage_progress: float | None = None,
    **gym_kwargs: Any,
) -> CircleSeekGymEnv:
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
        **gym_kwargs,
        obstacle_count=config.obstacle_count,
        obstacle_speed=obstacle_speed,
        obstacle_radius=obstacle_radius,
        min_target_distance=config.min_target_distance,
        max_steps=config.max_steps,
    )


def select_action(
    model: ActorCritic,
    observation: np.ndarray,
    *,
    deterministic: bool,
) -> np.ndarray:
    observation_tensor = torch.as_tensor(observation, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        if deterministic:
            (movement_logits, turn_logits), _ = model(observation_tensor)
            movement_action = movement_logits.argmax(dim=-1)
            turn_action = turn_logits.argmax(dim=-1)
            action = torch.stack((movement_action, turn_action), dim=-1)
        else:
            action, _, _, _ = model.get_action_and_value(observation_tensor)
        structured_action = action.squeeze(0)
    return structured_action_to_multibinary(structured_action)


def train_ppo(
    config: PPOConfig,
    checkpoint_path: Path | None = None,
    *,
    progress: bool = False,
) -> dict[str, Any]:
    rng = set_seed(config.seed)
    _, stage_progress = curriculum_stage(config, 0)
    env = build_env(config, stage_progress)
    observation_size = int(env.observation_space.shape[0])
    action_size = int(env.action_space.n)
    model = ActorCritic(observation_size, action_size, config.hidden_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    global_step = 0
    completed_returns: list[float] = []
    completed_training_returns: list[float] = []
    completed_lengths: list[int] = []
    initial_distances: list[float] = []
    final_distances: list[float] = []
    total_action_conflict_steps = 0
    total_action_conflicts = 0
    total_target_visible_steps = 0
    total_target_found_steps = 0
    total_turn_steps = 0
    total_no_vision_no_turn_steps = 0
    status_counts = {"success": 0, "collision": 0, "timeout": 0}
    loss_history: dict[str, list[float]] = {
        "policy_loss": [],
        "value_loss": [],
        "entropy": [],
        "approx_kl": [],
        "clip_fraction": [],
    }
    update_history: list[dict[str, Any]] = []
    stage_counts: list[dict[str, int]] = [
        {"stage": idx, "success": 0, "collision": 0, "timeout": 0}
        for idx in range(max(config.curriculum_stages if config.curriculum else 1, 1))
    ]

    progress_bar = tqdm(
        total=config.total_timesteps,
        desc="Training PPO",
        unit="step",
        disable=not progress,
    )
    while global_step < config.total_timesteps:
        stage_index, stage_progress = curriculum_stage(config, global_step)
        env = build_env(config, stage_progress)
        rollout_size = min(config.rollout_steps, config.total_timesteps - global_step)
        rollout_seed = config.seed + global_step + int(rng.integers(10_000))
        buffer, rollout_info = collect_rollout(
            model,
            rollout_size,
            env,
            seed=rollout_seed,
            gamma=config.gamma,
            gae_lambda=config.gae_lambda,
            distance_reward_coef=config.distance_reward_coef,
            action_conflict_penalty=config.action_conflict_penalty,
            target_visible_reward_coef=config.target_visible_reward_coef,
            target_found_reward_coef=config.target_found_reward_coef,
            no_vision_no_turn_penalty=config.no_vision_no_turn_penalty,
        )
        global_step += rollout_size
        progress_bar.update(rollout_size)

        completed_returns.extend(rollout_info["episode_returns"])
        completed_training_returns.extend(rollout_info["episode_training_returns"])
        completed_lengths.extend(rollout_info["episode_lengths"])
        initial_distances.extend(rollout_info["initial_distances"])
        final_distances.extend(rollout_info["final_distances"])
        total_action_conflict_steps += int(rollout_info["action_conflict_steps"])
        total_action_conflicts += int(rollout_info["action_conflicts"])
        total_target_visible_steps += int(rollout_info["target_visible_steps"])
        total_target_found_steps += int(rollout_info["target_found_steps"])
        total_turn_steps += int(rollout_info["turn_steps"])
        total_no_vision_no_turn_steps += int(rollout_info["no_vision_no_turn_steps"])
        for status in rollout_info["episode_statuses"]:
            status_counts[status] = status_counts.get(status, 0) + 1
            if status in stage_counts[stage_index]:
                stage_counts[stage_index][status] += 1

        advantages = buffer.advantages
        advantages = (advantages - advantages.mean()) / (
            advantages.std(unbiased=False) + 1e-8
        )

        batch_indices = np.arange(rollout_size)
        update_loss_history: dict[str, list[float]] = {
            "policy_loss": [],
            "value_loss": [],
            "entropy": [],
            "approx_kl": [],
            "clip_fraction": [],
        }
        kl_early_stopped = False
        completed_update_epochs = 0
        optimized_minibatches = 0
        for _ in range(config.update_epochs):
            rng.shuffle(batch_indices)
            completed_update_epochs += 1
            for start in range(0, rollout_size, config.minibatch_size):
                indices = batch_indices[start : start + config.minibatch_size]
                loss_terms = compute_ppo_loss(
                    model,
                    buffer.observations[indices],
                    buffer.actions[indices],
                    buffer.log_probs[indices],
                    advantages[indices],
                    buffer.returns[indices],
                    clip_coef=config.clip_coef,
                    value_coef=config.value_coef,
                    entropy_coef=config.entropy_coef,
                )

                optimizer.zero_grad()
                loss_terms["loss"].backward()
                nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                optimizer.step()
                optimized_minibatches += 1

                for key in loss_history:
                    value = float(loss_terms[key].detach().item())
                    loss_history[key].append(value)
                    update_loss_history[key].append(value)

                approx_kl = float(loss_terms["approx_kl"].detach().item())
                if should_stop_for_kl(approx_kl, config.target_kl):
                    kl_early_stopped = True
                    break

            if kl_early_stopped:
                break

        episode_count = len(completed_returns)
        rollout_episode_count = len(rollout_info["episode_returns"])
        rollout_successes = sum(
            1 for status in rollout_info["episode_statuses"] if status == "success"
        )
        update_history.append(
            {
                "update": len(update_history) + 1,
                "global_step": global_step,
                "stage": stage_index,
                "rollout_steps": rollout_size,
                "optimized_minibatches": optimized_minibatches,
                "completed_update_epochs": completed_update_epochs,
                "kl_early_stopped": kl_early_stopped,
                "rollout_episodes": rollout_episode_count,
                "rollout_mean_return": (
                    float(np.mean(rollout_info["episode_returns"]))
                    if rollout_info["episode_returns"]
                    else 0.0
                ),
                "rollout_success_rate": (
                    rollout_successes / rollout_episode_count
                    if rollout_episode_count
                    else 0.0
                ),
                "episodes": episode_count,
                "success_rate": status_counts["success"] / max(episode_count, 1),
                "collision_rate": status_counts["collision"] / max(episode_count, 1),
                "timeout_rate": status_counts["timeout"] / max(episode_count, 1),
                "mean_return": (
                    float(np.mean(completed_returns)) if completed_returns else 0.0
                ),
                "mean_training_return": (
                    float(np.mean(completed_training_returns))
                    if completed_training_returns
                    else 0.0
                ),
                "mean_episode_length": (
                    float(np.mean(completed_lengths)) if completed_lengths else 0.0
                ),
                "policy_loss": _mean_or_zero(update_loss_history["policy_loss"]),
                "value_loss": _mean_or_zero(update_loss_history["value_loss"]),
                "entropy": _mean_or_zero(update_loss_history["entropy"]),
                "approx_kl": _mean_or_zero(update_loss_history["approx_kl"]),
                "clip_fraction": _mean_or_zero(update_loss_history["clip_fraction"]),
            }
        )
        latest_update = update_history[-1]
        progress_bar.set_postfix(
            {
                "success": f"{latest_update['success_rate']:.3f}",
                "collision": f"{latest_update['collision_rate']:.3f}",
                "return": f"{latest_update['mean_return']:.3f}",
                "stage": stage_index,
            }
        )

    progress_bar.close()
    episode_count = len(completed_returns)
    metrics = {
        "timesteps": global_step,
        "episodes": episode_count,
        "successes": status_counts["success"],
        "collisions": status_counts["collision"],
        "timeouts": status_counts["timeout"],
        "success_rate": status_counts["success"] / max(episode_count, 1),
        "collision_rate": status_counts["collision"] / max(episode_count, 1),
        "timeout_rate": status_counts["timeout"] / max(episode_count, 1),
        "mean_return": float(np.mean(completed_returns)) if completed_returns else 0.0,
        "mean_training_return": (
            float(np.mean(completed_training_returns)) if completed_training_returns else 0.0
        ),
        "mean_episode_length": (
            float(np.mean(completed_lengths)) if completed_lengths else 0.0
        ),
        "mean_initial_distance": (
            float(np.mean(initial_distances)) if initial_distances else 0.0
        ),
        "mean_final_distance": (
            float(np.mean(final_distances)) if final_distances else 0.0
        ),
        "mean_distance_delta": (
            float(np.mean(np.asarray(initial_distances) - np.asarray(final_distances)))
            if initial_distances and final_distances
            else 0.0
        ),
        "action_conflict_rate": total_action_conflict_steps / max(global_step, 1),
        "mean_action_conflicts": total_action_conflicts / max(global_step, 1),
        "target_visible_rate": total_target_visible_steps / max(global_step, 1),
        "target_found_rate": total_target_found_steps / max(global_step, 1),
        "turn_action_rate": total_turn_steps / max(global_step, 1),
        "no_vision_no_turn_rate": total_no_vision_no_turn_steps / max(global_step, 1),
        "policy_loss": _mean_or_zero(loss_history["policy_loss"]),
        "value_loss": _mean_or_zero(loss_history["value_loss"]),
        "entropy": _mean_or_zero(loss_history["entropy"]),
        "approx_kl": _mean_or_zero(loss_history["approx_kl"]),
        "clip_fraction": _mean_or_zero(loss_history["clip_fraction"]),
        "target_kl": config.target_kl,
        "kl_early_stops": sum(
            1 for update in update_history if update["kl_early_stopped"]
        ),
        "curriculum": {
            "enabled": config.curriculum,
            "stages": max(config.curriculum_stages if config.curriculum else 1, 1),
            "stage_counts": stage_counts,
            "final_obstacle_speed": config.obstacle_speed,
            "final_obstacle_radius": config.obstacle_radius,
        },
        "updates": update_history,
    }

    if checkpoint_path is not None:
        save_checkpoint(
            checkpoint_path,
            model,
            config,
            observation_size,
            action_size,
            metrics,
        )

    return metrics


def save_checkpoint(
    path: Path,
    model: ActorCritic,
    config: PPOConfig,
    observation_size: int,
    action_size: int,
    metrics: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": asdict(config),
            "observation_size": observation_size,
            "action_size": action_size,
            "action_format": STRUCTURED_ACTION_FORMAT,
            "movement_action_count": MOVEMENT_ACTION_COUNT,
            "turn_action_count": TURN_ACTION_COUNT,
            "training_metrics": metrics,
        },
        path,
    )


def load_checkpoint(path: Path) -> tuple[ActorCritic, PPOConfig, dict[str, Any]]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    config = PPOConfig(**checkpoint["config"])
    action_format = checkpoint.get("action_format")
    model = ActorCritic(
        int(checkpoint["observation_size"]),
        int(checkpoint["action_size"]),
        config.hidden_size,
    )
    state_dict = checkpoint["model_state_dict"]
    if action_format is None and "policy_head.weight" in state_dict:
        state_dict = _convert_legacy_bernoulli_state_dict(state_dict)
    elif action_format != STRUCTURED_ACTION_FORMAT:
        raise ValueError(f"Unsupported PPO action format: {action_format!r}.")
    model.load_state_dict(state_dict)
    model.eval()
    return model, config, dict(checkpoint.get("training_metrics", {}))


def _convert_legacy_bernoulli_state_dict(
    state_dict: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    converted = {
        key: value
        for key, value in state_dict.items()
        if not key.startswith("policy_head.")
    }
    policy_weight = state_dict["policy_head.weight"]
    policy_bias = state_dict["policy_head.bias"]

    movement_matrix = torch.as_tensor(
        MOVEMENT_ACTIONS,
        dtype=policy_weight.dtype,
        device=policy_weight.device,
    )
    turn_matrix = torch.as_tensor(
        TURN_ACTIONS,
        dtype=policy_weight.dtype,
        device=policy_weight.device,
    )
    converted["movement_policy_head.weight"] = movement_matrix @ policy_weight[:4]
    converted["movement_policy_head.bias"] = movement_matrix @ policy_bias[:4]
    converted["turn_policy_head.weight"] = turn_matrix @ policy_weight[4:6]
    converted["turn_policy_head.bias"] = turn_matrix @ policy_bias[4:6]
    return converted


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


def _mean_or_zero(values: list[float]) -> float:
    return float(np.mean(values)) if values else 0.0
