# Roadmap

This roadmap is focused on implementing PPO from the paper
`paper/Proximal Policy Optimization Algorithms.pdf`
and using Circle Seeker as the first training environment.

## Phase 1: Environment Contract

Goal: make the environment reliable enough for training loops.

- Convert or wrap `CircleSeekEnv` as a real `gymnasium.Env`.
- Define explicit `observation_space` and `action_space`.
- Keep deterministic reset behavior through seeds.
- Add tests for space shapes, action validity, termination, truncation, and reward ranges.
- Add a deterministic evaluation mode so PPO progress can be measured consistently.

## Phase 2: Baselines and Metrics

Goal: establish what PPO must beat.

- Record random-policy success rate, collision rate, timeout rate, mean return, and mean episode length.
- Add a simple scripted heuristic baseline that moves toward the target.
- Save baseline results in a reproducible format such as JSON or CSV.

## Phase 3: PPO Core

Goal: implement the algorithm in readable, inspectable modules.

- Add a policy/value network for discrete actions.
- Add a rollout buffer with observations, actions, rewards, dones, values, log probabilities, and advantages.
- Implement Generalized Advantage Estimation (GAE).
- Implement the PPO clipped surrogate policy loss.
- Add value loss, entropy bonus, gradient clipping, and mini-batch updates.
- Add tests for GAE, discounted returns, buffer shapes, and PPO loss tensor shapes.

## Phase 4: Training Pipeline

Goal: train a first PPO agent end to end.

- Add a CLI training script with configurable seed, total timesteps, learning rate, rollout length, batch size, epochs, gamma, lambda, clip range, value coefficient, and entropy coefficient.
- Save checkpoints and training metrics.
- Add evaluation during training using fixed seeds.
- Keep one minimal default config that can run locally without extra setup.

## Phase 5: Analysis and Comparison

Goal: verify whether the implementation behaves like PPO should.

- Plot training curves for return, success rate, losses, entropy, and approximate KL.
- Compare the from-scratch PPO agent against random and heuristic baselines.
- Optionally compare against Stable-Baselines3 PPO after the custom implementation works.
- Save example gameplay videos or GIFs for trained policies.

## Phase 6: Documentation

Goal: make the implementation explainable.

- Document how each PPO component maps to the paper.
- Add command examples for training, evaluation, plotting, and rendering a trained policy.
- Update the README with first results once PPO training is stable.
