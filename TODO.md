# TODO

## Paper Setup

- [x] Copy the PPO paper into the repository.
- [x] Identify `Proximal Policy Optimization Algorithms.pdf` as the main implementation target.
- [x] Read the PPO paper and extract implementation requirements:
  - Use the clipped surrogate objective, not the KL-penalty variant, as the first implementation target.
  - Compute probability ratios from stored old action log probabilities: `ratio = exp(new_log_prob - old_log_prob)`.
  - Maximize `min(ratio * advantage, clip(ratio, 1 - epsilon, 1 + epsilon) * advantage)`.
  - Add a squared value-function loss against computed returns.
  - Add an entropy bonus to keep discrete-action exploration from collapsing too early.
  - Use truncated Generalized Advantage Estimation with `gamma = 0.99` and `lambda = 0.95`.
  - Optimize each rollout for multiple epochs with shuffled mini-batches and Adam.
  - Track approximate KL and clip fraction as diagnostics; use KL early stopping later if updates become unstable.
  - Start with paper defaults where reasonable: `epsilon = 0.2`, `epochs = 10`, `learning_rate = 3e-4`.
  - Adapt the policy head to Circle Seeker's multi-binary action space with independent Bernoulli distributions.

## Environment

- [x] Add Gymnasium as a dependency.
- [x] Wrap `CircleSeekEnv` as `gymnasium.Env`.
- [x] Define `observation_space` and `action_space`.
- [ ] Keep the existing pygame renderer usable after the Gymnasium wrapper.
- [x] Add tests for Gymnasium-compatible reset and step signatures.
- [ ] Add deterministic evaluation seeds.

## Baselines

- [x] Add a random-policy evaluation script.
- [x] Add a target-seeking heuristic baseline.
- [x] Record baseline metrics: success rate, collision rate, timeout rate, mean return, and mean episode length.

## PPO Implementation

- [x] Choose the neural network dependency: PyTorch.
- [x] Implement a Bernoulli policy network for the multi-binary action space.
- [x] Implement a value network or shared actor-critic network.
- [x] Implement rollout buffer storage.
- [x] Implement environment rollout collection.
- [x] Implement return and advantage computation.
- [x] Implement PPO clipped policy loss.
- [x] Implement value loss and entropy bonus.
- [x] Add mini-batch optimization over collected rollouts.
- [x] Add checkpoint save/load.
- [x] Add unit tests for actor-critic outputs.
- [x] Add unit tests for buffer shapes and GAE math.
- [x] Add unit tests for environment rollout collection.
- [x] Add unit tests for loss outputs.

## Training and Evaluation

- [x] Add `train_ppo.py`.
- [x] Add `evaluate_ppo.py`.
- [x] Save metrics to disk.
- [ ] Add plots for returns, success rate, entropy, value loss, policy loss, and approximate KL.
- [ ] Render or record a trained policy run.

## Documentation

- [ ] Update README with PPO training commands.
- [ ] Document how the code maps to the PPO paper.
- [ ] Add first reproducible training result.
- [ ] Add a preview GIF or video after training is stable.
