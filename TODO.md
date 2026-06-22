# TODO

## Research Setup

- [x] Copy the research paper folder into the repository.
- [x] Identify `Proximal Policy Optimization Algorithms.pdf` as the main implementation target.
- [ ] Read the PPO paper and extract implementation requirements:
  - clipped surrogate objective
  - value function loss
  - entropy bonus
  - GAE or advantage estimation choice
  - policy update epochs and mini-batches
  - KL diagnostics or early stopping criteria

## Environment

- [x] Add Gymnasium as a dependency.
- [x] Wrap `CircleSeekEnv` as `gymnasium.Env`.
- [x] Define `observation_space` and `action_space`.
- [ ] Keep the existing pygame renderer usable after the Gymnasium wrapper.
- [x] Add tests for Gymnasium-compatible reset and step signatures.
- [ ] Add deterministic evaluation seeds.

## Baselines

- [ ] Add a random-policy evaluation script.
- [ ] Add a target-seeking heuristic baseline.
- [ ] Record baseline metrics: success rate, collision rate, timeout rate, mean return, and mean episode length.

## PPO Implementation

- [ ] Choose the neural network dependency, most likely PyTorch.
- [ ] Implement a categorical policy network for the discrete action space.
- [ ] Implement a value network or shared actor-critic network.
- [ ] Implement rollout collection.
- [ ] Implement return and advantage computation.
- [ ] Implement PPO clipped policy loss.
- [ ] Implement value loss and entropy bonus.
- [ ] Add mini-batch optimization over collected rollouts.
- [ ] Add checkpoint save/load.
- [ ] Add unit tests for buffer shapes, GAE math, and loss outputs.

## Training and Evaluation

- [ ] Add `train_ppo.py`.
- [ ] Add `evaluate_policy.py`.
- [ ] Save metrics to disk.
- [ ] Add plots for returns, success rate, entropy, value loss, policy loss, and approximate KL.
- [ ] Render or record a trained policy run.

## Documentation

- [ ] Update README with PPO training commands.
- [ ] Document how the code maps to the PPO paper.
- [ ] Add first reproducible training result.
- [ ] Add a preview GIF or video after training is stable.
