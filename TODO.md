# TODO

## V1 Release

- [x] Tag the first complete PPO workflow as `v1.0.0`.
- [x] Document the V1 scope, benchmark snapshot, and reproduction commands.
- [x] Keep trained checkpoints out of git and document how to regenerate them.
- [x] Publish the GitHub release for `v1.0.0`.

## V2: Reproducible Benchmarks

- [ ] Add a benchmark matrix script for multi-seed PPO training and evaluation.
- [ ] Save benchmark outputs as JSON or CSV under `results/benchmarks/`.
- [ ] Aggregate mean and standard deviation for success, collision, timeout, return, episode length, obstacle proximity, and action diagnostics.
- [ ] Re-run random, heuristic, and PPO V1 evaluations on the same seed matrix.
- [ ] Update `docs/benchmarks/` with the multi-seed V1 results.

## V2: PPO Training Improvements

- [ ] Add vectorized environment support for faster and less correlated rollouts.
- [ ] Evaluate observation normalization for the partial-visibility observation vector.
- [ ] Test learning-rate annealing against the current fixed learning rate.
- [ ] Compare deterministic and sampled evaluation to detect brittle argmax policies.
- [ ] Track value explained variance or another value-function quality diagnostic.

## V2: External Baseline

- [ ] Add an optional Stable-Baselines3 PPO comparison after the custom implementation remains the primary path.
- [ ] Keep the external baseline in a separate script or optional dependency group.
- [ ] Compare SB3 and from-scratch PPO on the same environment, seeds, and metrics.

## V2: Documentation

- [ ] Refresh the README benchmark table after the multi-seed run.
- [ ] Document the benchmark matrix commands and expected output files.
- [ ] Keep `ROADMAP.md` aligned with the active V2 plan.
