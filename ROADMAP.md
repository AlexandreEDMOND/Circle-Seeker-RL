# Roadmap

Circle Seeker RL reached its first complete implementation milestone with
`v1.0.0`: a custom 2D environment, Gymnasium wrapper, from-scratch PPO training
loop, structured action policy, reproducible commands, tests, and visual
artifacts.

The next roadmap is focused on making the PPO results easier to reproduce,
compare, and improve without expanding the project beyond its small inspectable
scope.

## V1: Complete PPO Workflow

Status: complete and tagged as `v1.0.0`.

Delivered:

- deterministic Circle Seeker environment with moving obstacles and cone vision
- Gymnasium-compatible wrapper with explicit observation and action spaces
- random and target-seeking heuristic baselines
- from-scratch PPO implementation with rollout collection, GAE, clipped policy loss, value loss, entropy bonus, gradient clipping, KL diagnostics, and checkpointing
- structured categorical PPO action policy that avoids contradictory movement and turn actions
- periodic evaluation and best-checkpoint selection
- training metrics, plotting script, README media, and benchmark notes
- unit tests and GitHub Actions CI

## Phase 1: Benchmark Reproducibility

Goal: turn the V1 snapshot into repeatable evidence.

- Add a script that runs train/evaluate matrices across multiple train and eval seeds.
- Store raw per-run metrics under `results/benchmarks/`.
- Add an aggregation step for mean and standard deviation by policy and config.
- Re-run random, heuristic, and PPO V1 using the same obstacle-heavy environment settings.
- Update benchmark docs with the aggregate table and exact commands.

Success criteria:

- A single command can reproduce the benchmark matrix.
- The benchmark report includes raw output paths, seed lists, and aggregate metrics.
- PPO V1 is compared against the heuristic baseline on more than one seed.

## Phase 2: Training Throughput

Goal: make PPO experiments faster and less noisy before deeper tuning.

- Add vectorized environment rollout collection while keeping the single-env path readable.
- Keep checkpoint and metric formats compatible with V1 where practical.
- Measure wall-clock time and sample throughput for single-env vs vectorized runs.

Success criteria:

- Vectorized training passes the existing PPO tests plus focused rollout-shape tests.
- A short benchmark documents the speedup and confirms comparable evaluation behavior.

## Phase 3: PPO Quality Diagnostics

Goal: understand why a training run succeeds or fails.

- Track value-function quality, such as explained variance.
- Compare sampled and deterministic evaluation for the same checkpoint.
- Add optional observation normalization experiments.
- Test learning-rate annealing against the fixed V1 learning rate.

Success criteria:

- Checkpoints contain enough metrics to explain policy collapse, excessive turning, unsafe obstacle proximity, or weak value learning.
- Each training change is evaluated against the V1 benchmark matrix before being kept.

## Phase 4: External Reference Baseline

Goal: compare the from-scratch PPO implementation against a mature PPO implementation.

- Add Stable-Baselines3 as an optional baseline path, not a replacement for the custom PPO code.
- Use the same Gymnasium environment, seeds, obstacle settings, and evaluation metrics.
- Document differences in action modeling, normalization, and training defaults.

Success criteria:

- The project can show where the custom implementation is competitive and where it differs from a standard PPO baseline.
- The external baseline remains isolated from the core from-scratch implementation.

## Phase 5: Documentation Refresh

Goal: keep the public project state accurate as V2 work lands.

- Update the README benchmark table after aggregate V2 results exist.
- Keep release notes in `CHANGELOG.md`.
- Keep this roadmap and `TODO.md` focused on active work, not completed V1 history.
