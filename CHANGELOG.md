# Changelog

## v1.0.0 - PPO V1 training workflow

Released from commit `605aaa6`.

This release marks the first complete Circle Seeker RL workflow: a custom
environment, Gymnasium adapter, from-scratch PPO implementation, reproducible
training commands, benchmark snapshot, visual artifacts, and tests.

### Included

- 2D circle-seeking environment with moving polygon obstacles and cone-of-vision observations.
- Gymnasium-compatible `CircleSeekGymEnv` with explicit observation and action spaces.
- Random and target-seeking heuristic baseline evaluation.
- From-scratch PPO implementation with rollout storage, GAE, clipped policy loss, value loss, entropy bonus, gradient clipping, checkpointing, and checkpoint evaluation.
- Structured categorical PPO action policy for movement and turning while preserving the environment's 6-bit `MultiBinary` action API.
- Optional KL early stopping, curriculum training, obstacle proximity shaping, periodic evaluation, and best-checkpoint saving.
- Training metric plotting and README media generation scripts.
- Benchmark snapshot for the obstacle-heavy task.
- Unit tests and GitHub Actions CI.

### Reproduce The V1 PPO Checkpoint

The trained checkpoint is intentionally ignored by git under `checkpoints/`.
Regenerate the documented V1 short run with:

```bash
uv run python -m src.train_ppo \
  --total-timesteps 300000 \
  --rollout-steps 2048 \
  --update-epochs 4 \
  --minibatch-size 256 \
  --hidden-size 128 \
  --learning-rate 0.00025 \
  --seed 202 \
  --obstacle-count 4 \
  --max-steps 300 \
  --min-target-distance 120 \
  --distance-reward-coef 0.2 \
  --target-visible-reward-coef 0.02 \
  --target-found-reward-coef 0.2 \
  --no-vision-no-turn-penalty 0.005 \
  --obstacle-proximity-penalty-coef 0.05 \
  --obstacle-proximity-threshold 60 \
  --target-kl 0.03 \
  --curriculum \
  --curriculum-stages 5 \
  --curriculum-start-obstacle-speed 0.0 \
  --curriculum-start-obstacle-radius 0.0 \
  --eval-every 50000 \
  --eval-episodes 10 \
  --eval-seed 900 \
  --checkpoint checkpoints/v1/ppo_v1_short_last.pt \
  --best-checkpoint checkpoints/v1/ppo_v1_short_best.pt \
  --no-progress
```

Evaluate the best checkpoint with:

```bash
uv run python -m src.evaluate_ppo checkpoints/v1/ppo_v1_short_best.pt --episodes 50 --seed 900
```

### Known Gaps

- PPO V1 is documented from a short local run; V2 should validate it across multiple train and evaluation seeds.
- Vectorized environments are not implemented yet.
- Advanced experiment tracking is intentionally out of scope for V1.
