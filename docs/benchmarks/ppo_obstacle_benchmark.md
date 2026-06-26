# PPO Obstacle Benchmark

This benchmark records a short local tuning pass for the obstacle-heavy task.
It is intentionally reproducible and modest rather than presented as a solved
result.

## Environment

- Evaluation episodes: `30`
- Evaluation seed: `900`
- Obstacle count: `4`
- Max steps: `300`
- Minimum target distance: `120`

## Results

| Policy | Train seeds | Success rate | Collision rate | Timeout rate | Mean return | Mean episode length |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Random | n/a | 0.000 | 0.500 | 0.500 | -7.071 | 207.6 |
| Target-seeking heuristic | n/a | 0.767 | 0.233 | 0.000 | 4.847 | 51.9 |
| PPO simple checkpoint, no-obstacle eval | 123 | 0.200 | 0.000 | 0.800 | -0.582 | 259.1 |
| PPO obstacles, direct 20k tuning | 101, 202, 303 | 0.011 | 0.344 | 0.644 | -5.565 | 222.7 |

The PPO obstacle row is the mean over three independently trained checkpoints.
The best individual PPO obstacle seed was `202`, with `0.033` success rate,
`0.300` collision rate, and mean return `-4.826`.

The short PPO obstacle tuning run does not yet beat the scripted heuristic and
only slightly improves over random return. That makes the next implementation
target clear: improve the action policy/evaluation behavior before scaling
training time.

Implementation note: these benchmark numbers were produced before PPO switched
from six independent Bernoulli action bits to a structured categorical policy.
The environment still receives the same 6-bit `MultiBinary` action, but PPO now
samples movement and turn categories separately and converts them at the
environment boundary. New obstacle results should be recorded separately from
the table above.

## Reproduction Commands

Baseline evaluation:

```bash
uv run python -m src.evaluate_baselines --episodes 30 --seed 900 --obstacle-count 4 --max-steps 300 --min-target-distance 120
```

PPO simple checkpoint evaluation:

```bash
uv run python -m src.evaluate_ppo checkpoints/ppo_simple.pt --episodes 30 --seed 900
```

PPO obstacle tuning and evaluation:

```bash
mkdir -p checkpoints/benchmarks results/benchmarks

for seed in 101 202 303; do
  uv run python -m src.train_ppo \
    --total-timesteps 20000 \
    --rollout-steps 1024 \
    --update-epochs 4 \
    --minibatch-size 128 \
    --hidden-size 64 \
    --seed "$seed" \
    --obstacle-count 4 \
    --max-steps 300 \
    --min-target-distance 120 \
    --distance-reward-coef 0.2 \
    --target-visible-reward-coef 0.02 \
    --target-found-reward-coef 0.2 \
    --no-vision-no-turn-penalty 0.005 \
    --action-conflict-penalty 0.02 \
    --target-kl 0.03 \
    --checkpoint "checkpoints/benchmarks/ppo_obstacles_seed${seed}.pt" \
    > "results/benchmarks/train_obstacles_seed${seed}.json"

  uv run python -m src.evaluate_ppo \
    "checkpoints/benchmarks/ppo_obstacles_seed${seed}.pt" \
    --episodes 30 \
    --seed 900 \
    > "results/benchmarks/eval_obstacles_seed${seed}.json"
done
```

Curriculum variant tried during tuning:

```bash
uv run python -m src.train_ppo \
  --total-timesteps 30000 \
  --rollout-steps 1024 \
  --update-epochs 4 \
  --minibatch-size 128 \
  --hidden-size 64 \
  --seed 101 \
  --obstacle-count 4 \
  --max-steps 300 \
  --min-target-distance 80 \
  --distance-reward-coef 0.2 \
  --target-visible-reward-coef 0.02 \
  --target-found-reward-coef 0.2 \
  --no-vision-no-turn-penalty 0.005 \
  --action-conflict-penalty 0.02 \
  --target-kl 0.03 \
  --curriculum \
  --curriculum-stages 4 \
  --curriculum-start-obstacle-speed 0.0 \
  --curriculum-start-obstacle-radius 0.0 \
  --checkpoint checkpoints/benchmarks/ppo_obstacles_curriculum_seed101.pt
```

The curriculum variant reached `0.000` deterministic success over 30 evaluation
episodes and collapsed to a no-movement deterministic policy, so it was not used
for the table.
