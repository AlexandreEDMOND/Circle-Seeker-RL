# Circle Seeker RL

[![CI](https://github.com/AlexandreEDMOND/Circle-Seeker-RL/actions/workflows/ci.yml/badge.svg)](https://github.com/AlexandreEDMOND/Circle-Seeker-RL/actions/workflows/ci.yml)

Circle Seeker RL is a small Python reinforcement learning project focused on
implementing Proximal Policy Optimization (PPO) from the paper on a custom 2D
environment.

An agent moves in a top-down 2D world and must reach a green circular target while avoiding moving red polygon obstacles. The agent has its own oriented cone of vision: it can see to the environment bounds, but polygon obstacles block line of sight. The project includes clean environment mechanics, visual debugging, baseline policies, and early from-scratch PPO components.

## Project Goals

- Build a simple, readable RL environment from scratch.
- Keep the API close to Gymnasium: `reset()`, `step()`, observations, rewards, `terminated`, `truncated`, and `info`.
- Provide a pygame renderer to inspect the simulation.
- Support manual keyboard control and a random policy baseline.
- Implement PPO from the original paper before comparing against library baselines.
- Keep the codebase small enough that the PPO objective, advantage estimation,
  rollout collection, and evaluation loop remain easy to inspect.

## Preview

The pygame window displays:

- blue circle: agent
- green circle: target
- red polygons: moving obstacles
- translucent blue cone: agent field of vision
- HUD: current step, reward, cumulative reward, distance to target, episode status

![Circle Seeker RL environment](docs/media/environment_vision.png)

Target-seeking baseline movement:

![Target-seeking agent movement](docs/media/agent_scan_movement.gif)

Trained PPO policy playback:

![Trained PPO agent movement](docs/media/ppo_scan_trained.gif)

Random policy vs trained PPO trajectory on the same seeded environment:

![Random policy vs trained PPO trajectory](docs/media/trajectory_scan_comparison.png)

## Installation

Prerequisites:

- Python 3.11+
- uv
- ffmpeg, only if you want to regenerate the README GIFs

Create the local virtual environment and install dependencies:

```bash
uv sync
```

This creates a local `.venv` and installs the dependencies from `pyproject.toml` / `uv.lock`.

If you prefer pip, the same runtime dependencies are also listed in `requirements.txt`.

## Run

Manual control:

```bash
uv run python src/manual_play.py
```

Controls:

- Arrow keys: move the agent, including diagonal movement when multiple keys are held
- `A` / `D`: rotate the agent's vision direction
- `R`: reset the environment
- `ESC`: quit

Random policy:

```bash
uv run python src/random_play.py
```

Baseline evaluation:

```bash
uv run python -m src.evaluate_baselines --episodes 100 --seed 123
```

Visual baseline playback:

```bash
uv run python -m src.watch_baseline --policy heuristic --seed 123
uv run python -m src.watch_baseline --policy random --seed 123
```

Save baseline metrics to disk:

```bash
uv run python -m src.evaluate_baselines --episodes 100 --seed 123 --output results/baselines.json
```

Train a PPO checkpoint:

```bash
uv run python -m src.train_ppo --total-timesteps 100000 --checkpoint checkpoints/ppo.pt
```

Evaluate a PPO checkpoint:

```bash
uv run python -m src.evaluate_ppo checkpoints/ppo.pt --episodes 50 --seed 123
```

Visual PPO playback:

```bash
uv run python -m src.watch_ppo checkpoints/ppo.pt --seed 123
```

For an easier first PPO run, train without obstacles and allow closer target spawns:

```bash
uv run python -m src.train_ppo --total-timesteps 50000 --rollout-steps 1024 --update-epochs 4 --minibatch-size 128 --obstacle-count 0 --max-steps 300 --min-target-distance 80 --distance-reward-coef 0.2 --target-visible-reward-coef 0.02 --target-found-reward-coef 0.2 --no-vision-no-turn-penalty 0.005 --action-conflict-penalty 0.02 --checkpoint checkpoints/ppo_simple.pt
uv run python -m src.evaluate_ppo checkpoints/ppo_simple.pt --episodes 20 --seed 456
uv run python -m src.watch_ppo checkpoints/ppo_simple.pt --seed 123
```

## Test

Run the full test suite:

```bash
uv run pytest
```

Run a quick source compilation check:

```bash
uv run python -m compileall src
```

Run a small environment smoke test:

```bash
uv run python -c 'from src.env import CircleSeekEnv; env = CircleSeekEnv(); obs = env.reset(seed=123); print(obs.shape); print(env.step(CircleSeekEnv.ACTION_NOOP)[1:])'
```

## Environment API

Main class: `CircleSeekEnv` in `src/env.py`.
Gymnasium adapter: `CircleSeekGymEnv` in `src/gym_env.py`.

```python
from src.env import CircleSeekEnv
from src.gym_env import CircleSeekGymEnv

env = CircleSeekEnv()
observation = env.reset(seed=123)
observation, reward, terminated, truncated, info = env.step(CircleSeekEnv.ACTION_NOOP)

gym_env = CircleSeekGymEnv()
observation, info = gym_env.reset(seed=123)
observation, reward, terminated, truncated, info = gym_env.step(CircleSeekEnv.ACTION_NOOP)
```

Actions:

The environment uses a 6-bit `MultiBinary` action vector so movement and orientation can happen at the same time.

| Index | Meaning |
| --- | --- |
| `0` | move up |
| `1` | move down |
| `2` | move left |
| `3` | move right |
| `4` | turn vision left |
| `5` | turn vision right |

Episode endings:

- `success`: the agent reaches the target
- `collision`: the agent touches an obstacle
- `timeout`: the maximum number of steps is reached

Rewards:

- `+10` for reaching the target
- `-10` for colliding with an obstacle
- `-0.01` step penalty
- optional small shaping bonus when moving closer to the target

Observation vector:

- normalized ray distances across the agent's vision cone
- target visibility flag
- target angle relative to the agent's orientation, only when visible
- normalized distance to target, only when visible
- agent orientation as `cos(theta), sin(theta)`

## Repository Structure

```text
.
├── .github/workflows/ci.yml
├── docs/
│   └── media/
│       ├── agent_scan_movement.gif
│       ├── environment_vision.png
│       ├── ppo_scan_trained.gif
│       └── trajectory_scan_comparison.png
├── scripts/
│   └── generate_readme_media.py
├── src/
│   ├── __init__.py
│   ├── env.py
│   ├── evaluate_ppo.py
│   ├── gym_env.py
│   ├── renderer.py
│   ├── ppo.py
│   ├── train_ppo.py
│   ├── watch_ppo.py
│   ├── manual_play.py
│   └── random_play.py
├── tests/
│   └── test_env.py
├── paper/
│   └── Proximal Policy Optimization Algorithms.pdf
├── ROADMAP.md
├── TODO.md
├── LICENSE
├── pyproject.toml
├── requirements.txt
├── uv.lock
└── README.md
```

## Current Scope

Implemented:

- 2D environment mechanics
- moving polygon obstacles with wall and obstacle-to-obstacle bounce
- oriented cone-of-vision raycasting with obstacle occlusion
- minimum spawn distance between agent and target
- reward function and episode termination
- partial numeric observations for future RL training
- Gymnasium adapter with explicit observation and action spaces
- random and target-seeking heuristic baseline evaluation
- from-scratch PPO actor-critic implementation with rollout buffer, GAE returns,
  environment rollout collection, clipped policy objective, value loss, entropy
  bonus, mini-batch updates, checkpoint saving, and checkpoint evaluation
- pygame visualization
- manual and random-play scripts
- PyTorch actor-critic model
- PPO rollout buffer with GAE returns and advantages
- unit tests and GitHub Actions CI
- local PPO paper copy under `paper/`

Not included yet:

- vectorized environments
- advanced experiment tracking
- tuned hyperparameters for the obstacle-heavy task

## Paper Reference

The copied paper file is under `paper/`.
The PPO implementation target is:

- `paper/Proximal Policy Optimization Algorithms.pdf`

## Roadmap

The implementation roadmap now targets a from-scratch PPO implementation for this environment.

See `ROADMAP.md` for the staged plan and `TODO.md` for the current task list.

## License

This project is licensed under the MIT License. See `LICENSE` for details.
