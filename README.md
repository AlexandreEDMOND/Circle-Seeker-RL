# Circle Seeker RL

[![CI](https://github.com/AlexandreEDMOND/Circle-Seeker-RL/actions/workflows/ci.yml/badge.svg)](https://github.com/AlexandreEDMOND/Circle-Seeker-RL/actions/workflows/ci.yml)

Circle Seeker RL is a small Python reinforcement learning project focused on
implementing Proximal Policy Optimization (PPO) on a custom 2D environment.

An agent moves in a top-down 2D world and must reach a green circular target while avoiding moving red circular obstacles. This first version focuses on clean environment mechanics, visual debugging, and a Gymnasium-like API. It does not train a model yet.

## Project Goals

- Build a simple, readable RL environment from scratch.
- Keep the API close to Gymnasium: `reset()`, `step()`, observations, rewards, `terminated`, `truncated`, and `info`.
- Provide a pygame renderer to inspect the simulation.
- Support manual keyboard control and a random policy baseline.
- Implement PPO from the research paper before comparing against library baselines.
- Keep the codebase small enough that the PPO objective, advantage estimation,
  rollout collection, and evaluation loop remain easy to inspect.

## Preview

The pygame window displays:

- blue circle: agent
- green circle: target
- red circles: moving obstacles
- HUD: current step, reward, cumulative reward, distance to target, episode status

## Installation

Prerequisites:

- Python 3.11+
- uv

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

- Arrow keys: move the agent
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
uv run python -c 'from src.env import CircleSeekEnv; env = CircleSeekEnv(); obs = env.reset(seed=123); print(obs.shape); print(env.step(0)[1:])'
```

## Environment API

Main class: `CircleSeekEnv` in `src/env.py`.
Gymnasium adapter: `CircleSeekGymEnv` in `src/gym_env.py`.

```python
from src.env import CircleSeekEnv
from src.gym_env import CircleSeekGymEnv

env = CircleSeekEnv()
observation = env.reset(seed=123)
observation, reward, terminated, truncated, info = env.step(0)

gym_env = CircleSeekGymEnv()
observation, info = gym_env.reset(seed=123)
observation, reward, terminated, truncated, info = gym_env.step(0)
```

Actions:

| Action | Meaning |
| --- | --- |
| `0` | no-op |
| `1` | up |
| `2` | down |
| `3` | left |
| `4` | right |

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

- normalized agent position
- target position relative to the agent
- for each obstacle: relative position and velocity
- normalized distance to target

## Repository Structure

```text
.
├── .github/workflows/ci.yml
├── src/
│   ├── __init__.py
│   ├── env.py
│   ├── gym_env.py
│   ├── renderer.py
│   ├── manual_play.py
│   └── random_play.py
├── tests/
│   └── test_env.py
├── research/
│   └── Papier de recherche IA/
│       ├── Attention is all you need.pdf
│       ├── DeepSeekk-R1.pdf
│       └── Proximal Policy Optimization Algorithms.pdf
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
- moving circular obstacles with wall bounce
- reward function and episode termination
- numeric observations for future RL training
- Gymnasium adapter with explicit observation and action spaces
- random and target-seeking heuristic baseline evaluation
- pygame visualization
- manual and random-play scripts
- unit tests and GitHub Actions CI
- local research paper copies under `research/Papier de recherche IA/`

Not included yet:

- Gymnasium inheritance
- PPO rollout buffer
- policy/value neural networks
- clipped surrogate objective and GAE
- training and evaluation scripts
- saved experiment tracking

## Research References

The copied research files are under `research/Papier de recherche IA/`.
The PPO implementation target is:

- `research/Papier de recherche IA/Proximal Policy Optimization Algorithms.pdf`

## Roadmap

The implementation roadmap now targets a from-scratch PPO implementation for this environment.

See `ROADMAP.md` for the staged plan and `TODO.md` for the current task list.

## License

This project is licensed under the MIT License. See `LICENSE` for details.
