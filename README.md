# Circle Seeker RL

[![CI](https://github.com/AlexandreEDMOND/Circle-Seeker-RL/actions/workflows/ci.yml/badge.svg)](https://github.com/AlexandreEDMOND/Circle-Seeker-RL/actions/workflows/ci.yml)

Circle Seeker RL is a small Python reinforcement learning environment prototype.

An agent moves in a top-down 2D world and must reach a green circular target while avoiding moving red circular obstacles. This first version focuses on clean environment mechanics, visual debugging, and a Gymnasium-like API. It does not train a model yet.

## Project Goals

- Build a simple, readable RL environment from scratch.
- Keep the API close to Gymnasium: `reset()`, `step()`, observations, rewards, `terminated`, `truncated`, and `info`.
- Provide a pygame renderer to inspect the simulation.
- Support manual keyboard control and a random policy baseline.
- Keep the codebase easy to extend later with Gymnasium and Stable-Baselines3.

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

```python
from src.env import CircleSeekEnv

env = CircleSeekEnv()
observation = env.reset(seed=123)
observation, reward, terminated, truncated, info = env.step(0)
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
│   ├── renderer.py
│   ├── manual_play.py
│   └── random_play.py
├── tests/
│   └── test_env.py
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
- pygame visualization
- manual and random-play scripts
- unit tests and GitHub Actions CI

Not included yet:

- Gymnasium inheritance
- Stable-Baselines3 integration
- neural network training
- saved experiment tracking

## Roadmap

Planned improvements:

- Wrap `CircleSeekEnv` as a real `gymnasium.Env`.
- Add deterministic evaluation scripts and baseline metrics.
- Train a first PPO agent with Stable-Baselines3.
- Save training curves and example gameplay videos.
- Add a README preview GIF once the training loop is available.

## License

This project is licensed under the MIT License. See `LICENSE` for details.
