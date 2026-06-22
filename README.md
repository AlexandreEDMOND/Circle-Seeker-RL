# Circle Seeker RL

Premiere ebauche d'un environnement 2D pour du Reinforcement Learning en Python.

Un agent controle un petit cercle bleu vu de dessus. Il doit atteindre une cible verte placee aleatoirement, tout en evitant des obstacles rouges mobiles qui rebondissent sur les murs.

Cette version ne contient pas encore d'entrainement RL ni de reseau de neurones. Le but est de poser une base claire : environnement, visualisation pygame, controle manuel et API proche de Gymnasium.

## Installation avec uv

Prerequis :

- Python 3.11+
- uv

Cree le `.venv` local et installe les dependances :

```bash
uv sync
```

## Lancer le mode manuel

```bash
uv run python src/manual_play.py
```

Controles :

- Fleches : deplacer l'agent
- R : reset l'environnement
- ESC : quitter

## Lancer la politique aleatoire

```bash
uv run python src/random_play.py
```

## Verification rapide

```bash
uv run python -m compileall src
uv run python -c 'from src.env import CircleSeekEnv; env = CircleSeekEnv(); obs = env.reset(seed=123); print(obs.shape); print(env.step(0))'
```

## API de l'environnement

La classe principale est `CircleSeekEnv` dans `src/env.py`.

Actions :

- `0` : noop
- `1` : up
- `2` : down
- `3` : left
- `4` : right

Methodes principales :

- `reset(seed=None) -> observation`
- `step(action) -> observation, reward, terminated, truncated, info`
- `get_observation() -> observation`

Recompenses :

- `+10` si l'agent atteint la cible
- `-10` si l'agent touche un obstacle
- `-0.01` a chaque step
- petit bonus si l'agent se rapproche de la cible

Observation numerique :

- position normalisee de l'agent
- position relative de la cible par rapport a l'agent
- pour chaque obstacle : position relative et vitesse
- distance normalisee a la cible

## Prochaines etapes possibles

- Ajouter des tests unitaires sur les collisions et les rewards.
- Ajouter une classe compatible directement avec `gymnasium.Env`.
- Brancher une politique RL avec Stable-Baselines3 plus tard.
