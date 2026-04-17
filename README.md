# Reinforcement Learning Project — highway-v0

**CentraleSupelec — Mention IA**
Course instructor: Hédi Hadiji

**Group members:** Ikram Firdaous, Aya [NOM], [Prénom NOM], [Prénom NOM]

---

## Overview

This project trains and evaluates reinforcement learning agents on the `highway-v0` environment from the [highway-env](https://highway-env.farama.org/) collection. It covers three components: a DQN agent implemented from scratch, a reference model trained with Stable-Baselines3 (SB3), and a Double-DQN (DDQN) extension.

All agents share the same environment configuration and are evaluated identically: 50 episodes over three fixed seeds (42, 43, 44), with mean reward and standard deviation reported.

---

## Environment

| Parameter | Value |
|---|---|
| Environment | `highway-v0` |
| Observation | Kinematics — 10 vehicles × 5 features |
| Action space | `DiscreteMetaAction` — 5 actions |
| Lanes | 4 |
| Traffic density | 45 vehicles, density 1.0 |
| Episode duration | 30 s |
| Collision reward | -1.5 |
| High-speed reward | 0.7 |
| Reward normalization | enabled |

The environment configuration is fixed in `core/config.py` and shared across all agents.

---

## Repository Structure

```
core/
  config.py            # shared environment config and hyperparameters
  env.py               # make_env() factory
  dqn_model_cnn.py     # CNN Q-network architecture
  replay_buffer.py     # uniform experience replay
  dqn_agent.py         # DQNAgent (select_action, update, save, load)
  ddqn_agent.py        # DDQNAgent (overrides update with Double-DQN target)
  train_dqn.py         # DQN training loop
  train_ddqn.py        # Double-DQN training loop
  train_sb3.py         # Stable-Baselines3 training
  evaluate.py          # evaluation over 50 runs × 3 seeds

main.py                # CLI entry point (train / eval / ddqn / sb3 / all)
render_episode.py      # rollout rendering and video export

results/
  checkpoints/         # saved model weights (.pt and .zip)
  metrics/             # per-episode rewards and losses (JSON)
  plots/               # training curves
  videos/              # recorded rollouts (GIF/MP4)
  comparison.json      # evaluation summary
  comparison.txt       # evaluation summary (human-readable)
```

---

## Architecture

### DQN (custom implementation)

The Q-network processes the (10, 5) kinematic observation as a 2D spatial input, passed through two convolutional layers (1→16→32 channels, kernel 3×3, padding 1) followed by a fully-connected head (256 hidden units). This design, inspired by Mnih et al. (2015), allows the network to capture spatial relationships between vehicles rather than treating the observation as a flat vector.

Key components:
- **Replay buffer**: uniform sampling, capacity 50 000 transitions
- **Target network**: hard sync every 500 gradient steps
- **Epsilon-greedy**: linear decay from 1.0 to 0.05 (multiplicative factor 0.99/episode)
- **Optimizer**: Adam, learning rate 5×10⁻⁴, batch size 128, γ = 0.99

### Double-DQN (extension)

DDQNAgent inherits DQNAgent and overrides `update()`. The sole algorithmic change is in the target computation: the policy network selects the best next action while the target network evaluates it. This decoupling reduces the positive bias introduced by the max operator in standard DQN (van Hasselt et al., 2016). Gradient clipping (max norm 10) is added for stability.

### Stable-Baselines3

SB3 DQN with `MlpPolicy` is trained under an identical budget (40 000 steps, ≈ 1 300 episodes) and identical hyperparameters. The MlpPolicy processes the flattened observation without convolutional layers, which removes the spatial inductive bias present in the custom architecture.

---

## Hyperparameters

| Hyperparameter | Value |
|---|---|
| Learning rate | 5×10⁻⁴ |
| Discount factor γ | 0.99 |
| Replay buffer size | 50 000 |
| Batch size | 128 |
| Target network update | every 500 steps |
| ε start / end | 1.0 / 0.05 |
| ε decay (per episode) | 0.99 |
| Training episodes (DQN/DDQN) | 2 000 |
| Training timesteps (SB3) | 40 000 |

---

## Results

Evaluation: 50 episodes per model, cycling over seeds 42, 43, 44.

| Model | Mean reward | Std | Runs |
|---|---|---|---|
| DQN seed=42 | 15.157 | 8.391 | 50 |
| DQN seed=43 | 21.540 | 0.549 | 50 |
| DQN seed=44 | 21.771 | 1.183 | 50 |
| **DQN average** | **19.489** | — | 150 |
| DDQN seed=42 | 22.414 | 2.038 | 50 |
| DDQN seed=43 | 16.964 | 8.189 | 50 |
| DDQN seed=44 | 17.857 | 7.252 | 50 |
| **DDQN average** | **19.078** | — | 150 |
| SB3 seed=42 | 13.330 | 4.893 | 50 |
| SB3 seed=43 | 20.455 | 0.000 | 50 |
| SB3 seed=44 | 20.562 | 0.150 | 50 |
| **SB3 average** | **18.116** | — | 150 |

Training curves are saved in `results/plots/training_curves.png`. Rollout videos are in `results/videos/`.

---

## Reproducing the Results

**Installation:**

```bash
conda create -n highway-rl python=3.12
conda activate highway-rl
pip install torch gymnasium highway-env stable-baselines3 matplotlib
```

**Training:**

```bash
# Train all agents on all seeds
python main.py --mode all

# Train a single agent on a specific seed
python main.py --mode sb3 --seed 43
python main.py --mode ddqn --seed 42
```

**Evaluation:**

```bash
python main.py --mode eval
```

**Rendering a rollout:**

```bash
python render_episode.py --agent dqn --seed 43 --render video
python render_episode.py --agent sb3 --seed 44 --render human
```

---

## References

- Mnih, V. et al. (2015). Human-level control through deep reinforcement learning. *Nature*, 518, 529–533.
- van Hasselt, H., Guez, A., Silver, D. (2016). Deep Reinforcement Learning with Double Q-learning. *AAAI*.
- Leurent, E. (2018). highway-env: An environment for autonomous driving decision-making. GitHub.
- Raffin, A. et al. (2021). Stable-Baselines3: Reliable Reinforcement Learning Implementations. *JMLR*, 22(268).
