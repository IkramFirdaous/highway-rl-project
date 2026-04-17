# Individual Report — Ikram Firdaous

**Project:** Reinforcement Learning on highway-v0
**Course:** Mention IA — CentraleSupelec
**Instructor:** Hédi Hadiji

---

## 1. Personal Contribution

My contribution focused on two components: the design and implementation of the custom Q-network architecture, and the overall training and evaluation orchestration.

### 1.1 CNN Q-Network Architecture

The core design decision was to treat the (10, 5) kinematics observation as a 2D spatial matrix rather than a flat vector. Each row represents a vehicle (10 total) and each column a feature (presence, x, y, vx, vy). Processing this structure with convolutional layers allows the network to detect spatial patterns across vehicles — for example, recognizing a gap in the lane ahead — in a way that a flat MLP cannot.

The architecture consists of two convolutional layers (1→16 channels, then 16→32, kernel 3×3, padding 1), each followed by ReLU activation, feeding into a two-layer fully-connected head with 256 hidden units. No batch normalization is used: it is absent from the original Mnih et al. (2015) paper, and in practice it introduces instability in RL due to the non-stationarity of the training data distribution.

The output dimension of the convolutional block is computed dynamically at initialization via a dummy forward pass. This makes the architecture portable across different observation shapes without requiring manual dimension calculation.

### 1.2 Training Loop and Orchestration

I implemented the training loop in `core/train_dqn.py` and the main entry point in `main.py`. The training loop follows the standard DQN procedure: at each step, the agent selects an action under epsilon-greedy, stores the transition in the replay buffer, and triggers a gradient update. Epsilon is decayed multiplicatively per episode (factor 0.99) until reaching the minimum value of 0.05.

The `main.py` file exposes a command-line interface (`--mode train/eval/ddqn/sb3/all`, `--seed`) that allows any combination of training, evaluation, and plotting in a single invocation. Checkpoints and metrics are saved deterministically per seed to `results/checkpoints/` and `results/metrics/`.

---

## 2. Experiments

I trained the DQN agent on seeds 42, 43, and 44 for 2 000 episodes each. The initial training budget of 500 episodes was insufficient: the agent had not yet collected enough experience for the replay buffer to drive stable learning. Extending to 2 000 episodes produced a clear improvement in convergence, particularly on seeds 43 and 44.

I also ran a diagnostic training run where the device was logged at initialization to confirm GPU usage. Training on a CUDA device reduced wall time substantially compared to CPU.

---

## 3. Results and Analysis

| Seed | Mean reward | Std |
|------|-------------|-----|
| 42   | 15.157      | 8.391 |
| 43   | 21.540      | 0.549 |
| 44   | 21.771      | 1.183 |
| Average | 19.489  | —   |

The results reveal a clear disparity between seed 42 and the other two seeds. Seeds 43 and 44 show tight distributions (std below 1.2), indicating a reliable policy that avoids collisions in most evaluation episodes. Seed 42 produces a much wider distribution (std 8.4), suggesting that the initial traffic configuration for this seed generates scenarios where the agent's greedy policy fails on a non-trivial fraction of episodes.

The likely cause is that seed 42 initializes vehicles in a configuration where the ego vehicle has little room to maneuver in the first few time steps, leading to early termination by collision. A policy that maximizes speed in normal conditions may not be robust enough to handle these constrained initial states.

---

## 4. What Worked and What Failed

**What worked:**
- The CNN architecture converged reliably on seeds 43 and 44, achieving mean rewards above 21 with low variance.
- Multiplicative epsilon decay produced a smooth exploration-exploitation transition.
- The modular design (separate `config.py`, `env.py`, `dqn_agent.py`) made it straightforward to reuse components for the DDQN extension.

**What failed:**
- An early bug introduced duplicate initialization in `DQNAgent.__init__`: the device, policy network, and target network were each created twice in sequence, with an inconsistent print statement between the two. This did not cause incorrect behavior (the second initialization simply overwrote the first), but it indicated a lack of care in the code. This was corrected.
- The initial epsilon decay schedule (0.99 per episode over 500 episodes) was too slow: epsilon remained above 0.2 for the entire run, preventing the agent from fully exploiting its learned policy during evaluation.

---

## 5. Limitations and Perspectives

The main limitation of the current DQN is its sensitivity to initialization. Seed 42 consistently underperforms regardless of whether DQN or DDQN is used (DQN: 15.16, DDQN: 22.41), while seeds 43 and 44 converge reliably. This variance across seeds is a known issue in DQN training and can be partially addressed by longer training, prioritized experience replay, or ensemble evaluation.

A second limitation is the training budget. 2 000 episodes of 30 seconds at 15 steps/second represent approximately 60 000 environment steps — on the lower end for DQN on a continuous-state environment. Training longer would likely reduce variance across seeds.

From an architectural perspective, the CNN was chosen to preserve the spatial structure of the observation. A potential improvement would be a self-attention mechanism over the vehicle rows, which would better capture permutation invariance: the ordering of vehicles in the observation matrix is arbitrary, and a CNN treats row order as meaningful. A transformer-based Q-network would handle this more naturally.

---

## References

- Mnih, V. et al. (2015). Human-level control through deep reinforcement learning. *Nature*, 518, 529–533.
- van Hasselt, H., Guez, A., Silver, D. (2016). Deep Reinforcement Learning with Double Q-learning. *AAAI*.
