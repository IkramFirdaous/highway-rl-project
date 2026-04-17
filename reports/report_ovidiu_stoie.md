# Individual Report — [Prénom NOM]

**Project:** Reinforcement Learning on highway-v0
**Course:** Mention IA — CentraleSupelec
**Instructor:** Hédi Hadiji

---

## 1. Personal Contribution

My contribution covered three components: the experience replay buffer, the evaluation framework, and the results analysis pipeline. I implemented `core/replay_buffer.py` and `core/evaluate.py`, and I wrote the comparison logging in `main.py` (`log_comparison`). I also analyzed the final results and identified the key behavioral patterns across agents and seeds.

### 1.1 Experience Replay Buffer

The replay buffer stores transitions as a fixed-capacity deque. When the buffer is full, new transitions overwrite the oldest ones (FIFO policy). Sampling is uniform without replacement over the current buffer contents.

The implementation is intentionally minimal. The `sample` method returns five NumPy arrays with explicit dtypes (`float32` for states, rewards, and dones; `int64` for actions). This type discipline avoids silent casting issues when the arrays are converted to PyTorch tensors: `torch.FloatTensor` on a `float64` array performs an implicit copy, which can slow down the training loop under large batch sizes.

```python
def sample(self, batch_size):
    batch = random.sample(self.buffer, batch_size)
    states, actions, rewards, next_states, dones = zip(*batch)
    return (
        np.array(states,      dtype=np.float32),
        np.array(actions,     dtype=np.int64),
        np.array(rewards,     dtype=np.float32),
        np.array(next_states, dtype=np.float32),
        np.array(dones,       dtype=np.float32),
    )
```

### 1.2 Evaluation Framework

The evaluation function runs exactly `n_episodes` episodes, cycling through the provided seeds in round-robin order:

```python
for i in range(n_episodes):
    seed = seeds[i % len(seeds)]
```

With `n_episodes = 50` and `seeds = [42, 43, 44]`, this yields 17 episodes on seed 42, 17 on seed 43, and 16 on seed 44. The seeds are not reshuffled between episodes, which ensures that the evaluation is deterministic and reproducible across runs.

An earlier version of the evaluation ran a separate loop per seed (e.g., `for seed in seeds: for _ in range(n_episodes // len(seeds)): ...`), which produced 48 episodes instead of 50 due to integer division. The current implementation corrects this.

### 1.3 Results Analysis

I collected the outputs from all nine model-seed combinations and formatted them into `results/comparison.json` and `results/comparison.txt`. The comparison table includes per-seed mean and standard deviation, as well as the cross-seed average for each algorithm.

---

## 2. Experiments

I ran the full evaluation for all three agents after training was complete: DQN (seeds 42/43/44), DDQN (seeds 42/43/44), and SB3 (seeds 42/43/44). Each evaluation call executes 50 independent episodes.

I also investigated the anomalous result on SB3 seed 43 (std = 0.0), which I identified during the analysis pass. The zero standard deviation indicates a perfectly deterministic evaluation: the same initial state (fixed by the seed) combined with a deterministic policy (greedy, epsilon = 0) and a deterministic environment produces identical trajectories across all 50 episodes. This is not a sign of failure — the agent does obtain a non-trivial mean reward of 20.45 — but it does indicate that the policy has not generalized beyond this specific initial configuration.

---

## 3. Results and Analysis

### 3.1 Cross-Seed Comparison

| Algorithm | Avg mean | Avg std |
|-----------|----------|---------|
| DQN       | 19.489   | 3.374   |
| DDQN      | 19.078   | 5.826   |
| SB3       | 18.116   | 1.681   |

The three algorithms perform comparably on average (within 1.4 reward units of each other), but differ substantially in cross-seed variance. DQN achieves consistent results on seeds 43 and 44 (std below 1.2) while failing on seed 42 (std 8.4). DDQN shows the inverse pattern: it is the best-performing single model (seed 42, mean 22.4) but also one of the most variable (seeds 43 and 44, std 8.2 and 7.3). SB3 has the lowest average standard deviation but also the lowest mean reward.

### 3.2 The Seed 42 Failure Mode

Seed 42 is consistently the hardest initial configuration across all three algorithms. Qualitatively, this seed appears to place the ego vehicle in a dense traffic cluster at the start of the episode, leaving little margin for acceleration or lane changes before a collision becomes likely. Agents trained to maximize speed reward adopt an aggressive policy that works well in open traffic but fails in this constrained initial state.

This is a classical failure mode in DQN: the policy is locally optimal for the distribution of states seen during training, but not robust to the tail of the state distribution. Seed 42 represents such a tail event.

### 3.3 Convergence Pattern

The metrics saved in `results/metrics/` show that DQN training on seeds 43 and 44 exhibits a characteristic two-phase pattern: a flat exploration phase for the first ~200 episodes (rewards near zero, high collision rate), followed by a sharp transition to a regime of sustained positive rewards. This transition corresponds to the point where epsilon has decayed sufficiently that the greedy policy begins to dominate, and the replay buffer has accumulated enough high-reward transitions to provide useful gradient signal.

On seed 42, this transition is delayed and the post-transition variance remains high, consistent with the evaluation results.

---

## 4. What Worked and What Failed

**What worked:**
- The evaluation cycle over seeds (round-robin) produced a balanced comparison: each algorithm is evaluated against the same distribution of initial conditions.
- The dtype discipline in the replay buffer prevented silent type casting errors that had caused unexpected slowdowns in earlier prototyping.

**What failed:**
- The initial evaluation loop had an off-by-one error: 48 episodes were run instead of 50 due to integer division when distributing episodes across seeds. This was detected by checking `len(rewards)` in the output and corrected.
- An earlier version of `evaluate.py` closed the environment inside the episode loop instead of after it. This caused gymnasium to throw a warning on every episode because the environment was reset after being closed. The environment is now closed once per agent-seed pair.

---

## 5. Limitations and Perspectives

The main limitation of the current evaluation setup is that 50 episodes over 3 seeds may not provide sufficient statistical power to conclude that one algorithm is significantly better than another. The confidence intervals around the mean rewards overlap for all three algorithms. A proper statistical comparison would require more seeds, more episodes per seed, or a paired test that accounts for the correlation structure across seeds.

A second limitation is that the evaluation metric — cumulative episode reward — is a composite of speed reward, collision penalty, and lane change penalty. It does not distinguish between an agent that achieves high reward by driving fast and one that achieves it by surviving long without crashing. Decomposing the reward signal into its components would provide a more interpretable picture of each policy's behavior.

From a practical perspective, the most actionable direction for improving the evaluation is to add a collision rate metric alongside the reward. In highway driving, safety is the primary objective; reward is a proxy for it. An agent with a slightly lower mean reward but a much lower collision rate may be preferable in practice.

---

## References

- Mnih, V. et al. (2015). Human-level control through deep reinforcement learning. *Nature*, 518, 529–533.
- Leurent, E. (2018). highway-env: An environment for autonomous driving decision-making. GitHub.
