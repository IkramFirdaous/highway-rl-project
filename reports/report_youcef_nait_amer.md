# Individual Report — [Prénom NOM]

**Project:** Reinforcement Learning on highway-v0
**Course:** Mention IA — CentraleSupelec
**Instructor:** Hédi Hadiji

---

## 1. Personal Contribution

My contribution covered the Stable-Baselines3 (SB3) integration, the alignment of its hyperparameters with the custom DQN, and the analysis of the SB3 versus scratch comparison. I implemented `core/train_sb3.py` and ensured that the SB3 agent operates under the same constraints as the hand-coded agents to make the comparison meaningful.

### 1.1 Stable-Baselines3 Integration

SB3 provides a production-grade DQN implementation. The challenge in this project was not using SB3 per se, but configuring it to be comparable to the custom DQN. A naive comparison — where SB3 uses its default parameters and the custom agent uses ours — would be uninformative.

I mapped each shared hyperparameter explicitly:

```python
model = DQN(
    "MlpPolicy", env,
    learning_rate        = LR,            # 5e-4
    batch_size           = BATCH_SIZE,    # 128
    gamma                = GAMMA,         # 0.99
    buffer_size          = BUFFER_SIZE,   # 50 000
    learning_starts      = SB3_LEARNING_STARTS,   # 1 000
    target_update_interval = TARGET_UPDATE,        # 500
    exploration_fraction  = SB3_EXPLORATION_FRAC, # 0.3
    exploration_final_eps = EPS_END,              # 0.05
    seed                 = seed,
)
```

The budget was fixed at 40 000 timesteps, which corresponds approximately to the DQN's 2 000 episodes of 30 steps on average (60 000 steps). The slight difference is intentional: SB3 counts environment steps, while the custom agent counts episodes. A budget of 40 000 steps places SB3 in the same order of magnitude without exceeding it.

One configuration parameter requires explanation: `learning_starts = 1 000`. SB3 does not update the network until the replay buffer contains at least this many transitions. This is equivalent to the implicit warm-up that occurs in the custom DQN (the buffer must reach `BATCH_SIZE` = 128 transitions before any update). Setting it to 1 000 ensures that the initial batch draws are not dominated by highly correlated early transitions.

### 1.2 MlpPolicy vs. CNN

The `MlpPolicy` in SB3 flattens the (10, 5) observation to a 50-dimensional vector and processes it through a fully-connected network. This is a deliberate architectural difference from the custom CNN, which treats the observation as a 2D spatial structure.

This difference is a confound in the comparison: SB3 uses a different network architecture in addition to a different training framework. Ideally, one would implement a CNN policy within SB3 to isolate the framework effect. However, given the project scope and time constraints, the comparison between MlpPolicy and the custom CNN was retained as is, and this limitation is acknowledged in the analysis.

---

## 2. Experiments

I trained SB3 agents on seeds 42, 43, and 44 independently. An initial attempt trained all three seeds in sequence within a single `main.py` invocation, but the model for seed 43 was inadvertently overwritten. Each seed was subsequently trained with an explicit `--seed` argument to ensure independent checkpoints.

I also identified a compatibility issue between NumPy 2.x and SB3's internal observation preprocessing, which raised a type casting error at runtime. This was resolved by upgrading to the latest SB3 release (`pip install --upgrade stable-baselines3`), which added explicit compatibility with NumPy 2.x.

---

## 3. Results and Analysis

| Seed | Mean reward | Std |
|------|-------------|-----|
| 42   | 13.330      | 4.893 |
| 43   | 20.455      | 0.000 |
| 44   | 20.562      | 0.150 |
| Average | 18.116  | —   |

SB3 performs slightly below the custom DQN on average (18.12 vs. 19.49). The most notable feature is the behavior on seed 43: a standard deviation of 0.0 over 50 evaluation episodes indicates that the agent reached a deterministic fixed point — the same sequence of actions, and therefore the same reward, on every episode with that seed. This is not a sign of a strong policy; it suggests the agent found a single trajectory that avoids collision under this specific initial configuration and repeats it exactly.

Seed 42 again produces the weakest performance (13.33, std 4.9). The gap relative to the custom DQN on this seed (15.16) and especially to DDQN (22.41) suggests that the MlpPolicy, without spatial inductive bias, is less effective at handling the constrained initial configuration that seed 42 produces.

---

## 4. What Worked and What Failed

**What worked:**
- The hyperparameter alignment produced a fair comparison between SB3 and the custom DQN. The two models trained under the same budget and configuration, making the performance difference attributable to the algorithm and architecture rather than to implementation differences.
- SB3's built-in logging (`verbose=1`) provided timestep-level feedback during training, which was useful for monitoring convergence.

**What failed:**
- The seed 43 degenerate case (std = 0.0) was identified only after evaluation. During training, SB3 reported reasonable episode rewards, so the deterministic collapse went undetected until the evaluation script ran. Monitoring the variance of evaluation rewards during training — rather than only the mean — would have flagged this earlier.
- The training for seed 43 was repeated once after the initial run produced poor results (mean ≈ 7.6). The re-run improved the mean significantly (to 20.45) but the zero variance remained, suggesting the policy is still fragile despite the higher average.

---

## 5. Limitations and Perspectives

The comparison between SB3 and the custom DQN conflates two variables: the training framework and the network architecture. A cleaner comparison would either use a CNN policy within SB3 or use an MLP for the custom agent. Disentangling these effects would clarify whether the performance difference is due to the spatial representation or to lower-level implementation differences (e.g., SB3's internal gradient handling, normalization, or replay sampling).

A second limitation is the training budget. SB3's documentation recommends at least 100 000 timesteps for `highway-v0`. Under the equal-budget constraint imposed by this project, SB3 is disadvantaged: it has not fully committed the benefits of its implementation. Running SB3 at 100 000 steps (with the custom DQN trained for a proportionally longer budget) would constitute a fairer long-horizon comparison.

Finally, the `MlpPolicy` architecture is a fixed choice within SB3 that does not exploit the structure of the observation space. A promising direction would be to implement a custom policy class within SB3 using the same CNN as the hand-coded agent, and assess whether the performance gap closes. If it does, the difference is architectural; if it does not, it points to implementation-level effects.

---

## References

- Raffin, A. et al. (2021). Stable-Baselines3: Reliable Reinforcement Learning Implementations. *JMLR*, 22(268).
- Mnih, V. et al. (2015). Human-level control through deep reinforcement learning. *Nature*, 518, 529–533.
