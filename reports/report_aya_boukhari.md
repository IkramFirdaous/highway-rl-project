# Individual Report — Aya [NOM]

**Project:** Reinforcement Learning on highway-v0
**Course:** Mention IA — CentraleSupelec
**Instructor:** Hédi Hadiji

---

## 1. Personal Contribution

My contribution is the Double-DQN extension. I implemented `core/ddqn_agent.py`, ran the comparative experiments between DQN and DDQN, and analyzed the results to determine whether the Double-DQN modification produced a measurable improvement on the highway-v0 benchmark.

### 1.1 Double-DQN Implementation

Standard DQN computes the target Q-value as:

```
y = r + γ · max_a' Q_target(s', a')
```

The max operator is applied to the same target network that evaluates the action, which introduces a systematic positive bias: the agent tends to overestimate Q-values, particularly in the early phases of training where the Q-function is poorly calibrated. This overestimation can cause the policy to prefer actions whose Q-values are inflated by noise.

Double-DQN (van Hasselt et al., 2016) decouples action selection from action evaluation:

```
a* = argmax_a Q_policy(s', a)
y  = r + γ · Q_target(s', a*)
```

The policy network selects the action; the target network evaluates it. This removes the positive bias while preserving the stability benefits of the target network.

The implementation is minimal by design: `DDQNAgent` inherits the full `DQNAgent` class and only overrides `update()`. The three changed lines are in the target computation block:

```python
best_actions = self.policy_net(next_states_t).argmax(dim=1)
next_q       = self.target_net(next_states_t).gather(1, best_actions.unsqueeze(1)).squeeze(1)
targets      = rewards_t + self.gamma * next_q * (1 - dones_t)
```

I also added gradient clipping (`clip_grad_norm_` with max norm 10) to the DDQN update. In the standard DQN, gradients were not clipped. Because the DDQN target is typically smaller (less overestimation), the loss landscape is slightly different and large gradient steps were occasionally observed in early training. Gradient clipping eliminates this instability without affecting converged performance.

### 1.2 Experimental Design

The hypothesis was: *Q-value overestimation in standard DQN degrades policy quality on highway-v0, and DDQN will reduce this bias and improve average performance.*

To test this, I trained DDQN under exactly the same conditions as DQN: same hyperparameters, same seeds, same number of episodes (2 000), same evaluation protocol (50 runs over seeds 42/43/44). The only variables are the target computation and the gradient clipping.

---

## 2. Results and Analysis

| Seed | DQN mean | DDQN mean | Difference |
|------|----------|-----------|------------|
| 42   | 15.157   | 22.414    | +7.257     |
| 43   | 21.540   | 16.964    | −4.576     |
| 44   | 21.771   | 17.857    | −3.914     |
| Average | 19.489 | 19.078   | −0.411     |

The result is a negative finding: DDQN does not outperform DQN on this benchmark when averaged across seeds. The global averages are nearly identical (19.49 vs. 19.08), and the per-seed comparison shows that DDQN gains substantially on seed 42 but regresses on seeds 43 and 44.

This pattern is interpretable. On seed 42, where standard DQN struggles (mean 15.16, std 8.4), the Q-value overestimation bias was likely the dominant failure mode: the agent overvalued certain high-speed actions and selected them even when they led to collision. DDQN corrected this and achieved a mean of 22.4 — the highest single-seed result in the entire experiment.

On seeds 43 and 44, however, DQN had already converged to a good policy with low variance. The DDQN modification introduced higher variance on these seeds (std 8.2 and 7.3 vs. DQN's 0.5 and 1.2). This suggests that the DDQN correction is either unnecessary in these favorable initial conditions, or that it slightly destabilizes the optimization for these seeds. The higher variance may also reflect greater sensitivity to random weight initialization.

---

## 3. What Worked and What Failed

**What worked:**
- The inheritance-based design kept the DDQN implementation clean and auditable. Any reader can identify the exact change relative to DQN by reading `ddqn_agent.py` alone.
- Gradient clipping eliminated the early training instabilities that were observed in preliminary runs without it.
- The result on seed 42 validates the theoretical motivation: in a scenario where overestimation was the binding constraint, DDQN produced a clear improvement.

**What failed:**
- The hypothesis that DDQN would uniformly improve performance was not confirmed. Seeds 43 and 44 regressed, which was not anticipated.
- An earlier attempt trained DDQN for only 1 000 episodes. The curves showed apparent convergence, but evaluation revealed that the agent had not fully committed to a stable policy: rewards were higher on average but with much larger variance (std above 9). Extending to 2 000 episodes reduced variance on seed 42 but did not eliminate the regression on 43 and 44.

---

## 4. Limitations and Perspectives

The main limitation of this extension is that a single hyperparameter set was used for both DQN and DDQN. In principle, DDQN benefits from tuning the target update frequency and learning rate independently, because the reduced target magnitude changes the effective scale of gradient updates. A proper ablation would sweep these parameters for each algorithm separately.

The result also raises an open question: is the instability on seeds 43 and 44 specific to the DDQN modification, or is it a general consequence of the training budget being near the boundary of stable convergence? A longer training run (5 000 episodes) would help distinguish between these two explanations.

More broadly, this experiment illustrates a recurring challenge in deep RL: algorithmic improvements that are theoretically well-motivated do not always translate to uniform gains across environment instances. The overestimation bias addressed by DDQN is only one of many sources of policy instability. In an environment where traffic randomness dominates, bias reduction may be less impactful than variance reduction — which would point toward different extensions such as prioritized replay or ensemble Q-functions.

---

## References

- van Hasselt, H., Guez, A., Silver, D. (2016). Deep Reinforcement Learning with Double Q-learning. *AAAI*.
- Mnih, V. et al. (2015). Human-level control through deep reinforcement learning. *Nature*, 518, 529–533.
