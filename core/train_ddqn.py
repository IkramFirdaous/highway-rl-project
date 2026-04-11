import json
import os
import numpy as np
from core.env import make_env
from core.ddqn_agent import DDQNAgent
from core.config import N_EPISODES, EPS_START, EPS_END, EPS_DECAY


def train_ddqn(seed=0):
    env     = make_env(seed=seed)
    agent   = DDQNAgent()
    epsilon = EPS_START

    episode_rewards = []
    episode_losses  = []

    try:
        for ep in range(N_EPISODES):
            state, _ = env.reset()
            total_reward = 0
            ep_losses    = []
            done         = False

            while not done:
                action = agent.select_action(state, epsilon)
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated

                agent.buffer.push(state, action, reward, next_state, float(done))
                loss = agent.update()
                if loss is not None:
                    ep_losses.append(loss)

                state         = next_state
                total_reward += reward

            epsilon = max(EPS_END, epsilon * EPS_DECAY)
            episode_rewards.append(total_reward)
            episode_losses.append(np.mean(ep_losses) if ep_losses else 0.0)

            if ep % 50 == 0:
                mean_r = np.mean(episode_rewards[-50:])
                print(f"ep {ep:4d} | reward={mean_r:.3f} | eps={epsilon:.3f}")

    except KeyboardInterrupt:
        print(f"\nInterrupted at episode {len(episode_rewards)} — saving...")

    os.makedirs("results/checkpoints", exist_ok=True)
    os.makedirs("results/metrics", exist_ok=True)
    agent.save(f"results/checkpoints/ddqn_seed{seed}.pt")

    metrics = {"rewards": episode_rewards, "losses": episode_losses}
    with open(f"results/metrics/ddqn_seed{seed}.json", "w") as f:
        json.dump(metrics, f)

    env.close()
    return agent, episode_rewards
