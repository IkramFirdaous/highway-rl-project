import numpy as np
from core.config import EVAL_EPISODES, EVAL_SEEDS

def evaluate(agent_fn, env_fn, n_episodes=EVAL_EPISODES, seeds=EVAL_SEEDS):
    all_rewards = []
    for seed in seeds:
        env = env_fn(seed=seed)
        for ep in range(n_episodes // len(seeds)):
            state, _ = env.reset()
            total, done = 0, False
            while not done:
                action = agent_fn(state)
                state, reward, term, trunc, _ = env.step(action)
                done = term or trunc
                total += reward
            all_rewards.append(total)
    return np.mean(all_rewards), np.std(all_rewards)