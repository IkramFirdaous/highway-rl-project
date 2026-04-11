import os
from stable_baselines3 import DQN
from core.env import make_env


def train_sb3(seed=42, total_timesteps=15_000):
    os.makedirs("results/checkpoints", exist_ok=True)
    env = make_env(seed=seed)
    model = DQN(
        "MlpPolicy",
        env,
        verbose=1,
        seed=seed,
        learning_rate=1e-3,
        batch_size=64,
        gamma=0.99,
        learning_starts=1000,
        target_update_interval=1000,
        exploration_fraction=0.3,
        exploration_final_eps=0.05,
    )
    try:
        model.learn(total_timesteps=total_timesteps)
    except KeyboardInterrupt:
        print("\nInterrupted — saving SB3 checkpoint...")
    model.save("results/checkpoints/sb3_dqn")
    env.close()
    return model