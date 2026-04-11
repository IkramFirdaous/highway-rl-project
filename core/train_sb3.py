import os
from stable_baselines3 import DQN
from core.env import make_env
from core.config import (
    LR, GAMMA, BUFFER_SIZE, BATCH_SIZE, TARGET_UPDATE, EPS_END,
    SB3_TIMESTEPS, SB3_LEARNING_STARTS, SB3_EXPLORATION_FRAC,
)


def train_sb3(seed=42, total_timesteps=SB3_TIMESTEPS):
    os.makedirs("results/checkpoints", exist_ok=True)
    env = make_env(seed=seed)
    model = DQN(
        "MlpPolicy",
        env,
        verbose=1,
        seed=seed,
        learning_rate=LR,
        batch_size=BATCH_SIZE,
        gamma=GAMMA,
        buffer_size=BUFFER_SIZE,
        learning_starts=SB3_LEARNING_STARTS,
        target_update_interval=TARGET_UPDATE,
        exploration_fraction=SB3_EXPLORATION_FRAC,
        exploration_final_eps=EPS_END,
    )
    try:
        model.learn(total_timesteps=total_timesteps)
    except KeyboardInterrupt:
        print("\nInterrupted — saving SB3 checkpoint...")
    model.save(f"results/checkpoints/sb3_dqn_seed{seed}")
    env.close()
    return model
