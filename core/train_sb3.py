from stable_baselines3 import DQN
from core.env import make_env
from core.config import EVAL_SEEDS

def train_sb3():
    env = make_env()
    model = DQN("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=100_000)
    model.save("results/checkpoints/sb3_dqn")
    return model