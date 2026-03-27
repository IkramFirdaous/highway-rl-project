import gymnasium as gym
import highway_env
from core.config import SHARED_CORE_ENV_ID, SHARED_CORE_CONFIG

def make_env(seed=None):
    env = gym.make(SHARED_CORE_ENV_ID)
    env.unwrapped.configure(SHARED_CORE_CONFIG)
    if seed is not None:
        env.reset(seed=seed)
    return env