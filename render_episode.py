import os
import argparse
import gymnasium as gym
import highway_env
import numpy as np
from core.config import SHARED_CORE_CONFIG, SHARED_CORE_ENV_ID


def make_render_env(render_mode):
    env = gym.make(SHARED_CORE_ENV_ID, render_mode=render_mode)
    env.unwrapped.configure(SHARED_CORE_CONFIG)
    return env


def run_dqn_episode(checkpoint, render_mode, seed):
    import torch
    from core.dqn_agent import DQNAgent
    agent = DQNAgent()
    agent.load(checkpoint)
    agent.policy_net.eval()

    env = make_render_env(render_mode)
    state, _ = env.reset(seed=seed)
    frames, total_reward, done = [], 0, False

    while not done:
        if render_mode == "rgb_array":
            frames.append(env.render())
        action = agent.select_action(state, epsilon=0.0)
        state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        total_reward += reward

    env.close()
    print(f"Total reward: {total_reward:.3f}")
    return frames


def run_sb3_episode(checkpoint, render_mode, seed):
    from stable_baselines3 import DQN
    model = DQN.load(checkpoint)

    env = make_render_env(render_mode)
    state, _ = env.reset(seed=seed)
    frames, total_reward, done = [], 0, False

    while not done:
        if render_mode == "rgb_array":
            frames.append(env.render())
        action, _ = model.predict(state, deterministic=True)
        state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        total_reward += reward

    env.close()
    print(f"Total reward: {total_reward:.3f}")
    return frames


def save_video(frames, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation

    fig, ax = plt.subplots()
    ax.axis("off")
    img = ax.imshow(frames[0])

    def update(frame):
        img.set_data(frame)
        return [img]

    ani = animation.FuncAnimation(fig, update, frames=frames, interval=50, blit=True)
    ani.save(path, writer="pillow", fps=15)
    plt.close()
    print(f"Video saved: {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", choices=["dqn", "sb3"], default="dqn")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--render", choices=["human", "video"], default="human")
    args = parser.parse_args()

    render_mode = "human" if args.render == "human" else "rgb_array"

    os.makedirs("results/videos", exist_ok=True)

    if args.agent == "dqn":
        checkpoint = f"results/checkpoints/dqn_seed{args.seed}.pt"
        frames = run_dqn_episode(checkpoint, render_mode, args.seed)
    else:
        checkpoint = "results/checkpoints/sb3_dqn"
        frames = run_sb3_episode(checkpoint, render_mode, args.seed)

    if args.render == "video" and frames:
        save_video(frames, f"results/videos/{args.agent}_seed{args.seed}.gif")
