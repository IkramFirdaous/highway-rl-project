import argparse
import matplotlib.pyplot as plt

from core.train_dqn import train
from core.train_sb3 import train_sb3
from core.evaluate import evaluate
from core.env import make_env
from core.config import EVAL_SEEDS

def run_dqn(seeds):
    all_rewards = []
    for seed in seeds:
        print(f"\n=== DQN training seed={seed} ===")
        _, rewards = train(seed=seed)
        all_rewards.append(rewards)
    return all_rewards


def run_sb3():
    print("\n=== SB3 training ===")
    model = train_sb3()
    return model


def eval_dqn(seeds):
    results = []
    for seed in seeds:
        from core.dqn_agent import DQNAgent
        agent = DQNAgent()
        agent.load(f"results/checkpoints/dqn_seed{seed}.pt")
        mean, std = evaluate(
            agent_fn=lambda s: agent.select_action(s, epsilon=0.0),
            env_fn=make_env,
        )
        print(f"DQN seed={seed} | mean={mean:.3f} std={std:.3f}")
        results.append((mean, std))
    return results


def eval_sb3():
    from stable_baselines3 import DQN as SB3DQN
    model = SB3DQN.load("results/checkpoints/sb3_dqn")
    mean, std = evaluate(
        agent_fn=lambda s: model.predict(s, deterministic=True)[0],
        env_fn=make_env,
    )
    print(f"SB3  | mean={mean:.3f} std={std:.3f}")
    return mean, std


def plot_training_curves(all_rewards, seeds):
    plt.figure()
    for rewards, seed in zip(all_rewards, seeds):
        plt.plot(rewards, alpha=0.7, label=f"seed={seed}")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("DQN training curves")
    plt.legend()
    plt.tight_layout()
    plt.savefig("results/plots/dqn_training_curves.png")
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train", "eval", "all"], default="all")
    args = parser.parse_args()

    seeds = EVAL_SEEDS

    if args.mode in ("train", "all"):
        dqn_rewards = run_dqn(seeds)
        plot_training_curves(dqn_rewards, seeds)
        run_sb3()

    if args.mode in ("eval", "all"):
        dqn_results = eval_dqn(seeds)
        sb3_mean, sb3_std = eval_sb3()

        print("\n=== Comparison ===")
        for (mean, std), seed in zip(dqn_results, seeds):
            print(f"  DQN seed={seed}: {mean:.3f} +/- {std:.3f}")
        print(f"  SB3:           {sb3_mean:.3f} +/- {sb3_std:.3f}")
