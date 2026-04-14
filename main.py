import argparse
import json
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from core.train_dqn import train
from core.train_ddqn import train_ddqn
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


def run_ddqn(seeds):
    all_rewards = []
    for seed in seeds:
        print(f"\n=== Double-DQN training seed={seed} ===")
        _, rewards = train_ddqn(seed=seed)
        all_rewards.append(rewards)
    return all_rewards


def run_sb3(seeds):
    for seed in seeds:
        print(f"\n=== SB3 training seed={seed} ===")
        train_sb3(seed=seed)


def eval_dqn(seeds):
    from core.dqn_agent import DQNAgent
    results = {}
    for seed in seeds:
        path = f"results/checkpoints/dqn_seed{seed}.pt"
        if not os.path.exists(path):
            print(f"DQN seed={seed} | checkpoint not found, skipping")
            continue
        agent = DQNAgent()
        agent.load(path)
        agent.policy_net.eval()
        mean, std, rewards = evaluate(
            agent_fn=lambda s, a=agent: a.select_action(s, epsilon=0.0),
            env_fn=make_env,
        )
        print(f"DQN  seed={seed} | mean={mean:.3f} std={std:.3f} ({len(rewards)} runs)")
        results[f"dqn_seed{seed}"] = {"mean": mean, "std": std, "rewards": rewards}
    return results


def eval_ddqn(seeds):
    from core.ddqn_agent import DDQNAgent
    results = {}
    for seed in seeds:
        path = f"results/checkpoints/ddqn_seed{seed}.pt"
        if not os.path.exists(path):
            print(f"DDQN seed={seed} | checkpoint not found, skipping")
            continue
        agent = DDQNAgent()
        agent.load(path)
        agent.policy_net.eval()
        mean, std, rewards = evaluate(
            agent_fn=lambda s, a=agent: a.select_action(s, epsilon=0.0),
            env_fn=make_env,
        )
        print(f"DDQN seed={seed} | mean={mean:.3f} std={std:.3f} ({len(rewards)} runs)")
        results[f"ddqn_seed{seed}"] = {"mean": mean, "std": std, "rewards": rewards}
    return results


def eval_sb3(seeds):
    from stable_baselines3 import DQN as SB3DQN
    results = {}
    for seed in seeds:
        path = f"results/checkpoints/sb3_dqn_seed{seed}"
        if not os.path.exists(path + ".zip"):
            print(f"SB3  seed={seed} | checkpoint not found, skipping")
            continue
        model = SB3DQN.load(path)
        mean, std, rewards = evaluate(
            agent_fn=lambda s, m=model: m.predict(s, deterministic=True)[0],
            env_fn=make_env,
        )
        print(f"SB3  seed={seed} | mean={mean:.3f} std={std:.3f} ({len(rewards)} runs)")
        results[f"sb3_seed{seed}"] = {"mean": mean, "std": std, "rewards": rewards}
    return results


def plot_training_curves(all_rewards_dict):
    os.makedirs("results/plots", exist_ok=True)
    plt.figure(figsize=(10, 5))
    styles = {"dqn": ("blue", "-"), "ddqn": ("red", "--"), "sb3": ("green", ":")}
    for label, (rewards_list, seeds) in all_rewards_dict.items():
        color, ls = styles.get(label, ("gray", "-"))
        for rewards, seed in zip(rewards_list, seeds):
            plt.plot(rewards, alpha=0.6, color=color, linestyle=ls,
                     label=f"{label.upper()} seed={seed}")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Training curves — DQN vs Double-DQN")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig("results/plots/training_curves.png")
    plt.close()
    print("Saved: results/plots/training_curves.png")


def log_comparison(all_results):
    os.makedirs("results", exist_ok=True)
    log = {}
    for name, r in all_results.items():
        log[name] = {"mean": round(r["mean"], 4), "std": round(r["std"], 4),
                     "n_runs": len(r["rewards"])}
    with open("results/comparison.json", "w") as f:
        json.dump(log, f, indent=2)

    lines = ["\n=== Evaluation (50 runs per model) ===",
             f"{'Model':<22} {'Mean':>8} {'Std':>8} {'Runs':>6}",
             "-" * 48]
    for name, r in all_results.items():
        lines.append(f"{name:<22} {r['mean']:>8.3f} {r['std']:>8.3f} {len(r['rewards']):>6}")

    for prefix, label in [("dqn_", "DQN avg"), ("ddqn_", "DDQN avg"), ("sb3_", "SB3 avg")]:
        means = [r["mean"] for k, r in all_results.items() if k.startswith(prefix)]
        if means:
            lines.append(f"\n{label:<22} {sum(means)/len(means):>8.3f}")

    summary = "\n".join(lines)
    print(summary)
    with open("results/comparison.txt", "w") as f:
        f.write(summary)
    print("\nSaved: results/comparison.json, results/comparison.txt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train", "eval", "all", "sb3", "ddqn"], default="all")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    seeds = [args.seed] if args.seed is not None else EVAL_SEEDS

    if args.mode == "sb3":
        run_sb3(seeds)

    if args.mode == "ddqn":
        ddqn_rewards = run_ddqn(seeds)
        plot_training_curves({"ddqn": (ddqn_rewards, seeds)})

    if args.mode in ("train", "all"):
        dqn_rewards  = run_dqn(seeds)
        ddqn_rewards = run_ddqn(seeds)
        run_sb3(seeds)
        plot_training_curves({
            "dqn":  (dqn_rewards,  seeds),
            "ddqn": (ddqn_rewards, seeds),
        })

    if args.mode in ("eval", "all"):
        all_results = {}
        all_results.update(eval_dqn(seeds))
        all_results.update(eval_ddqn(seeds))
        all_results.update(eval_sb3(seeds))
        log_comparison(all_results)
