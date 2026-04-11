import argparse
import json
import os
import matplotlib
matplotlib.use("Agg")
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


def run_sb3(seeds):
    for seed in seeds:
        print(f"\n=== SB3 training seed={seed} ===")
        train_sb3(seed=seed)


def eval_dqn(seeds):
    results = {}
    for seed in seeds:
        from core.dqn_agent import DQNAgent
        agent = DQNAgent()
        agent.load(f"results/checkpoints/dqn_seed{seed}.pt")
        agent.policy_net.eval()
        mean, std, rewards = evaluate(
            agent_fn=lambda s, a=agent: a.select_action(s, epsilon=0.0),
            env_fn=make_env,
        )
        print(f"DQN seed={seed} | mean={mean:.3f} std={std:.3f} ({len(rewards)} runs)")
        results[f"dqn_seed{seed}"] = {"mean": mean, "std": std, "rewards": rewards}
    return results


def eval_sb3(seeds):
    from stable_baselines3 import DQN as SB3DQN
    results = {}
    for seed in seeds:
        path = f"results/checkpoints/sb3_dqn_seed{seed}"
        if not os.path.exists(path + ".zip"):
            print(f"SB3 seed={seed} | checkpoint not found, skipping")
            continue
        model = SB3DQN.load(path)
        mean, std, rewards = evaluate(
            agent_fn=lambda s, m=model: m.predict(s, deterministic=True)[0],
            env_fn=make_env,
        )
        print(f"SB3  seed={seed} | mean={mean:.3f} std={std:.3f} ({len(rewards)} runs)")
        results[f"sb3_seed{seed}"] = {"mean": mean, "std": std, "rewards": rewards}
    return results


def plot_training_curves(all_rewards, seeds):
    os.makedirs("results/plots", exist_ok=True)
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
    print("Saved: results/plots/dqn_training_curves.png")


def log_comparison(dqn_results, sb3_results):
    os.makedirs("results", exist_ok=True)
    all_results = {**dqn_results, **sb3_results}

    # JSON
    log = {}
    for name, r in all_results.items():
        log[name] = {"mean": r["mean"], "std": r["std"], "n_runs": len(r["rewards"])}
    with open("results/comparison.json", "w") as f:
        json.dump(log, f, indent=2)

    # Lisible
    lines = ["\n=== Evaluation (50 runs) ==="]
    lines.append(f"{'Model':<20} {'Mean':>8} {'Std':>8} {'Runs':>6}")
    lines.append("-" * 46)
    for name, r in all_results.items():
        lines.append(f"{name:<20} {r['mean']:>8.3f} {r['std']:>8.3f} {len(r['rewards']):>6}")

    dqn_means = [r["mean"] for k, r in all_results.items() if k.startswith("dqn")]
    sb3_means  = [r["mean"] for k, r in all_results.items() if k.startswith("sb3")]
    if dqn_means:
        lines.append(f"\n{'DQN average':<20} {sum(dqn_means)/len(dqn_means):>8.3f}")
    if sb3_means:
        lines.append(f"{'SB3 average':<20} {sum(sb3_means)/len(sb3_means):>8.3f}")

    summary = "\n".join(lines)
    print(summary)
    with open("results/comparison.txt", "w") as f:
        f.write(summary)
    print("\nSaved: results/comparison.json, results/comparison.txt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train", "eval", "all", "sb3"], default="all")
    args = parser.parse_args()

    seeds = EVAL_SEEDS

    if args.mode == "sb3":
        run_sb3(seeds)

    if args.mode in ("train", "all"):
        dqn_rewards = run_dqn(seeds)
        plot_training_curves(dqn_rewards, seeds)
        run_sb3(seeds)

    if args.mode in ("eval", "all"):
        dqn_results = eval_dqn(seeds)
        sb3_results  = eval_sb3(seeds)
        log_comparison(dqn_results, sb3_results)
