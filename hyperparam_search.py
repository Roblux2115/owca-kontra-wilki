import argparse
import json
import os
from pathlib import Path

from stable_baselines3 import PPO

from sheep_env import SheepLevelEnv
from state_client import StateReporter


BASE_DIR = Path(__file__).resolve().parent
RESULTS_PATH = BASE_DIR / "hyperparam_results.json"

PARAM_GRID = [
    {"learning_rate": 1e-4, "ent_coef": 0.005, "gamma": 0.99},
    {"learning_rate": 3e-4, "ent_coef": 0.01, "gamma": 0.99},
    {"learning_rate": 1e-3, "ent_coef": 0.02, "gamma": 0.98},
]


def evaluate_model(model, mode, episodes):
    env = SheepLevelEnv(render_mode=None, mode=mode)
    rewards = []

    for _ in range(episodes):
        obs, _ = env.reset()
        total_reward = 0.0
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            if terminated or truncated:
                break
        rewards.append(total_reward)

    env.close()
    return sum(rewards) / len(rewards)


def run_search(mode, timesteps, eval_episodes):
    reporter = StateReporter(f"{mode}_hyperparam_search")
    reporter.update(
        status="starting",
        mode=mode,
        trials=len(PARAM_GRID),
        timesteps_per_trial=timesteps,
        eval_episodes=eval_episodes,
    )

    results = []
    best_result = None

    for trial_idx, params in enumerate(PARAM_GRID, start=1):
        reporter.config(**params, timesteps=timesteps, eval_episodes=eval_episodes, mode=mode)
        reporter.update(status="training_trial", trial=trial_idx, params=params)

        env = SheepLevelEnv(render_mode=None, mode=mode)
        model = PPO(
            "MlpPolicy",
            env,
            verbose=0,
            n_steps=1024,
            batch_size=64,
            n_epochs=5,
            learning_rate=params["learning_rate"],
            ent_coef=params["ent_coef"],
            gamma=params["gamma"],
        )
        model.learn(total_timesteps=timesteps)
        env.close()

        mean_reward = evaluate_model(model, mode, eval_episodes)
        result = {
            "trial": trial_idx,
            "mode": mode,
            "timesteps": timesteps,
            "eval_episodes": eval_episodes,
            "mean_reward": round(mean_reward, 3),
            "params": params,
        }
        results.append(result)

        if best_result is None or result["mean_reward"] > best_result["mean_reward"]:
            best_result = result

        reporter.event("hyperparam_trial_finished", **result)
        reporter.update(
            status="trial_finished",
            trial=trial_idx,
            last_mean_reward=result["mean_reward"],
            best_mean_reward=best_result["mean_reward"],
            best_params=best_result["params"],
        )
        print(f"Trial {trial_idx}/{len(PARAM_GRID)}: reward={result['mean_reward']} params={params}")

    output = {
        "mode": mode,
        "results": results,
        "best": best_result,
    }
    save_results(output)
    reporter.update(status="done", best_mean_reward=best_result["mean_reward"], best_params=best_result["params"])
    reporter.event("hyperparam_search_finished", mode=mode, best=best_result)

    print("\nBest result:")
    print(json.dumps(best_result, indent=2))
    print(f"\nSaved: {RESULTS_PATH}")


def save_results(output):
    existing = []
    if RESULTS_PATH.exists() and RESULTS_PATH.stat().st_size > 0:
        with open(RESULTS_PATH, encoding="utf-8") as f:
            try:
                loaded = json.load(f)
                existing = loaded if isinstance(loaded, list) else [loaded]
            except json.JSONDecodeError:
                existing = []

    existing.append(output)
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(existing, f, indent=2)


def parse_args():
    parser = argparse.ArgumentParser(description="Prosty tuning hiperparametrow PPO.")
    parser.add_argument("--mode", choices=["normal", "boss"], default="normal")
    parser.add_argument("--timesteps", type=int, default=10_000)
    parser.add_argument("--eval-episodes", type=int, default=5)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
    run_search(args.mode, args.timesteps, args.eval_episodes)
