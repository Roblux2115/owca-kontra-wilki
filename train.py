"""
train.py – Trening agenta PPO do projektowania poziomów
=========================================================
Uruchomienie:
    python train.py

Wymagania:
    pip install gymnasium stable-baselines3

Po treningu zapisuje plik sheep_ppo.zip w tym samym folderze.
"""

from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback
import os

from sheep_env import SheepLevelEnv


# ─── Callback wypisujący postęp co N kroków ──────────────────────────────────
class ProgressCallback(BaseCallback):
    def __init__(self, print_every=10_000):
        super().__init__()
        self.print_every = print_every
        self.best_reward = -999

    def _on_step(self) -> bool:
        if self.num_timesteps % self.print_every == 0:
            # Odczytaj ostatnie nagrody z loggera SB3
            if len(self.model.ep_info_buffer) > 0:
                rewards = [ep["r"] for ep in self.model.ep_info_buffer]
                mean_r  = sum(rewards) / len(rewards)
                print(f"  Krok {self.num_timesteps:>8}  "
                      f"śr. nagroda = {mean_r:6.2f}  "
                      f"epizody = {len(self.model.ep_info_buffer)}")
        return True


# ─── Główny trening ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=== Trening agenta – projektowanie poziomów ===\n")

    # Środowisko treningowe
    env = SheepLevelEnv(render_mode=None)

    # Walidacja API Gymnasium (opcjonalna, ale warto na początku)
    print("Sprawdzanie środowiska...")
    check_env(env, warn=True)
    print("OK\n")

    # Model PPO
    # MlpPolicy = sieć neuronowa (MLP) na wejście wektorowe
    # Obserwacja (4,13,20) zostanie automatycznie spłaszczona do wektora 1040
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,       # zachęca do eksploracji
    )

    print("Parametry modelu:")
    print(f"  Polityka:         MlpPolicy")
    print(f"  Przestrzeń akcji: {env.action_space.n}")
    print(f"  Obserwacja:       {env.observation_space.shape}")
    print(f"  Kroki treningu:   200 000\n")

    # Trening
    print("Start treningu...\n")
    model.learn(
        total_timesteps=200_000,
        callback=ProgressCallback(print_every=10_000),
    )

    # Zapis
    save_path = os.path.join(os.path.dirname(__file__), "sheep_ppo")
    model.save(save_path)
    print(f"\nModel zapisany: {save_path}.zip")

    # Szybki test wyuczonego modelu
    print("\n── Test wyuczonego modelu (10 epizodów) ──")
    env2 = SheepLevelEnv(render_mode=None)
    rewards = []
    for ep in range(10):
        obs, _ = env2.reset()
        total_r = 0.0
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, r, terminated, truncated, info = env2.step(action)
            total_r += r
            if terminated or truncated:
                break
        rewards.append(total_r)
        print(f"  ep {ep+1:2d}: drzewa={info['obstacles']:2d}  "
              f"wrogowie={info['enemies']}  "
              f"nagroda={total_r:6.2f}")

    print(f"\nŚrednia nagroda: {sum(rewards)/len(rewards):.2f}")
    print("Gotowe! Uruchom play_trained.py żeby zobaczyć agenta w akcji.")
    env2.close()