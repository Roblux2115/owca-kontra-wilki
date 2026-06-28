from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import BaseCallback
import os

from sheep_env import SheepLevelEnv
from state_client import StateReporter


class ProgressCallback(BaseCallback):
    def __init__(self, print_every=10_000, reporter=None):
        super().__init__()
        self.print_every = print_every
        self.best_reward = -999
        self.reporter = reporter

    def _on_step(self) -> bool:
        if self.num_timesteps % self.print_every == 0:
            if len(self.model.ep_info_buffer) > 0:
                rewards = [ep["r"] for ep in self.model.ep_info_buffer]
                mean_r = sum(rewards) / len(rewards)
                self.best_reward = max(self.best_reward, mean_r)
                print(f"  Krok {self.num_timesteps:>8}  sr. nagroda = {mean_r:6.2f}")
                if self.reporter:
                    self.reporter.update(
                        status="training",
                        timesteps=self.num_timesteps,
                        mean_reward=round(mean_r, 3),
                        best_mean_reward=round(self.best_reward, 3),
                        episodes=len(self.model.ep_info_buffer),
                    )
        return True


if __name__ == "__main__":
    print("=== Trening agenta PPO - areny bossow ===\n")
    reporter = StateReporter("boss")
    reporter.update(status="starting", total_timesteps=200_000)

    env = SheepLevelEnv(render_mode=None, mode="boss")

    print("Sprawdzanie srodowiska boss...")
    check_env(env, warn=True)
    print("OK\n")

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
        ent_coef=0.01,
    )

    print("Start treningu boss arenas...\n")
    model.learn(
        total_timesteps=200_000,
        callback=ProgressCallback(print_every=10_000, reporter=reporter),
    )

    save_path = os.path.join(os.path.dirname(__file__), "boss_ppo")
    model.save(save_path)
    print(f"\nModel boss zapisany: {save_path}.zip")
    reporter.update(status="done", model_path=save_path + ".zip")
    reporter.event("model_saved", path=save_path + ".zip")
    env.close()
