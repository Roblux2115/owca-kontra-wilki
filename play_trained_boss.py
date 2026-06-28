import time
import json
import os
import random
import pygame

from stable_baselines3 import PPO
from sheep_env import SheepLevelEnv, idx_to_colrow, N_CELLS, BOSS_AMMO_CELLS
from play_trained import level_to_json
from state_client import StateReporter


SAVE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SAVE_DIR, "boss_ppo")
JSON_PATH = os.path.join(SAVE_DIR, "boss_levels.json")

STEP_DELAY = 0.12
PAUSE_END = 2.0
MIN_REWARD = 12.0


def choose_boss_ammo_cell(obstacle_cells):
    candidates = sorted(BOSS_AMMO_CELLS - set(obstacle_cells))
    if not candidates:
        return -1
    return random.choice(candidates)


def load_saved_levels():
    if not os.path.exists(JSON_PATH) or os.path.getsize(JSON_PATH) == 0:
        return []
    try:
        with open(JSON_PATH, encoding="utf-8-sig") as f:
            data = json.load(f)
    except json.JSONDecodeError:
        print("  Uwaga: boss_levels.json byl pusty/uszkodzony - zaczynam od pustej listy")
        return []
    return data if isinstance(data, list) else []


def save_levels(levels):
    with open(JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(levels, f, indent=2)


def run():
    reporter = StateReporter("boss_generator")
    reporter.update(status="starting", model_path=MODEL_PATH + ".zip")

    if not os.path.exists(MODEL_PATH + ".zip"):
        print(f"Nie znaleziono modelu: {MODEL_PATH}.zip")
        print("Najpierw uruchom: python train_boss.py")
        reporter.update(status="missing_model")
        return

    print("Wczytywanie modelu boss...")
    model = PPO.load(MODEL_PATH)
    print("OK\n")
    reporter.update(status="generating")

    pygame.init()
    env = SheepLevelEnv(render_mode="human", mode="boss")

    episode = 0
    running = True

    while running:
        episode += 1
        obs, info = env.reset()
        total_r = 0.0
        step = 0

        print(f"--- Boss epizod {episode} ---")

        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                if event.type == pygame.KEYDOWN and event.key == pygame.K_q:
                    running = False

            if not running:
                break

            action, _ = model.predict(obs, deterministic=False)
            obs, r, terminated, truncated, info = env.step(int(action))
            total_r += r
            step += 1

            if int(action) < N_CELLS:
                c, row = idx_to_colrow(int(action))
                print(f"  krok {step:2d}: drzewo   @ ({c:2d},{row:2d})")
            elif int(action) < 2 * N_CELLS:
                c, row = idx_to_colrow(int(action) - N_CELLS)
                print(f"  krok {step:2d}: wrog ignorowany @ ({c:2d},{row:2d})")
            elif int(action) < 3 * N_CELLS:
                c, row = idx_to_colrow(int(action) - 2 * N_CELLS)
                print(f"  krok {step:2d}: amunicja @ ({c:2d},{row:2d})")
            else:
                print(f"  krok {step:2d}: DONE")

            time.sleep(STEP_DELAY)

            if terminated or truncated:
                print(f"  -> drzewa={info['obstacles']}  "
                      f"amunicja={info.get('ammo', 0)}  "
                      f"nagroda={total_r:.2f}")
                reporter.update(
                    status="generating",
                    episode=episode,
                    last_reward=round(total_r, 3),
                    obstacles=info["obstacles"],
                    ammo=info.get("ammo", 0),
                )

                ammo_cell = choose_boss_ammo_cell(env._obstacle_cells)
                data = level_to_json(env._obstacle_cells, set(), ammo_cell)
                data["type"] = "boss"
                data["reward"] = round(total_r, 3)

                if total_r >= MIN_REWARD:
                    all_levels = load_saved_levels()
                    all_levels.append(data)
                    save_levels(all_levels)
                    print(f"  OK Boss arena {len(all_levels)} zapisana -> boss_levels.json")
                    reporter.event("boss_level_saved", reward=round(total_r, 3), saved_levels=len(all_levels))
                else:
                    print(f"  Boss arena odrzucona ({total_r:.2f} < {MIN_REWARD})")
                    reporter.event("boss_level_rejected", reward=round(total_r, 3), min_reward=MIN_REWARD)

                pause_start = time.time()
                while time.time() - pause_start < PAUSE_END and running:
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            running = False
                        if event.type == pygame.KEYDOWN and event.key == pygame.K_q:
                            running = False
                    env.render()
                break

    env.close()
    reporter.update(status="stopped", episodes=episode)
    print("Zamknieto.")


if __name__ == "__main__":
    run()
