
import time
import json
import os
import pygame

from stable_baselines3 import PPO
from sheep_env import (SheepLevelEnv, idx_to_colrow, N_CELLS, MAX_STEPS,
                       bfs_path_length, GRID_COLS, GRID_ROWS)
from state_client import StateReporter


SAVE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SAVE_DIR, "sheep_ppo")
JSON_PATH  = os.path.join(SAVE_DIR, "ai_levels.json")

STEP_DELAY = 0.12   # sekundy między krokami agenta (zmniejsz dla szybszego podglądu)
PAUSE_END  = 2.0    # sekundy pauzy po zaprojektowaniu poziomu


def level_to_json(obstacle_cells, enemy_cells, ammo_cell=-1):
    """Konwertuje komórki siatki → współrzędne pikseli dla main.py."""
    from sheep_env import BORDER, CELL_W, CELL_H

    obstacles = []
    for idx in obstacle_cells:
        c, r = idx_to_colrow(idx)
        x = BORDER + c * CELL_W
        y = BORDER + r * CELL_H
        obstacles.append({"x": x, "y": y, "w": CELL_W, "h": CELL_H})

    enemies = []
    for idx in enemy_cells:
        c, r = idx_to_colrow(idx)
        x = BORDER + c * CELL_W + CELL_W // 2
        y = BORDER + r * CELL_H + CELL_H // 2
        enemies.append({"x": x, "y": y})

    ammo = None
    if ammo_cell >= 0:
        c, r = idx_to_colrow(ammo_cell)
        ammo = {
            "x": BORDER + c * CELL_W + CELL_W // 2,
            "y": BORDER + r * CELL_H + CELL_H // 2,
        }

    return {"obstacles": obstacles, "enemies": enemies, "ammo": ammo}


def _print_reward_breakdown(env):
    """Wypisuje szczegółowy breakdown nagrody końcowej."""
    import math
    obs_cells   = env._obstacle_cells
    enemy_cells = env._enemy_cells
    ammo_cell   = env._ammo_cell

    path_len = bfs_path_length(obs_cells)
    max_path = GRID_COLS + GRID_ROWS

    print("  ┌─ Breakdown nagrody ─────────────────────────────")

    # BFS
    if path_len == -1:
        print("  │  BFS: nieprzejezdna mapa          -20.00")
        print("  └─────────────────────────────────────────────")
        return
    bfs_r = round(10.0 * path_len / max_path, 2)
    print(f"  │  BFS długość ścieżki ({path_len:2d}/{max_path})       +{bfs_r:.2f}")

    # Wrogowie
    n_e = len(enemy_cells)
    if n_e == 0:         e_r = -2.0
    elif 3 <= n_e <= 4:  e_r =  5.0
    elif n_e <= 6:       e_r =  3.0
    else:                e_r =  1.0
    sign = "+" if e_r >= 0 else ""
    print(f"  │  Wrogowie ({n_e})                       {sign}{e_r:.2f}")

    # Kara wrogowie blisko spawnu
    for idx in enemy_cells:
        c, r = idx_to_colrow(idx)
        d = math.sqrt(c**2 + r**2)
        if d <= 3.5:
            print(f"  │    wróg ({c},{r}) za blisko spawnu    -2.00")

    # Przeszkody
    n_o = len(obs_cells)
    if n_o == 0:        o_r = -2.0
    elif 5 <= n_o <= 8: o_r =  3.0
    elif n_o <= 12:     o_r =  1.5
    else:               o_r =  0.0
    sign = "+" if o_r >= 0 else ""
    print(f"  │  Drzewa ({n_o})                        {sign}{o_r:.2f}")

    # Kara drzewa blisko spawnu
    for idx in obs_cells:
        c, r = idx_to_colrow(idx)
        if c <= 2 and r <= 2:
            print(f"  │    drzewo ({c},{r}) blisko spawnu     -1.00")

    # Amunicja
    if ammo_cell < 0:
        print(f"  │  Amunicja: brak                   -3.00")
    else:
        print(f"  │  Amunicja: obecna                 +3.00")
        ac, ar = idx_to_colrow(ammo_cell)
        d_spawn = math.sqrt(ac**2 + ar**2)
        d_exit  = math.sqrt((ac-(GRID_COLS-1))**2 + (ar-GRID_ROWS//2)**2)
        print(f"  │    pozycja ({ac},{ar})  dystans od spawnu={d_spawn:.1f}")
        if d_spawn <= 3.0:
            print(f"  │    za blisko spawnu               -2.00")
        if d_spawn >= 14.0:
            print(f"  │    za daleko od spawnu            -2.00")
        if 4.0 <= d_spawn <= 12.0:
            print(f"  │    dobra strefa (4-12)            +1.00")
        if ammo_cell in obs_cells or ammo_cell in enemy_cells:
            print(f"  │    na przeszkodzie/wrogu          -2.00")

    print("  └─────────────────────────────────────────────")


def run():
    reporter = StateReporter("normal_generator")
    reporter.update(status="starting", model_path=MODEL_PATH + ".zip")

    # Wczytaj model
    if not os.path.exists(MODEL_PATH + ".zip"):
        print(f"Nie znaleziono modelu: {MODEL_PATH}.zip")
        print("Najpierw uruchom: python train.py")
        reporter.update(status="missing_model")
        return

    print("Wczytywanie modelu...")
    model = PPO.load(MODEL_PATH)
    print("OK\n")
    reporter.update(status="generating")

    pygame.init()
    env = SheepLevelEnv(render_mode="human")

    episode = 0
    running = True

    while running:
        episode += 1
        obs, info = env.reset()
        total_r   = 0.0
        step      = 0

        print(f"── Epizod {episode} ──")

        while running:
            # Obsługa zamknięcia okna
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                if event.type == pygame.KEYDOWN and event.key == pygame.K_q:
                    running = False

            if not running:
                break

            # Akcja agenta
            action, _ = model.predict(obs, deterministic=False)
            obs, r, terminated, truncated, info = env.step(int(action))
            total_r += r
            step    += 1

            # Wypisz co agent zrobił
            if int(action) < N_CELLS:
                c, row = idx_to_colrow(int(action))
                print(f"  krok {step:2d}: drzewo  @ ({c:2d},{row:2d})")
            elif int(action) < 2 * N_CELLS:
                c, row = idx_to_colrow(int(action) - N_CELLS)
                print(f"  krok {step:2d}: wróg    @ ({c:2d},{row:2d})")
            else:
                print(f"  krok {step:2d}: DONE")

            time.sleep(STEP_DELAY)

            if terminated or truncated:
                print(f"  → drzewa={info['obstacles']}  "
                      f"wrogowie={info['enemies']}  "
                      f"amunicja={info.get('ammo',0)}  "
                      f"nagroda={total_r:.2f}")
                reporter.update(
                    status="generating",
                    episode=episode,
                    last_reward=round(total_r, 3),
                    obstacles=info["obstacles"],
                    enemies=info["enemies"],
                    ammo=info.get("ammo", 0),
                )
                _print_reward_breakdown(env)
                print()

                # Zapisz tylko poziomy z nagrodą >= MIN_REWARD
                MIN_REWARD = 20.0
                data = level_to_json(env._obstacle_cells, env._enemy_cells, env._ammo_cell)
                data["reward"] = round(total_r, 3)  # zapisz nagrodę do JSON
                if total_r >= MIN_REWARD:
                    if os.path.exists(JSON_PATH):
                        with open(JSON_PATH) as f:
                            all_levels = json.load(f)
                    else:
                        all_levels = []
                    all_levels.append(data)
                    with open(JSON_PATH, "w") as f:
                        json.dump(all_levels, f, indent=2)
                    print(f"  ✓ Poziom {len(all_levels)} zapisany (nagroda={total_r:.2f}) → ai_levels.json")
                    reporter.event("level_saved", reward=round(total_r, 3), saved_levels=len(all_levels))
                else:
                    print(f"  ✗ Poziom odrzucony (nagroda={total_r:.2f} < {MIN_REWARD})")
                    reporter.event("level_rejected", reward=round(total_r, 3), min_reward=MIN_REWARD)

                # Pauza żeby zobaczyć gotowy poziom
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
    print("Zamknięto.")


if __name__ == "__main__":
    run()
