"""
sheep_env.py – Środowisko Gymnasium do uczenia agenta projektowania poziomów
=============================================================================
Instalacja:
    pip install gymnasium numpy pygame

Użycie:
    from sheep_env import SheepLevelEnv

    # Trening (bez okna)
    env = SheepLevelEnv(render_mode=None)

    # Podgląd na żywo
    env = SheepLevelEnv(render_mode="human")

    obs, info = env.reset()
    obs, reward, terminated, truncated, info = env.step(action)

Idea projektowania:
    - Mapa podzielona na siatkę GRID_COLS × GRID_ROWS komórek
    - Agent w każdym kroku wybiera jedną z 781 akcji:
        [0 .. 259]   → postaw drzewo (przeszkodę) w komórce i
        [260 .. 519] → postaw wroga w komórce i
        [520 .. 779] → postaw amunicję w komórce i
        [780]        → zakończ projektowanie
    - Nagroda końcowa zależy od jakości poziomu (BFS, wrogowie, amunicja, rozmieszczenie)
"""

import math
import random
import numpy as np
import gymnasium as gym
from gymnasium import spaces

try:
    import pygame
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False

# ─── PARAMETRY SIATKI ────────────────────────────────────────────────────────
MAP_W, MAP_H  = 600, 400
BORDER        = 22
GRID_COLS     = 20
GRID_ROWS     = 13
CELL_W        = (MAP_W - 2 * BORDER) // GRID_COLS   # 27 px
CELL_H        = (MAP_H - 2 * BORDER) // GRID_ROWS   # 27 px
N_CELLS       = GRID_COLS * GRID_ROWS               # 260

MAX_STEPS     = 35
MAX_OBSTACLES = 12
MAX_ENEMIES   = 6
MAX_AMMO      = 1    # dokładnie 1 kłębek wełny na mapie

# Indeks komórki: idx = col + row * GRID_COLS
def idx_to_colrow(idx: int):
    return idx % GRID_COLS, idx // GRID_COLS

def colrow_to_idx(col: int, row: int) -> int:
    return col + row * GRID_COLS

# Komórki chronione (spawn gracza i wyjście)
SPAWN_CELL  = colrow_to_idx(0, 0)
EXIT_CELLS  = {
    colrow_to_idx(GRID_COLS - 1, r)
    for r in range(GRID_ROWS // 2 - 1, GRID_ROWS // 2 + 2)
}
PROTECTED_CELLS = {SPAWN_CELL} | EXIT_CELLS

# Kanały obserwacji
CH_OBSTACLE, CH_ENEMY, CH_SPAWN, CH_EXIT, CH_AMMO = 0, 1, 2, 3, 4
N_CHANNELS = 5


# ─── BFS – sprawdzenie przejezdności ─────────────────────────────────────────
def bfs_path_length(obstacle_cells: set) -> int:
    """
    BFS od (col=0, row=0) do prawego środka mapy.
    Zwraca długość najkrótszej ścieżki lub -1 jeśli nieprzejezdna.
    """
    goal_c = GRID_COLS - 1
    goals  = {
        (goal_c, GRID_ROWS // 2 - 1),
        (goal_c, GRID_ROWS // 2),
        (goal_c, GRID_ROWS // 2 + 1),
    }
    visited = {(0, 0)}
    queue   = [((0, 0), 0)]

    while queue:
        (c, r), dist = queue.pop(0)
        if (c, r) in goals:
            return dist
        for dc, dr in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nc, nr = c + dc, r + dr
            if (0 <= nc < GRID_COLS and 0 <= nr < GRID_ROWS
                    and (nc, nr) not in visited
                    and colrow_to_idx(nc, nr) not in obstacle_cells):
                visited.add((nc, nr))
                queue.append(((nc, nr), dist + 1))
    return -1


# ─── ŚRODOWISKO ──────────────────────────────────────────────────────────────
class SheepLevelEnv(gym.Env):
    """
    Przestrzeń obserwacji:
        Box(0, 1, shape=(N_CHANNELS=5, GRID_ROWS=13, GRID_COLS=20), float32)
        Kanał 0 – przeszkody (drzewa)
        Kanał 1 – wrogowie (wilki)
        Kanał 2 – spawn gracza
        Kanał 3 – wyjście (brama)
        Kanał 4 – amunicja (kłębek wełny)

    Przestrzeń akcji:
        Discrete(3 * N_CELLS + 1 = 781)
        0   … 259  → postaw drzewo w komórce i
        260 … 519  → postaw wroga  w komórce i - N_CELLS
        520 … 779  → postaw amunicję w komórce i - 2*N_CELLS
        780        → zakończ (done)
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, render_mode=None):
        super().__init__()
        self.render_mode = render_mode

        self.action_space = spaces.Discrete(3 * N_CELLS + 1)
        # 1D wektor – MlpPolicy nie wymaga 3D tensora, brak warningow
        self.observation_space = spaces.Box(
            low=0.0, high=1.0,
            shape=(N_CHANNELS * GRID_ROWS * GRID_COLS,),   # 5*13*20 = 1300
            dtype=np.float32,
        )

        self._obstacle_cells: set = set()
        self._enemy_cells:    set = set()
        self._ammo_cell:      int = -1    # -1 = brak amunicji
        self._steps: int = 0

        self._screen = None
        self._clock  = None

        if render_mode == "human" and not PYGAME_AVAILABLE:
            raise RuntimeError("Zainstaluj pygame: pip install pygame")

    # ── Obserwacja ───────────────────────────────────────────────────────────
    def _get_obs(self) -> np.ndarray:
        obs = np.zeros((N_CHANNELS, GRID_ROWS, GRID_COLS), dtype=np.float32)
        for idx in self._obstacle_cells:
            c, r = idx_to_colrow(idx)
            obs[CH_OBSTACLE, r, c] = 1.0
        for idx in self._enemy_cells:
            c, r = idx_to_colrow(idx)
            obs[CH_ENEMY, r, c] = 1.0
        # Spawn
        obs[CH_SPAWN, 0, 0] = 1.0
        # Exit
        for idx in EXIT_CELLS:
            c, r = idx_to_colrow(idx)
            obs[CH_EXIT, r, c] = 1.0
        # Amunicja
        if self._ammo_cell >= 0:
            c, r = idx_to_colrow(self._ammo_cell)
            obs[CH_AMMO, r, c] = 1.0
        return obs.flatten()

    def _get_info(self) -> dict:
        return {
            "obstacles": len(self._obstacle_cells),
            "enemies":   len(self._enemy_cells),
            "ammo":      1 if self._ammo_cell >= 0 else 0,
            "steps":     self._steps,
        }

    # ── Reset ────────────────────────────────────────────────────────────────
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._obstacle_cells = set()
        self._enemy_cells    = set()
        self._ammo_cell      = -1
        self._steps          = 0
        return self._get_obs(), self._get_info()

    # ── Step ─────────────────────────────────────────────────────────────────
    def step(self, action: int):
        self._steps += 1
        action      = int(action)
        reward      = 0.0
        terminated  = False

        DONE = 3 * N_CELLS
        if action == DONE or self._steps >= MAX_STEPS:
            terminated = True
            reward     = self._compute_final_reward()

        elif action < N_CELLS:                   # postaw drzewo
            if (action not in PROTECTED_CELLS
                    and action not in self._obstacle_cells
                    and action not in self._enemy_cells
                    and action != self._ammo_cell
                    and len(self._obstacle_cells) < MAX_OBSTACLES):
                self._obstacle_cells.add(action)
                reward = 0.1
            else:
                reward = -0.05

        elif action < 2 * N_CELLS:               # postaw wroga
            enemy_idx = action - N_CELLS
            if (enemy_idx not in PROTECTED_CELLS
                    and enemy_idx not in self._obstacle_cells
                    and enemy_idx not in self._enemy_cells
                    and enemy_idx != self._ammo_cell
                    and len(self._enemy_cells) < MAX_ENEMIES):
                self._enemy_cells.add(enemy_idx)
                reward = 0.1
            else:
                reward = -0.05

        else:                                    # postaw amunicję
            ammo_idx = action - 2 * N_CELLS
            if (ammo_idx not in PROTECTED_CELLS
                    and ammo_idx not in self._obstacle_cells
                    and ammo_idx not in self._enemy_cells
                    and self._ammo_cell == -1):   # tylko 1 kłębek
                self._ammo_cell = ammo_idx
                reward = 0.1
            else:
                reward = -0.05

        obs  = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return obs, reward, terminated, False, info

    # ── Nagroda końcowa ──────────────────────────────────────────────────────
    def _compute_final_reward(self) -> float:
        """
        Składowe nagrody:
          BFS przejezdność        → max +10 (proporcjonalnie do długości ścieżki)
          Brak ścieżki            → -20 (kara krytyczna)
          Liczba wrogów           → +5 za optymum (3-4), -2 za brak
          Liczba przeszkód        → +3 za optymum (5-8), -2 za brak
          Drzewo blisko spawnu    → -1 za każde w promieniu 2
          Wróg blisko spawnu      → -2 za każdego w promieniu 3.5
          Amunicja obecna         → +3
          Amunicja za blisko      → -2 (promień <= 3 od spawnu)
          Amunicja za daleko      → -2 (promień >= 14 od spawnu)
          Brak amunicji           → -3
        """
        path_len = bfs_path_length(self._obstacle_cells)

        # Krytyczna kara – mapa nieprzejezdna
        if path_len == -1:
            return -20.0

        reward = 0.0

        # 1. Długość ścieżki BFS
        max_path = GRID_COLS + GRID_ROWS   # ~33
        reward  += 10.0 * (path_len / max_path)

        # 2. Wrogowie – optimum 3-4
        n_e = len(self._enemy_cells)
        if n_e == 0:
            reward -= 2.0
        elif 3 <= n_e <= 4:
            reward += 5.0
        elif n_e <= 6:
            reward += 3.0
        else:
            reward += 1.0

        # 3. Przeszkody – optimum 5-8
        n_o = len(self._obstacle_cells)
        if n_o == 0:
            reward -= 2.0
        elif 5 <= n_o <= 8:
            reward += 3.0
        elif n_o <= 12:
            reward += 1.5

        # 4. Kara za drzewa blisko spawnu (promień 2)
        for idx in self._obstacle_cells:
            c, r = idx_to_colrow(idx)
            if c <= 2 and r <= 2:
                reward -= 1.0

        # 5. Kara za wrogów za blisko spawnu (promień 3.5)
        for idx in self._enemy_cells:
            c, r = idx_to_colrow(idx)
            if math.sqrt(c**2 + r**2) <= 3.5:
                reward -= 2.0

        # 6. Amunicja
        if self._ammo_cell < 0:
            reward -= 3.0   # brak amunicji – duża kara
        else:
            reward += 3.0   # amunicja jest na mapie
            ac, ar = idx_to_colrow(self._ammo_cell)
            dist_spawn = math.sqrt(ac**2 + ar**2)
            dist_exit  = math.sqrt((ac - (GRID_COLS-1))**2 + (ar - GRID_ROWS//2)**2)

            # Za blisko spawnu (gracz zbiera za łatwo zanim trafi na wrogów)
            if dist_spawn <= 3.0:
                reward -= 2.0

            # Za daleko – praktycznie nieosiągalna
            if dist_spawn >= 14.0:
                reward -= 2.0

            # Bonus za amunicję w "środku" mapy (między 4 a 12 od spawnu)
            if 4.0 <= dist_spawn <= 12.0:
                reward += 1.0

            # Kara jeśli amunicja leży NA przeszkodzie lub wrogu
            if (self._ammo_cell in self._obstacle_cells
                    or self._ammo_cell in self._enemy_cells):
                reward -= 2.0

        return round(reward, 3)

    # ── Render ───────────────────────────────────────────────────────────────
    def render(self):
        if self.render_mode == "human":
            self._render_frame()
        elif self.render_mode == "rgb_array":
            return self._render_to_array()

    def _init_pygame(self):
        if self._screen is None:
            pygame.init()
            self._screen = pygame.display.set_mode((MAP_W, MAP_H))
            pygame.display.set_caption("SheepLevelEnv")
            self._clock  = pygame.time.Clock()

    def _render_frame(self):
        self._init_pygame()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close(); return

        surf = self._screen
        surf.fill((95, 175, 55))

        # Linie siatki
        for c in range(GRID_COLS + 1):
            x = BORDER + c * CELL_W
            pygame.draw.line(surf, (75, 150, 40), (x, BORDER), (x, MAP_H - BORDER))
        for r in range(GRID_ROWS + 1):
            y = BORDER + r * CELL_H
            pygame.draw.line(surf, (75, 150, 40), (BORDER, y), (MAP_W - BORDER, y))

        # Płot – ramka
        pygame.draw.rect(surf, (155, 95, 35), (0, 0, MAP_W, MAP_H), BORDER)

        # Strefa spawnu (niebieskawa)
        spawn_rect = pygame.Rect(BORDER, BORDER, CELL_W * 2, CELL_H * 2)
        pygame.draw.rect(surf, (100, 160, 240), spawn_rect, border_radius=3)
        pygame.draw.rect(surf, (60, 100, 200),  spawn_rect, width=2, border_radius=3)

        # Strefa exit (złota)
        for idx in EXIT_CELLS:
            c, r = idx_to_colrow(idx)
            er = pygame.Rect(BORDER + c*CELL_W + 1, BORDER + r*CELL_H + 1,
                             CELL_W - 2, CELL_H - 2)
            pygame.draw.rect(surf, (220, 180, 50), er, border_radius=2)

        # Drzewa
        for idx in self._obstacle_cells:
            c, r = idx_to_colrow(idx)
            tr = pygame.Rect(BORDER + c*CELL_W + 2, BORDER + r*CELL_H + 2,
                             CELL_W - 4, CELL_H - 4)
            pygame.draw.rect(surf, (40, 110, 30),  tr, border_radius=5)
            pygame.draw.rect(surf, (25,  75, 15),  tr, width=1, border_radius=5)

        # Wrogowie
        for idx in self._enemy_cells:
            c, r = idx_to_colrow(idx)
            cx = BORDER + c*CELL_W + CELL_W//2
            cy = BORDER + r*CELL_H + CELL_H//2
            rad = min(CELL_W, CELL_H)//2 - 2
            pygame.draw.circle(surf, (200, 40, 40), (cx, cy), rad)
            pygame.draw.circle(surf, (140, 10, 10), (cx, cy), rad, 1)

        # Amunicja (różowy owal)
        if self._ammo_cell >= 0:
            c, r = idx_to_colrow(self._ammo_cell)
            ammo_r = pygame.Rect(BORDER + c*CELL_W + 3, BORDER + r*CELL_H + 3,
                                 CELL_W - 6, CELL_H - 6)
            pygame.draw.ellipse(surf, (235, 170, 210), ammo_r)
            pygame.draw.ellipse(surf, (170,  90, 150), ammo_r, 1)

        # UI
        font = pygame.font.SysFont(None, 22)
        ammo_txt = "tak" if self._ammo_cell >= 0 else "brak"
        surf.blit(font.render(
            f"Krok {self._steps}/{MAX_STEPS}   "
            f"Drzewa: {len(self._obstacle_cells)}/{MAX_OBSTACLES}   "
            f"Wrogowie: {len(self._enemy_cells)}/{MAX_ENEMIES}   "
            f"Amunicja: {ammo_txt}",
            True, (255, 255, 255)), (BORDER + 4, 4))

        pygame.display.flip()
        self._clock.tick(self.metadata["render_fps"])

    def _render_to_array(self) -> np.ndarray:
        self._init_pygame()
        self._render_frame()
        return np.transpose(
            np.array(pygame.surfarray.pixels3d(self._screen)), axes=(1, 0, 2))

    def close(self):
        if self._screen is not None and PYGAME_AVAILABLE:
            pygame.quit()
            self._screen = None


# ─── SZYBKI TEST ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=== Test SheepLevelEnv ===\n")

    env = SheepLevelEnv(render_mode=None)

    # Walidacja Gymnasium API
    from gymnasium.utils.env_checker import check_env
    check_env(env, warn=True)
    print("check_env: OK\n")

    # 5 losowych epizodów
    for ep in range(5):
        obs, info = env.reset()
        total_r = 0.0
        while True:
            action = env.action_space.sample()
            obs, r, terminated, truncated, info = env.step(action)
            total_r += r
            if terminated or truncated:
                break
        path = bfs_path_length(env._obstacle_cells)
        print(f"Ep {ep+1}: kroki={info['steps']:2d}  "
              f"drzewa={info['obstacles']:2d}  "
              f"wrogowie={info['enemies']}  "
              f"BFS={path:3d}  "
              f"nagroda={total_r:6.2f}")

    env.close()
    print("\nŚrodowisko działa poprawnie!")
    print("\n── Przykład użycia ze Stable Baselines3 ──")
    print("""
from stable_baselines3 import PPO
from sheep_env import SheepLevelEnv

env   = SheepLevelEnv(render_mode=None)
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=100_000)
model.save("sheep_ppo")

# Podgląd wyuczonego agenta
env_vis = SheepLevelEnv(render_mode="human")
obs, _  = env_vis.reset()
for _ in range(MAX_STEPS):
    action, _ = model.predict(obs, deterministic=True)
    obs, r, done, _, _ = env_vis.step(action)
    if done:
        break
env_vis.close()
    """)