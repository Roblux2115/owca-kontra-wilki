
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
MAX_AMMO      = 1

def idx_to_colrow(idx: int):
    return idx % GRID_COLS, idx // GRID_COLS

def colrow_to_idx(col: int, row: int) -> int:
    return col + row * GRID_COLS

SPAWN_CELL  = colrow_to_idx(0, 0)
EXIT_CELLS  = {
    colrow_to_idx(GRID_COLS - 1, r)
    for r in range(GRID_ROWS // 2 - 1, GRID_ROWS // 2 + 2)
}
PROTECTED_CELLS = {SPAWN_CELL} | EXIT_CELLS
BOSS_AMMO_CELLS = {
    colrow_to_idx(c, r)
    for c in range(3, 8)
    for r in range(1, 6)
}

CH_OBSTACLE, CH_ENEMY, CH_SPAWN, CH_EXIT, CH_AMMO = 0, 1, 2, 3, 4
N_CHANNELS = 5

def bfs_path_length(obstacle_cells: set) -> int:

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


class SheepLevelEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, render_mode=None, mode="normal"):
        super().__init__()
        self.render_mode = render_mode
        if mode not in ("normal", "boss"):
            raise ValueError("mode must be 'normal' or 'boss'")
        self.mode = mode

        self.action_space = spaces.Discrete(3 * N_CELLS + 1)
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

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._obstacle_cells = set()
        self._enemy_cells    = set()
        self._ammo_cell      = -1
        self._steps          = 0
        return self._get_obs(), self._get_info()

    def step(self, action: int):
        self._steps += 1
        action      = int(action)
        reward      = 0.0
        terminated  = False

        DONE = 3 * N_CELLS
        if action == DONE or self._steps >= MAX_STEPS:
            terminated = True
            reward     = self._compute_final_reward()

        elif action < N_CELLS:
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
            if (self.mode != "boss"
                    and enemy_idx not in PROTECTED_CELLS
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
                    and (self.mode != "boss" or ammo_idx in BOSS_AMMO_CELLS)
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

    def _compute_final_reward(self) -> float:
        if self.mode == "boss":
            return self._compute_boss_reward()

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

            # Kara jeśli amunicja leży na przeszkodzie lub wrogu
            if (self._ammo_cell in self._obstacle_cells
                    or self._ammo_cell in self._enemy_cells):
                reward -= 2.0

        return round(reward, 3)

    def _compute_boss_reward(self) -> float:
        path_len = bfs_path_length(self._obstacle_cells)
        if path_len == -1:
            return -20.0

        reward = 0.0
        boss_c, boss_r = GRID_COLS - 4, GRID_ROWS // 2

        n_o = len(self._obstacle_cells)
        if 3 <= n_o <= 7:
            reward += 5.0
        elif 1 <= n_o <= 10:
            reward += 2.0
        else:
            reward -= 2.0

        for idx in self._obstacle_cells:
            c, r = idx_to_colrow(idx)
            dist_spawn = math.sqrt(c**2 + r**2)
            dist_boss = math.sqrt((c - boss_c)**2 + (r - boss_r)**2)
            if dist_spawn <= 3.0:
                reward -= 2.0
            if dist_boss <= 2.5:
                reward -= 2.0
            if 4.0 <= dist_boss <= 7.0:
                reward += 0.4

        if self._ammo_cell < 0:
            reward -= 8.0
        else:
            reward += 6.0
            ac, ar = idx_to_colrow(self._ammo_cell)
            dist_spawn = math.sqrt(ac**2 + ar**2)
            dist_boss = math.sqrt((ac - boss_c)**2 + (ar - boss_r)**2)

            if self._ammo_cell in BOSS_AMMO_CELLS:
                reward += 6.0
            else:
                reward -= 6.0

            if dist_boss <= 3.0:
                reward -= 3.0
            elif 6.0 <= dist_boss <= 12.0:
                reward += 2.0

        if self._enemy_cells:
            reward -= 3.0 * len(self._enemy_cells)

        # Lekki bonus za przejrzysta arene.
        reward += min(path_len, GRID_COLS + GRID_ROWS) / 6.0
        return round(reward, 3)

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

        # Płot
        pygame.draw.rect(surf, (155, 95, 35), (0, 0, MAP_W, MAP_H), BORDER)

        # Strefa spawnu
        spawn_rect = pygame.Rect(BORDER, BORDER, CELL_W * 2, CELL_H * 2)
        pygame.draw.rect(surf, (100, 160, 240), spawn_rect, border_radius=3)
        pygame.draw.rect(surf, (60, 100, 200),  spawn_rect, width=2, border_radius=3)

        # Strefa exit
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

        # Amunicja
        if self._ammo_cell >= 0:
            c, r = idx_to_colrow(self._ammo_cell)
            ammo_r = pygame.Rect(BORDER + c*CELL_W + 3, BORDER + r*CELL_H + 3,
                                 CELL_W - 6, CELL_H - 6)
            pygame.draw.ellipse(surf, (235, 170, 210), ammo_r)
            pygame.draw.ellipse(surf, (170,  90, 150), ammo_r, 1)

        font = pygame.font.SysFont(None, 22)
        ammo_txt = "tak" if self._ammo_cell >= 0 else "brak"
        surf.blit(font.render(
            f"Krok {self._steps}/{MAX_STEPS}   "
            f"Tryb: {self.mode}   "
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


if __name__ == "__main__":
    print("=== Test SheepLevelEnv ===\n")

    env = SheepLevelEnv(render_mode=None)

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
