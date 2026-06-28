
import pygame, random, math, os, json

pygame.init()
pygame.joystick.init()

WIDTH, HEIGHT = 600, 400
screen = pygame.display.set_mode((WIDTH, HEIGHT), pygame.FULLSCREEN | pygame.SCALED)
pygame.display.set_caption("Owca Kontra Wilki")
clock = pygame.time.Clock()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SPR_DIR  = os.path.join(BASE_DIR, "sprites")

def load(name):
    return pygame.image.load(os.path.join(SPR_DIR, f"{name}.png")).convert_alpha()

spr = {n: load(n) for n in [
    "player","enemy","enemy_shooter","enemy_boss","enemy_bullet",
    "tree_s","tree_m","tree_l","ammo","bullet","gate","fence","grass"
]}

TREE_SPRITES = [spr["tree_s"], spr["tree_m"], spr["tree_l"]]
GRASS_TILE   = spr["grass"];   GW, GH = GRASS_TILE.get_size()
FENCE_TILE   = spr["fence"];   FW, FH = FENCE_TILE.get_size()
fence_v      = pygame.transform.rotate(FENCE_TILE, 90)
FVW, FVH     = fence_v.get_size()

def draw_grass():
    for ty in range(0, HEIGHT, GH):
        for tx in range(0, WIDTH, GW):
            screen.blit(GRASS_TILE, (tx, ty))

def draw_fence():
    for tx in range(0, WIDTH, FW):
        screen.blit(FENCE_TILE, (tx, 0))
        screen.blit(FENCE_TILE, (tx, HEIGHT - FH))
    for ty in range(FH, HEIGHT - FH, FVH):
        screen.blit(fence_v, (0, ty))
        screen.blit(fence_v, (WIDTH - FVW, ty))

def draw_health_ui(hp):
    hp = max(0, min(3, hp))
    label = font.render(f"HP: {hp}/3", True, (120, 35, 45))
    screen.blit(label, (BORDER+5, BORDER+5))

    for i in range(3):
        x = BORDER + 72 + i * 18
        y = BORDER + 8
        color = (220, 60, 80) if i < hp else (120, 115, 110)
        pygame.draw.circle(screen, color, (x + 4, y + 4), 4)
        pygame.draw.circle(screen, color, (x + 10, y + 4), 4)
        pygame.draw.polygon(screen, color, [(x, y + 6), (x + 14, y + 6), (x + 7, y + 15)])
        pygame.draw.polygon(screen, (80, 45, 45), [(x, y + 6), (x + 14, y + 6), (x + 7, y + 15)], 1)

BORDER          = 22
IFRAME_DURATION = 60   # klatki nietykalności po trafieniu
JOY_DEADZONE    = 0.35


def init_joystick():
    if pygame.joystick.get_count() == 0:
        return None
    pad = pygame.joystick.Joystick(0)
    pad.init()
    print(f"Pad wykryty: {pad.get_name()}")
    return pad


joystick = init_joystick()


def joy_axis(pad, axis):
    if not pad or pad.get_numaxes() <= axis:
        return 0.0
    value = pad.get_axis(axis)
    return value if abs(value) >= JOY_DEADZONE else 0.0


def joy_button(pad, button):
    return bool(pad and pad.get_numbuttons() > button and pad.get_button(button))



class Enemy:
    KIND   = "normal"
    SIZE   = 32
    SPEED  = 2.0
    HP     = 1
    SPRITE = "enemy"

    def __init__(self, x, y):
        self.x, self.y = float(x), float(y)
        self.hp     = self.HP
        self.invuln = 0

    @property
    def rect(self):
        return pygame.Rect(int(self.x), int(self.y), self.SIZE, self.SIZE)

    def update(self, player_pos, obs_rects, enemy_bullets):
        dx = player_pos[0] - self.x
        dy = player_pos[1] - self.y
        dist = math.hypot(dx, dy)
        if dist:
            self.x += self.SPEED * dx / dist
            self.y += self.SPEED * dy / dist
        self.x = max(BORDER, min(self.x, WIDTH  - BORDER - self.SIZE))
        self.y = max(BORDER, min(self.y, HEIGHT - BORDER - self.SIZE))
        if self.invuln > 0:
            self.invuln -= 1

    def hit_by(self, bullet_rect):
        if self.invuln > 0:
            return False
        if self.rect.colliderect(bullet_rect):
            self.hp     -= 1
            self.invuln  = 8
            return True
        return False

    def draw(self, surface):
        img = spr[self.SPRITE]
        if self.invuln > 0 and (self.invuln % 4) < 2:
            img = img.copy()
            img.fill((255,255,255,120), special_flags=pygame.BLEND_RGBA_ADD)
        surface.blit(img, (int(self.x), int(self.y)))


class ShooterEnemy(Enemy):
    KIND   = "shooter"
    SPEED  = 1.3
    HP     = 2
    SPRITE = "enemy_shooter"
    SHOOT_COOLDOWN = 90

    def __init__(self, x, y):
        super().__init__(x, y)
        self.shoot_timer = random.randint(30, self.SHOOT_COOLDOWN)

    def update(self, player_pos, obs_rects, enemy_bullets):
        dx = player_pos[0] - self.x
        dy = player_pos[1] - self.y
        dist = math.hypot(dx, dy)

        if dist < 140:
            self.x -= self.SPEED * dx / dist if dist else 0
            self.y -= self.SPEED * dy / dist if dist else 0
        elif dist > 200:
            self.x += self.SPEED * dx / dist if dist else 0
            self.y += self.SPEED * dy / dist if dist else 0

        self.x = max(BORDER, min(self.x, WIDTH  - BORDER - self.SIZE))
        self.y = max(BORDER, min(self.y, HEIGHT - BORDER - self.SIZE))

        self.shoot_timer -= 1
        if self.shoot_timer <= 0 and dist:
            self.shoot_timer = self.SHOOT_COOLDOWN
            spd = 4.5
            enemy_bullets.append(EnemyBullet(
                self.x + self.SIZE//2, self.y + self.SIZE//2,
                spd * dx / dist, spd * dy / dist,
            ))
        if self.invuln > 0:
            self.invuln -= 1


class BossEnemy(Enemy):
    KIND   = "boss"
    SIZE   = 64
    SPEED  = 1.2
    HP     = 20
    SPRITE = "enemy_boss"
    SHOOT_COOLDOWN = 50

    def __init__(self):
        super().__init__(WIDTH - 100, HEIGHT // 2 - 32)
        self.shoot_timer = self.SHOOT_COOLDOWN
        self.phase = 1

    def update(self, player_pos, obs_rects, enemy_bullets):
        dx = player_pos[0] - self.x
        dy = player_pos[1] - self.y
        dist = math.hypot(dx, dy)

        self.phase = 2 if self.hp < self.HP // 2 else 1
        spd = self.SPEED * (1.5 if self.phase == 2 else 1.0)

        if dist:
            self.x += spd * dx / dist
            self.y += spd * dy / dist

        self.x = max(BORDER, min(self.x, WIDTH  - BORDER - self.SIZE))
        self.y = max(BORDER, min(self.y, HEIGHT - BORDER - self.SIZE))

        cooldown = 35 if self.phase == 2 else self.SHOOT_COOLDOWN
        self.shoot_timer -= 1
        if self.shoot_timer <= 0 and dist:
            self.shoot_timer = cooldown
            spd_b = 5.0
            base  = math.atan2(dy, dx)
            angles = [base] if self.phase == 1 else [base-0.3, base, base+0.3]
            for a in angles:
                enemy_bullets.append(EnemyBullet(
                    self.x + self.SIZE//2, self.y + self.SIZE//2,
                    spd_b * math.cos(a), spd_b * math.sin(a),
                ))
        if self.invuln > 0:
            self.invuln -= 1

    def draw(self, surface):
        img = spr[self.SPRITE]
        if self.invuln > 0 and (self.invuln % 4) < 2:
            img = img.copy()
            img.fill((200,0,0,100), special_flags=pygame.BLEND_RGBA_ADD)
        surface.blit(img, (int(self.x), int(self.y)))


class EnemyBullet:
    def __init__(self, x, y, vx, vy):
        self.x, self.y = float(x), float(y)
        self.vx, self.vy = vx, vy

    @property
    def rect(self):
        return pygame.Rect(int(self.x)-4, int(self.y)-4, 8, 8)

    def update(self):
        self.x += self.vx
        self.y += self.vy

    def alive(self):
        return BORDER < self.x < WIDTH-BORDER and BORDER < self.y < HEIGHT-BORDER

    def draw(self, surface):
        surface.blit(spr["enemy_bullet"], (int(self.x)-4, int(self.y)-4))


#GENERATOR MAPY
PROTECTED_ZONES = [
    pygame.Rect(28, 28, 100, 100),
    pygame.Rect(WIDTH-110, HEIGHT//2-55, 100, 100),
    pygame.Rect(0, 0, WIDTH, BORDER+8),
    pygame.Rect(0, HEIGHT-BORDER-8, WIDTH, BORDER+8),
    pygame.Rect(0, 0, BORDER+8, HEIGHT),
    pygame.Rect(WIDTH-BORDER-8, 0, BORDER+8, HEIGHT),
]

def rect_protected(r):
    return any(r.colliderect(z) for z in PROTECTED_ZONES)

def check_path(obs_rects):
    G = 12; cols, rows = WIDTH//G, HEIGHT//G
    def blocked(gx, gy):
        r = pygame.Rect(gx*G+1, gy*G+1, G-2, G-2)
        if r.left<BORDER or r.top<BORDER or r.right>WIDTH-BORDER or r.bottom>HEIGHT-BORDER:
            return True
        return any(r.colliderect(o) for o in obs_rects)
    sx, sy   = 50//G, 50//G
    gx2, gy2 = (WIDTH-70)//G, (HEIGHT//2)//G
    visited  = {(sx,sy)}; q = [(sx,sy)]
    while q:
        cx,cy = q.pop(0)
        if cx==gx2 and cy==gy2: return True
        for ddx,ddy in [(-1,0),(1,0),(0,-1),(0,1)]:
            nx,ny = cx+ddx, cy+ddy
            if 0<=nx<cols and 0<=ny<rows and (nx,ny) not in visited and not blocked(nx,ny):
                visited.add((nx,ny)); q.append((nx,ny))
    return False

#ZMIENNE GLOBALNE
player_pos    = [50, 50]
player_hp     = 3
player_invuln = 0
player_size   = 32
speed         = 4

obstacles      = []
player_bullets = []
enemy_bullets  = []
enemies        = []
has_ammo       = False
ammo_rect      = None
door_rect      = None
level          = 0

font     = pygame.font.SysFont(None, 26)
font_big = pygame.font.SysFont(None, 48)
font_menu = pygame.font.SysFont(None, 34)

LEVEL_JSON      = os.path.join(BASE_DIR, 'ai_levels.json')
BOSS_LEVEL_JSON = os.path.join(BASE_DIR, 'boss_levels.json')
ai_mode      = False   # True = poziomy od agenta AI, False = losowe
ai_level_idx = 0       # który poziom z listy wczytać następny
boss_level_idx = 0


def load_ai_level():
    """Wczytuje kolejny poziom z ai_levels.json (cyklicznie)."""
    global obstacles, player_pos, enemies, ammo_rect
    global player_bullets, enemy_bullets, has_ammo, door_rect, level, ai_level_idx

    if not os.path.exists(LEVEL_JSON):
        print("Brak ai_levels.json – uruchom play_trained.py żeby wygenerować poziomy")
        generate_map()
        return

    with open(LEVEL_JSON) as f:
        all_levels = json.load(f)

    if not all_levels:
        print("ai_levels.json jest pusty – przełączam na losowy")
        generate_map()
        return

    data = all_levels[ai_level_idx % len(all_levels)]
    ai_level_idx += 1

    level    += 1
    obstacles      = []; player_bullets = []; enemy_bullets = []
    has_ammo  = False;  enemies = [];          door_rect = None
    player_pos[:] = [50, 50]

    total = len(all_levels)
    idx   = (ai_level_idx - 1) % total + 1
    print(f"Wczytuję poziom AI {idx}/{total}")

    # Wczytaj drzewa
    for o in data.get("obstacles", []):
        r = pygame.Rect(o["x"], o["y"], o["w"], o["h"])
        obstacles.append((r, random.choice(TREE_SPRITES)))

    # Wczytaj wrogów
    for e in data.get("enemies", []):
        cls = ShooterEnemy if random.random() < 0.4 else Enemy
        enemies.append(cls(e["x"], e["y"]))

    # Wczytaj amunicję
    if data.get("ammo"):
        a = data["ammo"]
        ammo_rect = pygame.Rect(a["x"] - 11, a["y"] - 11, 22, 22)
    else:
        ammo_rect = pygame.Rect(WIDTH//2-11, HEIGHT//2-11, 22, 22)


def load_ai_boss_level():
    global obstacles, player_pos, enemies, ammo_rect
    global player_bullets, enemy_bullets, has_ammo, door_rect, level, boss_level_idx

    if not os.path.exists(BOSS_LEVEL_JSON):
        print("Brak boss_levels.json - uruchom play_trained_boss.py zeby wygenerowac areny")
        generate_map()
        return

    with open(BOSS_LEVEL_JSON) as f:
        all_levels = json.load(f)

    if not all_levels:
        print("boss_levels.json jest pusty - przelaczam na losowego bossa")
        generate_map()
        return

    data = all_levels[boss_level_idx % len(all_levels)]
    boss_level_idx += 1

    level += 1
    obstacles = []; player_bullets = []; enemy_bullets = []
    has_ammo = False; enemies = []; door_rect = None
    player_pos[:] = [50, 50]

    total = len(all_levels)
    idx = (boss_level_idx - 1) % total + 1
    print(f"Wczytuje poziom BOSS AI {idx}/{total}")

    for o in data.get("obstacles", []):
        r = pygame.Rect(o["x"], o["y"], o["w"], o["h"])
        obstacles.append((r, random.choice(TREE_SPRITES)))

    enemies.append(BossEnemy())

    if data.get("ammo"):
        a = data["ammo"]
        ammo_rect = pygame.Rect(a["x"] - 11, a["y"] - 11, 22, 22)
    else:
        ammo_rect = pygame.Rect(WIDTH//2-11, HEIGHT//2-11, 22, 22)


def load_next_ai_level():
    if (level + 1) % 5 == 0:
        load_ai_boss_level()
    else:
        load_ai_level()


def generate_map():
    global obstacles, player_pos, enemies, ammo_rect
    global player_bullets, enemy_bullets, has_ammo, door_rect, level

    level    += 1
    obstacles      = []; player_bullets = []; enemy_bullets = []
    has_ammo  = False;  enemies = [];          door_rect = None
    player_pos[:] = [50, 50]

    is_boss = (level % 5 == 0)

    # Przeszkody
    for _ in range(500):
        if len(obstacles) >= min(3 + level, 12):
            break
        t = random.choice(TREE_SPRITES)
        tw, th = t.get_size()
        x = random.randint(BORDER+5, WIDTH -BORDER-tw-5)
        y = random.randint(BORDER+5, HEIGHT-BORDER-th-5)
        c = pygame.Rect(x, y, tw, th)
        if rect_protected(c): continue
        if any(c.inflate(10,10).colliderect(o[0]) for o in obstacles): continue
        obstacles.append((c, t))

    obs_rects = [o[0] for o in obstacles]
    for safety in range(25):
        if check_path(obs_rects): break
        obstacles.pop(random.randint(0, len(obstacles)-1))
        obs_rects = [o[0] for o in obstacles]

    # Wrogowie
    if is_boss:
        enemies.append(BossEnemy())
    else:
        num_s = min(level//2, 3)
        num_n = random.randint(1, max(1, 3 - level//3))
        for i in range(num_s + num_n):
            for _ in range(80):
                ex = random.randint(200, 520)
                ey = random.randint(BORDER+10, HEIGHT-BORDER-40)
                er = pygame.Rect(ex, ey, 32, 32)
                if not any(er.colliderect(o[0]) for o in obstacles):
                    enemies.append(ShooterEnemy(ex, ey) if i < num_s else Enemy(ex, ey))
                    break

    # Amunicja
    for _ in range(300):
        ax = random.randint(BORDER+20, WIDTH -BORDER-30)
        ay = random.randint(BORDER+20, HEIGHT-BORDER-30)
        ar = pygame.Rect(ax, ay, 22, 22)
        if not any(ar.colliderect(o[0]) for o in obstacles) and not rect_protected(ar):
            ammo_rect = ar; break
    else:
        ammo_rect = pygame.Rect(WIDTH//2-11, HEIGHT//2-11, 22, 22)


generate_map()
level = 0

#PĘTLA GRY
running = True
paused = False
resume_btn = pygame.Rect(WIDTH//2 - 90, HEIGHT//2 - 25, 180, 42)
quit_btn = pygame.Rect(WIDTH//2 - 90, HEIGHT//2 + 30, 180, 42)
while running:

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.JOYDEVICEADDED:
            joystick = init_joystick()
        if event.type == pygame.JOYDEVICEREMOVED:
            joystick = init_joystick()
        if event.type == pygame.JOYBUTTONDOWN:
            if event.button in (7, 9):
                paused = not paused
            elif paused and event.button == 0:
                paused = False
            elif paused and event.button in (1, 6):
                running = False
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                paused = not paused
            elif paused and event.key in (pygame.K_RETURN, pygame.K_SPACE):
                paused = False
            elif paused and event.key == pygame.K_q:
                running = False
            elif not paused and event.key == pygame.K_TAB:
                ai_mode = not ai_mode
                mode_name = 'AI' if ai_mode else 'LOSOWY'
                print(f'Tryb poziomów: {mode_name}')
        if paused and event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if resume_btn.collidepoint(event.pos):
                paused = False
            elif quit_btn.collidepoint(event.pos):
                running = False

    if paused:
        overlay = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 150))
        screen.blit(overlay, (0, 0))

        title = font_big.render("PAUZA", True, (245, 238, 220))
        screen.blit(title, (WIDTH//2 - title.get_width()//2, HEIGHT//2 - 90))

        pygame.draw.rect(screen, (80, 135, 70), resume_btn, border_radius=6)
        pygame.draw.rect(screen, (130, 65, 60), quit_btn, border_radius=6)
        pygame.draw.rect(screen, (245, 238, 220), resume_btn, width=2, border_radius=6)
        pygame.draw.rect(screen, (245, 238, 220), quit_btn, width=2, border_radius=6)

        resume_txt = font_menu.render("Wznow", True, (255, 250, 235))
        quit_txt = font_menu.render("Wyjdz", True, (255, 250, 235))
        screen.blit(resume_txt, (resume_btn.centerx - resume_txt.get_width()//2,
                                 resume_btn.centery - resume_txt.get_height()//2))
        screen.blit(quit_txt, (quit_btn.centerx - quit_txt.get_width()//2,
                               quit_btn.centery - quit_txt.get_height()//2))

        hint = font.render("ESC/Start - wznow    Q/B - wyjdz", True, (225, 218, 200))
        screen.blit(hint, (WIDTH//2 - hint.get_width()//2, quit_btn.bottom + 18))
        pygame.display.flip()
        clock.tick(30)
        continue

    keys = pygame.key.get_pressed()
    hat_x = hat_y = 0
    if joystick and joystick.get_numhats() > 0:
        hat_x, hat_y = joystick.get_hat(0)

    nr = pygame.Rect(player_pos[0], player_pos[1], player_size, player_size)
    if keys[pygame.K_w]: nr.y -= speed
    if keys[pygame.K_s]: nr.y += speed
    if keys[pygame.K_a]: nr.x -= speed
    if keys[pygame.K_d]: nr.x += speed
    pad_x = joy_axis(joystick, 0)
    pad_y = joy_axis(joystick, 1)
    if hat_x:
        pad_x = float(hat_x)
    if hat_y:
        pad_y = float(-hat_y)
    nr.x += int(round(speed * pad_x))
    nr.y += int(round(speed * pad_y))
    nr.left   = max(nr.left,   BORDER)
    nr.right  = min(nr.right,  WIDTH -BORDER)
    nr.top    = max(nr.top,    BORDER)
    nr.bottom = min(nr.bottom, HEIGHT-BORDER)
    if not any(nr.colliderect(o[0]) for o in obstacles):
        player_pos = [nr.x, nr.y]

    player_rect = pygame.Rect(player_pos[0], player_pos[1], player_size, player_size)
    if player_invuln > 0:
        player_invuln -= 1

    if ammo_rect and player_rect.colliderect(ammo_rect):
        has_ammo = True; ammo_rect = None

    if has_ammo:
        cx, cy = player_pos[0]+16, player_pos[1]+16
        if keys[pygame.K_UP]:    player_bullets.append([cx, cy,  0.0, -7.0])
        if keys[pygame.K_DOWN]:  player_bullets.append([cx, cy,  0.0,  7.0])
        if keys[pygame.K_LEFT]:  player_bullets.append([cx, cy, -7.0,  0.0])
        if keys[pygame.K_RIGHT]: player_bullets.append([cx, cy,  7.0,  0.0])
        aim_x = joy_axis(joystick, 2)
        aim_y = joy_axis(joystick, 3)
        if abs(aim_x) > abs(aim_y):
            if aim_x > 0: player_bullets.append([cx, cy,  7.0,  0.0])
            if aim_x < 0: player_bullets.append([cx, cy, -7.0,  0.0])
        elif abs(aim_y) > 0:
            if aim_y > 0: player_bullets.append([cx, cy,  0.0,  7.0])
            if aim_y < 0: player_bullets.append([cx, cy,  0.0, -7.0])
        if joy_button(joystick, 3): player_bullets.append([cx, cy,  0.0, -7.0])
        if joy_button(joystick, 0): player_bullets.append([cx, cy,  0.0,  7.0])
        if joy_button(joystick, 2): player_bullets.append([cx, cy, -7.0,  0.0])
        if joy_button(joystick, 1): player_bullets.append([cx, cy,  7.0,  0.0])

    obs_rects = [o[0] for o in obstacles]
    player_bullets = [
        b for b in ([b.__setitem__(0, b[0]+b[2]) or b.__setitem__(1, b[1]+b[3]) or b
                     for b in player_bullets])
        if BORDER < b[0] < WIDTH-BORDER and BORDER < b[1] < HEIGHT-BORDER
        and not any(pygame.Rect(int(b[0])-4,int(b[1])-4,8,8).colliderect(o) for o in obs_rects)
    ]

    new_eb = []
    for eb in enemy_bullets:
        eb.update()
        if eb.alive():
            new_eb.append(eb)
        if eb.rect.colliderect(player_rect) and player_invuln == 0:
            player_hp    -= 1
            player_invuln = IFRAME_DURATION
    enemy_bullets = new_eb

    new_enemies = []
    for e in enemies:
        e.update(player_pos, obs_rects, enemy_bullets)
        for b in player_bullets:
            e.hit_by(pygame.Rect(int(b[0])-4, int(b[1])-4, 8, 8))
        if e.hp > 0:
            new_enemies.append(e)
        if e.rect.colliderect(player_rect) and player_invuln == 0:
            player_hp    -= 1
            player_invuln = IFRAME_DURATION
    enemies = new_enemies

    if not enemies and door_rect is None:
        door_rect = pygame.Rect(WIDTH-66, HEIGHT//2-22, 40, 44)
    if door_rect and player_rect.colliderect(door_rect):
        if ai_mode: load_next_ai_level()
        else:       generate_map()

    if player_hp <= 0:
        level = 0; player_hp = 3
        if ai_mode: load_next_ai_level()
        else:       generate_map()

    draw_grass()
    draw_fence()

    for obs_rect, obs_spr in obstacles:
        screen.blit(obs_spr, (obs_rect.x, obs_rect.y))

    if ammo_rect:
        screen.blit(spr["ammo"], (ammo_rect.x, ammo_rect.y))
    if door_rect:
        screen.blit(spr["gate"], (door_rect.x, door_rect.y))

    for e in enemies:
        e.draw(screen)

    for eb in enemy_bullets:
        eb.draw(screen)

    if not (player_invuln > 0 and (player_invuln % 8) < 4):
        screen.blit(spr["player"], (player_pos[0], player_pos[1]))

    for b in player_bullets:
        screen.blit(spr["bullet"], (int(b[0])-4, int(b[1])-4))

    draw_health_ui(player_hp)

    lvl_s = font.render(f"Poziom: {level}", True, (60,40,10))
    screen.blit(lvl_s, (WIDTH//2 - lvl_s.get_width()//2, BORDER+5))

    mode_col = (20, 120, 200) if ai_mode else (80, 80, 80)
    mode_txt = font.render("Tryb: AI [TAB]" if ai_mode else "Tryb: losowy [TAB]", True, mode_col)
    screen.blit(mode_txt, (WIDTH - mode_txt.get_width() - BORDER - 5, BORDER + 5))

    if level % 5 == 4:
        warn = font.render("⚠ Następny poziom: BOSS!", True, (200,30,30))
        screen.blit(warn, (WIDTH//2 - warn.get_width()//2, BORDER+22))

    boss = next((e for e in enemies if e.KIND == "boss"), None)
    if boss:
        BAR_W, BAR_H = 300, 18
        bx = WIDTH//2 - BAR_W//2
        by = HEIGHT - BORDER - BAR_H - 6
        pygame.draw.rect(screen, (60,10,10),   (bx-2, by-2, BAR_W+4, BAR_H+4), border_radius=4)
        pygame.draw.rect(screen, (40,40,40),   (bx,   by,   BAR_W,   BAR_H),   border_radius=3)
        fill  = int(BAR_W * boss.hp / boss.HP)
        color = (180,20,20) if boss.phase == 1 else (220,80,20)
        if fill > 0:
            pygame.draw.rect(screen, color, (bx, by, fill, BAR_H), border_radius=3)
        pygame.draw.rect(screen, (200,160,30), (bx-2, by-2, BAR_W+4, BAR_H+4),
                         width=2, border_radius=4)
        label = "BOSS" if boss.phase == 1 else "BOSS ☠ FAZA II"
        bt = font.render(f"{label}  {boss.hp}/{boss.HP}", True, (255,230,180))
        screen.blit(bt, (WIDTH//2 - bt.get_width()//2, by - 20))

    if not has_ammo and ammo_rect:
        screen.blit(font.render("Zbierz kłębek wełny! [WSAD=ruch  strzałki=strzał]",
                                True,(120,60,10)), (BORDER+5, HEIGHT-BORDER-24))
    elif door_rect:
        screen.blit(font.render("Brama otwarta! Wyjdź →", True,(40,120,20)),
                    (BORDER+5, HEIGHT-BORDER-24))

    pygame.display.flip()
    clock.tick(60)

pygame.quit()
