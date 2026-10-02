"""
game.py
-------
Stage 5 (Game Engine Development) of the project methodology.

An ancient-ruins-themed, 3-lane endless runner built with Pygame, visually
styled after the "temple runner" genre (jungle ruins, wood/stone bridge,
glowing jungle on either side, gold coin gems, skull/pillar framing) --
built entirely from original procedurally-drawn shapes (no copied art
assets, sprites, or logos from any commercial game).

Core gameplay:
  - 3-lane scrolling track (left / center / right)
  - Procedurally generated obstacles and coins
  - Jump (arc) and Duck (crouch hitbox) actions
  - AABB collision detection
  - Increasing difficulty/speed curve over time
  - Live distance / score / coin counters + lives

Control surface (Stage 4 maps gestures onto exactly these functions, and
keyboard input uses the same functions, so both input sources are
first-class and interchangeable):
    cmd_left() / cmd_right() / cmd_jump() / cmd_duck()
    cmd_start()   -- begin the run, or resume after a pause
    cmd_stop()    -- pause the run (freezes the character/world)
    cmd_restart() -- reset after game over
"""

import math
import random
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Tuple

import pygame


# ---------------------------------------------------------------------------
# Configuration / constants
# ---------------------------------------------------------------------------

SCREEN_W, SCREEN_H = 900, 600
FPS = 60

LANE_COUNT = 3
LANE_WIDTH = 220
TRACK_LEFT = (SCREEN_W - LANE_WIDTH * LANE_COUNT) // 2
GROUND_Y = SCREEN_H - 120
HORIZON_Y = 90

PLAYER_W, PLAYER_H = 70, 100
PLAYER_DUCK_H = 55

JUMP_VELOCITY = -17.0
GRAVITY = 0.85

BASE_SCROLL_SPEED = 7.0
MAX_SCROLL_SPEED = 22.0
SPEED_RAMP_PER_SEC = 0.06  # how fast difficulty ramps up

OBSTACLE_MIN_GAP_MS = 700
OBSTACLE_MAX_GAP_MS = 1400
COIN_ROW_CHANCE = 0.55

STARTING_LIVES = 3
INVINCIBILITY_MS = 1200

# ---------------------------------------------------------------------------
# Ancient-ruins / jungle-runner color palette (original art direction --
# warm wood & stone path, glowing teal jungle either side, gold accents)
# ---------------------------------------------------------------------------

COLOR_SKY_TOP = (8, 28, 26)          # dark jungle canopy / night sky
COLOR_SKY_BOTTOM = (18, 55, 48)      # lighter toward the horizon

COLOR_GLOW_DARK = (10, 60, 55)       # jungle glow, far / dim
COLOR_GLOW_BRIGHT = (35, 190, 165)   # jungle glow, near / bright (teal)

COLOR_STONE_BORDER = (96, 86, 74)    # mossy stone edge of the bridge
COLOR_STONE_BORDER_DARK = (60, 52, 44)
COLOR_MOSS = (58, 120, 72)

COLOR_PLANK_LIGHT = (134, 92, 52)
COLOR_PLANK_DARK = (96, 63, 34)
COLOR_PLANK_SEAM = (58, 38, 20)

COLOR_GOLD = (232, 184, 68)
COLOR_GOLD_DARK = (150, 108, 30)
COLOR_GOLD_TEXT_SHADOW = (40, 26, 8)

COLOR_SKULL = (222, 214, 190)
COLOR_SKULL_SHADOW = (150, 140, 118)

COLOR_PLAYER_SKIN = (222, 178, 140)
COLOR_PLAYER_SHIRT = (196, 164, 112)
COLOR_PLAYER_SHIRT_DUCK = (168, 128, 78)
COLOR_PLAYER_PANTS = (64, 52, 44)

COLOR_OBSTACLE_LOW = (128, 122, 112)     # carved stone block -> must JUMP
COLOR_OBSTACLE_LOW_DARK = (86, 82, 74)
COLOR_OBSTACLE_HIGH = (94, 66, 40)       # hanging wooden beam -> must DUCK
COLOR_OBSTACLE_HIGH_MOSS = (66, 110, 62)

COLOR_COIN_GOLD = (255, 210, 70)
COLOR_COIN_GOLD_DARK = (196, 148, 30)

COLOR_TEXT = (250, 244, 226)
COLOR_TEXT_WARN = (235, 90, 80)


def lane_center_x(lane_index: int) -> int:
    return TRACK_LEFT + LANE_WIDTH * lane_index + LANE_WIDTH // 2


def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def _lerp_color(c1: Tuple[int, int, int], c2: Tuple[int, int, int], t: float) -> Tuple[int, int, int]:
    t = max(0.0, min(1.0, t))
    return (
        int(_lerp(c1[0], c2[0], t)),
        int(_lerp(c1[1], c2[1], t)),
        int(_lerp(c1[2], c2[2], t)),
    )


class ObstacleType(Enum):
    LOW = "low"    # carved stone block on the ground -> player must JUMP
    HIGH = "high"  # hanging wooden beam at head height -> player must DUCK


@dataclass
class Obstacle:
    lane: int
    kind: ObstacleType
    y: float = float(GROUND_Y)
    passed: bool = False

    def rect(self) -> pygame.Rect:
        cx = lane_center_x(self.lane)
        if self.kind == ObstacleType.LOW:
            w, h = 60, 60
            top = GROUND_Y - h
        else:  # HIGH: barrier at head height, gap below to duck through
            w, h = 90, 40
            top = GROUND_Y - PLAYER_H - 5
        return pygame.Rect(cx - w // 2, int(top), w, h)


@dataclass
class Coin:
    lane: int
    y_offset: int  # vertical offset above ground (for jump-collectable coins)
    collected: bool = False

    def rect(self) -> pygame.Rect:
        cx = lane_center_x(self.lane)
        size = 26
        top = GROUND_Y - self.y_offset - size
        return pygame.Rect(cx - size // 2, top, size, size)


@dataclass
class ScrollingItem:
    """Wraps an Obstacle or Coin with a world z-distance that we scroll
    toward the player; converted to a screen Y position each frame."""
    kind: str  # "obstacle" or "coin"
    payload: object
    distance: float  # world units ahead of the player, decreases each frame


class PlayerState(Enum):
    RUNNING = "running"
    JUMPING = "jumping"
    DUCKING = "ducking"
    IDLE = "idle"  # standing still, before the run has started / while paused


class Player:
    def __init__(self):
        self.lane = 1  # 0=left,1=center,2=right
        self.state = PlayerState.IDLE
        self.y = float(GROUND_Y - PLAYER_H)
        self.vy = 0.0
        self.duck_timer_ms = 0
        self.invincible_ms = 0
        self.stride_phase = 0.0  # for a simple running-leg animation

    @property
    def height(self) -> int:
        return PLAYER_DUCK_H if self.state == PlayerState.DUCKING else PLAYER_H

    def rect(self) -> pygame.Rect:
        cx = lane_center_x(self.lane)
        h = self.height
        top = self.y if self.state == PlayerState.JUMPING else GROUND_Y - h
        return pygame.Rect(cx - PLAYER_W // 2, int(top), PLAYER_W, h)

    def move_lane(self, delta: int):
        self.lane = max(0, min(LANE_COUNT - 1, self.lane + delta))

    def set_lane(self, lane: int):
        self.lane = max(0, min(LANE_COUNT - 1, lane))

    def jump(self):
        if self.state in (PlayerState.RUNNING, PlayerState.IDLE):
            self.state = PlayerState.JUMPING
            self.vy = JUMP_VELOCITY
            self.y = float(GROUND_Y - PLAYER_H)

    def duck(self, duration_ms: int = 550):
        if self.state != PlayerState.JUMPING:
            self.state = PlayerState.DUCKING
            self.duck_timer_ms = duration_ms

    def update(self, dt_ms: int, running: bool):
        if self.state == PlayerState.JUMPING:
            self.vy += GRAVITY
            self.y += self.vy
            if self.y >= GROUND_Y - PLAYER_H:
                self.y = float(GROUND_Y - PLAYER_H)
                self.vy = 0.0
                self.state = PlayerState.RUNNING if running else PlayerState.IDLE
        elif self.state == PlayerState.DUCKING:
            self.duck_timer_ms -= dt_ms
            if self.duck_timer_ms <= 0:
                self.state = PlayerState.RUNNING if running else PlayerState.IDLE
        elif running and self.state == PlayerState.IDLE:
            self.state = PlayerState.RUNNING
        elif not running and self.state == PlayerState.RUNNING:
            self.state = PlayerState.IDLE

        if self.state == PlayerState.RUNNING:
            self.stride_phase += dt_ms * 0.02

        if self.invincible_ms > 0:
            self.invincible_ms -= dt_ms


class EndlessRunnerGame:
    """Self-contained game object. `main.py` drives it: create once,
    call the cmd_* functions from the CV pipeline (or keyboard),
    call `update(dt_ms)` and `draw(surface)` every frame."""

    def __init__(self, screen: Optional[pygame.Surface] = None, seed: Optional[int] = None):
        self.screen = screen or pygame.display.get_surface()
        self.font_big = pygame.font.SysFont("georgia", 46, bold=True)
        self.font_med = pygame.font.SysFont("georgia", 26, bold=True)
        self.font_small = pygame.font.SysFont("georgia", 18)
        self.font_tiny = pygame.font.SysFont("consolas", 14)
        if seed is not None:
            random.seed(seed)
        self.reset()

    # ---------------- lifecycle ----------------

    def reset(self):
        self.player = Player()
        self.items: List[ScrollingItem] = []
        self.score = 0.0
        self.coins = 0
        self.distance_m = 0.0
        self.lives = STARTING_LIVES
        self.elapsed_ms = 0
        self.game_over = False
        self.started = False   # waiting for a START gesture/key before play begins
        self.paused = False    # frozen mid-run after a STOP gesture/key
        self.spawn_timer_ms = 0
        self.next_spawn_gap = random.randint(OBSTACLE_MIN_GAP_MS, OBSTACLE_MAX_GAP_MS)
        self.scroll_speed = BASE_SCROLL_SPEED
        self.spawn_distance = 1400  # world units ahead where new items appear
        self.last_gesture_label = "-"

    # ---------------- Stage 4: command interface ----------------
    # These are the ONLY entry points the gesture-recognition / keyboard
    # layer should call. Keeping them tiny & explicit is what Stage 4
    # ("gesture-to-game command mapping") in the report refers to.

    def _controls_active(self) -> bool:
        return self.started and not self.paused and not self.game_over

    def cmd_left(self):
        if self._controls_active():
            self.player.move_lane(-1)
            self.last_gesture_label = "LEFT"

    def cmd_right(self):
        if self._controls_active():
            self.player.move_lane(+1)
            self.last_gesture_label = "RIGHT"

    def cmd_jump(self):
        if self._controls_active():
            self.player.jump()
            self.last_gesture_label = "JUMP"

    def cmd_duck(self):
        if self._controls_active():
            self.player.duck()
            self.last_gesture_label = "DUCK"

    def cmd_start(self):
        """Thumbs-up gesture (or Enter key): begin the run, resume from a
        pause, or start a fresh run after game-over."""
        if self.game_over:
            self.reset()
            self.started = True
        elif not self.started:
            self.started = True
        elif self.paused:
            self.paused = False
        self.last_gesture_label = "START"

    def cmd_stop(self):
        """Peace-sign gesture (or P key): pause the run in place. The
        character freezes; obstacles/coins stop scrolling until resumed."""
        if self.started and not self.game_over and not self.paused:
            self.paused = True
        self.last_gesture_label = "STOP"

    def cmd_restart(self):
        self.reset()

    # ---------------- world generation ----------------

    def _spawn_row(self):
        lane = random.randint(0, LANE_COUNT - 1)
        kind = random.choice([ObstacleType.LOW, ObstacleType.HIGH])
        obstacle = Obstacle(lane=lane, kind=kind)
        self.items.append(
            ScrollingItem(kind="obstacle", payload=obstacle, distance=self.spawn_distance)
        )

        # Sometimes add a coin trail in one of the *other* lanes so it's
        # always possible to dodge the obstacle and still grab coins.
        if random.random() < COIN_ROW_CHANCE:
            free_lanes = [l for l in range(LANE_COUNT) if l != lane]
            coin_lane = random.choice(free_lanes)
            for i in range(3):
                coin = Coin(lane=coin_lane, y_offset=0)
                self.items.append(
                    ScrollingItem(
                        kind="coin",
                        payload=coin,
                        distance=self.spawn_distance + i * 90,
                    )
                )

    # ---------------- main loop steps ----------------

    def update(self, dt_ms: int):
        # The player's idle/running animation still needs updating even
        # while waiting to start / paused, but the WORLD (scrolling,
        # spawning, scoring, collisions) only advances while active.
        self.player.update(dt_ms, running=self._controls_active())

        if not self._controls_active():
            return

        self.elapsed_ms += dt_ms
        self.score += self.scroll_speed * dt_ms * 0.01
        self.distance_m += self.scroll_speed * dt_ms * 0.004

        # Difficulty ramp
        self.scroll_speed = min(
            MAX_SCROLL_SPEED,
            BASE_SCROLL_SPEED + (self.elapsed_ms / 1000.0) * SPEED_RAMP_PER_SEC,
        )

        # Spawn new obstacle/coin rows periodically
        self.spawn_timer_ms += dt_ms
        if self.spawn_timer_ms >= self.next_spawn_gap:
            self.spawn_timer_ms = 0
            self.next_spawn_gap = random.randint(OBSTACLE_MIN_GAP_MS, OBSTACLE_MAX_GAP_MS)
            self._spawn_row()

        # Advance all scrolling items toward the player
        still_active: List[ScrollingItem] = []
        player_rect = self.player.rect()

        for item in self.items:
            item.distance -= self.scroll_speed * (dt_ms / 16.6667)  # normalize to ~60fps units

            if item.distance <= 0:
                if item.kind == "obstacle":
                    obstacle: Obstacle = item.payload
                    if obstacle.lane == self.player.lane and self.player.invincible_ms <= 0:
                        hit = player_rect.colliderect(obstacle.rect())
                        if obstacle.kind == ObstacleType.LOW and self.player.state == PlayerState.JUMPING:
                            hit = False
                        if obstacle.kind == ObstacleType.HIGH and self.player.state == PlayerState.DUCKING:
                            hit = False
                        if hit:
                            self._on_hit()
                else:  # coin
                    coin: Coin = item.payload
                    if coin.lane == self.player.lane:
                        self.coins += 1
                        self.score += 25
                continue  # drop item, it has scrolled past

            still_active.append(item)

        self.items = still_active

    def _on_hit(self):
        self.lives -= 1
        self.player.invincible_ms = INVINCIBILITY_MS
        if self.lives <= 0:
            self.game_over = True

    # ---------------- rendering ----------------

    def _distance_to_screen_y(self, distance: float) -> int:
        t = max(0.0, min(1.0, distance / self.spawn_distance))
        y = HORIZON_Y + (GROUND_Y - HORIZON_Y) * (1 - t) ** 1.6
        return int(y)

    def _border_width_at(self, t: float) -> float:
        """Width of the mossy stone border at perspective depth t (0=far
        at the horizon, 1=near/bottom of screen)."""
        return _lerp(8, 34, t)

    def _plank_edge_x_at(self, t: float) -> Tuple[float, float]:
        """Left/right x of the wooden plank path itself (not the stone
        border) at perspective depth t."""
        top_half = 40.0
        bottom_half = LANE_WIDTH * LANE_COUNT / 2.0
        half = _lerp(top_half, bottom_half, t)
        cx = SCREEN_W / 2.0
        return cx - half, cx + half

    def _draw_background(self, surf: pygame.Surface):
        for y in range(SCREEN_H):
            t = y / SCREEN_H
            color = _lerp_color(COLOR_SKY_TOP, COLOR_SKY_BOTTOM, t)
            pygame.draw.line(surf, color, (0, y), (SCREEN_W, y))

    def _draw_hanging_vines(self, surf: pygame.Surface):
        """A handful of simple hanging vine squiggles from the top edge,
        purely decorative jungle-canopy framing."""
        vine_xs = [60, 180, SCREEN_W - 90, SCREEN_W - 200, SCREEN_W // 2 + 260]
        t = pygame.time.get_ticks() * 0.0006
        for i, x0 in enumerate(vine_xs):
            sway = math.sin(t + i) * 6
            points = []
            length = 70 + (i % 3) * 25
            for step in range(0, length, 8):
                px = x0 + math.sin((step * 0.15) + t * 2 + i) * 5 + sway
                py = step
                points.append((px, py))
            if len(points) >= 2:
                pygame.draw.lines(surf, COLOR_MOSS, False, points, 4)
                pygame.draw.circle(surf, (70, 130, 80), points[-1], 5)

    def _draw_track(self, surf: pygame.Surface):
        # 1) Side jungle glow (from the stone border out to the screen
        #    edges), scanline by scanline, dark near the horizon and
        #    bright teal near the player -- approximates the glowing
        #    jungle/water either side of the bridge in temple-runner style
        #    games, built from plain gradients rather than any copied art.
        for y in range(HORIZON_Y, GROUND_Y + 40):
            t = (y - HORIZON_Y) / (GROUND_Y + 40 - HORIZON_Y)
            plank_l, plank_r = self._plank_edge_x_at(t)
            border_w = self._border_width_at(t)
            outer_l = plank_l - border_w
            outer_r = plank_r + border_w
            glow_color = _lerp_color(COLOR_GLOW_DARK, COLOR_GLOW_BRIGHT, t * 0.9)
            if outer_l > 0:
                pygame.draw.line(surf, glow_color, (0, y), (int(outer_l), y))
            if outer_r < SCREEN_W:
                pygame.draw.line(surf, glow_color, (int(outer_r), y), (SCREEN_W, y))

        # 2) Mossy stone border (drawn as two polygons hugging the plank
        #    trapezoid).
        t0, t1 = 0.0, 1.0
        pl0, pr0 = self._plank_edge_x_at(t0)
        pl1, pr1 = self._plank_edge_x_at(t1)
        bw0, bw1 = self._border_width_at(t0), self._border_width_at(t1)
        left_border = [
            (pl0 - bw0, HORIZON_Y), (pl0, HORIZON_Y),
            (pl1, GROUND_Y + 40), (pl1 - bw1, GROUND_Y + 40),
        ]
        right_border = [
            (pr0, HORIZON_Y), (pr0 + bw0, HORIZON_Y),
            (pr1 + bw1, GROUND_Y + 40), (pr1, GROUND_Y + 40),
        ]
        pygame.draw.polygon(surf, COLOR_STONE_BORDER, left_border)
        pygame.draw.polygon(surf, COLOR_STONE_BORDER, right_border)
        pygame.draw.polygon(surf, COLOR_STONE_BORDER_DARK, left_border, 2)
        pygame.draw.polygon(surf, COLOR_STONE_BORDER_DARK, right_border, 2)

        # small moss patches along the stone border for texture
        for i in range(10):
            t = i / 9.0
            pl, pr = self._plank_edge_x_at(t)
            bw = self._border_width_at(t)
            y = int(HORIZON_Y + (GROUND_Y + 40 - HORIZON_Y) * t)
            r = max(2, int(_lerp(2, 7, t)))
            if i % 2 == 0:
                pygame.draw.circle(surf, COLOR_MOSS, (int(pl - bw * 0.5), y), r)
            else:
                pygame.draw.circle(surf, COLOR_MOSS, (int(pr + bw * 0.5), y), r)

        # 3) Wooden plank path fill.
        plank_poly = [
            (pl0, HORIZON_Y), (pr0, HORIZON_Y),
            (pr1, GROUND_Y + 40), (pl1, GROUND_Y + 40),
        ]
        pygame.draw.polygon(surf, COLOR_PLANK_LIGHT, plank_poly)

        # 4) Plank seams (perpendicular boards receding into the distance).
        n_planks = 14
        for i in range(n_planks):
            t = i / (n_planks - 1)
            y = self._distance_to_screen_y(self.spawn_distance * (1 - t))
            pl, pr = self._plank_edge_x_at(t)
            shade = COLOR_PLANK_DARK if i % 2 == 0 else COLOR_PLANK_LIGHT
            band_bottom_t = min(1.0, t + (1.0 / n_planks))
            y_bottom = self._distance_to_screen_y(self.spawn_distance * (1 - band_bottom_t))
            pl_b, pr_b = self._plank_edge_x_at(band_bottom_t)
            pygame.draw.polygon(
                surf, shade, [(pl, y), (pr, y), (pr_b, y_bottom), (pl_b, y_bottom)]
            )
            pygame.draw.line(surf, COLOR_PLANK_SEAM, (pl, y), (pr, y), max(1, int(_lerp(1, 3, t))))

        # 5) Lane divider hints (very subtle, plank-colored, not bright
        #    lines like a road -- keeps the "wood bridge" look).
        for i in range(1, LANE_COUNT):
            top_x = _lerp(pl0, pr0, i / LANE_COUNT)
            bot_x = _lerp(pl1, pr1, i / LANE_COUNT)
            pygame.draw.line(surf, COLOR_PLANK_SEAM, (top_x, HORIZON_Y), (bot_x, GROUND_Y + 40), 2)

    def _draw_stone_pillars(self, surf: pygame.Surface):
        """Static carved-stone pillar framing near the top corners --
        decorative only, echoes the ruined-temple archway look."""
        for side in (-1, 1):
            base_x = SCREEN_W // 2 + side * (LANE_WIDTH * LANE_COUNT // 2 + 55)
            pillar_w = 46
            top_y = 0
            bot_y = 130
            rect = pygame.Rect(0, top_y, pillar_w, bot_y)
            rect.centerx = base_x
            pygame.draw.rect(surf, COLOR_STONE_BORDER, rect)
            pygame.draw.rect(surf, COLOR_STONE_BORDER_DARK, rect, 3)
            cap = pygame.Rect(0, 0, pillar_w + 22, 20)
            cap.centerx = base_x
            cap.top = bot_y - 6
            pygame.draw.rect(surf, COLOR_STONE_BORDER_DARK, cap)
            for gy in range(20, bot_y - 10, 22):
                pygame.draw.line(
                    surf, COLOR_STONE_BORDER_DARK,
                    (rect.left + 6, gy), (rect.right - 6, gy), 2,
                )

    def _draw_skull_decorations(self, surf: pygame.Surface):
        """A couple of static skull decorations framing the bottom
        corners, echoing the reference art's foreground skull motif --
        drawn from simple primitives, not traced/copied artwork."""
        for cx, cy, scale in [(50, SCREEN_H - 45, 1.15), (SCREEN_W - 46, SCREEN_H - 90, 0.8)]:
            r = int(24 * scale)
            skull_rect = pygame.Rect(0, 0, int(r * 1.7), int(r * 1.5))
            skull_rect.center = (cx, cy)
            pygame.draw.ellipse(surf, COLOR_SKULL, skull_rect)
            pygame.draw.ellipse(surf, COLOR_SKULL_SHADOW, skull_rect, 2)
            eye_r = max(3, int(r * 0.22))
            pygame.draw.circle(surf, (20, 16, 12), (cx - int(r * 0.35), cy - int(r * 0.05)), eye_r)
            pygame.draw.circle(surf, (20, 16, 12), (cx + int(r * 0.35), cy - int(r * 0.05)), eye_r)
            nose = [
                (cx, cy + int(r * 0.05)),
                (cx - int(r * 0.12), cy + int(r * 0.28)),
                (cx + int(r * 0.12), cy + int(r * 0.28)),
            ]
            pygame.draw.polygon(surf, (20, 16, 12), nose)
            jaw_rect = pygame.Rect(0, 0, int(r * 1.1), int(r * 0.5))
            jaw_rect.center = (cx, cy + int(r * 0.62))
            pygame.draw.ellipse(surf, COLOR_SKULL_SHADOW, jaw_rect)

    def _draw_player(self, surf: pygame.Surface):
        rect = self.player.rect()
        blinking = self.player.invincible_ms > 0 and (self.player.invincible_ms // 100) % 2 == 0
        if blinking:
            return

        shirt = COLOR_PLAYER_SHIRT_DUCK if self.player.state == PlayerState.DUCKING else COLOR_PLAYER_SHIRT

        leg_w = 14
        stride = math.sin(self.player.stride_phase) * 10 if self.player.state == PlayerState.RUNNING else 0
        leg_y = rect.bottom - 26
        pygame.draw.rect(surf, COLOR_PLAYER_PANTS, (rect.centerx - 18 + stride, leg_y, leg_w, 26), border_radius=4)
        pygame.draw.rect(surf, COLOR_PLAYER_PANTS, (rect.centerx + 4 - stride, leg_y, leg_w, 26), border_radius=4)

        torso_h = rect.height - 26
        torso_rect = pygame.Rect(rect.left, rect.top, rect.width, max(20, torso_h))
        pygame.draw.rect(surf, shirt, torso_rect, border_radius=12)

        pygame.draw.rect(
            surf, COLOR_OBSTACLE_HIGH,
            (torso_rect.left + 6, torso_rect.top + 6, 10, torso_rect.height - 12),
            border_radius=4,
        )

        head_r = 16
        head_center = (rect.centerx, torso_rect.top - head_r + 6)
        pygame.draw.circle(surf, COLOR_PLAYER_SKIN, head_center, head_r)
        eye_y = head_center[1] - 2
        pygame.draw.circle(surf, (30, 24, 18), (head_center[0] - 6, eye_y), 2)
        pygame.draw.circle(surf, (30, 24, 18), (head_center[0] + 6, eye_y), 2)

    def _draw_items(self, surf: pygame.Surface):
        for item in sorted(self.items, key=lambda it: -it.distance):
            screen_y = self._distance_to_screen_y(item.distance)
            scale = 0.4 + 0.6 * (1 - min(1.0, item.distance / self.spawn_distance))
            cx = lane_center_x(item.payload.lane)

            if item.kind == "obstacle":
                obstacle: Obstacle = item.payload
                if obstacle.kind == ObstacleType.LOW:
                    w, h = int(60 * scale), int(60 * scale)
                    rect = pygame.Rect(cx - w // 2, screen_y - h, w, h)
                    pygame.draw.rect(surf, COLOR_OBSTACLE_LOW, rect, border_radius=4)
                    pygame.draw.rect(surf, COLOR_OBSTACLE_LOW_DARK, rect, 3, border_radius=4)
                    if w > 20:
                        pygame.draw.line(
                            surf, COLOR_OBSTACLE_LOW_DARK,
                            (rect.left + 4, rect.centery), (rect.right - 4, rect.centery), 2,
                        )
                else:
                    w, h = int(90 * scale), int(40 * scale)
                    rect = pygame.Rect(cx - w // 2, screen_y - h - 90, w, h)
                    pygame.draw.rect(surf, COLOR_OBSTACLE_HIGH, rect, border_radius=6)
                    pygame.draw.rect(surf, (60, 42, 24), rect, 3, border_radius=6)
                    for mx in range(rect.left + 6, rect.right - 6, 14):
                        pygame.draw.circle(surf, COLOR_OBSTACLE_HIGH_MOSS, (mx, rect.top + 4), 3)
            else:
                coin: Coin = item.payload
                size = max(4, int(15 * scale))
                cy = screen_y - 40
                pts = [(cx, cy - size), (cx + size, cy), (cx, cy + size), (cx - size, cy)]
                pygame.draw.polygon(surf, COLOR_COIN_GOLD, pts)
                pygame.draw.polygon(surf, COLOR_COIN_GOLD_DARK, pts, 2)
                if size > 6:
                    pygame.draw.line(surf, (255, 245, 200), (cx - size // 2, cy - size // 3), (cx, cy - size), 1)

    # ---------------- HUD ----------------

    def _draw_gold_plaque(self, surf, rect, text, font, text_color=COLOR_GOLD):
        pygame.draw.rect(surf, (28, 20, 12), rect, border_radius=10)
        pygame.draw.rect(surf, COLOR_GOLD_DARK, rect, 3, border_radius=10)
        pygame.draw.rect(surf, COLOR_GOLD, rect.inflate(-6, -6), 1, border_radius=8)
        txt = font.render(text, True, text_color)
        surf.blit(txt, txt.get_rect(center=rect.center))

    def _draw_pause_button(self, surf):
        cx, cy, r = SCREEN_W - 42, SCREEN_H - 42, 26
        pygame.draw.circle(surf, (28, 20, 12), (cx, cy), r)
        pygame.draw.circle(surf, COLOR_GOLD_DARK, (cx, cy), r, 3)
        if self._controls_active():
            pygame.draw.rect(surf, COLOR_GOLD, (cx - 8, cy - 9, 5, 18))
            pygame.draw.rect(surf, COLOR_GOLD, (cx + 3, cy - 9, 5, 18))
        else:
            pygame.draw.polygon(surf, COLOR_GOLD, [(cx - 6, cy - 9), (cx - 6, cy + 9), (cx + 9, cy)])

    def _draw_hud(self, surf: pygame.Surface):
        dist_rect = pygame.Rect(0, 14, 170, 40)
        dist_rect.centerx = SCREEN_W // 2
        self._draw_gold_plaque(surf, dist_rect, f"{int(self.distance_m)} m", self.font_med)

        coin_rect = pygame.Rect(SCREEN_W - 150, 14, 134, 36)
        self._draw_gold_plaque(surf, coin_rect, f"\u25C6 {self.coins}", self.font_small)

        score_txt = self.font_small.render(f"Score {int(self.score)}", True, COLOR_TEXT)
        surf.blit(score_txt, (SCREEN_W - score_txt.get_width() - 16, 54))

        hearts = "\u2665 " * self.lives
        lives_txt = self.font_small.render(hearts.strip(), True, COLOR_TEXT_WARN)
        surf.blit(lives_txt, (16, SCREEN_H - 34))

        dbg_txt = self.font_tiny.render(f"cmd: {self.last_gesture_label}", True, (200, 200, 190))
        surf.blit(dbg_txt, (12, 12))

        self._draw_pause_button(surf)

        if not self.started and not self.game_over:
            self._draw_center_overlay(
                surf, "TEMPLE DASH", "Show a THUMBS-UP to begin  (or press Enter)"
            )
        elif self.paused:
            self._draw_center_overlay(
                surf, "PAUSED", "Show a THUMBS-UP to resume  (or press Enter)"
            )
        elif self.game_over:
            self._draw_center_overlay(
                surf,
                "GAME OVER",
                f"Score {int(self.score)}   Coins {self.coins}   Distance {int(self.distance_m)}m",
                sub2="Show THUMBS-UP, or press Enter, to run again",
                title_color=COLOR_TEXT_WARN,
            )

    def _draw_center_overlay(self, surf, title, subtitle, sub2=None, title_color=COLOR_GOLD):
        overlay = pygame.Surface((SCREEN_W, SCREEN_H), pygame.SRCALPHA)
        overlay.fill((5, 10, 8, 165))
        surf.blit(overlay, (0, 0))
        title_txt = self.font_big.render(title, True, title_color)
        sub_txt = self.font_small.render(subtitle, True, COLOR_TEXT)
        surf.blit(title_txt, title_txt.get_rect(center=(SCREEN_W // 2, SCREEN_H // 2 - 30)))
        surf.blit(sub_txt, sub_txt.get_rect(center=(SCREEN_W // 2, SCREEN_H // 2 + 20)))
        if sub2:
            sub2_txt = self.font_small.render(sub2, True, COLOR_TEXT)
            surf.blit(sub2_txt, sub2_txt.get_rect(center=(SCREEN_W // 2, SCREEN_H // 2 + 50)))

    def draw(self, surf: Optional[pygame.Surface] = None):
        surf = surf or self.screen
        self._draw_background(surf)
        self._draw_track(surf)
        self._draw_stone_pillars(surf)
        self._draw_hanging_vines(surf)
        self._draw_items(surf)
        self._draw_player(surf)
        self._draw_skull_decorations(surf)
        self._draw_hud(surf)
