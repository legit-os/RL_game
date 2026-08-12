"""
physics.py — Pure 2-Player Game Physics Engine

This is the shared core used by BOTH:
  - env/brawl_sniper_env.py (Gymnasium wrapper for RL training)
  - backend/game_loop.py (live WebSocket gameplay)

All functions are pure: state in → state out. No side effects, no globals.
"""

import numpy as np
from dataclasses import dataclass, field


# ─── Game Constants ───────────────────────────────────────────────────────────

MAP_SIZE = 10.0            # Bounded area: [-10.0, 10.0] on both axes
AGENT_SPEED = 0.1          # Units per tick
BULLET_SPEED = 0.35        # Units per tick
HITBOX_RADIUS = 0.35       # Collision radius (~3.5% of map width)
MAX_HP = 100.0
MAX_AMMO = 3.0
RELOAD_TICKS = 30          # Ticks to reload 1 ammo slot (~0.5s at 60fps)
DAMAGE_PER_HIT = 34.0      # Flat damage (3-shot kill on 100 HP)
BULLETS_PER_PLAYER = 5     # Max active bullets per player
TOTAL_BULLETS = BULLETS_PER_PLAYER * 2  # 10 total


# ─── Game State ───────────────────────────────────────────────────────────────

@dataclass
class PlayerState:
    pos: np.ndarray          # [x, y] float32
    vel: np.ndarray          # [x, y] float32
    hp: float
    ammo: float
    reload_timer: int


@dataclass
class GameState:
    p1: PlayerState
    p2: PlayerState
    bullets: np.ndarray      # Shape: (10, 5) → [x, y, vx, vy, owner_id]
    tick_count: int = 0
    p1_damage_dealt: float = 0.0  # Accumulated this tick (for reward calc)
    p2_damage_dealt: float = 0.0
    p1_damage_taken: float = 0.0
    p2_damage_taken: float = 0.0
    p1_shots_fired: int = 0
    p2_shots_fired: int = 0
    p1_hits: int = 0
    p2_hits: int = 0
    p1_dead: bool = False
    p2_dead: bool = False


# ─── Factory ──────────────────────────────────────────────────────────────────

def create_state(
    p1_pos: np.ndarray | None = None,
    p2_pos: np.ndarray | None = None,
) -> GameState:
    """Create a fresh game state with optional spawn positions."""
    return GameState(
        p1=PlayerState(
            pos=np.array(p1_pos if p1_pos is not None else [-5.0, 0.0], dtype=np.float32),
            vel=np.zeros(2, dtype=np.float32),
            hp=MAX_HP,
            ammo=MAX_AMMO,
            reload_timer=0,
        ),
        p2=PlayerState(
            pos=np.array(p2_pos if p2_pos is not None else [5.0, 0.0], dtype=np.float32),
            vel=np.zeros(2, dtype=np.float32),
            hp=MAX_HP,
            ammo=MAX_AMMO,
            reload_timer=0,
        ),
        bullets=np.zeros((TOTAL_BULLETS, 5), dtype=np.float32),
        tick_count=0,
        p1_damage_dealt=0.0,
        p2_damage_dealt=0.0,
        p1_damage_taken=0.0,
        p2_damage_taken=0.0,
        p1_shots_fired=0,
        p2_shots_fired=0,
        p1_hits=0,
        p2_hits=0,
        p1_dead=False,
        p2_dead=False,
    )


# ─── Core Tick ────────────────────────────────────────────────────────────────

def tick(state: GameState, action1: np.ndarray, action2: np.ndarray) -> GameState:
    """
    Advance the game by one tick. Pure function.

    Args:
        state: Current game state
        action1: Player 1 action [move_x, move_y, aim_x, aim_y, shoot_trigger]
        action2: Player 2 action [move_x, move_y, aim_x, aim_y, shoot_trigger]

    Returns:
        The same GameState object, mutated in place for performance.
        Per-tick event counters are reset at the start of each tick.
    """
    state.tick_count += 1

    # Reset per-tick event counters
    state.p1_damage_dealt = 0.0
    state.p2_damage_dealt = 0.0
    state.p1_damage_taken = 0.0
    state.p2_damage_taken = 0.0
    state.p1_shots_fired = 0
    state.p2_shots_fired = 0
    state.p1_hits = 0
    state.p2_hits = 0

    # --- 1. MOVEMENT (both players) ---
    _apply_movement(state.p1, action1)
    _apply_movement(state.p2, action2)

    # --- 2. AMMO RELOAD (both players) ---
    _apply_reload(state.p1)
    _apply_reload(state.p2)

    # --- 3. SHOOTING (both players) ---
    # Player 1 gets bullet slots 0-4, Player 2 gets slots 5-9
    p1_fired = _apply_shooting(state.p1, action1, state.bullets, 0, BULLETS_PER_PLAYER, owner_id=1)
    p2_fired = _apply_shooting(state.p2, action2, state.bullets, BULLETS_PER_PLAYER, TOTAL_BULLETS, owner_id=2)
    state.p1_shots_fired = p1_fired
    state.p2_shots_fired = p2_fired

    # --- 4. BULLET PHYSICS & COLLISION ---
    _advance_bullets(state)

    # --- 5. DEATH CHECK ---
    state.p1_dead = state.p1.hp <= 0
    state.p2_dead = state.p2.hp <= 0

    return state


# ─── Movement ─────────────────────────────────────────────────────────────────

def _apply_movement(player: PlayerState, action: np.ndarray) -> None:
    """Move a player, clamping diagonal speed to unit circle."""
    move_vector = action[0:2].astype(np.float32)
    move_mag = np.linalg.norm(move_vector)
    if move_mag > 1.0:
        move_vector = move_vector / move_mag

    player.vel = move_vector * AGENT_SPEED
    player.pos = player.pos + player.vel
    player.pos = np.clip(player.pos, -MAP_SIZE, MAP_SIZE)


# ─── Reload ───────────────────────────────────────────────────────────────────

def _apply_reload(player: PlayerState) -> None:
    """Tick-based ammo reload state machine."""
    if player.ammo < MAX_AMMO:
        player.reload_timer -= 1
        if player.reload_timer <= 0:
            player.ammo += 1.0
            player.reload_timer = RELOAD_TICKS if player.ammo < MAX_AMMO else 0


# ─── Shooting ─────────────────────────────────────────────────────────────────

def _apply_shooting(
    player: PlayerState,
    action: np.ndarray,
    bullets: np.ndarray,
    slot_start: int,
    slot_end: int,
    owner_id: int,
) -> int:
    """
    Attempt to fire a bullet for this player.
    Returns: number of shots fired (0 or 1).
    """
    shoot_trigger = action[4]
    aim_vector = action[2:4].astype(np.float32)
    aim_mag = np.linalg.norm(aim_vector)

    if shoot_trigger > 0.0 and player.ammo >= 1.0 and aim_mag > 0.1:
        # Find empty slot in this player's range
        player_bullets = bullets[slot_start:slot_end]
        empty_mask = np.all(player_bullets == 0, axis=1)
        empty_indices = np.where(empty_mask)[0]

        if len(empty_indices) > 0:
            idx = slot_start + empty_indices[0]
            direction = aim_vector / aim_mag
            bullets[idx] = [
                player.pos[0],
                player.pos[1],
                direction[0] * BULLET_SPEED,
                direction[1] * BULLET_SPEED,
                owner_id,
            ]
            player.ammo -= 1.0

            # Start reload cycle if not already running
            if player.reload_timer <= 0:
                player.reload_timer = RELOAD_TICKS

            return 1

    return 0


# ─── Bullet Physics ──────────────────────────────────────────────────────────

def _advance_bullets(state: GameState) -> None:
    """Advance all bullets, check collisions, despawn out-of-bounds."""
    bullets = state.bullets

    # Find active bullets (any non-zero row)
    active_mask = np.any(bullets != 0, axis=1)
    if not np.any(active_mask):
        return

    # Advance positions
    bullets[active_mask, 0:2] += bullets[active_mask, 2:4]

    # --- Collision: Player 1's bullets (owner=1) vs Player 2 ---
    p1_bullets_mask = active_mask & (bullets[:, 4] == 1)
    if np.any(p1_bullets_mask):
        dist_to_p2 = np.linalg.norm(bullets[:, 0:2] - state.p2.pos, axis=1)
        p1_hits = p1_bullets_mask & (dist_to_p2 < HITBOX_RADIUS)
        if np.any(p1_hits):
            hit_count = int(np.sum(p1_hits))
            damage = hit_count * DAMAGE_PER_HIT
            state.p2.hp -= damage
            state.p1_damage_dealt += damage
            state.p2_damage_taken += damage
            state.p1_hits += hit_count
            bullets[p1_hits] = 0.0

    # --- Collision: Player 2's bullets (owner=2) vs Player 1 ---
    p2_bullets_mask = active_mask & (bullets[:, 4] == 2)
    if np.any(p2_bullets_mask):
        dist_to_p1 = np.linalg.norm(bullets[:, 0:2] - state.p1.pos, axis=1)
        p2_hits = p2_bullets_mask & (dist_to_p1 < HITBOX_RADIUS)
        if np.any(p2_hits):
            hit_count = int(np.sum(p2_hits))
            damage = hit_count * DAMAGE_PER_HIT
            state.p1.hp -= damage
            state.p2_damage_dealt += damage
            state.p1_damage_taken += damage
            state.p2_hits += hit_count
            bullets[p2_hits] = 0.0

    # --- Despawn out-of-bounds ---
    oob = (np.abs(bullets[:, 0]) > MAP_SIZE) | (np.abs(bullets[:, 1]) > MAP_SIZE)
    # Only clear active bullets that went OOB (don't touch inactive zeros)
    oob_active = oob & active_mask
    if np.any(oob_active):
        bullets[oob_active] = 0.0


# ─── Observation ──────────────────────────────────────────────────────────────

def get_obs(state: GameState, perspective: int) -> np.ndarray:
    """
    Get the 30-float normalized observation from a player's perspective.

    Args:
        state: Current game state
        perspective: 1 for player 1's view, 2 for player 2's view

    Returns:
        np.ndarray of shape (30,) dtype float32

    Layout:
        [0]    hp_norm
        [1]    ammo_norm
        [2]    reload_norm
        [3:5]  vel_x, vel_y (normalized by AGENT_SPEED)
        [5:7]  rel_enemy_x, rel_enemy_y (normalized)
        [7]    distance_to_enemy (normalized)
        [8]    angle_to_enemy (normalized to [-1, 1])
        [9]    enemy_hp_norm
        [10:30] 5 bullet slots × [rel_x, rel_y, norm_vx, norm_vy]
    """
    if perspective == 1:
        me = state.p1
        them = state.p2
    else:
        me = state.p2
        them = state.p1

    obs = np.zeros(30, dtype=np.float32)

    # 1. Self State (5)
    obs[0] = me.hp / MAX_HP
    obs[1] = me.ammo / MAX_AMMO
    obs[2] = me.reload_timer / RELOAD_TICKS if RELOAD_TICKS > 0 else 0.0
    obs[3:5] = me.vel / AGENT_SPEED if AGENT_SPEED > 0 else 0.0

    # 2. Relative Enemy State (5)
    rel_enemy = them.pos - me.pos
    dist_enemy = np.linalg.norm(rel_enemy)
    angle_enemy = np.arctan2(rel_enemy[1], rel_enemy[0]) / np.pi

    obs[5:7] = rel_enemy / (MAP_SIZE * 2.0)
    obs[7] = dist_enemy / (MAP_SIZE * 2.0)
    obs[8] = angle_enemy
    obs[9] = them.hp / MAX_HP

    # 3. Bullet State (20) — all active bullets, positions relative to self
    bullets = state.bullets
    active = np.any(bullets[:, :4] != 0, axis=1)  # Check x,y,vx,vy (not owner)

    if np.any(active):
        rel_bullets = np.zeros((TOTAL_BULLETS, 4), dtype=np.float32)
        rel_bullets[active, 0:2] = (
            bullets[active, 0:2] - me.pos
        ) / (MAP_SIZE * 2.0)
        rel_bullets[active, 2:4] = bullets[active, 2:4] / BULLET_SPEED

        # We only have 5 observation slots — pick the 5 closest bullets
        # (could be own bullets or enemy bullets — the agent learns to distinguish
        #  by velocity direction relative to itself)
        active_indices = np.where(active)[0]
        if len(active_indices) <= 5:
            obs_bullets = rel_bullets[active_indices]
            # Pad to 5
            padded = np.zeros((5, 4), dtype=np.float32)
            padded[:len(active_indices)] = obs_bullets
            obs[10:30] = padded.flatten()
        else:
            # Pick 5 closest to self
            distances = np.linalg.norm(rel_bullets[active_indices, 0:2], axis=1)
            closest_5 = active_indices[np.argsort(distances)[:5]]
            obs[10:30] = rel_bullets[closest_5].flatten()

    return obs


# ─── Serialization for WebSocket ──────────────────────────────────────────────

def state_to_dict(state: GameState) -> dict:
    """
    Serialize game state to a compact dict for WebSocket transmission.
    Uses short keys to minimize payload size.
    """
    # Collect active bullets as list of [x, y]
    active_mask = np.any(state.bullets[:, :4] != 0, axis=1)
    bullet_list = state.bullets[active_mask, 0:2].tolist() if np.any(active_mask) else []

    return {
        "t": state.tick_count,
        "p": [
            round(float(state.p1.pos[0]), 2),
            round(float(state.p1.pos[1]), 2),
            round(float(state.p1.hp), 1),
        ],
        "e": [
            round(float(state.p2.pos[0]), 2),
            round(float(state.p2.pos[1]), 2),
            round(float(state.p2.hp), 1),
        ],
        "b": [
            [round(float(bx), 2), round(float(by), 2)]
            for bx, by in bullet_list
        ],
    }
