"""
rule_bot.py — Hardcoded baseline AI for the "Normal Bot" mode.

Provides a deterministic opponent to benchmark the RL agent against.
Uses the same interface as ai_inference.py: predict(observation) → action.

Strategy:
  - Walk toward the player when far away
  - Strafe sideways when at medium range
  - Aim directly at the player and shoot when in range
"""

import numpy as np


class RuleBot:
    """
    A simple rule-based bot that reads the 30-float observation
    (from its own perspective) and returns a 5-float action.
    """

    def __init__(self, aggression: float = 0.7, fire_range: float = 0.4):
        """
        Args:
            aggression: 0.0 = passive, 1.0 = always charge. Controls approach distance.
            fire_range: Normalized distance threshold to start shooting (0.0–1.0).
        """
        self.aggression = aggression
        self.fire_range = fire_range
        self._strafe_dir = 1.0  # 1 or -1, flips periodically
        self._strafe_timer = 0

    def predict(self, obs: np.ndarray) -> np.ndarray:
        """
        Given a 30-float observation, return a 5-float action.

        Observation layout (from this bot's perspective):
            [0]    hp_norm
            [1]    ammo_norm
            [2]    reload_norm
            [3:5]  vel_x, vel_y
            [5:7]  rel_enemy_x, rel_enemy_y (normalized)
            [7]    distance_to_enemy (normalized)
            [8]    angle_to_enemy (normalized [-1, 1])
            [9]    enemy_hp_norm
            [10:30] bullet data (ignored by rule bot)
        """
        # Extract key features
        my_hp = obs[0]
        my_ammo = obs[1]
        rel_x = obs[5]
        rel_y = obs[6]
        distance = obs[7]
        enemy_hp = obs[9]

        # --- Movement ---
        move_x, move_y = 0.0, 0.0

        if distance > self.fire_range:
            # Walk toward enemy
            move_x = rel_x * 3.0  # Scale up for stronger movement signal
            move_y = rel_y * 3.0
        else:
            # Strafe sideways at close range
            self._strafe_timer += 1
            if self._strafe_timer > 20:
                self._strafe_dir *= -1.0
                self._strafe_timer = 0

            # Perpendicular to enemy direction
            move_x = -rel_y * self._strafe_dir * 2.0
            move_y = rel_x * self._strafe_dir * 2.0

        # Clamp movement
        move_mag = np.sqrt(move_x**2 + move_y**2)
        if move_mag > 1.0:
            move_x /= move_mag
            move_y /= move_mag

        # --- Aiming ---
        # Aim directly at enemy position
        aim_x = rel_x
        aim_y = rel_y
        aim_mag = np.sqrt(aim_x**2 + aim_y**2)
        if aim_mag > 0.01:
            aim_x /= aim_mag
            aim_y /= aim_mag

        # --- Shooting ---
        shoot = -1.0  # Don't shoot by default

        if distance < self.fire_range and my_ammo > 0.1 and aim_mag > 0.01:
            shoot = 1.0  # Fire!

        # Low HP retreat behavior
        if my_hp < 0.3 and enemy_hp > 0.5:
            # Run away (reverse direction)
            move_x = -rel_x * 3.0
            move_y = -rel_y * 3.0
            move_mag = np.sqrt(move_x**2 + move_y**2)
            if move_mag > 1.0:
                move_x /= move_mag
                move_y /= move_mag

        return np.array([move_x, move_y, aim_x, aim_y, shoot], dtype=np.float32)

    def reset(self):
        """Reset internal state for a new game."""
        self._strafe_dir = 1.0
        self._strafe_timer = 0
