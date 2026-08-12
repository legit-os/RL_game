"""
brawl_sniper_env.py — Gymnasium wrapper around the shared physics engine.

This is a THIN wrapper. All game logic lives in backend/physics.py.
This file handles:
  - Gymnasium API compliance (reset, step, observation_space, action_space)
  - Reward annealing (dense rewards decay to 0 over 50M steps)
  - Curriculum learning (Random -> Hardcoded -> PFSP ONNX)
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import sys
import os
import glob
from collections import deque

# Add project root to path so we can import backend.physics
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from backend.physics import (
    GameState,
    create_state,
    tick,
    get_obs,
    MAX_HP,
)

# Global ONNX session cache to prevent memory leaks across episodes
_ONNX_SESSIONS = {}

def get_onnx_session(path: str):
    import onnxruntime as ort
    if path not in _ONNX_SESSIONS:
        _ONNX_SESSIONS[path] = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    return _ONNX_SESSIONS[path]


class BrawlSniperEnv(gym.Env):
    """
    Gymnasium environment for training a sniper RL agent.
    """
    metadata = {"render_modes": []}

    def __init__(self, max_steps: int = 500, n_envs: int = 8):
        super().__init__()
        self.max_steps = max_steps
        self.n_envs = n_envs
        self.env_steps = 0  # Local step counter

        # Action: [move_x, move_y, aim_x, aim_y, shoot_trigger]
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(5,), dtype=np.float32)

        # Observation: 30 relative floats (VecFrameStack will wrap this to 90)
        self.observation_space = spaces.Box(low=-2.0, high=2.0, shape=(30,), dtype=np.float32)

        self.state: GameState | None = None
        self._enemy_vel_direction = np.zeros(2, dtype=np.float32)
        
        # State for Phase 3 (PFSP)
        self.p2_obs_stack = deque(maxlen=3)
        self.opponent_session = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Randomize Player 2 spawn position
        spawn_x = self.np_random.uniform(2.0, 8.0)
        spawn_y = self.np_random.uniform(-5.0, 5.0)

        self.state = create_state(
            p1_pos=[-5.0, 0.0],
            p2_pos=[spawn_x, spawn_y],
        )

        self._enemy_vel_direction = np.zeros(2, dtype=np.float32)
        
        # Initialize Player 2's frame stack for ONNX inference
        obs2 = get_obs(self.state, perspective=2)
        for _ in range(3):
            self.p2_obs_stack.append(obs2)
            
        # Sample opponent for Phase 3
        global_step = self.env_steps * self.n_envs
        if global_step >= 15_000_000:
            self.opponent_session = self._sample_opponent()
        else:
            self.opponent_session = None

        return get_obs(self.state, perspective=1), {}

    def step(self, action: np.ndarray):
        self.env_steps += 1
        
        # --- Generate enemy action ---
        enemy_action = self._get_enemy_action()

        # --- Tick the physics engine ---
        tick(self.state, action1=action, action2=enemy_action)
        
        # Update P2 frame stack for next tick
        obs2 = get_obs(self.state, perspective=2)
        self.p2_obs_stack.append(obs2)

        # --- Reward shaping ---
        reward = self._compute_reward()

        # --- Termination ---
        terminated = self.state.p2_dead  # Agent wins
        truncated = self.state.tick_count >= self.max_steps

        if truncated and not terminated:
            reward -= 1.0  # Timeout penalty

        # If agent dies, also terminate (agent lost)
        if self.state.p1_dead:
            reward -= 10.0
            terminated = True

        obs = get_obs(self.state, perspective=1)
        return obs, reward, terminated, truncated, {}

    def _sample_opponent(self):
        """Samples an ONNX model for PFSP."""
        pool_dir = os.path.join(PROJECT_ROOT, "models", "opponent_pool")
        if not os.path.isdir(pool_dir):
            return None
            
        models = glob.glob(os.path.join(pool_dir, "*.onnx"))
        if not models:
            return None
            
        # Sort by modification time to identify the latest
        models.sort(key=os.path.getmtime)
        
        # 80% latest, 20% random historical
        if self.np_random.random() < 0.8:
            chosen = models[-1]
        else:
            chosen = self.np_random.choice(models)
            
        try:
            return get_onnx_session(chosen)
        except Exception:
            return None

    def _get_enemy_action(self) -> np.ndarray:
        global_step = self.env_steps * self.n_envs
        
        if global_step < 5_000_000:
            # Phase 1: Foundation (Random Walk)
            return self._random_walk_action()
        elif global_step < 15_000_000 or self.opponent_session is None:
            # Phase 2: Hardcoded Combat (or Fallback if no models in Phase 3)
            return self._hardcoded_action()
        else:
            # Phase 3: Fictitious Self-Play (ONNX Inference)
            try:
                stacked_obs = np.concatenate(self.p2_obs_stack).astype(np.float32)
                # Expand dims for batch size = 1
                action = self.opponent_session.run(None, {"observation": np.expand_dims(stacked_obs, axis=0)})[0][0]
                return action
            except Exception:
                # Fallback to hardcoded on inference crash
                return self._hardcoded_action()

    def _random_walk_action(self) -> np.ndarray:
        """Phase 1 dummy opponent."""
        if self.state.tick_count % 15 == 1 or self.state.tick_count == 1:
            angle = self.np_random.uniform(0, 2 * np.pi)
            self._enemy_vel_direction = np.array([np.cos(angle), np.sin(angle)], dtype=np.float32) * 0.6
        return np.array([self._enemy_vel_direction[0], self._enemy_vel_direction[1], 0.0, 0.0, -1.0], dtype=np.float32)

    def _hardcoded_action(self) -> np.ndarray:
        """Phase 2 aggressive opponent."""
        p1 = self.state.p1
        p2 = self.state.p2
        rel_pos = p1.pos - p2.pos
        dist = np.linalg.norm(rel_pos)
        
        # Move towards player if far, else strafe
        if dist > 4.0:
            move_dir = rel_pos / (dist + 1e-8)
        else:
            move_dir = np.array([-rel_pos[1], rel_pos[0]], dtype=np.float32) / (dist + 1e-8)
            if self.state.tick_count % 60 < 30:
                move_dir = -move_dir
                
        # Aim exactly at player
        aim_dir = rel_pos / (dist + 1e-8)
        
        # Shoot if reloaded and in range
        shoot = 1.0 if dist < 8.0 and p2.ammo >= 1.0 else -1.0
        
        return np.array([move_dir[0], move_dir[1], aim_dir[0], aim_dir[1], shoot], dtype=np.float32)

    def _compute_reward(self) -> float:
        """Reward shaping with annealing over 50M steps."""
        global_step = self.env_steps * self.n_envs
        # Decay from 1.0 to 0.0 over 50M steps
        anneal_factor = max(0.0, 1.0 - (global_step / 50_000_000.0))
        
        reward = -0.01  # Time pressure (always active)

        # Dense rewards (decay over time)
        if self.state.p1_damage_dealt > 0:
            reward += (self.state.p1_damage_dealt / MAX_HP * 3.4) * anneal_factor
        if self.state.p1_damage_taken > 0:
            reward -= (self.state.p1_damage_taken / MAX_HP * 2.0) * anneal_factor
        if self.state.p1_shots_fired > 0:
            reward -= 0.05 * anneal_factor

        # Sparse terminal rewards (never decay)
        if self.state.p2_dead:
            reward += 10.0

        return reward
