"""
env.py — Custom Game Environment & Vectorized Wrapper

Zero dependencies beyond numpy + our physics engine.
No gymnasium, no gym, no SB3.

BrawlEnv: Single environment with reset()/step() interface.
VecEnv:   Runs N BrawlEnv instances in threads for batched rollouts.
"""

import numpy as np
import os
import sys
from collections import deque
from concurrent.futures import ThreadPoolExecutor

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from backend.physics import (
    GameState,
    create_state,
    tick,
    get_obs,
    state_to_dict,
    MAX_HP,
    MAX_BULLET_RANGE,
)

# Global ONNX session cache
_ONNX_SESSIONS = {}

def get_onnx_session(path: str):
    import onnxruntime as ort
    if path not in _ONNX_SESSIONS:
        _ONNX_SESSIONS[path] = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    return _ONNX_SESSIONS[path]


class BrawlEnv:
    """
    Minimal game environment. No gym dependency.
    
    obs shape: (30,) per frame, (91,) with frame stacking + time feature
    action shape: (5,) — [move_x, move_y, aim_x, aim_y, shoot_trigger]
    """

    # --- Reward Constants ---
    REWARD_DAMAGE_DEALT    = 6.0   # Per hit (was 5.0)
    REWARD_NEAR_MISS_MAX   = 1.5   # Max per bullet trajectory (was 2.0)
    REWARD_WIN_ELIMINATION = 25.0   # Victory bonus (was 20.0)
    REWARD_SHOT_FIRED      = 0.05  # Per trigger pull (was 0.20)
    REWARD_FINISHER_BONUS  = 3.0   # Consecutive hit within 120 ticks
    PENALTY_DAMAGE_TAKEN   = -4.0  # Per hit taken (was -2.0, teaches self-preservation)
    PENALTY_DEFEAT         = -10.0 # Dying in battle
    PENALTY_TIMEOUT        = -30.0 # Match timeout (was -10.0)

    def __init__(self, bot_name: str = None, max_steps: int = 3600, frame_stack: int = 3):
        self.max_steps = max_steps
        self.frame_stack = frame_stack
        self.bot_name = bot_name
        self.env_steps = 0

        from backend.opponent_pool import OpponentPool
        bot_dir = os.path.join(PROJECT_ROOT, "models", bot_name) if bot_name else ""
        self.pool = OpponentPool(bot_dir)
        self.active_opponent = None

        self.obs_dim = 30 * frame_stack + 1  # 91 with 3 frames + time_remaining
        self.act_dim = 5

        self.state: GameState | None = None
        self._obs_stack = deque(maxlen=frame_stack)
        self.p2_obs_stack = deque(maxlen=frame_stack)

        # Reward tracking
        self._prev_dist = 0.0
        self._last_hit_tick = 0  # For finisher momentum bonus

    def reset(self) -> np.ndarray:
        """Reset environment. Returns stacked observation."""
        self.env_steps = 0

        # Reload pool periodically
        if hasattr(self, '_total_resets'):
            self._total_resets += 1
            if self._total_resets % 100 == 0:
                self.pool._reload_pool()
        else:
            self._total_resets = 0

        self.active_opponent = self.pool.sample_opponent()

        self.state = create_state(
            p1_pos=[0.0, 15.0],
            p2_pos=[0.0, -15.0],
        )

        # Reset reward tracking
        self._prev_dist = 0.0
        self._last_hit_tick = 0

        obs1 = get_obs(self.state, perspective=1)

        # Initialize frame stacks
        self._obs_stack.clear()
        for _ in range(self.frame_stack):
            self._obs_stack.append(obs1.copy())

        obs2 = get_obs(self.state, perspective=2)
        self.p2_obs_stack.clear()
        for _ in range(self.frame_stack):
            self.p2_obs_stack.append(obs2.copy())

        return self._get_stacked_obs()

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, dict]:
        """
        Step the environment.
        
        Returns: (obs, reward, done, info)
        """
        self.env_steps += 1

        # Generate enemy action
        stacked_p2 = np.concatenate(list(self.p2_obs_stack)).astype(np.float32)
        
        # Append time_remaining so self-play ONNX opponents have correct input
        time_remaining = 1.0 - (self.state.tick_count / self.max_steps)
        stacked_p2_91 = np.append(stacked_p2, np.float32(time_remaining))
        
        enemy_action = self.pool.predict_opponent(self.active_opponent, stacked_p2_91)
        enemy_action[0:4] = -enemy_action[0:4]  # Rotate P2 action to world space

        # Tick physics
        tick(self.state, action1=action, action2=enemy_action)

        # Update P2 frame stack
        obs2 = get_obs(self.state, perspective=2)
        self.p2_obs_stack.append(obs2)

        # Get P1 observation
        obs = get_obs(self.state, perspective=1)
        self._obs_stack.append(obs.copy())

        # Compute reward
        reward = self._compute_reward(action, obs)

        # Termination
        terminated = self.state.p2_dead or self.state.p1_dead
        truncated = self.state.tick_count >= self.max_steps
        done = terminated or truncated

        # Timeout penalty (stalling is 3x worse than dying in combat)
        if truncated and not self.state.p2_dead:
            reward += self.PENALTY_TIMEOUT

        # Death penalty
        if self.state.p1_dead and not self.state.p2_dead:
            reward += self.PENALTY_DEFEAT

        info = {"is_success": bool(self.state.p2_dead and not self.state.p1_dead)}
        return self._get_stacked_obs(), reward, done, info

    def get_render_state(self) -> dict | None:
        """Get the current game state as a dict for live visualization."""
        if self.state is None:
            return None
        return state_to_dict(self.state)

    def _get_stacked_obs(self) -> np.ndarray:
        """Stack the last N frames + append time_remaining feature."""
        stacked = np.concatenate(list(self._obs_stack)).astype(np.float32)
        tick = self.state.tick_count if self.state else 0
        time_remaining = 1.0 - (tick / self.max_steps)
        return np.append(stacked, np.float32(time_remaining))

    def _compute_reward(self, action: np.ndarray, obs: np.ndarray) -> float:
        """
        Combat ledger v2:
        - Escalating time urgency (accelerates after tick 1800)
        - Damage dealt / taken
        - Near-miss trajectory reward
        - Finisher momentum bonus (consecutive hits within 120 ticks)
        - Victory / defeat / timeout
        """
        # 1. Escalating Urgency (gentle early, harsh late)
        tc = self.state.tick_count
        if tc > self.max_steps // 2:
            progress = (tc - self.max_steps // 2) / (self.max_steps // 2)
            reward = -0.005 * (1.0 + progress)  # -0.005 → -0.010
        else:
            reward = -0.001

        # 2. Firing (Base Reward — reduced to prevent trigger spam)
        shots_fired = self.state.p1_shots_fired
        if shots_fired > 0:
            reward += shots_fired * self.REWARD_SHOT_FIRED

        # 3. Damage Dealt + Finisher Momentum
        damage_dealt = self.state.p1_damage_dealt
        if damage_dealt > 0:
            reward += (damage_dealt / 34.0) * self.REWARD_DAMAGE_DEALT
            # Finisher bonus: rapid follow-up hit within 120 ticks
            if self._last_hit_tick > 0 and (tc - self._last_hit_tick) < 120:
                reward += self.REWARD_FINISHER_BONUS
            self._last_hit_tick = tc

        # 4. Damage Taken
        damage_taken = self.state.p1_damage_taken
        if damage_taken > 0:
            reward += (damage_taken / 34.0) * self.PENALTY_DAMAGE_TAKEN

        # 5. Near-Miss Trajectory Reward (reduced to prevent farming)
        near_miss_score = self.state.p1_near_miss_score
        if near_miss_score > 0:
            reward += near_miss_score * (self.REWARD_NEAR_MISS_MAX / 5.0)

        # 6. Victory
        if self.state.p2_dead and not self.state.p1_dead:
            reward += self.REWARD_WIN_ELIMINATION

        return reward


class VecEnv:
    """
    Vectorized environment running N BrawlEnv instances with ThreadPoolExecutor.
    
    Each env runs in a separate thread (GIL released during numpy ops).
    Provides batched reset/step for efficient rollout collection.
    """

    def __init__(self, env_fn, n_envs: int = 4):
        self.n_envs = n_envs
        self.envs = [env_fn() for _ in range(n_envs)]
        self.obs_dim = self.envs[0].obs_dim
        self.act_dim = self.envs[0].act_dim
        self._executor = ThreadPoolExecutor(max_workers=n_envs)

    def reset(self) -> np.ndarray:
        """Reset all environments. Returns (n_envs, obs_dim)."""
        futures = [self._executor.submit(env.reset) for env in self.envs]
        obs = np.array([f.result() for f in futures], dtype=np.float32)
        return obs

    def step(self, actions: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict]]:
        """
        Step all environments.
        
        Args:
            actions: (n_envs, act_dim)
            
        Returns:
            obs: (n_envs, obs_dim)
            rewards: (n_envs,)
            dones: (n_envs,) 
            infos: list of dicts
        """
        def _step_env(args):
            env, action = args
            obs, reward, done, info = env.step(action)
            if done:
                # Auto-reset on done
                new_obs = env.reset()
                info["terminal_obs"] = obs
                return new_obs, reward, done, info
            return obs, reward, done, info

        futures = [
            self._executor.submit(_step_env, (self.envs[i], actions[i]))
            for i in range(self.n_envs)
        ]
        results = [f.result() for f in futures]

        obs = np.array([r[0] for r in results], dtype=np.float32)
        rewards = np.array([r[1] for r in results], dtype=np.float32)
        dones = np.array([r[2] for r in results], dtype=bool)
        infos = [r[3] for r in results]

        return obs, rewards, dones, infos

    def get_render_states(self) -> list[dict | None]:
        """Get current game states from all envs for visualization."""
        return [env.get_render_state() for env in self.envs]

    def close(self):
        """Shutdown the thread pool."""
        self._executor.shutdown(wait=False)
