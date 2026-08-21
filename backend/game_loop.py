"""
game_loop.py — Drift-compensated 60fps async game loop.

Runs as an asyncio task. Supports both human-play and spectator modes.

Human play: Player 1 = human (WebSocket input), Player 2 = bot
Spectator:  Player 1 = bot A (RL/rule), Player 2 = bot B (RL/rule)
"""

import asyncio
import time
import numpy as np
import os
from datetime import datetime

from backend.physics import (
    GameState,
    create_state,
    tick,
    get_obs,
    state_to_dict,
)
from backend.curriculum_bots import Level1Bot, Level2Bot, Level3Bot, Level4Bot, Level5Bot
from backend.ai_inference import ONNXBot, FallbackBot


TICK_RATE = 60
TICK_DURATION = 1.0 / TICK_RATE


class GameSession:
    """
    Manages a single game session between two players.

    Modes:
      - "rule_bot": Human (P1) vs Rule Bot (P2)
      - "rl_bot": Human (P1) vs RL Bot (P2)
      - "spectate_rl_vs_rule": RL Bot (P1) vs Rule Bot (P2) — no human input
      - "spectate_rl_vs_rl": RL Bot (P1) vs RL Bot (P2) — no human input
    """

    def __init__(self, mode: str = "rule_bot", model_path: str | None = None, record: bool = False, selected_bot: str = None, opponent: str = None):
        """
        Args:
            mode: Game mode string
            model_path: Path to .onnx model for RL bot(s)
            record: Whether to record P1 gameplay data
            selected_bot: The name of the specific rule bot to use
        """
        self.mode = mode
        self.state: GameState = create_state()
        self.is_running = False
        self.is_spectating = mode.startswith("spectate")
        self.record = record
        self.p1_score = 0.0
        self._last_hit_tick = 0

        # Human input (updated by WebSocket handler in play modes)
        self._human_action = np.zeros(5, dtype=np.float32)
        
        # Buffers for imitation learning
        self._obs_buffer = []
        self._action_buffer = []

        # --- Initialize bots based on mode ---
        self.p1_bot = None  # Only used in spectator mode
        self.p2_bot = None  # Always used (opponent)

        if mode == "spectate_rl_vs_rule":
            # P1 = RL bot, P2 = Rule bot (or explicitly chosen opponent)
            self.p1_bot = self._make_bot("rl", model_path)
            self.p2_bot = self._make_bot(opponent if opponent else "rule", None)
        elif mode == "spectate_rl_vs_rl":
            # P1 = RL bot, P2 = RL bot (same model, two instances)
            self.p1_bot = self._make_bot("rl", model_path)
            self.p2_bot = self._make_bot("rl", model_path)
        elif mode == "rl_bot":
            # Human P1, RL P2
            self.p2_bot = self._make_bot("rl", model_path)
        elif mode == "rule_bot" or mode == "record":
            # Human P1, Rule P2 (user can specify which rule bot)
            self.p2_bot = self._make_bot(selected_bot if selected_bot else "rule", None)
        else:
            self.p2_bot = FallbackBot()

    def _make_bot(self, bot_type: str, model_path: str | None):
        """Create a bot instance."""
        if bot_type == "rl" and model_path:
            try:
                return ONNXBot(model_path)
            except Exception as e:
                print(f"[WARN] ONNX model unavailable ({e}), falling back to Level4Bot")
                return Level4Bot()
                
        # Rule bots & aliases
        if bot_type in ("level1", "lazy"):
            return Level1Bot()
        elif bot_type in ("level2", "sniper"):
            return Level2Bot()
        elif bot_type in ("level3", "evasive"):
            return Level3Bot()
        elif bot_type in ("level4", "rule"):
            return Level4Bot()
        elif bot_type in ("level5", "aggressive"):
            return Level5Bot()
            
        return Level4Bot() if bot_type == "rule" else FallbackBot()

    def set_human_input(self, mx: float, my: float, ax: float, ay: float, st: float):
        """
        Called by the WebSocket handler when new input arrives.
        Thread-safe for asyncio (single-threaded event loop).
        """
        self._human_action[0] = mx
        self._human_action[1] = my
        self._human_action[2] = ax
        self._human_action[3] = ay
        self._human_action[4] = st

    async def run(self, send_callback):
        """
        Main game loop. Runs at exactly 60 ticks per second
        with drift compensation.

        Args:
            send_callback: async function(dict) to send game state to client
        """
        self.is_running = True

        # Reset all bots
        if self.p1_bot:
            self.p1_bot.reset()
        if self.p2_bot:
            self.p2_bot.reset()

        self.state = create_state()

        next_tick_time = time.perf_counter()

        while self.is_running:
            now = time.perf_counter()

            if now >= next_tick_time:
                # --- Get Player 1 action ---
                if self.is_spectating and self.p1_bot:
                    # Spectator mode: P1 is a bot
                    p1_obs = get_obs(self.state, perspective=1)
                    action1 = self.p1_bot.predict(p1_obs)
                else:
                    # Human play mode: read latest joystick input
                    action1 = self._human_action.copy()

                # --- Get Player 2 action ---
                p2_obs = get_obs(self.state, perspective=2)
                action2 = self.p2_bot.predict(p2_obs).copy()
                action2[0:4] = -action2[0:4]  # Rotate from P2 perspective to world space
                if self.record and not self.is_spectating:
                    p1_obs = get_obs(self.state, perspective=1)
                    self._obs_buffer.append(p1_obs)
                    self._action_buffer.append(action1.copy())

                # --- Tick physics ---
                tick(self.state, action1=action1, action2=action2)
                
                # --- Compute score for P1 (Human) ---
                if not self.is_spectating:
                    tc = self.state.tick_count
                    # 1. Escalating Urgency
                    if tc > 1800:
                        progress = (tc - 1800) / 1800.0
                        self.p1_score += -0.005 * (1.0 + progress)
                    else:
                        self.p1_score += -0.001
                    
                    # 2. Shots Fired
                    if self.state.p1_shots_fired > 0:
                        self.p1_score += self.state.p1_shots_fired * 0.05
                    
                    # 3. Damage Dealt
                    if self.state.p1_damage_dealt > 0:
                        self.p1_score += (self.state.p1_damage_dealt / 34.0) * 6.0
                        if self._last_hit_tick > 0 and (tc - self._last_hit_tick) < 120:
                            self.p1_score += 3.0
                        self._last_hit_tick = tc
                        
                    # 4. Damage Taken
                    if self.state.p1_damage_taken > 0:
                        self.p1_score += (self.state.p1_damage_taken / 34.0) * -4.0
                        
                    # 5. Near Miss
                    if self.state.p1_near_miss_score > 0:
                        self.p1_score += self.state.p1_near_miss_score * (1.5 / 5.0)

                # --- Send state to client ---
                state_dict = state_to_dict(self.state)
                try:
                    await send_callback(state_dict)
                except Exception:
                    # Client disconnected
                    self.is_running = False
                    break

                # --- Check game over ---
                timeout = self.state.tick_count >= 3600
                if self.state.p1_dead or self.state.p2_dead or timeout:
                    if self.is_spectating:
                        # Brief pause to show the death, then reset
                        await asyncio.sleep(1.0)
                        self.state = create_state()
                        if self.p1_bot:
                            self.p1_bot.reset()
                        if self.p2_bot:
                            self.p2_bot.reset()
                        next_tick_time = time.perf_counter() + TICK_DURATION
                        continue
                    else:
                        # Apply win/loss/timeout final reward
                        if self.state.p2_dead and not self.state.p1_dead:
                            self.p1_score += 25.0
                            winner = 1
                        elif self.state.p1_dead and not self.state.p2_dead:
                            self.p1_score -= 10.0
                            winner = 2
                        elif timeout:
                            self.p1_score -= 30.0
                            winner = 0
                        else:
                            winner = 3 # Draw (both died)

                        # Send final state with game over flag and score
                        state_dict["go"] = winner
                        state_dict["score"] = round(self.p1_score, 2)
                        try:
                            await send_callback(state_dict)
                        except Exception:
                            pass
                        self.is_running = False
                        break

                # --- Schedule next tick ---
                next_tick_time += TICK_DURATION

                # If we fell behind, skip to catch up (prevent spiral)
                if next_tick_time < now:
                    missed = int((now - next_tick_time) / TICK_DURATION)
                    next_tick_time += missed * TICK_DURATION

            # Sleep until next tick (yield to event loop)
            sleep_time = max(0.0, next_tick_time - time.perf_counter())
            if sleep_time > 0:
                await asyncio.sleep(sleep_time)
            else:
                await asyncio.sleep(0)  # Yield to event loop even if behind

    def stop(self):
        """Stop the game loop."""
        self.is_running = False
        if self.record and len(self._obs_buffer) > 0:
            global _pending_recording
            _pending_recording = {
                "obs": list(self._obs_buffer),
                "act": list(self._action_buffer),
            }
            self._obs_buffer.clear()
            self._action_buffer.clear()


# Buffer holding the last recorded game pending user confirmation
_pending_recording = None


def save_pending_recording() -> dict:
    global _pending_recording
    if _pending_recording is None or len(_pending_recording.get("obs", [])) == 0:
        return {"status": "error", "message": "No pending recording found"}
    
    dataset_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "datasets"))
    os.makedirs(dataset_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(dataset_dir, f"imitation_data_{timestamp}.npz")
    
    obs_arr = np.array(_pending_recording["obs"], dtype=np.float32)
    act_arr = np.array(_pending_recording["act"], dtype=np.float32)
    
    np.savez_compressed(
        filename,
        observations=obs_arr,
        actions=act_arr,
    )
    steps = len(obs_arr)
    print(f"[Record] User saved {steps} steps to {filename}")
    
    _pending_recording = None
    return {"status": "saved", "filename": os.path.basename(filename), "steps": steps}


def discard_pending_recording() -> dict:
    global _pending_recording
    _pending_recording = None
    print("[Record] User discarded pending recording")
    return {"status": "discarded"}


def get_pending_recording_info() -> dict:
    global _pending_recording
    if _pending_recording and len(_pending_recording.get("obs", [])) > 0:
        return {"has_pending": True, "steps": len(_pending_recording.get("obs", []))}
    return {"has_pending": False, "steps": 0}
