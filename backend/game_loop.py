"""
game_loop.py — Drift-compensated 60fps async game loop.

Runs as an asyncio task. Supports both human-play and spectator modes.

Human play: Player 1 = human (WebSocket input), Player 2 = bot
Spectator:  Player 1 = bot A (RL/rule), Player 2 = bot B (RL/rule)
"""

import asyncio
import time
import numpy as np

from backend.physics import (
    GameState,
    create_state,
    tick,
    get_obs,
    state_to_dict,
)
from backend.rule_bot import RuleBot
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

    def __init__(self, mode: str = "rule_bot", model_path: str | None = None):
        """
        Args:
            mode: Game mode string
            model_path: Path to .onnx model for RL bot(s)
        """
        self.mode = mode
        self.state: GameState = create_state()
        self.is_running = False
        self.is_spectating = mode.startswith("spectate")

        # Human input (updated by WebSocket handler in play modes)
        self._human_action = np.zeros(5, dtype=np.float32)

        # --- Initialize bots based on mode ---
        self.p1_bot = None  # Only used in spectator mode
        self.p2_bot = None  # Always used (opponent)

        if mode == "spectate_rl_vs_rule":
            # P1 = RL bot, P2 = Rule bot
            self.p1_bot = self._make_bot("rl", model_path)
            self.p2_bot = RuleBot()
        elif mode == "spectate_rl_vs_rl":
            # P1 = RL bot, P2 = RL bot (same model, two instances)
            self.p1_bot = self._make_bot("rl", model_path)
            self.p2_bot = self._make_bot("rl", model_path)
        elif mode == "rl_bot":
            # Human P1, RL P2
            self.p2_bot = self._make_bot("rl", model_path)
        elif mode == "rule_bot":
            # Human P1, Rule P2
            self.p2_bot = RuleBot()
        else:
            self.p2_bot = FallbackBot()

    def _make_bot(self, bot_type: str, model_path: str | None):
        """Create a bot instance."""
        if bot_type == "rl" and model_path:
            try:
                return ONNXBot(model_path)
            except (RuntimeError, FileNotFoundError) as e:
                print(f"[WARN] ONNX model unavailable ({e}), falling back to rule bot")
                return RuleBot()
        return RuleBot() if bot_type == "rule" else FallbackBot()

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
                action2 = self.p2_bot.predict(p2_obs)

                # --- Tick physics ---
                tick(self.state, action1=action1, action2=action2)

                # --- Send state to client ---
                state_dict = state_to_dict(self.state)
                try:
                    await send_callback(state_dict)
                except Exception:
                    # Client disconnected
                    self.is_running = False
                    break

                # --- Check game over ---
                if self.state.p1_dead or self.state.p2_dead:
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
                        # Send final state with game over flag
                        state_dict["go"] = 1 if self.state.p2_dead else 2
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
