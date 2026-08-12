"""
main.py — FastAPI entry point & WebSocket connection manager.

Serves the frontend static files, handles WebSocket connections
for real-time gameplay and spectating, and exposes REST API
endpoints for the RL Studio (model management, training control).

Run with:
    uvicorn backend.main:app --host 0.0.0.0 --port 8000
"""

import json
import asyncio
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from backend.game_loop import GameSession
from backend import model_registry
from backend import trainer_worker
from backend.export_onnx import export_model

# ─── App Setup ────────────────────────────────────────────────────────────────

app = FastAPI(title="Brawl Sniper RL Studio")

FRONTEND_DIR = Path(__file__).parent.parent / "frontend"
MODEL_PATH = Path(__file__).parent.parent / "models" / "ppo_sniper_v1.onnx"


# ─── Health Check ─────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model_available": MODEL_PATH.exists(),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# REST API — Model Management
# ═══════════════════════════════════════════════════════════════════════════════

class BotCreateRequest(BaseModel):
    bot_name: str
    layers: list[int] = [128, 128]
    activation: str = "relu"
    learning_rate: float = 3e-4
    total_timesteps: int = 3_000_000
    n_envs: int = 8
    batch_size: int = 64
    n_steps: int = 2048


@app.get("/api/bots")
async def api_list_bots():
    """List all registered bots with metadata."""
    bots = model_registry.list_bots()
    # Augment with live training status
    for bot in bots:
        bot["is_training"] = trainer_worker.is_training(bot["bot_name"])
    return {"bots": bots}


@app.post("/api/bots")
async def api_create_bot(req: BotCreateRequest):
    """Register a new bot with architecture configuration."""
    try:
        metadata = model_registry.create_bot(req.bot_name, req.model_dump())
        return {"status": "created", "bot": metadata}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/api/bots/{bot_name}")
async def api_get_bot(bot_name: str):
    """Get metadata for a single bot."""
    bot = model_registry.get_bot(bot_name)
    if bot is None:
        raise HTTPException(status_code=404, detail=f"Bot '{bot_name}' not found")
    bot["is_training"] = trainer_worker.is_training(bot_name)
    return {"bot": bot}


@app.delete("/api/bots/{bot_name}")
async def api_delete_bot(bot_name: str):
    """Delete a bot and all its files."""
    # Stop training first if running
    if trainer_worker.is_training(bot_name):
        trainer_worker.stop_training(bot_name)

    if model_registry.delete_bot(bot_name):
        return {"status": "deleted", "bot_name": bot_name}
    raise HTTPException(status_code=404, detail=f"Bot '{bot_name}' not found")


# ═══════════════════════════════════════════════════════════════════════════════
# REST API — Training Control
# ═══════════════════════════════════════════════════════════════════════════════

@app.post("/api/bots/{bot_name}/train")
async def api_start_training(bot_name: str):
    """Start background training for a bot."""
    bot = model_registry.get_bot(bot_name)
    if bot is None:
        raise HTTPException(status_code=404, detail=f"Bot '{bot_name}' not found")

    if trainer_worker.is_training(bot_name):
        raise HTTPException(status_code=409, detail=f"Bot '{bot_name}' is already training")

    if bot.get("status") == "completed":
        raise HTTPException(status_code=409, detail=f"Bot '{bot_name}' has completed training. Delete and recreate to retrain.")

    bot_dir = model_registry.get_bot_dir(bot_name)
    started = trainer_worker.start_training(bot_name, bot_dir)

    if started:
        return {"status": "training_started", "bot_name": bot_name}
    raise HTTPException(status_code=500, detail="Failed to start training process")


@app.post("/api/bots/{bot_name}/stop")
async def api_stop_training(bot_name: str):
    """Stop a running training process."""
    if trainer_worker.stop_training(bot_name):
        return {"status": "stopped", "bot_name": bot_name}
    raise HTTPException(status_code=404, detail=f"No active training for '{bot_name}'")


@app.get("/api/bots/{bot_name}/status")
async def api_training_status(bot_name: str):
    """Poll training progress for a bot."""
    bot = model_registry.get_bot(bot_name)
    if bot is None:
        raise HTTPException(status_code=404, detail=f"Bot '{bot_name}' not found")

    return {
        "bot_name": bot_name,
        "status": bot.get("status", "unknown"),
        "current_step": bot.get("current_step", 0),
        "total_timesteps": bot.get("total_timesteps", 0),
        "avg_reward": bot.get("avg_reward", 0.0),
        "win_rate": bot.get("win_rate", 0.0),
        "is_training": trainer_worker.is_training(bot_name),
        "has_onnx": bot.get("has_onnx", False),
        "has_model": bot.get("has_model", False),
        "error_message": bot.get("error_message"),
    }


@app.post("/api/bots/{bot_name}/export")
async def api_export_onnx(bot_name: str):
    """Trigger ONNX export from the latest checkpoint."""
    bot = model_registry.get_bot(bot_name)
    if bot is None:
        raise HTTPException(status_code=404, detail=f"Bot '{bot_name}' not found")

    bot_dir = model_registry.get_bot_dir(bot_name)
    model_zip = model_registry.get_model_path(bot_name)

    if model_zip is None:
        raise HTTPException(status_code=400, detail="No trained model checkpoint found. Train the bot first.")

    import os
    onnx_path = os.path.join(bot_dir, "model.onnx")

    # Run export (this is fast, ~1-2 seconds)
    success = export_model(model_zip, onnx_path)

    if success:
        model_registry.update_bot(bot_name, {"has_onnx": True})
        return {"status": "exported", "bot_name": bot_name, "onnx_path": onnx_path}
    raise HTTPException(status_code=500, detail="ONNX export failed")


@app.get("/api/training/active")
async def api_active_training():
    """Get list of all currently training bots."""
    active = trainer_worker.get_all_active()
    results = []
    for name in active:
        bot = model_registry.get_bot(name)
        if bot:
            results.append({
                "bot_name": name,
                "current_step": bot.get("current_step", 0),
                "total_timesteps": bot.get("total_timesteps", 0),
                "avg_reward": bot.get("avg_reward", 0.0),
                "status": bot.get("status", "training"),
            })
    return {"active": results}


# ═══════════════════════════════════════════════════════════════════════════════
# WebSocket — Game & Spectator
# ═══════════════════════════════════════════════════════════════════════════════

@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()

    session: GameSession | None = None
    game_task: asyncio.Task | None = None

    try:
        # --- Step 1: Read mode selection ---
        init_msg = await ws.receive_text()
        init_data = json.loads(init_msg)
        mode = init_data.get("mode", "rule_bot")
        bot_name = init_data.get("bot", None)  # Optional: which trained bot to use

        is_spectating = mode.startswith("spectate")
        print(f"[WS] New session — mode: {mode}, bot: {bot_name}, spectate: {is_spectating}")

        # --- Step 2: Resolve model path ---
        # If a specific bot is requested, find its ONNX file
        model_path = None
        if bot_name:
            onnx_path = model_registry.get_onnx_path(bot_name)
            if onnx_path:
                model_path = onnx_path
            else:
                print(f"[WS] No ONNX found for bot '{bot_name}', using fallback")
        elif MODEL_PATH.exists():
            model_path = str(MODEL_PATH)

        # --- Step 3: Create game session ---
        session = GameSession(mode=mode, model_path=model_path)

        # --- Step 4: Define async send callback ---
        async def send_state(state_dict: dict):
            try:
                await ws.send_text(json.dumps(state_dict))
            except Exception:
                if session:
                    session.stop()

        # --- Step 5: Start game loop as background task ---
        game_task = asyncio.create_task(session.run(send_state))

        # --- Step 6: Handle input (or just wait in spectator mode) ---
        if is_spectating:
            # In spectator mode, no input needed — just wait for game to end
            await game_task
        else:
            # Human play mode — receive input stream
            session.is_running = True
            while session.is_running:
                try:
                    raw = await asyncio.wait_for(ws.receive_text(), timeout=0.5)
                    data = json.loads(raw)

                    # Skip init message echoes
                    if "mode" in data and "mx" not in data:
                        continue

                    session.set_human_input(
                        mx=float(data.get("mx", 0)),
                        my=float(data.get("my", 0)),
                        ax=float(data.get("ax", 0)),
                        ay=float(data.get("ay", 0)),
                        st=float(data.get("st", -1)),
                    )
                except asyncio.TimeoutError:
                    continue
                except (json.JSONDecodeError, ValueError) as e:
                    print(f"[WS] Invalid payload received: {e}")
                    continue

    except WebSocketDisconnect:
        print("[WS] Client disconnected")
    except Exception as e:
        print(f"[WS] Session error: {e}")
    finally:
        if session:
            session.stop()
        if game_task and not game_task.done():
            game_task.cancel()
        print("[WS] Session cleanly closed")


# ─── Static Files (Serve Frontend) ───────────────────────────────────────────
app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
