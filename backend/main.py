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
import os

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from backend.game_loop import GameSession, save_pending_recording, discard_pending_recording, get_pending_recording_info
from backend import model_registry
from backend import trainer_worker

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
    use_lr_scheduler: bool = True
    total_timesteps: int = 3_000_000
    n_envs: int = 8
    batch_size: int = 512
    n_steps: int = 4096
    n_epochs: int = 10
    curriculum_level: int = 1
    base_model: str = ""
    # PPO hyperparameters
    clip_range: float = 0.2
    ent_coef: float = 0.0
    vf_coef: float = 0.5
    gamma: float = 0.99
    gae_lambda: float = 0.95
    max_grad_norm: float = 0.5
    target_kl: float | None = None
    init_log_std: float = -0.5

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
        # Enforce architecture inheritance if a base model is selected
        if req.base_model:
            base_meta_path = os.path.join(model_registry.MODELS_DIR, req.base_model, "metadata.json")
            if os.path.exists(base_meta_path):
                with open(base_meta_path, "r") as f:
                    base_meta = json.load(f)
                req.layers = base_meta.get("layers", req.layers)
                req.activation = base_meta.get("activation", req.activation)
            else:
                raise ValueError(f"Base model '{req.base_model}' metadata not found.")

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


@app.get("/api/imitation/pending")
async def api_get_pending_imitation():
    """Check if there is a recorded match pending save."""
    return get_pending_recording_info()

@app.get("/api/imitation/models")
async def api_list_imitation_models():
    """List all available imitation models with their architecture metadata."""
    models_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models"))
    imitation_models = []
    if os.path.exists(models_dir):
        for f in os.listdir(models_dir):
            if f.startswith("imitation_") and os.path.isdir(os.path.join(models_dir, f)):
                # Only include it if model.pth exists (required to load PyTorch weights)
                if os.path.exists(os.path.join(models_dir, f, "model.pth")):
                    meta = {}
                    meta_path = os.path.join(models_dir, f, "metadata.json")
                    if os.path.exists(meta_path):
                        try:
                            with open(meta_path, "r") as fp:
                                meta = json.load(fp)
                        except Exception:
                            pass
                    imitation_models.append({
                        "name": f,
                        "layers": meta.get("layers", [128, 128, 64]),
                        "activation": meta.get("activation", "relu")
                    })
    imitation_models.sort(key=lambda x: x["name"], reverse=True)
    return {"models": imitation_models}


@app.post("/api/imitation/save")
async def api_save_imitation():
    """Save the pending recorded match to datasets/."""
    result = save_pending_recording()
    if result.get("status") == "error":
        raise HTTPException(status_code=400, detail=result.get("message"))
    return result


@app.post("/api/imitation/discard")
async def api_discard_imitation():
    """Discard the pending recorded match."""
    return discard_pending_recording()


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
        "matches_played": bot.get("matches_played", 0),
        "curriculum_level": bot.get("curriculum_level", 1),
        "level_win_rate": bot.get("level_win_rate", 0.0),
        "level_matches": bot.get("level_matches", 0),
        "avg_reward": bot.get("avg_reward", 0.0),
        "win_rate": bot.get("win_rate", 0.0),
        "is_training": trainer_worker.is_training(bot_name),
        "has_onnx": bot.get("has_onnx", False),
        "has_model": bot.get("has_model", False),
        "error_message": bot.get("error_message"),
    }

@app.get("/api/bots/{bot_name}/progress")
async def api_training_progress(bot_name: str):
    """Fetch the JSON Lines progress history for charting."""
    bot = model_registry.get_bot(bot_name)
    if bot is None:
        raise HTTPException(status_code=404, detail=f"Bot '{bot_name}' not found")
        
    progress_path = os.path.join(model_registry.MODELS_DIR, bot_name, "progress.json")
    if not os.path.exists(progress_path):
        return []
        
    data = []
    try:
        with open(progress_path, "r") as f:
            for line in f:
                if line.strip():
                    try:
                        data.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
    except Exception as e:
        print(f"Error reading progress.json: {e}")
        
    return data


@app.post("/api/bots/{bot_name}/export")
async def api_export_onnx(bot_name: str):
    """Trigger ONNX export from the latest checkpoint."""
    bot = model_registry.get_bot(bot_name)
    if bot is None:
        raise HTTPException(status_code=404, detail=f"Bot '{bot_name}' not found")

    bot_dir = model_registry.get_bot_dir(bot_name)
    model_pt = model_registry.get_model_path(bot_name)

    if model_pt is None:
        raise HTTPException(status_code=400, detail="No trained model checkpoint found. Train the bot first.")

    onnx_path = os.path.join(bot_dir, "model.onnx")

    from backend.export_onnx import export_model
    success = export_model(
        model_pt,
        onnx_path,
        obs_dim=90,
        act_dim=5,
        layers=bot.get("layers", [256, 256, 128]),
        activation=bot.get("activation", "silu"),
    )

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
                "matches_played": bot.get("matches_played", 0),
                "curriculum_level": bot.get("curriculum_level", 1),
                "level_win_rate": bot.get("level_win_rate", 0.0),
                "level_matches": bot.get("level_matches", 0),
                "avg_reward": bot.get("avg_reward", 0.0),
                "status": bot.get("status", "training"),
                "has_onnx": bot.get("has_onnx", False),
            })
    return {"active": results}


# ═══════════════════════════════════════════════════════════════════════════════
# WebSocket — Live Training Visualization
# ═══════════════════════════════════════════════════════════════════════════════

@app.websocket("/ws/training/{bot_name}")
async def ws_training_live(ws: WebSocket, bot_name: str):
    """Stream live game states from a training process for real-time visualization."""
    await ws.accept()
    print(f"[WS:Training] Client connected for bot '{bot_name}'")

    try:
        while True:
            q = trainer_worker.get_live_queue(bot_name)
            if q is None:
                # Bot is not training, send empty state and wait
                await ws.send_text(json.dumps({"status": "not_training"}))
                await asyncio.sleep(2.0)
                continue

            # Drain queue and send latest state
            state = None
            import queue
            try:
                while True:
                    state = q.get_nowait()
            except queue.Empty:
                pass
            except Exception:
                pass

            if state:
                await ws.send_text(json.dumps(state))
            
            await asyncio.sleep(0.033)  # ~30fps

    except WebSocketDisconnect:
        print(f"[WS:Training] Client disconnected for '{bot_name}'")
    except Exception as e:
        print(f"[WS:Training] Error: {e}")


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
        opponent = init_data.get("opponent", None)  # Optional: specific opponent to face
        snapshot = init_data.get("snapshot", None) # Optional: specific level snapshot

        is_spectating = mode.startswith("spectate")
        is_recording = init_data.get("record", False)
        print(f"[WS] New session — mode: {mode}, bot: {bot_name}, snapshot: {snapshot}, opponent: {opponent}, spectate: {is_spectating}, record: {is_recording}")

        # --- Step 2: Resolve model path ---
        model_path = None
        if bot_name and not is_recording:
            if snapshot:
                bot_dir = model_registry.get_bot_dir(bot_name)
                onnx_path = os.path.join(bot_dir, snapshot)
            else:
                onnx_path = model_registry.get_onnx_path(bot_name)
                
            if onnx_path and os.path.exists(onnx_path):
                model_path = onnx_path
            else:
                print(f"[WS] No ONNX found for bot '{bot_name}' snapshot '{snapshot}', using fallback")
        elif MODEL_PATH.exists():
            model_path = str(MODEL_PATH)

        # --- Step 3: Create game session ---
        session = GameSession(mode=mode, model_path=model_path, record=is_recording, selected_bot=bot_name, opponent=opponent)

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
            await game_task
        else:
            session.is_running = True
            while session.is_running:
                try:
                    raw = await asyncio.wait_for(ws.receive_text(), timeout=0.5)
                    data = json.loads(raw)

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
