"""
model_registry.py — CRUD manager for trained bot metadata.

Each bot lives in models/<bot_name>/ with:
  - metadata.json  (config, progress, stats)
  - model.pt       (native PyTorch checkpoint)
  - model.onnx     (production inference model)
"""

import os
import json
import shutil
from datetime import datetime, timezone
from typing import Any


MODELS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models"))


def _meta_path(bot_name: str) -> str:
    return os.path.join(MODELS_DIR, bot_name, "metadata.json")


def _bot_dir(bot_name: str) -> str:
    return os.path.join(MODELS_DIR, bot_name)


def _sanitize_name(name: str) -> str:
    """Sanitize bot name to be filesystem-safe."""
    return "".join(c if c.isalnum() or c in ("_", "-") else "_" for c in name).strip("_")


def list_bots() -> list[dict[str, Any]]:
    """
    Scan the models/ directory for registered bots.
    Returns a list of metadata dicts, each augmented with `has_onnx` and `has_model`.
    """
    os.makedirs(MODELS_DIR, exist_ok=True)
    bots = []
    for folder_name in sorted(os.listdir(MODELS_DIR)):
        meta_path = _meta_path(folder_name)
        if os.path.isfile(meta_path):
            try:
                with open(meta_path, "r") as f:
                    data = json.load(f)
                bot_dir = _bot_dir(folder_name)
                data["has_onnx"] = os.path.isfile(os.path.join(bot_dir, "model.onnx"))
                data["has_model"] = os.path.isfile(os.path.join(bot_dir, "model.pt"))
                
                snapshots = [f for f in os.listdir(bot_dir) if f.startswith("model_lvl_") and f.endswith(".onnx")]
                if data["has_onnx"]:
                    snapshots.append("model.onnx")
                data["snapshots"] = sorted(snapshots)
                
                bots.append(data)
            except (json.JSONDecodeError, IOError):
                continue
    return bots


def get_bot(bot_name: str) -> dict[str, Any] | None:
    """Get metadata for a single bot. Returns None if not found."""
    meta_path = _meta_path(bot_name)
    if not os.path.isfile(meta_path):
        return None
    try:
        with open(meta_path, "r") as f:
            data = json.load(f)
        bot_dir = _bot_dir(bot_name)
        data["has_onnx"] = os.path.isfile(os.path.join(bot_dir, "model.onnx"))
        data["has_model"] = os.path.isfile(os.path.join(bot_dir, "model.pt"))
        
        snapshots = [f for f in os.listdir(bot_dir) if f.startswith("model_lvl_") and f.endswith(".onnx")]
        if data["has_onnx"]:
            snapshots.append("model.onnx")
        data["snapshots"] = sorted(snapshots)
        
        return data
    except (json.JSONDecodeError, IOError):
        return None


def create_bot(bot_name: str, config: dict[str, Any]) -> dict[str, Any]:
    """
    Register a new bot with the given configuration.

    Args:
        bot_name: Human-readable name (will be sanitized for filesystem)
        config: Dict with keys like 'layers', 'activation', 'learning_rate', etc.

    Returns:
        The created metadata dict.

    Raises:
        ValueError: If bot name is empty or already exists.
    """
    safe_name = _sanitize_name(bot_name)
    if not safe_name:
        raise ValueError("Bot name cannot be empty")

    bot_dir = _bot_dir(safe_name)
    if os.path.exists(bot_dir):
        raise ValueError(f"Bot '{safe_name}' already exists")

    os.makedirs(bot_dir, exist_ok=True)

    metadata = {
        "bot_name": safe_name,
        "display_name": bot_name,
        "layers": config.get("layers", [128, 128]),
        "activation": config.get("activation", "relu"),
        "learning_rate": float(config.get("learning_rate", 3e-4)),
        "use_lr_scheduler": bool(config.get("use_lr_scheduler", False)),
        "total_timesteps": int(config.get("total_timesteps", 3_000_000)),
        "n_envs": int(config.get("n_envs", 4)),
        "batch_size": int(config.get("batch_size", 512)),
        "n_steps": int(config.get("n_steps", 4096)),
        "current_step": 0,
        "matches_played": 0,
        "curriculum_level": 3 if config.get("base_model") else int(config.get("curriculum_level", 1)),
        "base_model": config.get("base_model", ""),
        "level_win_rate": 0.0,
        "level_matches": 0,
        "avg_reward": 0.0,
        "win_rate": 0.0,
        "status": "created",  # created | training | completed | error
        "created_at": datetime.now(timezone.utc).isoformat(),
        "has_onnx": False,
        "has_model": False,
        "error_message": None,
    }

    with open(_meta_path(safe_name), "w") as f:
        json.dump(metadata, f, indent=2)

    return metadata


def update_bot(bot_name: str, updates: dict[str, Any]) -> dict[str, Any] | None:
    """
    Partially update a bot's metadata.
    Only updates keys present in `updates`, preserves everything else.

    Returns the updated metadata, or None if bot not found.
    """
    meta_path = _meta_path(bot_name)
    if not os.path.isfile(meta_path):
        return None

    with open(meta_path, "r") as f:
        data = json.load(f)

    # Only update known mutable fields
    mutable_keys = {
        "current_step", "avg_reward", "win_rate", "status",
        "error_message", "has_onnx", "has_model",
    }
    for key, value in updates.items():
        if key in mutable_keys:
            data[key] = value

    with open(meta_path, "w") as f:
        json.dump(data, f, indent=2)

    return data


def delete_bot(bot_name: str) -> bool:
    """
    Delete a bot and all its files.
    Returns True if deleted, False if not found.
    """
    bot_dir = _bot_dir(bot_name)
    if not os.path.isdir(bot_dir):
        return False
    shutil.rmtree(bot_dir)
    return True


def get_bot_dir(bot_name: str) -> str:
    """Return the absolute path to a bot's directory."""
    return _bot_dir(bot_name)


def get_model_path(bot_name: str) -> str | None:
    """Return the path to the bot's .pt model file, or None if it doesn't exist."""
    path = os.path.join(_bot_dir(bot_name), "model.pt")
    return path if os.path.isfile(path) else None


def get_onnx_path(bot_name: str) -> str | None:
    """Return the path to the bot's .onnx file, or None if it doesn't exist."""
    path = os.path.join(_bot_dir(bot_name), "model.onnx")
    return path if os.path.isfile(path) else None
