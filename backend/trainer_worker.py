"""
trainer_worker.py — Background training process manager.

Uses multiprocessing.Process to isolate heavy PyTorch/SB3 training
from the FastAPI event loop, ensuring zero frame-rate impact on live games.

Each training job runs in its own process with its own SubprocVecEnv pool.
Progress is communicated via the filesystem (metadata.json).
"""

import os
import sys
import json
import time
import multiprocessing
import traceback
import shutil
from typing import Any


# Global registry of active training processes (in the main server process)
_active_processes: dict[str, multiprocessing.Process] = {}

# Absolute project root (for imports inside child processes)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _run_training_job(bot_dir: str):
    """
    The actual training function that runs inside an isolated child process.

    Reads config from metadata.json, creates parallel envs, trains PPO,
    and periodically saves checkpoints + updates metadata.
    """
    # Force UTF-8 encoding to prevent PyTorch ONNX exporter from crashing on Windows with emojis
    if sys.platform == 'win32':
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')

    # Ensure project root is importable inside the child process
    if PROJECT_ROOT not in sys.path:
        sys.path.insert(0, PROJECT_ROOT)

    meta_path = os.path.join(bot_dir, "metadata.json")

    try:
        # --- 1. Read config ---
        with open(meta_path, "r") as f:
            meta = json.load(f)

        bot_name = meta["bot_name"]
        print(f"[Trainer:{bot_name}] Starting training job...", flush=True)

        # Mark as training
        meta["status"] = "training"
        meta["error_message"] = None
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

        # --- 2. Lazy imports (heavy, only in child process) ---
        import torch
        import numpy as np
        from stable_baselines3 import PPO
        from stable_baselines3.common.env_util import make_vec_env
        from stable_baselines3.common.vec_env import SubprocVecEnv, VecFrameStack
        from stable_baselines3.common.callbacks import BaseCallback
        from env.brawl_sniper_env import BrawlSniperEnv

        # --- 3. Detect device ---
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cuda":
            gpu_name = torch.cuda.get_device_name(0)
            print(f"[Trainer:{bot_name}] Using GPU: {gpu_name}", flush=True)
        else:
            print(f"[Trainer:{bot_name}] Using CPU (no CUDA GPU detected)", flush=True)

        # --- 4. Create parallel environments ---
        n_envs = meta.get("n_envs", 8)
        # Cap to available CPUs minus 2 (reserve for game server)
        max_envs = max(1, multiprocessing.cpu_count() - 2)
        n_envs = min(n_envs, max_envs)

        print(f"[Trainer:{bot_name}] Spawning {n_envs} parallel environments...", flush=True)
        envs = make_vec_env(BrawlSniperEnv, n_envs=n_envs, vec_env_cls=SubprocVecEnv)
        
        # Apply Frame Stacking (3 frames) for superhuman reflex perception
        envs = VecFrameStack(envs, n_stack=3)

        # --- 5. Configure network architecture ---
        activation_map = {
            "relu": torch.nn.ReLU,
            "tanh": torch.nn.Tanh,
            "elu": torch.nn.ELU,
            "swish": torch.nn.SiLU,
            "silu": torch.nn.SiLU,
        }
        # Use Swish by default for continuous control
        activation_fn = activation_map.get(meta.get("activation", "swish"), torch.nn.SiLU)
        layers = meta.get("layers", [256, 256, 128])

        policy_kwargs = dict(
            activation_fn=activation_fn,
            net_arch=dict(pi=layers, vf=layers),
        )

        # --- Schedules ---
        def linear_schedule(initial_value: float, final_value: float):
            def func(progress_remaining: float) -> float:
                return progress_remaining * (initial_value - final_value) + final_value
            return func
            
        lr_schedule = linear_schedule(meta.get("learning_rate", 3e-4), 1e-5)
        total_steps = meta.get("total_timesteps", 50_000_000)

        # --- Entropy Decay Callback ---
        class EntCoefDecayCallback(BaseCallback):
            def __init__(self, initial=0.01, final=0.001, total_timesteps=total_steps):
                super().__init__()
                self.initial = initial
                self.final = final
                self.total = total_timesteps
                
            def _on_step(self) -> bool:
                progress = self.num_timesteps / max(1, self.total)
                current = self.initial - progress * (self.initial - self.final)
                self.model.ent_coef = max(self.final, current)
                return True
                
        ent_callback = EntCoefDecayCallback(total_timesteps=total_steps)

        # --- 6. Create or resume PPO model ---
        model_zip_path = os.path.join(bot_dir, "model.zip")
        current_step = meta.get("current_step", 0)

        if current_step > 0 and os.path.isfile(model_zip_path):
            print(f"[Trainer:{bot_name}] Resuming from step {current_step:,}...", flush=True)
            model = PPO.load(model_zip_path, env=envs, device=device)
        else:
            print(f"[Trainer:{bot_name}] Creating fresh PPO model...", flush=True)
            model = PPO(
                "MlpPolicy",
                envs,
                learning_rate=lr_schedule,
                n_steps=meta.get("n_steps", 4096),
                batch_size=meta.get("batch_size", 512),
                n_epochs=10,
                gamma=0.99,
                gae_lambda=0.95,
                clip_range=0.2,
                ent_coef=0.01,
                policy_kwargs=policy_kwargs,
                verbose=0,
                device=device,
            )

        # --- 7. Training loop in chunks ---
        chunk_size = 100_000  # Save checkpoint every 100K steps
        steps_done = current_step

        print(f"[Trainer:{bot_name}] Training {total_steps:,} total steps "
              f"(starting from {steps_done:,})...", flush=True)

        while steps_done < total_steps:
            chunk_start = time.perf_counter()
            remaining = total_steps - steps_done
            this_chunk = min(chunk_size, remaining)

            model.learn(
                total_timesteps=this_chunk,
                reset_num_timesteps=False,
                progress_bar=False,
                callback=ent_callback,
            )
            steps_done += this_chunk
            chunk_elapsed = time.perf_counter() - chunk_start
            steps_per_sec = this_chunk / chunk_elapsed if chunk_elapsed > 0 else 0

            # Save checkpoint and export ONNX
            model.save(model_zip_path)
            
            # Export ONNX mid-training so live spectating always has the latest weights
            try:
                from backend.export_onnx import export_model
                onnx_path = os.path.join(bot_dir, "model.onnx")
                if export_model(model_zip_path, onnx_path):
                    meta["has_onnx"] = True
                    # Sync to opponent pool for PFSP curriculum
                    pool_dir = os.path.join(PROJECT_ROOT, "models", "opponent_pool")
                    os.makedirs(pool_dir, exist_ok=True)
                    pool_path = os.path.join(pool_dir, f"{bot_name}_step_{steps_done}.onnx")
                    shutil.copy(onnx_path, pool_path)
            except Exception as e:
                print(f"[Trainer:{bot_name}] Mid-training ONNX export failed: {e}", flush=True)

            # Compute average reward from the latest rollout buffer
            avg_reward = 0.0
            if hasattr(model, "ep_info_buffer") and len(model.ep_info_buffer) > 0:
                rewards = [ep["r"] for ep in model.ep_info_buffer]
                avg_reward = float(np.mean(rewards))

            # Update metadata
            meta["current_step"] = steps_done
            meta["avg_reward"] = round(avg_reward, 2)
            meta["has_model"] = True
            meta["status"] = "training" if steps_done < total_steps else "completed"

            with open(meta_path, "w") as f:
                json.dump(meta, f, indent=2)

            pct = steps_done / total_steps * 100
            print(f"[Trainer:{bot_name}] Step {steps_done:,}/{total_steps:,} "
                  f"({pct:.1f}%) | {steps_per_sec:,.0f} steps/s | "
                  f"avg_reward={avg_reward:.2f}", flush=True)

        # --- 8. Final Export (if needed) ---
        print(f"[Trainer:{bot_name}] Training complete!", flush=True)

        # Cleanup
        envs.close()
        print(f"[Trainer:{bot_name}] [OK] All done!", flush=True)

    except Exception as e:
        # Write error to metadata so the frontend can display it
        print(f"[Trainer:{meta.get('bot_name', '?')}] [ERROR] Training failed: {e}", flush=True)
        traceback.print_exc()
        try:
            with open(meta_path, "r") as f:
                meta = json.load(f)
            meta["status"] = "error"
            meta["error_message"] = str(e)
            with open(meta_path, "w") as f:
                json.dump(meta, f, indent=2)
        except Exception:
            pass


def start_training(bot_name: str, bot_dir: str) -> bool:
    """
    Spawn an isolated background process for training.
    """
    if bot_name in _active_processes and _active_processes[bot_name].is_alive():
        return False

    p = multiprocessing.Process(
        target=_run_training_job,
        args=(bot_dir,),
        name=f"trainer-{bot_name}",
        daemon=False,
    )
    p.start()
    _active_processes[bot_name] = p
    print(f"[TrainerManager] Started training process for '{bot_name}' (PID: {p.pid})", flush=True)
    return True


def stop_training(bot_name: str) -> bool:
    """
    Terminate a running training process.
    """
    if bot_name not in _active_processes:
        return False

    p = _active_processes[bot_name]
    if p.is_alive():
        print(f"[TrainerManager] Terminating training for '{bot_name}'...", flush=True)
        p.terminate()
        p.join(timeout=5)
        if p.is_alive():
            p.kill()

    del _active_processes[bot_name]

    # Update metadata status
    from backend.model_registry import _meta_path
    meta_path = _meta_path(bot_name)
    try:
        with open(meta_path, "r") as f:
            meta = json.load(f)
        if meta["status"] == "training":
            meta["status"] = "created"  # Reset to created (can resume)
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)
    except Exception:
        pass

    return True


def is_training(bot_name: str) -> bool:
    """Check if a training process is currently alive for this bot."""
    if bot_name not in _active_processes:
        return False
    if not _active_processes[bot_name].is_alive():
        # Clean up dead processes
        del _active_processes[bot_name]
        return False
    return True


def get_all_active() -> list[str]:
    """Return list of bot names that are currently training."""
    # Clean up dead processes first
    dead = [name for name, p in _active_processes.items() if not p.is_alive()]
    for name in dead:
        del _active_processes[name]
    return list(_active_processes.keys())
