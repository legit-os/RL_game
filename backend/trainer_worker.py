"""
trainer_worker.py — Background training process manager.

Uses multiprocessing.Process to isolate heavy PyTorch training
from the FastAPI event loop. Native PPO — no SB3 dependency.

Each training job runs in its own process.
Progress is communicated via the filesystem (metadata.json + progress.json).
Live game states are sent via multiprocessing.Queue for WebSocket streaming.
"""

import os
import sys
import json
import time
import threading
import queue
import traceback
from typing import Any

# Global registry of active training threads
_active_processes: dict[str, threading.Thread] = {}

# Flags to gracefully stop training threads
_stop_flags: dict[str, bool] = {}

# Shared queues for live game state streaming (bot_name → Queue)
_live_queues: dict[str, Any] = {}

# Absolute project root (for imports inside child processes)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _run_training_job(bot_dir: str, live_queue: Any = None):
    """
    The actual training function that runs inside an isolated child process.

    Reads config from metadata.json, creates parallel envs, trains PPO,
    and periodically saves checkpoints + updates metadata.
    """
    # Force UTF-8 encoding on Windows
    if sys.platform == 'win32':
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')

    # Ensure project root is importable
    if PROJECT_ROOT not in sys.path:
        sys.path.insert(0, PROJECT_ROOT)

    meta_path = os.path.join(bot_dir, "metadata.json")

    try:
        # --- 1. Read config ---
        with open(meta_path, "r") as f:
            meta = json.load(f)

        bot_name = meta["bot_name"]
        print(f"[Trainer:{bot_name}] Starting training job...", flush=True)

        meta["status"] = "training"
        meta["error_message"] = None
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

        # --- 2. Lazy imports (heavy, only in child process) ---
        import torch
        import numpy as np
        from backend.models import Actor, Critic
        from backend.ppo import PPO
        from env.env import BrawlEnv, VecEnv

        # Limit PyTorch internal worker threads
        torch.set_num_threads(4)

        # --- 3. Device ---
        device = "cpu"
        print(f"[Trainer:{bot_name}] Using device: CPU (optimal for vectorized MLP PPO)", flush=True)

        # --- 4. Create parallel environments ---
        n_envs = int(meta.get("n_envs", 8))
        max_envs = max(1, (os.cpu_count() or 4) - 2)
        n_envs = min(n_envs, max_envs)

        print(f"[Trainer:{bot_name}] Spawning {n_envs} parallel environments...", flush=True)
        env = VecEnv(
            env_fn=lambda: BrawlEnv(bot_name=bot_name),
            n_envs=n_envs
        )

        # --- 5. Configure network architecture ---
        obs_dim = env.obs_dim   # 91 (3 × 30 stacked + 1 time feature)
        act_dim = env.act_dim   # 5

        activation_name = meta.get("activation", "swish")
        layers = meta.get("layers", [256, 256, 128])

        is_fine_tuning = bool(meta.get("base_model"))
        init_log_std = float(meta.get("init_log_std", -2.0 if is_fine_tuning else -0.5))

        actor = Actor(
            obs_dim=obs_dim,
            act_dim=act_dim,
            hidden_layers=layers,
            activation=activation_name,
            init_log_std=init_log_std,
        )
        critic = Critic(
            obs_dim=obs_dim,
            hidden_layers=layers,
            activation=activation_name,
        )

        # --- 6. PPO hyperparameters (all customizable from metadata) ---
        lr = float(meta.get("learning_rate", 3e-4))
        clip_range = float(meta.get("clip_range", 0.2))
        ent_coef = float(meta.get("ent_coef", 0.0))
        vf_coef = float(meta.get("vf_coef", 0.5))
        gamma = float(meta.get("gamma", 0.99))
        gae_lambda = float(meta.get("gae_lambda", 0.95))
        max_grad_norm = float(meta.get("max_grad_norm", 0.5))
        n_steps = int(meta.get("n_steps", 4096))
        batch_size = int(meta.get("batch_size", 512))
        n_epochs = int(meta.get("n_epochs", 10))
        target_kl = meta.get("target_kl", None)
        if target_kl is not None:
            target_kl = float(target_kl)

        ppo = PPO(
            actor=actor,
            critic=critic,
            lr=lr,
            gamma=gamma,
            gae_lambda=gae_lambda,
            clip_range=clip_range,
            ent_coef=ent_coef,
            vf_coef=vf_coef,
            max_grad_norm=max_grad_norm,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=n_epochs,
            target_kl=target_kl,
            device=device,
        )

        print(f"[Trainer:{bot_name}] PPO Config: lr={lr}, clip={clip_range}, ent={ent_coef}, "
              f"vf={vf_coef}, gamma={gamma}, gae_lambda={gae_lambda}, "
              f"n_steps={n_steps}, batch={batch_size}, epochs={n_epochs}", flush=True)

        # --- 7. Resume or initialize ---
        model_pt_path = os.path.join(bot_dir, "model.pt")
        current_step = meta.get("current_step", 0)
        total_steps = meta.get("total_timesteps", 50_000_000)

        if current_step > 0 and os.path.isfile(model_pt_path):
            print(f"[Trainer:{bot_name}] Resuming from step {current_step:,}...", flush=True)
            ppo.load(model_pt_path)
        elif meta.get("base_model"):
            # Initialize from imitation model
            base_model_name = meta["base_model"]
            base_model_path = os.path.join(PROJECT_ROOT, "models", base_model_name, "model.pth")

            if os.path.isfile(base_model_path):
                print(f"[Trainer:{bot_name}] Loading imitation base model '{base_model_name}'...", flush=True)
                try:
                    im_dict = torch.load(base_model_path, map_location=device, weights_only=True)

                    # --- Expand first layer weights from old obs_dim to new obs_dim ---
                    # The imitation model was trained with obs_dim=90.
                    # Our new model has obs_dim=91 (extra time feature).
                    # We zero-initialize the new input column to preserve pretrained weights.
                    first_layer_key = "backbone.0.weight"
                    if first_layer_key in im_dict:
                        old_weight = im_dict[first_layer_key]  # Shape: (hidden, 90)
                        old_in_features = old_weight.shape[1]
                        if old_in_features < obs_dim:
                            new_cols = obs_dim - old_in_features
                            expanded = torch.zeros(old_weight.shape[0], obs_dim, dtype=old_weight.dtype)
                            expanded[:, :old_in_features] = old_weight
                            im_dict[first_layer_key] = expanded
                            print(f"[Trainer:{bot_name}] Expanded {first_layer_key}: {old_in_features} → {obs_dim} (+{new_cols} zero-init cols)", flush=True)

                    # Load actor weights with the expanded dict
                    actor.load_state_dict(im_dict, strict=False)

                    # For the critic, copy backbone weights (with expansion)
                    critic_state = critic.state_dict()
                    for key in im_dict:
                        if key.startswith("backbone."):
                            critic_key = key
                            if critic_key in critic_state and im_dict[key].shape == critic_state[critic_key].shape:
                                critic_state[critic_key] = im_dict[key]
                    critic.load_state_dict(critic_state)

                    print(f"[Trainer:{bot_name}] Successfully loaded imitation model weights (expanded {old_in_features}→{obs_dim}).", flush=True)

                    # Save initial checkpoint + ONNX
                    ppo.save(model_pt_path)
                    _export_actor_onnx(actor, obs_dim, os.path.join(bot_dir, "model.onnx"), device)
                    _export_actor_onnx(actor, obs_dim, os.path.join(bot_dir, "model_lvl_1.onnx"), device)
                    _export_actor_onnx(actor, obs_dim, os.path.join(bot_dir, "model_lvl_2.onnx"), device)

                    meta["curriculum_level"] = 3
                    meta["has_onnx"] = True
                    meta["has_model"] = True
                    with open(meta_path, "w") as f:
                        json.dump(meta, f, indent=2)

                except Exception as e:
                    print(f"[Trainer:{bot_name}] Error loading base model: {e}", flush=True)
                    traceback.print_exc()
            else:
                print(f"[Trainer:{bot_name}] Base model not found at {base_model_path}", flush=True)

        # --- 8. Training state ---
        if "curriculum_level" not in meta:
            meta["curriculum_level"] = 1
        if "level_win_rate" not in meta:
            meta["level_win_rate"] = 0.0
        if "level_matches" not in meta:
            meta["level_matches"] = 0

        match_results = []
        matches_played = meta.get("matches_played", 0)
        last_save_time = time.time()
        last_meta_time = time.time()
        last_live_time = time.time()

        # --- 9. Define training callback ---
        def training_callback(ppo_instance, metrics, vec_env):
            nonlocal match_results, matches_played, last_save_time, last_meta_time, last_live_time

            if _stop_flags.get(bot_name):
                return False

            # Count new episodes and track wins
            ep_rewards = metrics.get("ep_rewards", [])
            ep_wins = metrics.get("ep_wins", [])

            # Track recent match results for curriculum
            # Dynamic window: must play at least as many matches as opponents in pool
            current_level = meta.get("curriculum_level", 1)
            pool_size = 5 + max(0, current_level - 1)  # rule bots + checkpoints
            required_matches = max(100, pool_size)

            for w in ep_wins[matches_played:]:
                matches_played += 1
                match_results.append(bool(w))
                if len(match_results) > required_matches:
                    match_results.pop(0)

            win_rate = (sum(match_results) / len(match_results)) if match_results else 0.0
            meta["level_win_rate"] = round(win_rate * 100, 1)
            meta["level_matches"] = len(match_results)
            meta["matches_played"] = matches_played
            meta["win_rate"] = round(win_rate, 3)
            meta["avg_reward"] = round(metrics.get("avg_reward", 0.0), 2)

            # --- Pool Expansion: 75% over dynamic window ---
            threshold = 0.75
            if len(match_results) >= required_matches and win_rate >= threshold:
                # Export checkpoint to pool
                try:
                    lvl_onnx = os.path.join(bot_dir, f"model_lvl_{current_level}.onnx")
                    _export_actor_onnx(ppo_instance.actor, obs_dim, lvl_onnx, device)
                    print(f"\n[Trainer:{bot_name}] Saved pool snapshot: model_lvl_{current_level}.onnx", flush=True)
                except Exception as e:
                    print(f"\n[Trainer:{bot_name}] Snapshot save failed: {e}", flush=True)

                meta["curriculum_level"] = current_level + 1
                match_results.clear()
                meta["level_win_rate"] = 0.0
                meta["level_matches"] = 0
                print(f"\n[Trainer:{bot_name}] POOL EXPANDED! Gen {current_level + 1} | Eval window was {required_matches} matches | Pool: {pool_size + 1} opponents\n", flush=True)

            now = time.time()

            # --- Periodic checkpoint save (every 30s) ---
            if now - last_save_time > 30.0:
                last_save_time = now
                ppo_instance.save(model_pt_path)

                try:
                    _export_actor_onnx(ppo_instance.actor, obs_dim, os.path.join(bot_dir, "model.onnx"), device)
                    meta["has_onnx"] = True
                except Exception as e:
                    print(f"[Trainer:{bot_name}] ONNX export failed: {e}", flush=True)

                meta["current_step"] = ppo_instance.num_timesteps
                meta["has_model"] = True
                meta["status"] = "training"

                pct = min(100.0, (ppo_instance.num_timesteps / max(1, total_steps)) * 100)
                fps = metrics.get("fps", 0)
                print(f"[Trainer:{bot_name}] Step {ppo_instance.num_timesteps:,}/{total_steps:,} "
                      f"({pct:.1f}%) | {fps:,.0f} fps | reward={metrics.get('avg_reward', 0):.2f} | "
                      f"win_rate={win_rate:.1%} | pl={metrics.get('policy_loss', 0):.4f} | "
                      f"vl={metrics.get('value_loss', 0):.4f} | ent={metrics.get('entropy', 0):.4f}", flush=True)

            # --- Metadata update (every 2s) ---
            if now - last_meta_time > 2.0:
                last_meta_time = now
                try:
                    with open(meta_path, "w") as f:
                        json.dump(meta, f, indent=2)
                except Exception:
                    pass

            return True  # Continue training

        # --- 10. Define step callback (for live visualization) ---
        def step_callback(vec_env):
            nonlocal last_live_time
            now = time.time()
            if live_queue is not None and now - last_live_time > 0.033:
                last_live_time = now
                try:
                    states = vec_env.get_render_states()
                    if states and states[0]:
                        try:
                            live_queue.put_nowait(states[0])
                        except queue.Full:
                            pass
                except Exception as e:
                    print(f"[Trainer:{bot_name}] Live queue error: {e}", flush=True)

        # --- 11. Run training ---
        remaining_steps = max(0, total_steps - current_step)
        progress_path = os.path.join(bot_dir, "progress.json")
        print(f"[Trainer:{bot_name}] Training started ({remaining_steps:,} steps remaining)...", flush=True)

        if remaining_steps > 0:
            ppo.learn(
                total_timesteps=remaining_steps,
                env=env,
                callbacks=[training_callback],
                step_callbacks=[step_callback],
                progress_path=progress_path,
                initial_timesteps=current_step,
            )

        # --- 12. Final checkpoint ---
        ppo.save(model_pt_path)
        try:
            _export_actor_onnx(ppo.actor, obs_dim, os.path.join(bot_dir, "model.onnx"), device)
            meta["has_onnx"] = True
        except Exception as e:
            print(f"[Trainer:{bot_name}] Final ONNX export failed: {e}", flush=True)

        meta["current_step"] = ppo.num_timesteps
        meta["status"] = "completed"
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

        env.close()
        print(f"[Trainer:{bot_name}] Training complete!", flush=True)

    except Exception as e:
        print(f"[Trainer] [ERROR] Training failed: {e}", flush=True)
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


def _export_actor_onnx(actor, obs_dim: int, onnx_path: str, device: str = "cpu"):
    """Export Actor to ONNX for production inference."""
    import torch
    actor.eval()
    dummy = torch.randn(1, obs_dim, dtype=torch.float32, device=device)
    os.makedirs(os.path.dirname(onnx_path) if os.path.dirname(onnx_path) else ".", exist_ok=True)
    try:
        torch.onnx.export(
            actor,
            dummy,
            onnx_path,
            export_params=True,
            opset_version=18,
            do_constant_folding=True,
            input_names=["observation"],
            output_names=["action"],
            dynamic_axes={"observation": {0: "batch_size"}, "action": {0: "batch_size"}},
        )
    finally:
        # Crucial fix for PyTorch 2.x: ONNX export internally uses Dynamo and leaves it
        # in an inconsistent state, causing KeyError: 'custom' in fx_traceback.annotate
        # during subsequent training steps (like optimizer.zero_grad).
        if hasattr(torch, "_dynamo"):
            torch._dynamo.reset()


def start_training(bot_name: str, bot_dir: str) -> bool:
    """Spawn a background thread for training."""
    if bot_name in _active_processes and _active_processes[bot_name].is_alive():
        return False

    # Create a live queue for this bot
    q = queue.Queue(maxsize=20)
    _live_queues[bot_name] = q
    _stop_flags[bot_name] = False

    t = threading.Thread(
        target=_run_training_job,
        args=(bot_dir, q),
        name=f"trainer-{bot_name}",
        daemon=False,
    )
    t.start()
    _active_processes[bot_name] = t
    print(f"[TrainerManager] Started training thread for '{bot_name}'", flush=True)
    return True


def stop_training(bot_name: str) -> bool:
    """Terminate a running training thread."""
    if bot_name not in _active_processes:
        return False

    t = _active_processes[bot_name]
    if t.is_alive():
        print(f"[TrainerManager] Terminating training for '{bot_name}'...", flush=True)
        _stop_flags[bot_name] = True
        t.join(timeout=5)

    del _active_processes[bot_name]
    if bot_name in _live_queues:
        del _live_queues[bot_name]
    if bot_name in _stop_flags:
        del _stop_flags[bot_name]

    # Update metadata status
    from backend.model_registry import _meta_path
    meta_path = _meta_path(bot_name)
    try:
        with open(meta_path, "r") as f:
            meta = json.load(f)
        if meta["status"] == "training":
            meta["status"] = "created"
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)
    except Exception:
        pass

    return True


def is_training(bot_name: str) -> bool:
    """Check if a training thread is currently alive for this bot."""
    if bot_name not in _active_processes:
        return False
    if not _active_processes[bot_name].is_alive():
        del _active_processes[bot_name]
        if bot_name in _live_queues:
            del _live_queues[bot_name]
        if bot_name in _stop_flags:
            del _stop_flags[bot_name]
        return False
    return True


def get_all_active() -> list[str]:
    """Return list of bot names that are currently training."""
    dead = [name for name, p in _active_processes.items() if not p.is_alive()]
    for name in dead:
        del _active_processes[name]
        if name in _live_queues:
            del _live_queues[name]
        if name in _stop_flags:
            del _stop_flags[name]
    return list(_active_processes.keys())


def get_live_queue(bot_name: str) -> Any:
    """Get the live game state queue for a training bot."""
    return _live_queues.get(bot_name)
