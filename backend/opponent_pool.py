import numpy as np
import os
import onnxruntime as ort
from backend.curriculum_bots import Level1Bot, Level2Bot, Level3Bot, Level4Bot, Level5Bot

import json

class OpponentPool:
    """
    PFSP (Prioritized Fictitious Self-Play) opponent pool.
    
    Sampling distribution:
      - 25% Latest generation (hardest single opponent)
      - 50% Recent meta (last 20 generations)
      - 25% Deep archive (rule bots + older generations)
    """
    def __init__(self, bot_dir: str):
        self.bot_dir = bot_dir
        self.meta_path = os.path.join(bot_dir, "metadata.json") if bot_dir else ""
        
        self.rule_bots = [Level1Bot(), Level2Bot(), Level3Bot(), Level4Bot(), Level5Bot()]
        self.onnx_models = []
        self._reload_pool()

    def _get_curriculum_level(self) -> int:
        if not self.meta_path or not os.path.exists(self.meta_path):
            return 1
        try:
            with open(self.meta_path, "r") as f:
                meta = json.load(f)
                return meta.get("curriculum_level", 1)
        except Exception:
            return 1

    def _reload_pool(self):
        """Scans the bot directory and loads all self-play checkpoints, sorted numerically."""
        self.onnx_models = []
        
        if self.bot_dir and os.path.exists(self.bot_dir):
            # Sort numerically by level number (not lexicographically)
            level_files = [
                f for f in os.listdir(self.bot_dir)
                if f.startswith("model_lvl_") and f.endswith(".onnx")
            ]
            level_files.sort(key=lambda f: int(f.replace("model_lvl_", "").replace(".onnx", "")))
            
            for f in level_files:
                path = os.path.join(self.bot_dir, f)
                session = ort.InferenceSession(path, providers=['CPUExecutionProvider'])
                self.onnx_models.append(session)
                
        level = self._get_curriculum_level()
        n_rule = len(self.rule_bots)
        n_onnx = len(self.onnx_models)
        total = n_rule + n_onnx
        print(f"[Opponent Pool] Level {level}: {total} Slots ({n_rule} Rule Bots, {n_onnx} Checkpoints). PFSP sampling active.", flush=True)

    def sample_opponent(self):
        """
        PFSP: Prioritized Fictitious Self-Play sampling.
        
        - 25% chance: Latest generation (the single hardest opponent)
        - 50% chance: Recent meta (last 20 generations)
        - 25% chance: Deep archive (rule bots + older generations)
        
        Falls back to uniform over rule bots if no ONNX models exist.
        """
        n_onnx = len(self.onnx_models)
        
        if n_onnx == 0:
            # No self-play models yet — uniform over rule bots
            return self.rule_bots[np.random.randint(0, len(self.rule_bots))]
        
        roll = np.random.random()
        
        if roll < 0.25:
            # 25%: Latest generation (absolute ceiling push)
            return self.onnx_models[-1]
        elif roll < 0.75:
            # 50%: Recent meta (last 20 generations)
            recent_start = max(0, n_onnx - 20)
            recent_pool = self.onnx_models[recent_start:]
            return recent_pool[np.random.randint(0, len(recent_pool))]
        else:
            # 25%: Deep archive (rule bots + older generations)
            archive = list(self.rule_bots)
            if n_onnx > 20:
                archive.extend(self.onnx_models[:n_onnx - 20])
            return archive[np.random.randint(0, len(archive))]

    @property
    def pool_size(self) -> int:
        """Total number of opponents in the pool."""
        return len(self.rule_bots) + len(self.onnx_models)

    def predict_opponent(self, opponent, obs: np.ndarray) -> np.ndarray:
        """
        Get action from an opponent (rule bot or ONNX model).
        
        Args:
            opponent: A rule bot instance or an ort.InferenceSession
            obs: Stacked observation (90,) for P2 or single frame (30,)
            
        Returns:
            action: np.ndarray (5,) in the opponent's local frame
        """
        if not isinstance(opponent, ort.InferenceSession):
            # Rule bot — only needs latest 30 features
            if obs.shape[0] >= 90:
                # The latest 30 features are at index 60:90. (Index 90 is time_remaining if length is 91)
                obs_to_pass = obs[60:90]
            else:
                obs_to_pass = obs
            act = opponent.predict(obs_to_pass).copy()
        else:
            # ONNX model inference — adapt obs to expected input dim
            input_name = opponent.get_inputs()[0].name
            output_name = opponent.get_outputs()[0].name
            expected_dim = opponent.get_inputs()[0].shape[1]
            
            # Truncate or pad obs to match the model's expected input size
            if obs.shape[0] > expected_dim:
                obs_adapted = obs[:expected_dim]
            elif obs.shape[0] < expected_dim:
                obs_adapted = np.zeros(expected_dim, dtype=np.float32)
                obs_adapted[:obs.shape[0]] = obs
            else:
                obs_adapted = obs
                
            obs_batch = np.expand_dims(obs_adapted, axis=0).astype(np.float32)
            action_batch = opponent.run([output_name], {input_name: obs_batch})[0]
            act = action_batch[0].copy()

        return act
