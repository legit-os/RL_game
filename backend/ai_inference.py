"""
ai_inference.py — ONNX Runtime wrapper for the trained RL agent.

Loads the exported .onnx model and provides the same predict() interface
as rule_bot.py, making them interchangeable in game_loop.py.
"""

import os
import numpy as np
from collections import deque

# Optional import — onnxruntime may not be installed during dev
try:
    import onnxruntime as ort
    HAS_ORT = True
except ImportError:
    HAS_ORT = False


class ONNXBot:
    """
    Runs the trained RL policy via ONNX Runtime for ultra-fast inference.

    Usage:
        bot = ONNXBot("models/ppo_sniper_v1.onnx")
        action = bot.predict(obs_30_floats)
    """

    def __init__(self, model_path: str):
        if not HAS_ORT:
            raise RuntimeError(
                "onnxruntime is not installed. Install with: pip install onnxruntime"
            )
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"ONNX model not found: {model_path}")

        # Create inference session with optimizations
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.intra_op_num_threads = 1  # Single thread for minimal latency

        self.session = ort.InferenceSession(
            model_path,
            sess_options,
            providers=["CPUExecutionProvider"],
        )

        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        
        # Check if the model expects stacked frames (90 features) or single (30)
        self.input_shape = self.session.get_inputs()[0].shape
        self.expected_features = self.input_shape[1] if len(self.input_shape) > 1 else 30
        
        # Rolling buffer for Frame Stacking
        self.obs_stack = deque(maxlen=3)

    def predict(self, obs: np.ndarray) -> np.ndarray:
        """
        Run inference on a single observation.

        Args:
            obs: 30-float observation array (shape: (30,) or (1, 30))

        Returns:
            5-float action array [move_x, move_y, aim_x, aim_y, shoot_trigger]
        """
        obs_flat = obs.flatten()
        
        # Initialize stack if empty
        if not self.obs_stack:
            for _ in range(3):
                self.obs_stack.append(obs_flat)
        else:
            self.obs_stack.append(obs_flat)
            
        if self.expected_features == 90:
            stacked_obs = np.concatenate(self.obs_stack).astype(np.float32)
        else:
            # Fallback for old 30-float models
            stacked_obs = obs_flat.astype(np.float32)

        # Ensure correct shape: (1, N) batch dimension
        stacked_obs = stacked_obs.reshape(1, -1)

        # Run inference (< 1ms typically)
        result = self.session.run([self.output_name], {self.input_name: stacked_obs})
        action = result[0].flatten()

        # Clip action to [-1, 1] (safety)
        action = np.clip(action, -1.0, 1.0).astype(np.float32)

        return action

    def reset(self):
        """Reset internal frame stack on new episode."""
        self.obs_stack.clear()


class FallbackBot:
    """
    A dummy bot used when no ONNX model is available.
    Just stands still. Used during development before training.
    """

    def predict(self, obs: np.ndarray) -> np.ndarray:
        return np.zeros(5, dtype=np.float32)

    def reset(self):
        pass
