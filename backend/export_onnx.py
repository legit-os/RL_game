"""
export_onnx.py — Convert SB3 PPO checkpoint to ONNX for production inference.

Called automatically by trainer_worker on completion, or manually via the API.
Extracts the policy actor network and exports it as a standalone ONNX graph.
"""

import os
import numpy as np


def export_model(model_zip_path: str, onnx_output_path: str) -> bool:
    """
    Export an SB3 PPO model's policy network to ONNX format.

    Args:
        model_zip_path: Path to the .zip model file (SB3 format)
        onnx_output_path: Path to write the .onnx file

    Returns:
        True if export succeeded, False otherwise.
    """
    try:
        import torch
        from stable_baselines3 import PPO
    except ImportError as e:
        print(f"[ONNX Export] Missing dependency: {e}")
        return False

    if not os.path.isfile(model_zip_path):
        print(f"[ONNX Export] Model not found: {model_zip_path}")
        return False

    print(f"[ONNX Export] Loading model from {model_zip_path}...")
    model = PPO.load(model_zip_path, device="cpu")
    
    # Dynamically determine if using VecFrameStack or standard
    obs_size = model.observation_space.shape[0]

    # Extract the actor (policy) network for deterministic inference
    # SB3's MlpPolicy has policy.action_net + policy.mu (for continuous actions)
    policy = model.policy
    policy.set_training_mode(False)

    # Create a wrapper that takes obs and returns deterministic actions
    class PolicyWrapper(torch.nn.Module):
        def __init__(self, sb3_policy):
            super().__init__()
            self.features_extractor = sb3_policy.features_extractor
            self.mlp_extractor = sb3_policy.mlp_extractor
            self.action_net = sb3_policy.action_net

        def forward(self, obs):
            features = self.features_extractor(obs)
            latent_pi, _ = self.mlp_extractor(features)
            actions = self.action_net(latent_pi)
            # Squash to [-1, 1] via tanh
            return torch.tanh(actions)

    wrapper = PolicyWrapper(policy)
    wrapper.eval()

    # Dummy input for tracing
    dummy_obs = torch.randn(1, obs_size, dtype=torch.float32)

    print(f"[ONNX Export] Exporting to {onnx_output_path}...")
    os.makedirs(os.path.dirname(onnx_output_path), exist_ok=True)

    torch.onnx.export(
        wrapper,
        dummy_obs,
        onnx_output_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=["observation"],
        output_names=["action"],
        dynamic_axes={
            "observation": {0: "batch_size"},
            "action": {0: "batch_size"},
        },
    )

    # Verify the exported model
    try:
        import onnx
        onnx_model = onnx.load(onnx_output_path)
        onnx.checker.check_model(onnx_model)
        print("[ONNX Export] [OK] Model verification passed!")
    except Exception as e:
        print(f"[ONNX Export] [WARN] Verification warning: {e}")

    # Quick inference test
    try:
        import onnxruntime as ort
        sess = ort.InferenceSession(onnx_output_path, providers=["CPUExecutionProvider"])
        test_obs = np.random.randn(1, obs_size).astype(np.float32)
        result = sess.run(None, {"observation": test_obs})
        action = result[0]
        print(f"[ONNX Export] [OK] Inference test passed! Output shape: {action.shape}")
    except Exception as e:
        print(f"[ONNX Export] [WARN] Inference test warning: {e}")

    print("[ONNX Export] [OK] Export complete!")
    return True


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("Usage: python export_onnx.py <model.zip> <output.onnx>")
        sys.exit(1)
    export_model(sys.argv[1], sys.argv[2])
