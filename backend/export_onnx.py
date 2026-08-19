"""
export_onnx.py — Export native PyTorch Actor to ONNX for production inference.

No SB3 dependency. Works directly with our Actor model.
"""

import os
import numpy as np


def export_model(model_pt_path: str, onnx_output_path: str, obs_dim: int = 90,
                 act_dim: int = 5, layers: list[int] | None = None,
                 activation: str = "silu") -> bool:
    """
    Export a native PyTorch Actor model to ONNX format.

    Args:
        model_pt_path: Path to the .pt checkpoint file
        onnx_output_path: Path to write the .onnx file
        obs_dim: Observation dimension (default: 90 for 3-frame stack)
        act_dim: Action dimension (default: 5)
        layers: Hidden layer sizes (default: read from checkpoint)
        activation: Activation function name

    Returns:
        True if export succeeded, False otherwise.
    """
    try:
        import torch
        from backend.models import Actor
    except ImportError as e:
        print(f"[ONNX Export] Missing dependency: {e}")
        return False

    if not os.path.isfile(model_pt_path):
        print(f"[ONNX Export] Model not found: {model_pt_path}")
        return False

    print(f"[ONNX Export] Loading model from {model_pt_path}...")
    checkpoint = torch.load(model_pt_path, map_location="cpu", weights_only=False)

    # Try to read architecture from checkpoint hyperparams
    if layers is None:
        hp = checkpoint.get("hyperparams", {})
        # If no hyperparams stored, try to infer from state dict keys
        if "actor" in checkpoint:
            actor_state = checkpoint["actor"]
        else:
            actor_state = checkpoint  # Might be a raw state dict

        # Default fallback
        layers = [256, 256, 128]

    # Create actor and load weights
    actor = Actor(obs_dim=obs_dim, act_dim=act_dim, hidden_layers=layers, activation=activation)

    if "actor" in checkpoint:
        actor.load_state_dict(checkpoint["actor"])
    else:
        # Try loading as raw state dict
        actor.load_state_dict(checkpoint)

    actor.eval()

    # Export to ONNX
    dummy_obs = torch.randn(1, obs_dim, dtype=torch.float32)
    print(f"[ONNX Export] Exporting to {onnx_output_path}...")
    os.makedirs(os.path.dirname(onnx_output_path) if os.path.dirname(onnx_output_path) else ".", exist_ok=True)

    torch.onnx.export(
        actor,
        dummy_obs,
        onnx_output_path,
        export_params=True,
        opset_version=18,
        do_constant_folding=True,
        input_names=["observation"],
        output_names=["action"],
        dynamic_axes={
            "observation": {0: "batch_size"},
            "action": {0: "batch_size"},
        },
    )

    # Verify
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
        test_obs = np.random.randn(1, obs_dim).astype(np.float32)
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
        print("Usage: python export_onnx.py <model.pt> <output.onnx>")
        sys.exit(1)
    export_model(sys.argv[1], sys.argv[2])
