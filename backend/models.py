"""
models.py — Actor & Critic Neural Networks for PPO

Fully configurable architecture. Separate models for policy and value function.
Can be independently customized, swapped, or pretrained.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal


def _build_mlp(input_dim: int, hidden_layers: list[int], activation: str) -> nn.Sequential:
    """Build an MLP backbone from config."""
    act_map = {
        "relu": nn.ReLU,
        "tanh": nn.Tanh,
        "elu": nn.ELU,
        "silu": nn.SiLU,
        "swish": nn.SiLU,
    }
    act_fn = act_map.get(activation, nn.SiLU)

    layers = []
    last_dim = input_dim
    for h_dim in hidden_layers:
        layers.extend([nn.Linear(last_dim, h_dim), act_fn()])
        last_dim = h_dim
    return nn.Sequential(*layers), last_dim


class Actor(nn.Module):
    """
    Policy network: obs → action distribution (Gaussian).
    
    Outputs action mean via tanh squashing.
    log_std is a learnable parameter (state-independent).
    """

    def __init__(self, obs_dim: int, act_dim: int, hidden_layers: list[int],
                 activation: str = "silu", init_log_std: float = -0.5):
        super().__init__()
        self.backbone, last_dim = _build_mlp(obs_dim, hidden_layers, activation)
        self.mean_head = nn.Linear(last_dim, act_dim)
        self.log_std = nn.Parameter(torch.ones(act_dim) * init_log_std)

        # Orthogonal init for stable training
        self._init_weights()

    def _init_weights(self):
        for module in self.backbone:
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                nn.init.zeros_(module.bias)
        nn.init.orthogonal_(self.mean_head.weight, gain=0.01)
        nn.init.zeros_(self.mean_head.bias)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Deterministic forward: obs → tanh(mean). Used for ONNX export and inference."""
        h = self.backbone(obs)
        return torch.tanh(self.mean_head(h))

    def get_distribution(self, obs: torch.Tensor) -> Normal:
        """Get the action distribution for given observations."""
        h = self.backbone(obs)
        mean = self.mean_head(h)
        std = torch.exp(self.log_std.clamp(-5, 2))
        return Normal(mean, std)

    def get_action(self, obs: torch.Tensor, deterministic: bool = False):
        """
        Sample an action from the policy.
        
        Returns: (action, log_prob)
            action: tanh-squashed action in [-1, 1]
            log_prob: log probability of the pre-squash action
        """
        dist = self.get_distribution(obs)
        if deterministic:
            raw_action = dist.mean
        else:
            raw_action = dist.rsample()

        action = torch.tanh(raw_action)
        # Log prob with tanh correction
        log_prob = dist.log_prob(raw_action).sum(dim=-1)
        log_prob -= (2 * (np.log(2) - raw_action - torch.nn.functional.softplus(-2 * raw_action))).sum(dim=-1)
        return action, log_prob

    def evaluate(self, obs: torch.Tensor, action: torch.Tensor):
        """
        Evaluate given actions under the current policy.
        
        Returns: (log_prob, entropy)
        """
        dist = self.get_distribution(obs)
        # Inverse tanh to recover raw action
        raw_action = torch.atanh(action.clamp(-0.999, 0.999))
        log_prob = dist.log_prob(raw_action).sum(dim=-1)
        log_prob -= (2 * (np.log(2) - raw_action - torch.nn.functional.softplus(-2 * raw_action))).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        return log_prob, entropy


class Critic(nn.Module):
    """
    Value network: obs → scalar value estimate.
    """

    def __init__(self, obs_dim: int, hidden_layers: list[int], activation: str = "silu"):
        super().__init__()
        self.backbone, last_dim = _build_mlp(obs_dim, hidden_layers, activation)
        self.value_head = nn.Linear(last_dim, 1)

        self._init_weights()

    def _init_weights(self):
        for module in self.backbone:
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                nn.init.zeros_(module.bias)
        nn.init.orthogonal_(self.value_head.weight, gain=1.0)
        nn.init.zeros_(self.value_head.bias)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """obs → value (scalar per batch element)."""
        h = self.backbone(obs)
        return self.value_head(h).squeeze(-1)
