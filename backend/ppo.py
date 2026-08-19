"""
ppo.py — Clean PPO Implementation in Native PyTorch

Takes separate Actor and Critic models as constructor args.
All hyperparameters are fully customizable.

Features:
  - GAE-λ advantage estimation
  - Clipped surrogate objective
  - Value function clipping
  - Entropy bonus
  - KL divergence early stopping (optional)
  - Gradient norm clipping
  - Live visualization hook via callback
  - JSON Lines progress logging
"""

import time
import json
import numpy as np
import torch
import torch.nn as nn
from dataclasses import dataclass, field


@dataclass
class RolloutBuffer:
    """Stores rollout data for PPO updates."""
    obs: list = field(default_factory=list)
    actions: list = field(default_factory=list)
    rewards: list = field(default_factory=list)
    dones: list = field(default_factory=list)
    log_probs: list = field(default_factory=list)
    values: list = field(default_factory=list)

    def clear(self):
        self.obs.clear()
        self.actions.clear()
        self.rewards.clear()
        self.dones.clear()
        self.log_probs.clear()
        self.values.clear()

    def add(self, obs, action, reward, done, log_prob, value):
        self.obs.append(obs)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
        self.log_probs.append(log_prob)
        self.values.append(value)

    def compute_gae(self, last_values: np.ndarray, gamma: float, gae_lambda: float):
        """
        Compute Generalized Advantage Estimation.
        
        Returns:
            advantages: np.ndarray (T, n_envs)
            returns: np.ndarray (T, n_envs)
        """
        T = len(self.rewards)
        n_envs = self.rewards[0].shape[0]

        advantages = np.zeros((T, n_envs), dtype=np.float32)
        last_gae = np.zeros(n_envs, dtype=np.float32)

        for t in reversed(range(T)):
            if t == T - 1:
                next_values = last_values
                next_non_terminal = 1.0 - self.dones[t].astype(np.float32)
            else:
                next_values = self.values[t + 1]
                next_non_terminal = 1.0 - self.dones[t].astype(np.float32)

            delta = self.rewards[t] + gamma * next_values * next_non_terminal - self.values[t]
            last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae
            advantages[t] = last_gae

        returns = advantages + np.array(self.values, dtype=np.float32)
        return advantages, returns

    def flatten(self):
        """Flatten (T, n_envs, ...) → (T*n_envs, ...)."""
        return {
            "obs": np.concatenate(self.obs, axis=0),
            "actions": np.concatenate(self.actions, axis=0),
            "log_probs": np.concatenate(self.log_probs, axis=0),
            "values": np.concatenate(self.values, axis=0),
        }


class PPO:
    """
    Proximal Policy Optimization with clipped objective.
    
    Takes separate Actor and Critic models. All hyperparameters customizable.
    
    Args:
        actor: Policy network (Actor)
        critic: Value network (Critic)
        lr: Learning rate for both networks
        gamma: Discount factor
        gae_lambda: GAE lambda
        clip_range: PPO clipping parameter
        ent_coef: Entropy coefficient
        vf_coef: Value function loss coefficient
        max_grad_norm: Max gradient norm for clipping
        n_steps: Rollout length per environment
        batch_size: Minibatch size for updates
        n_epochs: Number of PPO epochs per rollout
        target_kl: Optional KL divergence threshold for early stopping
        device: "cpu" or "cuda"
    """

    def __init__(
        self,
        actor,
        critic,
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: float = 0.2,
        ent_coef: float = 0.0,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        n_steps: int = 4096,
        batch_size: int = 512,
        n_epochs: int = 10,
        target_kl: float | None = None,
        device: str = "cpu",
    ):
        self.actor = actor.to(device)
        self.critic = critic.to(device)
        self.device = device

        # Hyperparameters
        self.lr = lr
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_range = clip_range
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.n_steps = n_steps
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.target_kl = target_kl

        # Single optimizer for both networks
        self.optimizer = torch.optim.Adam(
            list(self.actor.parameters()) + list(self.critic.parameters()),
            lr=lr, eps=1e-5
        )

        self.buffer = RolloutBuffer()
        self.num_timesteps = 0

        # Episode tracking
        self._ep_rewards = []
        self._ep_lengths = []
        self._current_ep_rewards = None  # Set during learn()

    def collect_rollouts(self, env, obs: np.ndarray, step_callbacks: list | None = None) -> np.ndarray:
        """
        Collect n_steps of experience from the vectorized environment.
        
        Args:
            env: VecEnv instance
            obs: Current observation (n_envs, obs_dim)
            
        Returns:
            Last observation after collection
        """
        self.buffer.clear()
        self.actor.eval()
        self.critic.eval()

        n_envs = env.n_envs

        # Initialize per-env reward accumulators if needed
        if self._current_ep_rewards is None:
            self._current_ep_rewards = np.zeros(n_envs, dtype=np.float32)

        with torch.no_grad():
            for step in range(self.n_steps):
                obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device)

                actions, log_probs = self.actor.get_action(obs_t)
                values = self.critic(obs_t)

                actions_np = actions.cpu().numpy()
                log_probs_np = log_probs.cpu().numpy()
                values_np = values.cpu().numpy()

                new_obs, rewards, dones, infos = env.step(actions_np)

                self.num_timesteps += n_envs

                # Track episode stats
                self._current_ep_rewards += rewards
                for i in range(n_envs):
                    if dones[i]:
                        self._ep_rewards.append(float(self._current_ep_rewards[i]))
                        self._current_ep_rewards[i] = 0.0
                        # Track wins
                        if infos[i].get("is_success", False):
                            self._ep_lengths.append(1)  # 1 = win
                        else:
                            self._ep_lengths.append(0)  # 0 = loss

                self.buffer.add(obs, actions_np, rewards, dones, log_probs_np, values_np)
                obs = new_obs

                if step_callbacks:
                    for scb in step_callbacks:
                        scb(env)

        # Compute last values for GAE
        with torch.no_grad():
            last_obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device)
            last_values = self.critic(last_obs_t).cpu().numpy()

        return obs, last_values

    def update(self, last_values: np.ndarray) -> dict:
        """
        Run PPO update using collected rollout buffer.
        
        Returns:
            Dictionary of training metrics
        """
        # Compute GAE
        advantages, returns = self.buffer.compute_gae(last_values, self.gamma, self.gae_lambda)

        # Flatten everything
        flat = self.buffer.flatten()
        T, n_envs = advantages.shape
        advantages_flat = advantages.reshape(-1)
        returns_flat = returns.reshape(-1)

        # Normalize advantages
        adv_mean = advantages_flat.mean()
        adv_std = advantages_flat.std() + 1e-8
        advantages_flat = (advantages_flat - adv_mean) / adv_std

        # Convert to tensors
        all_obs = torch.tensor(flat["obs"], dtype=torch.float32, device=self.device)
        all_actions = torch.tensor(flat["actions"], dtype=torch.float32, device=self.device)
        all_old_log_probs = torch.tensor(flat["log_probs"], dtype=torch.float32, device=self.device)
        all_old_values = torch.tensor(flat["values"], dtype=torch.float32, device=self.device)
        all_advantages = torch.tensor(advantages_flat, dtype=torch.float32, device=self.device)
        all_returns = torch.tensor(returns_flat, dtype=torch.float32, device=self.device)

        total_samples = len(all_obs)

        # Metrics accumulators
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        total_kl = 0.0
        total_clip_fraction = 0.0
        n_updates = 0
        early_stopped = False

        self.actor.train()
        self.critic.train()

        for epoch in range(self.n_epochs):
            # Random permutation for minibatch sampling
            indices = torch.randperm(total_samples, device=self.device)

            for start in range(0, total_samples, self.batch_size):
                end = min(start + self.batch_size, total_samples)
                mb_idx = indices[start:end]

                mb_obs = all_obs[mb_idx]
                mb_actions = all_actions[mb_idx]
                mb_old_log_probs = all_old_log_probs[mb_idx]
                mb_advantages = all_advantages[mb_idx]
                mb_returns = all_returns[mb_idx]
                mb_old_values = all_old_values[mb_idx]

                # Evaluate current policy on these obs/actions
                new_log_probs, entropy = self.actor.evaluate(mb_obs, mb_actions)
                new_values = self.critic(mb_obs)

                # Policy loss (clipped surrogate)
                ratio = torch.exp(new_log_probs - mb_old_log_probs)
                surr1 = ratio * mb_advantages
                surr2 = torch.clamp(ratio, 1.0 - self.clip_range, 1.0 + self.clip_range) * mb_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss (clipped)
                value_pred_clipped = mb_old_values + torch.clamp(
                    new_values - mb_old_values, -self.clip_range, self.clip_range
                )
                value_loss_unclipped = (new_values - mb_returns) ** 2
                value_loss_clipped = (value_pred_clipped - mb_returns) ** 2
                value_loss = 0.5 * torch.max(value_loss_unclipped, value_loss_clipped).mean()

                # Entropy loss
                entropy_loss = -entropy.mean()

                # Total loss
                loss = policy_loss + self.vf_coef * value_loss + self.ent_coef * entropy_loss

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    list(self.actor.parameters()) + list(self.critic.parameters()),
                    self.max_grad_norm
                )
                self.optimizer.step()

                # Track metrics
                with torch.no_grad():
                    approx_kl = (mb_old_log_probs - new_log_probs).mean().item()
                    clip_fraction = ((ratio - 1.0).abs() > self.clip_range).float().mean().item()

                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.mean().item()
                total_kl += approx_kl
                total_clip_fraction += clip_fraction
                n_updates += 1

            # KL early stopping
            if self.target_kl is not None and n_updates > 0:
                avg_kl = total_kl / n_updates
                if avg_kl > 1.5 * self.target_kl:
                    early_stopped = True
                    break

        if n_updates == 0:
            n_updates = 1  # Prevent division by zero

        # Explained variance
        with torch.no_grad():
            y_pred = all_old_values.cpu().numpy()
            y_true = all_returns.cpu().numpy()
            var_y = np.var(y_true)
            explained_var = 1 - np.var(y_true - y_pred) / (var_y + 1e-8) if var_y > 1e-8 else 0.0

        metrics = {
            "policy_loss": total_policy_loss / n_updates,
            "value_loss": total_value_loss / n_updates,
            "entropy": total_entropy / n_updates,
            "kl_divergence": total_kl / n_updates,
            "clip_fraction": total_clip_fraction / n_updates,
            "explained_variance": float(explained_var),
            "early_stopped": early_stopped,
        }

        return metrics

    def learn(
        self,
        total_timesteps: int,
        env,
        callbacks: list | None = None,
        step_callbacks: list | None = None,
        progress_path: str | None = None,
        initial_timesteps: int = 0,
    ):
        """
        Main training loop.
        
        Args:
            total_timesteps: Total environment steps to train for
            env: VecEnv instance
            callbacks: List of callback functions called each rollout with signature:
                       callback(ppo, metrics, env) -> bool (return False to stop)
            progress_path: Path to write JSON Lines progress log
            initial_timesteps: Starting timestep count (for resume)
        """
        self.num_timesteps = initial_timesteps
        obs = env.reset()
        self._current_ep_rewards = np.zeros(env.n_envs, dtype=np.float32)

        progress_file = None
        if progress_path:
            progress_file = open(progress_path, "at")

        start_time = time.time()
        last_log_time = time.time()
        iteration = 0

        try:
            while self.num_timesteps < initial_timesteps + total_timesteps:
                iteration += 1
                rollout_start = time.time()

                # 1. Collect rollouts
                obs, last_values = self.collect_rollouts(env, obs, step_callbacks=step_callbacks)

                # 2. PPO update
                metrics = self.update(last_values)

                # 3. Compute episode stats
                if self._ep_rewards:
                    recent_rewards = self._ep_rewards[-100:]
                    recent_wins = self._ep_lengths[-100:]
                    avg_reward = np.mean(recent_rewards)
                    win_rate = np.mean(recent_wins) if recent_wins else 0.0
                else:
                    avg_reward = 0.0
                    win_rate = 0.0

                elapsed = time.time() - start_time
                fps = self.num_timesteps / max(1, elapsed)

                # 4. Log progress
                now = time.time()
                if now - last_log_time > 2.0 or iteration <= 2:
                    last_log_time = now

                    log_entry = {
                        "time/total_timesteps": self.num_timesteps,
                        "time/iterations": iteration,
                        "time/fps": int(fps),
                        "time/time_elapsed": round(elapsed, 1),
                        "rollout/avg_reward": round(avg_reward, 3),
                        "rollout/win_rate": round(win_rate, 3),
                        "rollout/episodes": len(self._ep_rewards),
                        "train/policy_loss": round(metrics["policy_loss"], 5),
                        "train/value_loss": round(metrics["value_loss"], 5),
                        "train/entropy": round(metrics["entropy"], 5),
                        "train/kl_divergence": round(metrics["kl_divergence"], 6),
                        "train/clip_fraction": round(metrics["clip_fraction"], 4),
                        "train/explained_variance": round(metrics["explained_variance"], 4),
                    }

                    if progress_file:
                        progress_file.write(json.dumps(log_entry) + "\n")
                        progress_file.flush()

                # 5. Callbacks
                if callbacks:
                    callback_metrics = {
                        **metrics,
                        "avg_reward": avg_reward,
                        "win_rate": win_rate,
                        "num_timesteps": self.num_timesteps,
                        "iteration": iteration,
                        "fps": fps,
                        "ep_rewards": self._ep_rewards,
                        "ep_wins": self._ep_lengths,
                    }
                    for cb in callbacks:
                        if not cb(self, callback_metrics, env):
                            return  # Callback requested stop

        finally:
            if progress_file:
                progress_file.close()

    def save(self, path: str):
        """Save actor + critic state dicts and hyperparameters."""
        torch.save({
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "num_timesteps": self.num_timesteps,
            "hyperparams": {
                "lr": self.lr,
                "gamma": self.gamma,
                "gae_lambda": self.gae_lambda,
                "clip_range": self.clip_range,
                "ent_coef": self.ent_coef,
                "vf_coef": self.vf_coef,
                "max_grad_norm": self.max_grad_norm,
                "n_steps": self.n_steps,
                "batch_size": self.batch_size,
                "n_epochs": self.n_epochs,
            }
        }, path)

    def load(self, path: str):
        """Load actor + critic state dicts, handling input dimension expansion."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)

        # --- Handle obs_dim expansion (e.g., 90 → 91) ---
        first_key = "backbone.0.weight"
        for model_name, model, ckpt_key in [
            ("actor", self.actor, "actor"),
            ("critic", self.critic, "critic"),
        ]:
            if ckpt_key in checkpoint:
                saved_state = checkpoint[ckpt_key]
                if first_key in saved_state:
                    saved_w = saved_state[first_key]
                    current_w = model.state_dict()[first_key]
                    if saved_w.shape[1] < current_w.shape[1]:
                        expanded = torch.zeros_like(current_w)
                        expanded[:, :saved_w.shape[1]] = saved_w
                        saved_state[first_key] = expanded
                        print(f"[PPO] Expanded {model_name} {first_key}: {saved_w.shape[1]} → {current_w.shape[1]}", flush=True)
                model.load_state_dict(saved_state, strict=False)

        if "optimizer" in checkpoint:
            try:
                self.optimizer.load_state_dict(checkpoint["optimizer"])
            except Exception:
                pass  # Optimizer state may not match if architecture changed
        if "num_timesteps" in checkpoint:
            self.num_timesteps = checkpoint["num_timesteps"]
