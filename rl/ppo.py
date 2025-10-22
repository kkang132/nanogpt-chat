"""
Proximal Policy Optimization (PPO) components for chat RL

This module contains a minimal, self-contained PPO implementation designed to
work with the ChatEnvironment in rl/environment.py. It implements:
- PPOConfig dataclass with key hyperparameters
- RolloutBuffer for on-policy trajectory collection and GAE(λ)
- PPOAgent with policy/value network, action selection, and update step
- Logging statistics needed by acceptance criteria

The implementation is lightweight, avoids external RL libs, and uses PyTorch only.
"""
from dataclasses import dataclass
from typing import Dict, Tuple, Optional
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical


@dataclass
class PPOConfig:
    # rollout / optimization
    steps_per_rollout: int = 1024
    epochs: int = 4
    minibatch_size: int = 256
    gamma: float = 0.99
    lam: float = 0.95
    clip_range: float = 0.2
    vf_coef: float = 0.5
    ent_coef: float = 0.01
    max_grad_norm: float = 1.0
    lr: float = 3e-4
    # kl control
    kl_target: float = 0.02
    kl_stop: float = 0.2
    # misc
    device: str = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    advantage_norm: bool = True


class RolloutBuffer:
    def __init__(self, obs_shape: Tuple[int, ...], steps: int, action_dim: int, device: str):
        self.obs = torch.zeros((steps,) + obs_shape, dtype=torch.long, device=device)
        self.actions = torch.zeros((steps,), dtype=torch.long, device=device)
        self.logprobs = torch.zeros((steps,), dtype=torch.float32, device=device)
        self.values = torch.zeros((steps,), dtype=torch.float32, device=device)
        self.rewards = torch.zeros((steps,), dtype=torch.float32, device=device)
        self.dones = torch.zeros((steps,), dtype=torch.float32, device=device)
        self.advantages = torch.zeros((steps,), dtype=torch.float32, device=device)
        self.returns = torch.zeros((steps,), dtype=torch.float32, device=device)
        self.ptr = 0
        self.max_steps = steps

    def add(self, obs, action, logprob, value, reward, done):
        i = self.ptr
        self.obs[i] = torch.as_tensor(obs, device=self.obs.device)
        self.actions[i] = int(action)
        self.logprobs[i] = float(logprob)
        self.values[i] = float(value)
        self.rewards[i] = float(reward)
        self.dones[i] = float(done)
        self.ptr += 1

    def full(self):
        return self.ptr >= self.max_steps

    def compute_gae(self, last_value: float, gamma: float, lam: float):
        adv = 0.0
        for t in reversed(range(self.max_steps)):
            mask = 1.0 - self.dones[t].item()
            delta = self.rewards[t].item() + gamma * last_value * mask - self.values[t].item()
            adv = delta + gamma * lam * mask * adv
            self.advantages[t] = adv
            self.returns[t] = self.advantages[t] + self.values[t]
            last_value = self.values[t].item()
        return self.advantages, self.returns

    def get_minibatches(self, batch_size: int, advantage_norm: bool = True):
        n = self.max_steps
        idx = np.arange(n)
        np.random.shuffle(idx)
        for start in range(0, n, batch_size):
            mb_idx = idx[start:start+batch_size]
            mb_idx_t = torch.as_tensor(mb_idx, device=self.obs.device, dtype=torch.long)
            adv = self.advantages[mb_idx_t]
            if advantage_norm:
                adv = (adv - adv.mean()) / (adv.std(unbiased=False) + 1e-8)
            yield (
                self.obs[mb_idx_t],
                self.actions[mb_idx_t],
                self.logprobs[mb_idx_t],
                self.values[mb_idx_t],
                self.returns[mb_idx_t],
                adv,
            )


class PolicyValueNet(nn.Module):
    """
    A very small policy/value network that embeds token ids and mean-pools
    over the sequence to produce a state representation. It then outputs
    action logits over the vocabulary and a scalar value.
    """
    def __init__(self, vocab_size: int, embed_dim: int = 128):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.ln = nn.LayerNorm(embed_dim)
        self.policy = nn.Sequential(
            nn.Linear(embed_dim, embed_dim), nn.Tanh(),
            nn.Linear(embed_dim, vocab_size)
        )
        self.value = nn.Sequential(
            nn.Linear(embed_dim, embed_dim), nn.Tanh(),
            nn.Linear(embed_dim, 1)
        )

    def forward(self, obs_ids: torch.LongTensor):
        # obs_ids: [B, T]
        x = self.embed(obs_ids)  # [B, T, E]
        x = x.mean(dim=1)  # [B, E]
        x = self.ln(x)
        logits = self.policy(x)
        value = self.value(x).squeeze(-1)
        return logits, value


class PPOAgent:
    def __init__(self, vocab_size: int, obs_shape: Tuple[int, ...], config: PPOConfig):
        self.config = config
        self.device = torch.device(config.device)
        self.net = PolicyValueNet(vocab_size).to(self.device)
        self.optimizer = torch.optim.AdamW(self.net.parameters(), lr=config.lr)
        self.global_step = 0

    @torch.no_grad()
    def act(self, obs: np.ndarray) -> Tuple[int, float, float, Dict]:
        self.net.eval()
        obs_t = torch.as_tensor(obs, device=self.device, dtype=torch.long).unsqueeze(0)
        logits, value = self.net(obs_t)
        dist = Categorical(logits=logits)
        action = dist.sample()
        logprob = dist.log_prob(action)
        info = {
            'entropy': dist.entropy().mean().item()
        }
        return int(action.item()), float(logprob.item()), float(value.item()), info

    def update(self, buffer: RolloutBuffer) -> Dict[str, float]:
        self.net.train()
        stats = {
            'loss_policy': 0.0,
            'loss_value': 0.0,
            'entropy': 0.0,
            'kl': 0.0,
            'clip_frac': 0.0,
        }
        num_updates = 0
        old_params = {k: v.clone().detach() for k, v in self.net.state_dict().items()}

        for epoch in range(self.config.epochs):
            for obs, actions, old_logprobs, old_values, returns, adv in buffer.get_minibatches(self.config.minibatch_size, self.config.advantage_norm):
                logits, values = self.net(obs)
                dist = Categorical(logits=logits)
                logprobs = dist.log_prob(actions)
                entropy = dist.entropy().mean()

                # Policy loss with clipping
                ratio = (logprobs - old_logprobs).exp()
                unclipped = ratio * adv
                clipped = torch.clamp(ratio, 1.0 - self.config.clip_range, 1.0 + self.config.clip_range) * adv
                policy_loss = -torch.min(unclipped, clipped).mean()

                # Value loss (clipped)
                value_clipped = old_values + (values - old_values).clamp(-self.config.clip_range, self.config.clip_range)
                v_loss_unclipped = (values - returns).pow(2)
                v_loss_clipped = (value_clipped - returns).pow(2)
                value_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()

                loss = policy_loss + self.config.vf_coef * value_loss - self.config.ent_coef * entropy

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), self.config.max_grad_norm)
                self.optimizer.step()

                # KL approx between old and new policy
                with torch.no_grad():
                    kl = (old_logprobs - logprobs).mean().clamp_min(0.0)
                    clip_frac = (torch.gt(torch.abs(ratio - 1.0), self.config.clip_range)).float().mean()

                # Accumulate stats
                stats['loss_policy'] += policy_loss.item()
                stats['loss_value'] += value_loss.item()
                stats['entropy'] += entropy.item()
                stats['kl'] += kl.item()
                stats['clip_frac'] += clip_frac.item()
                num_updates += 1

                # Early stop on too-large KL
                if stats['kl'] / num_updates > self.config.kl_stop:
                    break
            if stats['kl'] / max(1, num_updates) > self.config.kl_stop:
                break

        # Average stats
        for k in list(stats.keys()):
            stats[k] = stats[k] / max(1, num_updates)
        return stats

    def save(self, path: str):
        torch.save({
            'state_dict': self.net.state_dict(),
            'config': self.config.__dict__,
            'global_step': self.global_step,
        }, path)

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.net.load_state_dict(ckpt['state_dict'])
        self.global_step = ckpt.get('global_step', 0)
