"""
PPO Trainer for nanoGPT Chat

Implements Proximal Policy Optimization (PPO) to fine-tune the GPT model
using reward signals from user feedback. The language model itself serves
as the policy: its logits over the vocabulary define the action distribution
at each timestep. A lightweight value head is added for advantage estimation.

This is an educational implementation. It prioritizes readability over
throughput. For the algorithm details, see Schulman et al., 2017:
https://arxiv.org/abs/1707.06347
"""

import copy
import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

logger = logging.getLogger(__name__)


@dataclass
class PPOConfig:
    """Hyperparameters for PPO training."""

    # Generation
    max_new_tokens: int = 64        # max response length in tokens
    temperature: float = 0.8        # sampling temperature during rollouts
    top_k: int = 50                 # top-k filtering during rollouts

    # PPO core
    clip_eps: float = 0.2           # surrogate objective clipping range
    gamma: float = 1.0              # discount factor (1.0 = no discounting, typical for LM)
    lam: float = 0.95               # GAE lambda for advantage estimation
    vf_coef: float = 0.5            # value loss coefficient
    entropy_coef: float = 0.01      # entropy bonus to encourage exploration

    # KL penalty
    kl_coef: float = 0.1            # coefficient for per-token KL penalty against reference model
    target_kl: Optional[float] = None  # if set, early-stop epoch when mean KL exceeds this

    # Optimization
    lr: float = 1e-5                # learning rate (low — we're fine-tuning, not training)
    ppo_epochs: int = 4             # number of optimization passes per batch of rollouts
    batch_size: int = 4             # number of prompts per rollout batch
    mini_batch_size: int = 2        # mini-batch size for PPO updates
    max_grad_norm: float = 0.5      # gradient clipping norm

    # Training loop
    total_rollout_steps: int = 100  # total number of rollout-then-update cycles
    log_interval: int = 10          # print stats every N rollout steps
    save_interval: int = 50         # save checkpoint every N rollout steps

    # Paths
    model_dir: str = "models"
    prompt_file: str = "data/prompts.jsonl"  # file of user prompts for rollouts


class ValueHead(nn.Module):
    """
    A single linear layer that projects the transformer's hidden states
    to scalar value estimates. Shares the backbone with the policy —
    no separate critic network.
    """

    def __init__(self, n_embd: int):
        super().__init__()
        self.linear = nn.Linear(n_embd, 1)
        # Small init so values start near zero
        nn.init.normal_(self.linear.weight, std=0.01)
        nn.init.zeros_(self.linear.bias)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states: (batch, seq_len, n_embd) from the transformer backbone.
        Returns:
            values: (batch, seq_len) scalar value estimates per token position.
        """
        return self.linear(hidden_states).squeeze(-1)


class PPOTrainer:
    """
    Proximal Policy Optimization trainer for the GPT chat model.

    The training loop repeats:
        1. Generate responses to a batch of prompts (rollout).
        2. Score each response with the reward model.
        3. Compute per-token advantages using GAE.
        4. Update the policy (GPT weights + value head) with the PPO clipped objective.

    The reference model is a frozen copy of the initial policy, used to compute
    a KL penalty that keeps the policy from drifting too far from the pretrained
    distribution.
    """

    def __init__(self, model, tokenizer, reward_model, config=None, device=None):
        """
        Args:
            model: a GPT instance (the policy to optimize).
            tokenizer: tiktoken encoder with encode/decode methods.
            reward_model: a RewardModel instance that scores (prompt, response) pairs.
            config: PPOConfig with hyperparameters.
            device: torch device. Auto-detected if None.
        """
        self.config = config or PPOConfig()
        self.device = device or self._detect_device()
        self.tokenizer = tokenizer

        # Policy model + value head
        self.model = model.to(self.device)
        self.value_head = ValueHead(model.config.n_embd).to(self.device)

        # Reference model: frozen copy for KL computation
        self.ref_model = copy.deepcopy(model).to(self.device)
        self.ref_model.eval()
        for p in self.ref_model.parameters():
            p.requires_grad = False

        self.reward_model = reward_model

        # Optimizer covers both GPT parameters and value head
        self.optimizer = torch.optim.AdamW(
            list(self.model.parameters()) + list(self.value_head.parameters()),
            lr=self.config.lr,
        )

        # Stats tracking
        self.stats_history: List[Dict] = []

    # ------------------------------------------------------------------
    # Device detection
    # ------------------------------------------------------------------

    @staticmethod
    def _detect_device() -> str:
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    # ------------------------------------------------------------------
    # Forward helpers
    # ------------------------------------------------------------------

    def _forward_full(self, input_ids: torch.Tensor):
        """
        Run the model and return full-sequence logits, hidden states, and values.

        The base GPT.forward() optimizes inference by only computing logits at
        the last position when targets=None. We need logits at every position
        for the PPO loss, so we pass a dummy targets tensor to force full
        computation, then discard the loss.
        """
        # Dummy targets just to get full logits — the cross-entropy loss is ignored.
        dummy_targets = input_ids  # shape doesn't affect logits, only loss
        logits, _ = self.model(input_ids, targets=dummy_targets)

        # Get hidden states by re-running the transformer backbone (without lm_head).
        # This duplicates some work, but keeps the code obvious.
        device = input_ids.device
        b, t = input_ids.size()
        pos = torch.arange(0, t, dtype=torch.long, device=device)
        tok_emb = self.model.transformer.wte(input_ids)
        pos_emb = self.model.transformer.wpe(pos)
        x = self.model.transformer.drop(tok_emb + pos_emb)
        for block in self.model.transformer.h:
            x = block(x)
        hidden_states = self.model.transformer.ln_f(x)

        values = self.value_head(hidden_states)
        return logits, hidden_states, values

    def _get_log_probs(self, logits: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """
        Compute log P(action | state) for each token position.

        Args:
            logits: (batch, seq_len, vocab_size)
            actions: (batch, seq_len) token ids that were sampled
        Returns:
            log_probs: (batch, seq_len)
        """
        log_probs = F.log_softmax(logits, dim=-1)
        # Gather the log-prob of the action that was actually taken
        return log_probs.gather(2, actions.unsqueeze(2)).squeeze(2)

    # ------------------------------------------------------------------
    # Rollout collection
    # ------------------------------------------------------------------

    def _load_prompts(self) -> List[str]:
        """
        Load user prompts for rollout generation.

        Reads from config.prompt_file (JSONL with a "prompt" field per line).
        Falls back to a small set of default prompts if the file doesn't exist.
        """
        if os.path.exists(self.config.prompt_file):
            prompts = []
            with open(self.config.prompt_file, "r") as f:
                for line in f:
                    data = json.loads(line)
                    prompts.append(data["prompt"])
            if prompts:
                return prompts

        # Fallback prompts for testing / bootstrapping
        return [
            "Hello, how are you?",
            "What is machine learning?",
            "Tell me a joke.",
            "Explain Python decorators.",
            "What is the meaning of life?",
            "How does a neural network learn?",
            "What are transformers in AI?",
            "Write a haiku about programming.",
        ]

    @torch.no_grad()
    def collect_rollouts(self) -> Dict[str, torch.Tensor]:
        """
        Generate a batch of responses and collect the data needed for PPO.

        For each prompt:
            1. Encode the prompt as "Human: {text}\\nAssistant:"
            2. Sample a response token-by-token, recording log-probs.
            3. Score the response with the reward model.
            4. Compute reference-model log-probs and per-token KL.

        Returns a dict with tensors for the full sequences, masks, log-probs,
        values, rewards, and advantages.
        """
        self.model.eval()
        prompts = self._load_prompts()

        # Sample batch_size prompts (with replacement if needed)
        rng = np.random.default_rng()
        batch_prompts = [prompts[i] for i in rng.integers(0, len(prompts), self.config.batch_size)]

        all_input_ids = []
        all_response_masks = []  # 1 where the token is part of the response
        all_rewards = []

        for prompt_text in batch_prompts:
            # Encode prompt in the chat format used by finetune.py
            formatted = f"Human: {prompt_text}\nAssistant:"
            prompt_ids = self.tokenizer.encode(formatted)
            prompt_tensor = torch.tensor([prompt_ids], dtype=torch.long, device=self.device)

            # Generate response by sampling token-by-token
            generated = self._generate(prompt_tensor, self.config.max_new_tokens)
            seq = generated[0]  # (total_len,)

            # Decode just the response portion for reward scoring
            response_ids = seq[len(prompt_ids):].tolist()
            response_text = self.tokenizer.decode(response_ids)

            # Score with reward model
            reward = self.reward_model.score_response(prompt_text, response_text)

            # Build response mask: 0 for prompt tokens, 1 for response tokens
            mask = torch.zeros(len(seq), dtype=torch.float32, device=self.device)
            mask[len(prompt_ids):] = 1.0

            all_input_ids.append(seq)
            all_response_masks.append(mask)
            all_rewards.append(reward)

        # Pad sequences to the same length
        max_len = max(len(s) for s in all_input_ids)
        padded_ids = torch.zeros(len(all_input_ids), max_len, dtype=torch.long, device=self.device)
        padded_masks = torch.zeros(len(all_input_ids), max_len, dtype=torch.float32, device=self.device)

        for i, (ids, mask) in enumerate(zip(all_input_ids, all_response_masks)):
            padded_ids[i, :len(ids)] = ids
            padded_masks[i, :len(mask)] = mask

        # Truncate to block_size if needed
        block_size = self.model.config.block_size
        if max_len > block_size:
            padded_ids = padded_ids[:, :block_size]
            padded_masks = padded_masks[:, :block_size]

        # Forward pass through policy to get log-probs and values
        logits, _, values = self._forward_full(padded_ids)
        # Shift: predict next token from current position
        # logits[:, t, :] predicts token at position t+1
        shift_logits = logits[:, :-1, :]
        shift_actions = padded_ids[:, 1:]
        shift_masks = padded_masks[:, 1:]
        shift_values = values[:, :-1]

        log_probs = self._get_log_probs(shift_logits, shift_actions)

        # Reference model log-probs for KL penalty
        ref_logits, _ = self.ref_model(padded_ids, targets=padded_ids)
        ref_log_probs = self._get_log_probs(ref_logits[:, :-1, :], shift_actions)

        # Per-token KL divergence: KL = log_prob - ref_log_prob
        kl_per_token = log_probs - ref_log_probs

        # Build per-token reward: only the last response token gets the score reward;
        # all response tokens get the KL penalty
        token_rewards = torch.zeros_like(shift_masks)
        for i in range(len(all_rewards)):
            # Find the last response token position
            response_positions = (shift_masks[i] > 0).nonzero(as_tuple=True)[0]
            if len(response_positions) > 0:
                last_pos = response_positions[-1]
                token_rewards[i, last_pos] = all_rewards[i]
            # KL penalty on all response tokens
            token_rewards[i] -= self.config.kl_coef * kl_per_token[i] * shift_masks[i]

        # Compute GAE advantages
        advantages = self._compute_gae(token_rewards, shift_values, shift_masks)
        returns = advantages + shift_values.detach()

        return {
            "input_ids": padded_ids,
            "log_probs": log_probs.detach(),
            "values": shift_values.detach(),
            "advantages": advantages,
            "returns": returns,
            "mask": shift_masks,
            "kl_per_token": kl_per_token.detach(),
            "rewards": torch.tensor(all_rewards, device=self.device),
        }

    def _generate(self, prompt: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
        """
        Sample tokens autoregressively. Like GPT.generate(), but without
        @torch.no_grad() on the method itself (the caller handles that).
        """
        idx = prompt
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.model.config.block_size else idx[:, -self.model.config.block_size:]
            logits, _ = self.model(idx_cond)
            logits = logits[:, -1, :] / self.config.temperature
            if self.config.top_k is not None:
                v, _ = torch.topk(logits, min(self.config.top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("Inf")
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, next_token), dim=1)
        return idx

    # ------------------------------------------------------------------
    # GAE computation
    # ------------------------------------------------------------------

    def _compute_gae(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Generalized Advantage Estimation (Schulman et al., 2015).

        Computes advantages by walking backwards through the sequence:
            delta_t = r_t + gamma * V(t+1) - V(t)
            A_t = delta_t + gamma * lambda * A(t+1)

        Only response tokens (where mask=1) contribute. Prompt tokens get zero advantage.

        Args:
            rewards: (batch, seq_len) per-token rewards (including KL penalty).
            values: (batch, seq_len) value estimates from the value head.
            mask: (batch, seq_len) binary mask, 1 for response tokens.
        Returns:
            advantages: (batch, seq_len) GAE-computed advantages, normalized.
        """
        batch_size, seq_len = rewards.shape
        advantages = torch.zeros_like(rewards)
        gamma = self.config.gamma
        lam = self.config.lam

        for b in range(batch_size):
            last_gae = 0.0
            for t in reversed(range(seq_len)):
                if mask[b, t] == 0:
                    continue
                next_value = values[b, t + 1] if t + 1 < seq_len and mask[b, t + 1] > 0 else 0.0
                delta = rewards[b, t] + gamma * next_value - values[b, t]
                last_gae = delta + gamma * lam * last_gae
                advantages[b, t] = last_gae

        # Normalize advantages over response tokens for stable training
        response_advantages = advantages[mask > 0]
        if len(response_advantages) > 1:
            mean = response_advantages.mean()
            std = response_advantages.std() + 1e-8
            advantages = (advantages - mean) / std * mask

        return advantages

    # ------------------------------------------------------------------
    # PPO update
    # ------------------------------------------------------------------

    def update(self, rollout: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Run ppo_epochs of mini-batch updates on the collected rollout data.

        For each mini-batch:
            1. Recompute log-probs and values under the current policy.
            2. Compute the probability ratio r = exp(log_prob - old_log_prob).
            3. Clip the ratio and take the min of clipped/unclipped surrogate.
            4. Add value loss and entropy bonus.
            5. Backprop and step.

        Returns a dict of mean loss components for logging.
        """
        self.model.train()

        input_ids = rollout["input_ids"]
        old_log_probs = rollout["log_probs"]
        old_values = rollout["values"]
        advantages = rollout["advantages"]
        returns = rollout["returns"]
        mask = rollout["mask"]

        batch_size = input_ids.size(0)
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        total_kl = 0.0
        n_updates = 0

        for epoch in range(self.config.ppo_epochs):
            # Shuffle batch indices for mini-batching
            indices = torch.randperm(batch_size, device=self.device)

            for start in range(0, batch_size, self.config.mini_batch_size):
                end = start + self.config.mini_batch_size
                mb_idx = indices[start:end]

                mb_input_ids = input_ids[mb_idx]
                mb_old_log_probs = old_log_probs[mb_idx]
                mb_advantages = advantages[mb_idx]
                mb_returns = returns[mb_idx]
                mb_mask = mask[mb_idx]

                # Forward pass under current policy
                logits, _, values = self._forward_full(mb_input_ids)
                shift_logits = logits[:, :-1, :]
                shift_actions = mb_input_ids[:, 1:]
                shift_values = values[:, :-1]

                new_log_probs = self._get_log_probs(shift_logits, shift_actions)

                # Entropy of the policy distribution (for the bonus)
                probs = F.softmax(shift_logits, dim=-1)
                entropy = -(probs * (probs + 1e-8).log()).sum(-1)
                masked_entropy = (entropy * mb_mask).sum() / mb_mask.sum().clamp(min=1)

                # Probability ratio
                ratio = torch.exp(new_log_probs - mb_old_log_probs)

                # Clipped surrogate objective
                surr1 = ratio * mb_advantages
                surr2 = torch.clamp(ratio, 1.0 - self.config.clip_eps, 1.0 + self.config.clip_eps) * mb_advantages
                policy_loss = -(torch.min(surr1, surr2) * mb_mask).sum() / mb_mask.sum().clamp(min=1)

                # Value loss (clipped MSE)
                value_loss = ((shift_values - mb_returns) ** 2 * mb_mask).sum() / mb_mask.sum().clamp(min=1)

                # Total loss
                loss = (
                    policy_loss
                    + self.config.vf_coef * value_loss
                    - self.config.entropy_coef * masked_entropy
                )

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    list(self.model.parameters()) + list(self.value_head.parameters()),
                    self.config.max_grad_norm,
                )
                self.optimizer.step()

                # Track stats
                with torch.no_grad():
                    approx_kl = ((ratio - 1) - (ratio.log()) * mb_mask).sum() / mb_mask.sum().clamp(min=1)
                    total_kl += approx_kl.item()

                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += masked_entropy.item()
                n_updates += 1

            # Early stopping on KL divergence if target_kl is set
            if self.config.target_kl is not None:
                mean_kl = total_kl / n_updates
                if mean_kl > self.config.target_kl:
                    logger.info(f"Early stopping PPO epoch {epoch}: mean KL {mean_kl:.4f} > target {self.config.target_kl}")
                    break

        stats = {
            "policy_loss": total_policy_loss / max(n_updates, 1),
            "value_loss": total_value_loss / max(n_updates, 1),
            "entropy": total_entropy / max(n_updates, 1),
            "approx_kl": total_kl / max(n_updates, 1),
            "mean_reward": rollout["rewards"].mean().item(),
        }
        return stats

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------

    def train(self) -> List[Dict[str, float]]:
        """
        Main training loop: alternate between collecting rollouts and updating.

        Returns the full stats history for analysis.
        """
        logger.info("Starting PPO training")
        logger.info(f"  device:              {self.device}")
        logger.info(f"  rollout steps:       {self.config.total_rollout_steps}")
        logger.info(f"  batch size:          {self.config.batch_size}")
        logger.info(f"  PPO epochs:          {self.config.ppo_epochs}")
        logger.info(f"  learning rate:       {self.config.lr}")
        logger.info(f"  clip epsilon:        {self.config.clip_eps}")
        logger.info(f"  KL coefficient:      {self.config.kl_coef}")

        for step in range(1, self.config.total_rollout_steps + 1):
            # 1. Collect rollouts
            rollout = self.collect_rollouts()

            # 2. Update policy
            stats = self.update(rollout)
            stats["step"] = step
            self.stats_history.append(stats)

            # 3. Log
            if step % self.config.log_interval == 0 or step == 1:
                logger.info(
                    f"Step {step}/{self.config.total_rollout_steps}  "
                    f"reward={stats['mean_reward']:.3f}  "
                    f"policy_loss={stats['policy_loss']:.4f}  "
                    f"value_loss={stats['value_loss']:.4f}  "
                    f"entropy={stats['entropy']:.4f}  "
                    f"kl={stats['approx_kl']:.4f}"
                )

            # 4. Save checkpoint
            if step % self.config.save_interval == 0:
                self.save_checkpoint(step)

        # Final save
        self.save_checkpoint(self.config.total_rollout_steps)
        logger.info("PPO training complete")
        return self.stats_history

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def save_checkpoint(self, step: int) -> str:
        """
        Save the policy model in the same format as finetune.py, so
        the chat server can load it without modification.

        Returns the path to the saved checkpoint.
        """
        os.makedirs(self.config.model_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(self.config.model_dir, f"ppo_{timestamp}_step{step}.pt")

        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "config": self.model.config,
                "value_head_state_dict": self.value_head.state_dict(),
                "ppo_config": self.config,
                "step": step,
                "stats_history": self.stats_history,
            },
            path,
        )
        logger.info(f"Saved checkpoint to {path}")
        return path
