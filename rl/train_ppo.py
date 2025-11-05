"""
Training loop for PPO on the ChatEnvironment.

Runs a simple on-policy loop: rollout -> compute advantages -> PPO update -> log -> checkpoint.
Satisfies acceptance criteria logging keys and stability measures.
"""
import os
import time
from typing import List
import numpy as np

import torch

from rl.environment import ChatEnvironment, MockTokenizer, MockModel
from rl.reward_model import create_reward_model
from rl.ppo import PPOConfig, RolloutBuffer, PPOAgent


def make_env(vocab_size: int = 128, max_length: int = 64):
    tokenizer = MockTokenizer(vocab_size=vocab_size, pad_token_id=0)
    model = MockModel(vocab_size=vocab_size)
    reward_model = create_reward_model("simple", default_reward=0.5)
    env = ChatEnvironment(model=model, tokenizer=tokenizer, reward_model=reward_model, max_length=max_length)
    return env, tokenizer


def train(
    total_steps: int = 4096,
    seed: int = 42,
    config: PPOConfig = PPOConfig(),
    ckpt_dir: str = "models/ppo",
    log_interval: int = 1,
):
    os.makedirs(ckpt_dir, exist_ok=True)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Environment and agent
    env, tokenizer = make_env(vocab_size=128, max_length=64)
    obs, _ = env.reset()
    obs_shape = obs.shape

    agent = PPOAgent(vocab_size=tokenizer.vocab_size, obs_shape=obs_shape, config=config)

    steps_done = 0
    episode_rewards: List[float] = []
    reward_window: List[float] = []

    rollout_id = 0
    while steps_done < total_steps:
        buffer = RolloutBuffer(obs_shape=obs_shape, steps=config.steps_per_rollout, action_dim=tokenizer.vocab_size, device=config.device)

        reward_sum = 0.0
        for t in range(config.steps_per_rollout):
            action, logprob, value, info = agent.act(obs)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            buffer.add(obs, action, logprob, value, reward, done)
            reward_sum += reward
            obs = next_obs
            steps_done += 1
            if done:
                episode_rewards.append(reward_sum)
                reward_window.append(reward_sum)
                obs, _ = env.reset()
                reward_sum = 0.0
            if steps_done >= total_steps:
                break

        # Bootstrap last value for GAE
        with torch.no_grad():
            _, last_v = agent.net(torch.as_tensor(obs, device=agent.device, dtype=torch.long).unsqueeze(0))
        buffer.compute_gae(last_value=float(last_v.item()), gamma=config.gamma, lam=config.lam)

        # Logging pre-update stats
        adv_mean = buffer.advantages.mean().item()
        reward_mean = np.mean(reward_window[-10:]) if reward_window else 0.0

        stats = agent.update(buffer)

        # Required logs
        log = {
            'reward_mean': float(reward_mean),
            'adv_mean': float(adv_mean),
            'loss_policy': float(stats['loss_policy']),
            'loss_value': float(stats['loss_value']),
            'kl': float(stats['kl']),
            'clip_frac': float(stats['clip_frac']),
            'entropy': float(stats['entropy']),
            'global_step': int(steps_done),
            'rollout': int(rollout_id),
        }
        if rollout_id % log_interval == 0:
            print(f"[PPO] step={log['global_step']} r_mean={log['reward_mean']:.4f} adv_mean={log['adv_mean']:.4f} "
                  f"pi_loss={log['loss_policy']:.4f} v_loss={log['loss_value']:.4f} kl={log['kl']:.4f} clip_frac={log['clip_frac']:.3f}")

        # Early stop on too-high KL
        if log['kl'] > config.kl_stop:
            print(f"[PPO] Early stop: KL {log['kl']:.4f} exceeded threshold {config.kl_stop}")
            break

        # Save checkpoint
        ckpt_path = os.path.join(ckpt_dir, f"ppo_step{steps_done}.pt")
        agent.save(ckpt_path)

        rollout_id += 1

    # Final checkpoint
    final_path = os.path.join(ckpt_dir, "ppo_final.pt")
    agent.save(final_path)
    print(f"Saved final checkpoint to {final_path}")


if __name__ == "__main__":
    # Small default run for sanity
    cfg = PPOConfig(steps_per_rollout=512, epochs=2, minibatch_size=128, lr=3e-4, clip_range=0.2)
    train(total_steps=2048, config=cfg)
