"""
Minimal evaluation script for PPO policy on held-out prompts.
Prints mean reward and response length stats before/after training.
"""
import os
import numpy as np
import torch
from rl.environment import ChatEnvironment, MockTokenizer, MockModel
from rl.reward_model import create_reward_model
from rl.ppo import PPOAgent, PPOConfig


def run_episode(env, agent, max_steps=64):
    obs, _ = env.reset()
    total_reward = 0.0
    length = 0
    for t in range(max_steps):
        action, logp, v, info = agent.act(obs)
        obs, r, terminated, truncated, _ = env.step(action)
        total_reward += r
        length += 1
        if terminated or truncated:
            break
    return total_reward, length


def evaluate(ckpt_path: str = None, episodes: int = 10):
    env = ChatEnvironment(MockModel(128), MockTokenizer(128), create_reward_model("simple"), max_length=64)
    dummy_agent = PPOAgent(vocab_size=128, obs_shape=(64,), config=PPOConfig())
    if ckpt_path and os.path.exists(ckpt_path):
        dummy_agent.load(ckpt_path)
    rewards = []
    lengths = []
    for _ in range(episodes):
        r, L = run_episode(env, dummy_agent)
        rewards.append(r)
        lengths.append(L)
    print(f"Eval: episodes={episodes} reward_mean={np.mean(rewards):.4f} reward_std={np.std(rewards):.4f} length_mean={np.mean(lengths):.2f}")


if __name__ == "__main__":
    evaluate()
