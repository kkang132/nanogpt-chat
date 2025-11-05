"""
RL Chat System Components

This package contains the reinforcement learning components for the chat system,
including reward models, environment interfaces, and training algorithms.
"""

from .reward_model import (
    RewardModel, SimpleRatingReward, MultiCriteriaReward, 
    LearnedRewardModel, create_reward_model
)
from .environment import ChatEnvironment
from .ppo import PPOConfig, RolloutBuffer, PPOAgent

__version__ = "0.2.0"
__all__ = [
    "RewardModel", "SimpleRatingReward", "MultiCriteriaReward", 
    "LearnedRewardModel", "create_reward_model", "ChatEnvironment",
    "PPOConfig", "RolloutBuffer", "PPOAgent"
]