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
from .ppo_trainer import PPOTrainer, PPOConfig, ValueHead

__version__ = "0.1.0"
__all__ = [
    "RewardModel", "SimpleRatingReward", "MultiCriteriaReward",
    "LearnedRewardModel", "create_reward_model", "ChatEnvironment",
    "PPOTrainer", "PPOConfig", "ValueHead",
]