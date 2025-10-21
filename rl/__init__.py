"""
RL Chat System Components

This package contains the reinforcement learning components for the chat system,
including reward models, environment interfaces, and training algorithms.
"""

from .reward_model import RewardModel
from .environment import ChatEnvironment

__version__ = "0.1.0"
__all__ = ["RewardModel", "ChatEnvironment"]
