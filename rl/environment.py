"""
Chat Environment for RL Training

This module implements the ChatEnvironment class, which serves as the RL
environment interface for PPO training. It defines the state and action spaces
and integrates with the reward model.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Optional, Tuple, Any

# Define a mock model and tokenizer for standalone testing
class MockModel:
    def __init__(self, vocab_size):
        self.vocab_size = vocab_size

    def __call__(self, inputs):
        # Simulate model outputting random token probabilities
        batch_size, sequence_length = inputs.shape
        return np.random.rand(batch_size, sequence_length, self.vocab_size)

class MockTokenizer:
    def __init__(self, vocab_size, pad_token_id=0):
        self.vocab_size = vocab_size
        self.pad_token_id = pad_token_id

    def encode(self, text):
        # Simulate tokenization
        return [ord(c) for c in text]

    def decode(self, token_ids):
        # Simulate decoding
        return "".join([chr(t) for t in token_ids if t != self.pad_token_id])

class ChatEnvironment(gym.Env):
    """
    Chat Environment for Reinforcement Learning.

    This environment interfaces a chat system with an RL algorithm like PPO.
    It follows the gymnasium.Env interface.
    """
    metadata = {'render.modes': ['human']}

    def __init__(self, model, tokenizer, reward_model, max_length=512):
        """
        Initialize the ChatEnvironment.

        Args:
            model: The language model for generating responses.
            tokenizer: The tokenizer for encoding and decoding text.
            reward_model: The model that provides reward signals.
            max_length (int): Maximum length of the conversation history.
        """
        super(ChatEnvironment, self).__init__()

        self.model = model
        self.tokenizer = tokenizer
        self.reward_model = reward_model
        self.max_length = max_length

        # Define action and observation spaces
        self.action_space = spaces.Discrete(self.tokenizer.vocab_size)
        self.observation_space = spaces.Box(
            low=0,
            high=self.tokenizer.vocab_size - 1,
            shape=(self.max_length,),
            dtype=np.int32
        )

        self.conversation_history = []
        self.current_step = 0

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[np.ndarray, dict]:
        """
        Reset the environment to an initial state.

        Args:
            seed (int, optional): The seed for the random number generator.
            options (dict, optional): Additional options for resetting the environment.

        Returns:
            A tuple containing the initial observation and an info dictionary.
        """
        super().reset(seed=seed)

        self.conversation_history = []
        self.current_step = 0

        initial_obs = np.full((self.max_length,), self.tokenizer.pad_token_id, dtype=np.int32)
        return initial_obs, {}

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """
        Execute one time step within the environment.

        Args:
            action (int): The action to take (token ID to generate).

        Returns:
            A tuple containing the new observation, reward, terminated flag, truncated flag, and info dictionary.
        """
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action: {action}")

        self.conversation_history.append(action)
        self.current_step += 1

        # Check for termination conditions
        terminated = self.current_step >= self.max_length
        truncated = False

        # Generate observation
        obs = np.full((self.max_length,), self.tokenizer.pad_token_id, dtype=np.int32)
        obs[:len(self.conversation_history)] = self.conversation_history

        # For now, let's use a placeholder for user_message and response
        user_message = "user input"
        response = self.tokenizer.decode(self.conversation_history)

        # Calculate reward
        reward = self.reward_model.score_response(user_message, response)

        info = {}

        return obs, reward, terminated, truncated, info

    def render(self, mode: str = 'human') -> None:
        """
        Render the environment's state.

        Args:
            mode (str): The mode to render with ('human' is supported).
        """
        if mode == 'human':
            print("Conversation History:")
            print(self.tokenizer.decode(self.conversation_history))
        else:
            super(ChatEnvironment, self).render()

    def close(self) -> None:
        """
        Perform any necessary cleanup.
        """
        print("Closing ChatEnvironment")