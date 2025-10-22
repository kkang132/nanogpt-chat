import unittest
import numpy as np
import gymnasium as gym
from gymnasium.spaces import Discrete, Box

from rl.environment import ChatEnvironment, MockModel, MockTokenizer
from rl.reward_model import SimpleRatingReward

class TestChatEnvironment(unittest.TestCase):
    def setUp(self):
        """Set up the test environment before each test."""
        self.model = MockModel(vocab_size=128)
        self.tokenizer = MockTokenizer(vocab_size=128, pad_token_id=0)
        self.reward_model = SimpleRatingReward(default_reward=0.5)
        self.env = ChatEnvironment(
            model=self.model,
            tokenizer=self.tokenizer,
            reward_model=self.reward_model,
            max_length=20
        )

    def test_initialization(self):
        """Test if the environment is initialized correctly."""
        self.assertIsInstance(self.env, gym.Env)
        self.assertIsInstance(self.env.action_space, Discrete)
        self.assertEqual(self.env.action_space.n, self.tokenizer.vocab_size)
        self.assertIsInstance(self.env.observation_space, Box)
        self.assertEqual(self.env.observation_space.shape, (self.env.max_length,))

    def test_reset(self):
        """Test the reset method."""
        obs, info = self.env.reset()
        self.assertIsInstance(obs, np.ndarray)
        self.assertEqual(obs.shape, (self.env.max_length,))
        self.assertTrue(np.all(obs == self.tokenizer.pad_token_id))
        self.assertEqual(len(self.env.conversation_history), 0)
        self.assertEqual(self.env.current_step, 0)
        self.assertEqual(info, {})

    def test_step(self):
        """Test the step method."""
        self.env.reset()
        action = 10  # Example action
        obs, reward, terminated, truncated, info = self.env.step(action)

        self.assertIsInstance(obs, np.ndarray)
        self.assertEqual(obs.shape, (self.env.max_length,))
        self.assertEqual(obs[0], action)
        self.assertEqual(self.env.conversation_history[0], action)

        self.assertIsInstance(reward, float)
        self.assertEqual(reward, 0.5)  # Default reward

        self.assertFalse(terminated)
        self.assertFalse(truncated)
        self.assertEqual(info, {})

    def test_episode_termination(self):
        """Test if the episode terminates correctly."""
        self.env.reset()
        for i in range(self.env.max_length):
            action = i + 1
            obs, reward, terminated, truncated, info = self.env.step(action)
            if i < self.env.max_length - 1:
                self.assertFalse(terminated)
            else:
                self.assertTrue(terminated)

    def test_invalid_action(self):
        """Test that an invalid action raises an error."""
        self.env.reset()
        with self.assertRaises(ValueError):
            self.env.step(self.tokenizer.vocab_size) # Action outside of space

    def test_render(self):
        """Test the render method."""
        self.env.reset()
        self.env.step(65) # ASCII for 'A'
        self.env.step(66) # ASCII for 'B'

        # Redirect stdout to capture render output
        import io
        from contextlib import redirect_stdout

        f = io.StringIO()
        with redirect_stdout(f):
            self.env.render()
        output = f.getvalue()

        self.assertIn("Conversation History:", output)
        self.assertIn("AB", output)

    def test_close(self):
        """Test the close method."""
        # Redirect stdout to capture close output
        import io
        from contextlib import redirect_stdout

        f = io.StringIO()
        with redirect_stdout(f):
            self.env.close()
        output = f.getvalue()

        self.assertIn("Closing ChatEnvironment", output)

if __name__ == "__main__":
    unittest.main()