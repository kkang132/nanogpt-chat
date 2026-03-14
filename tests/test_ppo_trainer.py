"""
Tests for the PPO trainer.

Uses a tiny GPT model (2 layers, 2 heads, 64-dim embeddings) so tests
run in seconds on CPU. The point is to verify that the PPO machinery
works end-to-end, not that the model learns anything meaningful.
"""

import os
import sys
import tempfile
import unittest

import torch

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from model import GPT, GPTConfig
from rl.ppo_trainer import PPOConfig, PPOTrainer, ValueHead
from rl.reward_model import SimpleRatingReward


class FakeTokenizer:
    """Minimal tokenizer that maps characters to integers. Good enough for tests."""

    vocab_size = 256

    def encode(self, text: str):
        return [ord(c) % self.vocab_size for c in text]

    def decode(self, ids):
        return "".join(chr(i % 128) for i in ids if i > 0)


def make_tiny_model() -> GPT:
    """Create a tiny GPT for fast testing."""
    config = GPTConfig(
        block_size=128,
        vocab_size=256,
        n_layer=2,
        n_head=2,
        n_embd=64,
        dropout=0.0,
        bias=True,
    )
    return GPT(config)


class TestValueHead(unittest.TestCase):
    def test_output_shape(self):
        head = ValueHead(n_embd=64)
        x = torch.randn(2, 10, 64)
        values = head(x)
        self.assertEqual(values.shape, (2, 10))

    def test_initial_values_near_zero(self):
        head = ValueHead(n_embd=64)
        x = torch.randn(1, 5, 64)
        values = head(x)
        # With small init, values should be close to zero
        self.assertTrue(values.abs().mean().item() < 1.0)


class TestPPOTrainer(unittest.TestCase):
    def setUp(self):
        self.model = make_tiny_model()
        self.tokenizer = FakeTokenizer()
        self.reward_model = SimpleRatingReward(default_reward=0.5)
        self.config = PPOConfig(
            max_new_tokens=8,
            batch_size=2,
            mini_batch_size=2,
            ppo_epochs=2,
            total_rollout_steps=2,
            log_interval=1,
            save_interval=2,
            lr=1e-4,
        )
        self.trainer = PPOTrainer(
            model=self.model,
            tokenizer=self.tokenizer,
            reward_model=self.reward_model,
            config=self.config,
            device="cpu",
        )

    def test_init(self):
        """Trainer initializes with policy, reference model, and value head."""
        self.assertIsNotNone(self.trainer.model)
        self.assertIsNotNone(self.trainer.ref_model)
        self.assertIsNotNone(self.trainer.value_head)
        # Reference model should be frozen
        for p in self.trainer.ref_model.parameters():
            self.assertFalse(p.requires_grad)

    def test_forward_full(self):
        """_forward_full returns logits, hidden states, and values."""
        ids = torch.randint(0, 256, (2, 16))
        logits, hidden, values = self.trainer._forward_full(ids)
        self.assertEqual(logits.shape, (2, 16, 256))
        self.assertEqual(hidden.shape, (2, 16, 64))
        self.assertEqual(values.shape, (2, 16))

    def test_get_log_probs(self):
        """_get_log_probs returns correct shape and valid log probabilities."""
        logits = torch.randn(2, 10, 256)
        actions = torch.randint(0, 256, (2, 10))
        log_probs = self.trainer._get_log_probs(logits, actions)
        self.assertEqual(log_probs.shape, (2, 10))
        # Log probs should be non-positive
        self.assertTrue((log_probs <= 0).all())

    def test_collect_rollouts(self):
        """collect_rollouts returns a dict with all expected keys and shapes."""
        rollout = self.trainer.collect_rollouts()
        expected_keys = {"input_ids", "log_probs", "values", "advantages", "returns", "mask", "kl_per_token", "rewards"}
        self.assertEqual(set(rollout.keys()), expected_keys)
        self.assertEqual(rollout["rewards"].shape, (2,))
        # log_probs, values, advantages, returns, mask should all have batch dim = 2
        self.assertEqual(rollout["log_probs"].shape[0], 2)

    def test_compute_gae(self):
        """GAE computation produces advantages with correct shape."""
        rewards = torch.tensor([[0.0, 0.0, 1.0, 0.0], [0.0, 0.5, 0.0, 0.0]])
        values = torch.tensor([[0.1, 0.2, 0.3, 0.1], [0.1, 0.2, 0.3, 0.1]])
        mask = torch.tensor([[0.0, 1.0, 1.0, 0.0], [1.0, 1.0, 0.0, 0.0]])
        advantages = self.trainer._compute_gae(rewards, values, mask)
        self.assertEqual(advantages.shape, (2, 4))
        # Prompt tokens (mask=0) should have zero advantage
        self.assertEqual(advantages[0, 0].item(), 0.0)
        self.assertEqual(advantages[0, 3].item(), 0.0)

    def test_update(self):
        """PPO update runs without error and returns stats."""
        rollout = self.trainer.collect_rollouts()
        stats = self.trainer.update(rollout)
        expected_stat_keys = {"policy_loss", "value_loss", "entropy", "approx_kl", "mean_reward"}
        self.assertEqual(set(stats.keys()), expected_stat_keys)
        # Losses should be finite
        for key in ["policy_loss", "value_loss"]:
            self.assertTrue(abs(stats[key]) < float("inf"))

    def test_train_loop(self):
        """Full training loop completes and produces stats."""
        with tempfile.TemporaryDirectory() as tmpdir:
            self.trainer.config.model_dir = tmpdir
            history = self.trainer.train()
            self.assertEqual(len(history), 2)  # total_rollout_steps = 2
            # Check that a checkpoint was saved
            checkpoints = [f for f in os.listdir(tmpdir) if f.startswith("ppo_")]
            self.assertGreater(len(checkpoints), 0)

    def test_save_checkpoint(self):
        """Checkpoint includes model_state_dict compatible with app.py loading."""
        with tempfile.TemporaryDirectory() as tmpdir:
            self.trainer.config.model_dir = tmpdir
            path = self.trainer.save_checkpoint(step=1)
            self.assertTrue(os.path.exists(path))
            checkpoint = torch.load(path, weights_only=False)
            self.assertIn("model_state_dict", checkpoint)
            self.assertIn("config", checkpoint)
            self.assertIn("value_head_state_dict", checkpoint)
            self.assertIn("ppo_config", checkpoint)

    def test_reference_model_stays_frozen(self):
        """Reference model weights don't change after an update."""
        ref_params_before = {
            k: v.clone() for k, v in self.trainer.ref_model.state_dict().items()
        }
        rollout = self.trainer.collect_rollouts()
        self.trainer.update(rollout)
        for k, v in self.trainer.ref_model.state_dict().items():
            self.assertTrue(torch.equal(v, ref_params_before[k]),
                            f"Reference model param {k} changed after update")


if __name__ == "__main__":
    unittest.main()
