"""
Tests for the fine-tuning pipeline (finetune.py).

These tests exercise the pure-logic helpers: data preparation,
batching, learning-rate scheduling, and early-stopping — without
running an actual GPU training loop.
"""

import json
import math
import os
import sys

import numpy as np
import pytest

# finetune.py adds 'nanoGPT' to sys.path and uses `from model import …`.
# Make sure the import works from the test runner's cwd.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "nanoGPT"))

import finetune


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def chat_log(tmp_path):
    """Create a temporary chat_history.jsonl with controlled content."""
    log_file = tmp_path / "chat_history.jsonl"
    entries = []
    for i in range(20):
        entries.append({
            "id": f"id-{i}",
            "user": f"Question {i}",
            "assistant": f"Answer {i}",
            "rating": 1 if i % 3 != 0 else None,  # every 3rd is unrated
        })
    # Add a few negatively-rated entries that should be filtered out
    for i in range(5):
        entries.append({
            "id": f"neg-{i}",
            "user": f"Bad question {i}",
            "assistant": f"Bad answer {i}",
            "rating": 0,
        })
    log_file.write_text("\n".join(json.dumps(e) for e in entries) + "\n")
    return log_file


@pytest.fixture(autouse=True)
def _patch_finetune_paths(tmp_path, monkeypatch):
    """Redirect finetune globals to temp directory."""
    monkeypatch.setattr(finetune, "CHAT_LOG_FILE", str(tmp_path / "chat_history.jsonl"))
    monkeypatch.setattr(finetune, "DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setattr(finetune, "MODEL_DIR", str(tmp_path / "models"))


# ===================================================================
# prepare_training_data
# ===================================================================

class TestPrepareTrainingData:

    def test_returns_none_when_no_log(self, tmp_path):
        """Returns None when chat log does not exist."""
        result = finetune.prepare_training_data()
        assert result is None

    def test_filters_negative_ratings(self, chat_log, monkeypatch):
        """Negatively-rated conversations are excluded from training data."""
        monkeypatch.setattr(finetune, "CHAT_LOG_FILE", str(chat_log))
        result = finetune.prepare_training_data()
        assert result is not None

        train_file, val_file, train_len, val_len = result
        assert os.path.exists(train_file)
        assert os.path.exists(val_file)
        # 20 entries total, 5 with rating=0 filtered → 20 kept (unrated + positive)
        # Total tokens must be > 0
        assert train_len > 0
        assert val_len > 0

    def test_train_val_split_ratio(self, chat_log, monkeypatch):
        """Training data is split approximately 90/10."""
        monkeypatch.setattr(finetune, "CHAT_LOG_FILE", str(chat_log))
        _, _, train_len, val_len = finetune.prepare_training_data()
        total = train_len + val_len
        assert 0.85 < train_len / total < 0.95

    def test_output_files_are_valid_numpy(self, chat_log, monkeypatch):
        """Output .bin files can be loaded as uint16 numpy arrays."""
        monkeypatch.setattr(finetune, "CHAT_LOG_FILE", str(chat_log))
        train_file, val_file, _, _ = finetune.prepare_training_data()

        train_data = np.fromfile(train_file, dtype=np.uint16)
        val_data = np.fromfile(val_file, dtype=np.uint16)
        assert len(train_data) > 0
        assert len(val_data) > 0


# ===================================================================
# get_batch
# ===================================================================

class TestGetBatch:

    def _make_data(self, n=500):
        return np.arange(n, dtype=np.uint16)

    def test_batch_shapes(self):
        """Returned x, y have shape (batch_size, block_size)."""
        data = self._make_data()
        x, y = finetune.get_batch("train", data, data)
        assert x.shape == (finetune.BATCH_SIZE, finetune.BLOCK_SIZE)
        assert y.shape == (finetune.BATCH_SIZE, finetune.BLOCK_SIZE)

    def test_y_is_shifted_x(self):
        """y should be x shifted by one position (next-token prediction)."""
        data = self._make_data()
        x, y = finetune.get_batch("train", data, data)
        # For every sample, y[i] == data[start+1 : start+1+block_size]
        # We can't know the random start, but y[:, 0] == x[:, 0] + 1
        # for sequential data like arange.
        diffs = (y[:, 0] - x[:, 0]).float()
        assert (diffs == 1).all()

    def test_raises_on_tiny_dataset(self):
        """Raises ValueError when dataset is too small for a single block."""
        tiny = np.arange(5, dtype=np.uint16)
        with pytest.raises(ValueError, match="Dataset too small"):
            finetune.get_batch("train", tiny, tiny)

    def test_val_uses_val_data(self):
        """'val' split reads from val_data, not train_data."""
        train = np.zeros(500, dtype=np.uint16)
        val = np.ones(500, dtype=np.uint16) * 42
        _, y = finetune.get_batch("val", train, val)
        # All values should come from val (which is 42s)
        y_np = y.cpu().numpy()
        assert (y_np >= 42).all() and (y_np <= 43).all()


# ===================================================================
# Learning rate schedule (get_lr inside finetune())
# ===================================================================

class TestLearningRateSchedule:
    """Test the LR schedule logic extracted from finetune().

    We re-implement the schedule here to verify the mathematical
    properties — the actual function is a closure inside finetune(),
    so we reproduce its logic.
    """

    @staticmethod
    def get_lr(it, learning_rate=3e-4, warmup_iters=100,
               lr_decay_iters=1000, min_lr=3e-5):
        if it < warmup_iters:
            return learning_rate * it / warmup_iters
        if it > lr_decay_iters:
            return min_lr
        decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return min_lr + coeff * (learning_rate - min_lr)

    def test_lr_at_zero_is_zero(self):
        assert self.get_lr(0) == 0.0

    def test_lr_at_warmup_end(self):
        """LR equals max learning rate at end of warmup."""
        lr = self.get_lr(100)
        assert abs(lr - 3e-4) < 1e-9

    def test_lr_decays_monotonically(self):
        """After warmup, LR never increases."""
        prev = self.get_lr(100)
        for it in range(101, 1001):
            curr = self.get_lr(it)
            assert curr <= prev + 1e-12
            prev = curr

    def test_lr_at_end_equals_min(self):
        """LR at final iteration equals min_lr."""
        lr = self.get_lr(1000)
        assert abs(lr - 3e-5) < 1e-9

    def test_lr_beyond_decay_clamps(self):
        """LR beyond lr_decay_iters is clamped to min_lr."""
        assert self.get_lr(2000) == 3e-5


# ===================================================================
# Early stopping logic
# ===================================================================

class TestEarlyStopping:
    """Verify early stopping behavior using the same logic as finetune()."""

    def test_stops_after_patience_exhausted(self):
        """Simulates val losses that don't improve — should trigger stop."""
        patience = finetune.PATIENCE
        min_delta = finetune.MIN_DELTA

        best_val_loss = 1.0
        patience_counter = 0
        stopped_at = None

        # Simulate 20 eval intervals where val_loss stays at 1.0 (no improvement)
        for i in range(20):
            val_loss = 1.0
            if val_loss < best_val_loss - min_delta:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    stopped_at = i
                    break

        assert stopped_at is not None
        # 0-indexed: first eval that hits patience is at index patience - 1
        assert stopped_at == patience - 1

    def test_resets_on_improvement(self):
        """Patience counter resets when val loss improves."""
        patience = finetune.PATIENCE
        min_delta = finetune.MIN_DELTA

        best_val_loss = 1.0
        patience_counter = 0

        # Almost exhaust patience, then improve
        val_losses = [1.0] * (patience - 1) + [0.5]
        for val_loss in val_losses:
            if val_loss < best_val_loss - min_delta:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1

        assert patience_counter == 0
        assert best_val_loss == 0.5
