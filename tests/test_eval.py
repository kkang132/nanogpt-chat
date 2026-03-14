"""
Tests for the evaluation pipeline (eval.py).

These tests exercise the pure-logic helpers — scoring functions,
answer extraction, checkpoint discovery, output formatting — without
loading real model checkpoints or running GPU inference.
"""

import json
import math
import sys
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

# Add project root so ``import eval`` resolves model.py etc.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import eval as eval_module  # noqa: E402

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _patch_eval_paths(tmp_path, monkeypatch):
    """Redirect eval globals to temp directory."""
    monkeypatch.setattr(eval_module, "MODEL_DIR", str(tmp_path / "models"))
    monkeypatch.setattr(eval_module, "DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setattr(
        eval_module, "RESULTS_FILE", str(tmp_path / "eval_results.jsonl")
    )


# ===================================================================
# discover_checkpoints
# ===================================================================


class TestDiscoverCheckpoints:
    def test_finds_finetuned_checkpoints(self, tmp_path):
        models = tmp_path / "models"
        models.mkdir()
        (models / "finetuned_20260101_120000.pt").touch()
        (models / "finetuned_20260102_120000.pt").touch()
        result = eval_module.discover_checkpoints(str(models))
        names = [Path(p).name for p in result]
        assert "finetuned_20260101_120000.pt" in names
        assert "finetuned_20260102_120000.pt" in names

    def test_finds_ppo_checkpoints(self, tmp_path):
        models = tmp_path / "models"
        models.mkdir()
        (models / "ppo_20260101_120000.pt").touch()
        result = eval_module.discover_checkpoints(str(models))
        assert len(result) == 1
        assert "ppo_20260101_120000.pt" in Path(result[0]).name

    def test_includes_base_model(self, tmp_path):
        models = tmp_path / "models"
        models.mkdir()
        (models / "gpt2_nano.pt").touch()
        result = eval_module.discover_checkpoints(str(models))
        assert len(result) == 1
        assert "gpt2_nano.pt" in Path(result[0]).name

    def test_base_model_comes_first(self, tmp_path):
        models = tmp_path / "models"
        models.mkdir()
        (models / "gpt2_nano.pt").touch()
        (models / "finetuned_20260101_120000.pt").touch()
        result = eval_module.discover_checkpoints(str(models))
        assert Path(result[0]).name == "gpt2_nano.pt"

    def test_empty_directory(self, tmp_path):
        models = tmp_path / "models"
        models.mkdir()
        assert eval_module.discover_checkpoints(str(models)) == []

    def test_ignores_non_checkpoint_files(self, tmp_path):
        models = tmp_path / "models"
        models.mkdir()
        (models / "notes.txt").touch()
        (models / "other_model.pt").touch()
        assert eval_module.discover_checkpoints(str(models)) == []


# ===================================================================
# score_length
# ===================================================================


class TestScoreLength:
    def test_empty_string_scores_zero(self):
        assert eval_module.score_length("") == 0.0

    def test_ideal_length_scores_one(self):
        text = "a" * 100
        assert eval_module.score_length(text) == 1.0

    def test_boundary_500_scores_one(self):
        assert eval_module.score_length("a" * 500) == 1.0

    def test_very_short_penalized(self):
        score = eval_module.score_length("Hi")
        assert 0.0 < score < 1.0

    def test_very_long_penalized(self):
        score = eval_module.score_length("a" * 800)
        assert 0.0 < score < 1.0

    def test_beyond_1000_scores_zero(self):
        assert eval_module.score_length("a" * 1001) == 0.0


# ===================================================================
# score_repetition
# ===================================================================


class TestScoreRepetition:
    def test_unique_text_scores_high(self):
        text = "the quick brown fox jumps over the lazy dog near a river"
        score = eval_module.score_repetition(text)
        assert score > 0.8

    def test_repeated_text_scores_low(self):
        text = " ".join(["the cat sat"] * 20)
        score = eval_module.score_repetition(text)
        assert score < 0.2

    def test_short_text_returns_one(self):
        assert eval_module.score_repetition("hi") == 1.0

    def test_empty_text_returns_one(self):
        assert eval_module.score_repetition("") == 1.0


# ===================================================================
# score_coherence
# ===================================================================


class TestScoreCoherence:
    def test_english_text_scores_high(self):
        text = "The weather is nice today and I feel great"
        score = eval_module.score_coherence(text)
        assert score > 0.9

    def test_gibberish_scores_low(self):
        text = "xkcd qwrty zbnm plgh krft"
        score = eval_module.score_coherence(text)
        assert score < 0.3

    def test_empty_string_scores_zero(self):
        assert eval_module.score_coherence("") == 0.0

    def test_numbers_only_scores_zero(self):
        assert eval_module.score_coherence("123 456 789") == 0.0


# ===================================================================
# score_format
# ===================================================================


class TestScoreFormat:
    def test_normal_response_scores_high(self):
        text = "The capital of France is Paris."
        assert eval_module.score_format(text) == 1.0

    def test_empty_response_scores_zero(self):
        assert eval_module.score_format("") == 0.0
        assert eval_module.score_format("   ") == 0.0

    def test_prompt_leakage_penalized(self):
        text = "Human: some repeated prompt"
        score = eval_module.score_format(text)
        assert score < 1.0

    def test_single_word_penalized(self):
        score = eval_module.score_format("Yes")
        assert score < 1.0


# ===================================================================
# extract_numeric_answer
# ===================================================================


class TestExtractNumericAnswer:
    def test_the_answer_is_pattern(self):
        assert eval_module.extract_numeric_answer("The answer is 42") == "42"

    def test_the_answer_is_case_insensitive_t(self):
        assert eval_module.extract_numeric_answer("the answer is 7") == "7"

    def test_hash_pattern(self):
        assert eval_module.extract_numeric_answer("#### 42") == "42"

    def test_comma_in_number(self):
        assert eval_module.extract_numeric_answer("The answer is 1,234") == "1234"

    def test_decimal_number(self):
        assert eval_module.extract_numeric_answer("The answer is 3.14") == "3.14"

    def test_no_number_returns_none(self):
        assert eval_module.extract_numeric_answer("no numbers here") is None

    def test_fallback_to_last_number(self):
        text = "First we get 10, then 20, and finally 42 total."
        assert eval_module.extract_numeric_answer(text) == "42"

    def test_negative_number(self):
        assert eval_module.extract_numeric_answer("The answer is -5") == "-5"


# ===================================================================
# compute_perplexity
# ===================================================================


class TestComputePerplexity:
    def test_zero_loss(self):
        assert eval_module.compute_perplexity(0.0) == pytest.approx(1.0)

    def test_known_loss(self):
        assert eval_module.compute_perplexity(3.0) == pytest.approx(
            math.exp(3.0), rel=1e-5
        )

    def test_high_loss(self):
        result = eval_module.compute_perplexity(10.0)
        assert result == pytest.approx(math.exp(10.0), rel=1e-5)


# ===================================================================
# compute_val_loss
# ===================================================================


class TestComputeValLoss:
    def test_returns_average_loss(self):
        """Mock model returns a fixed loss; verify averaging."""
        mock_model = MagicMock()
        mock_model.eval = MagicMock()
        mock_model.config = MagicMock()
        fixed_loss = torch.tensor(2.5)
        mock_model.return_value = (None, fixed_loss)

        data = np.arange(500, dtype=np.uint16)
        loss = eval_module.compute_val_loss(mock_model, data, num_batches=5)
        assert loss == pytest.approx(2.5, abs=0.01)

    def test_raises_on_tiny_data(self):
        mock_model = MagicMock()
        tiny = np.arange(5, dtype=np.uint16)
        with pytest.raises(ValueError, match="too small"):
            eval_module.compute_val_loss(mock_model, tiny)


# ===================================================================
# format_results_table
# ===================================================================


class TestFormatResultsTable:
    def test_single_result(self):
        results = [
            {
                "checkpoint": "gpt2_nano.pt",
                "eval_val_loss": 3.5,
                "perplexity": 33.12,
                "generation": {"avg_combined_score": 0.75},
                "gsm8k": {"accuracy": 0.02},
            }
        ]
        table = eval_module.format_results_table(results)
        assert "gpt2_nano.pt" in table
        assert "3.5000" in table
        assert "0.75" in table

    def test_none_values_show_dash(self):
        results = [
            {
                "checkpoint": "test.pt",
                "eval_val_loss": None,
                "perplexity": None,
                "generation": None,
                "gsm8k": None,
            }
        ]
        table = eval_module.format_results_table(results)
        # Should contain dashes for missing values
        assert table.count("\u2014") >= 4

    def test_long_name_truncated(self):
        results = [
            {
                "checkpoint": "a" * 40,
                "eval_val_loss": 1.0,
                "perplexity": 2.72,
                "generation": None,
                "gsm8k": None,
            }
        ]
        table = eval_module.format_results_table(results)
        assert "..." in table


# ===================================================================
# save_results
# ===================================================================


class TestSaveResults:
    def test_creates_file(self, tmp_path):
        path = str(tmp_path / "results.jsonl")
        results = [{"checkpoint": "test.pt", "eval_val_loss": 1.0}]
        eval_module.save_results(results, path)
        assert Path(path).exists()

    def test_appends_to_existing(self, tmp_path):
        path = str(tmp_path / "results.jsonl")
        eval_module.save_results([{"checkpoint": "a.pt"}], path)
        eval_module.save_results([{"checkpoint": "b.pt"}], path)
        with Path(path).open() as f:
            lines = f.readlines()
        assert len(lines) == 2

    def test_valid_jsonl(self, tmp_path):
        path = str(tmp_path / "results.jsonl")
        results = [
            {"checkpoint": "test.pt", "eval_val_loss": 1.5, "generation": None},
        ]
        eval_module.save_results(results, path)
        with Path(path).open() as f:
            for line in f:
                parsed = json.loads(line)
                assert "checkpoint" in parsed

    def test_strips_sample_responses(self, tmp_path):
        path = str(tmp_path / "results.jsonl")
        results = [
            {
                "checkpoint": "test.pt",
                "generation": {
                    "avg_combined_score": 0.5,
                    "sample_responses": [{"prompt": "p", "response": "r"}],
                },
            }
        ]
        eval_module.save_results(results, path)
        with Path(path).open() as f:
            saved = json.loads(f.readline())
        assert "sample_responses" not in saved["generation"]
