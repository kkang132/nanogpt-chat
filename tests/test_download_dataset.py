"""
Tests for the dataset download and preparation utilities.

Only the pure functions are tested here; the download functions that
call `load_dataset` are too network-dependent for unit tests.
"""

import pytest
from download_dataset import clean_gsm8k_answer


class TestCleanGsm8kAnswer:

    def test_removes_calculator_annotations(self):
        raw = "She has 48 apples. <<48/2=24>> She gives away 24."
        result = clean_gsm8k_answer(raw)
        assert "<<" not in result
        assert ">>" not in result
        assert "48/2=24" not in result
        assert "She has 48 apples." in result

    def test_replaces_final_answer_marker(self):
        raw = "Step one. Step two. #### 42"
        result = clean_gsm8k_answer(raw)
        assert "####" not in result
        assert "The answer is 42" in result

    def test_combined_annotations_and_answer(self):
        raw = "Total cost is <<10+5=15>>15 dollars. #### 15"
        result = clean_gsm8k_answer(raw)
        assert "<<" not in result
        assert "####" not in result
        assert "The answer is 15" in result

    def test_plain_text_unchanged(self):
        raw = "A simple answer with no annotations."
        assert clean_gsm8k_answer(raw) == raw

    def test_strips_whitespace(self):
        raw = "  Some text  #### 7  "
        result = clean_gsm8k_answer(raw)
        assert result == "Some text  The answer is 7"

    def test_multiple_calculator_annotations(self):
        raw = "<<2*3=6>>6 items and <<6+4=10>>10 total. #### 10"
        result = clean_gsm8k_answer(raw)
        assert result == "6 items and 10 total. The answer is 10"
