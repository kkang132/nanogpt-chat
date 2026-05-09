"""Pytest configuration for the nanogpt-chat test suite.

`app.py` instantiates a GPT model and loads a checkpoint at import time, which
makes it impossible to import inside unit tests without either real checkpoint
files on disk or carefully ordered mocks. This conftest installs a stub for
`nanoGPT.model` in `sys.modules` and intercepts `torch.load` *before* any test
module imports `app`, so tests can `from app import ...` without crashing.

After `app` has been imported once, the stubs are torn down and `torch.load`
is restored, so individual tests can still `@patch('app.torch.load', ...)` to
exercise the loading code path.
"""
from __future__ import annotations

import os
import sys
from unittest.mock import MagicMock

import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


def _install_app_stubs() -> None:
    """Make `import app` safe for unit tests."""
    if "app" in sys.modules:
        return

    fake_model_module = MagicMock(name="nanoGPT.model")
    fake_gpt_instance = MagicMock(name="GPTInstance")
    fake_gpt_instance.config.block_size = 1024
    fake_model_module.GPT.return_value = fake_gpt_instance
    sys.modules.setdefault("nanoGPT", MagicMock(name="nanoGPT"))
    sys.modules["nanoGPT.model"] = fake_model_module

    original_torch_load = torch.load

    def _fake_torch_load(*_args, **_kwargs):
        return {
            "config": fake_model_module.GPTConfig.return_value,
            "model_state_dict": {},
            "train_loss": 0.0,
        }

    torch.load = _fake_torch_load
    try:
        import app  # noqa: F401  (imports under stubs)
    finally:
        torch.load = original_torch_load


_install_app_stubs()
