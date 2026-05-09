"""
Tests for the Flask application endpoints and helpers.

The real app.py loads a PyTorch model at module level, which is expensive
and requires a checkpoint file.  We avoid that entirely by mocking the
model and tokenizer before importing the module, then exercising the
Flask test client.
"""

import json
import os
import sys
import types
import uuid
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Fixture: create a usable Flask app without loading any PyTorch model
# ---------------------------------------------------------------------------

@pytest.fixture()
def app(tmp_path, monkeypatch):
    """Import app.py with heavy dependencies stubbed out."""

    # 1. Prevent the real torch.load / model init from running.
    #    We monkey-patch the relevant symbols *before* the module is imported.
    mock_model = MagicMock()
    # model(idx_cond) must return (logits, loss) during generate_response
    # logits shape: (1, 1, vocab_size) – only last-token logits matter
    import torch
    fake_logits = torch.zeros(1, 1, 50257)
    fake_logits[0, 0, 0] = 10.0            # bias toward token 0
    mock_model.return_value = (fake_logits, None)
    mock_model.config = MagicMock(block_size=1024)
    mock_model.eval = MagicMock()
    mock_model.to = MagicMock(return_value=mock_model)
    mock_model.parameters = MagicMock(return_value=[torch.zeros(1)])

    # tiktoken encoder stub — encode returns fixed tokens, decode returns text
    mock_enc = MagicMock()
    mock_enc.encode.return_value = [0]
    mock_enc.decode.return_value = "Hello from the model"

    # Point logs / chat history to a temp directory so tests are isolated
    monkeypatch.setenv("NANOGPT_CHAT_DIR", str(tmp_path))

    # Stub torch.load so it doesn't need a real checkpoint file
    fake_state_dict = {}
    monkeypatch.setattr("torch.load", lambda *a, **kw: fake_state_dict)

    # Create a fake model checkpoint directory the module expects
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    (model_dir / "gpt2_nano.pt").touch()

    # We need to trick the module-level code.  The simplest approach is to
    # remove 'app' from sys.modules (if cached) and patch globals after import.
    sys.modules.pop("app", None)

    # Patch GPT constructor and tiktoken before importing app
    import model as gpt_model
    original_gpt_init = gpt_model.GPT.__init__

    with patch.object(gpt_model.GPT, "__init__", lambda self, config: None), \
         patch.object(gpt_model.GPT, "load_state_dict", lambda self, sd: None), \
         patch.object(gpt_model.GPT, "eval", lambda self: None), \
         patch.object(gpt_model.GPT, "to", lambda self, dev: self), \
         patch.object(gpt_model.GPT, "parameters", lambda self: [torch.zeros(1)]), \
         patch("tiktoken.get_encoding", return_value=mock_enc), \
         patch("glob.glob", return_value=[]):               # no finetuned checkpoints

        # Override _APP_DIR so paths resolve to tmp_path. Capture the real
        # `abspath` *before* patching: otherwise the lambda's else branch
        # calls `os.path.realpath`, whose CPython implementation in turn
        # calls `os.path.abspath`, recursing back into this lambda forever.
        _real_abspath = os.path.abspath
        monkeypatch.setattr(
            "os.path.abspath",
            lambda p: str(tmp_path) if p == os.path.dirname("") else _real_abspath(p),
        )

        import app as app_module

        # Replace the module-level model with our controllable mock
        app_module.model = mock_model
        app_module.enc = mock_enc
        app_module.device = "cpu"
        app_module.CHAT_LOG_FILE = str(tmp_path / "chat_history.jsonl")
        app_module.MODEL_DIR = str(model_dir)
        app_module.LOG_DIR = str(tmp_path / "logs")

    flask_app = app_module.app
    flask_app.config["TESTING"] = True
    # Disable rate limiting in tests
    app_module.limiter.enabled = False

    yield flask_app

    # Cleanup: remove cached module so next test gets a fresh import
    sys.modules.pop("app", None)


@pytest.fixture()
def client(app):
    return app.test_client()


@pytest.fixture()
def chat_log(app):
    """Return the path to the temporary chat log file."""
    import app as app_module
    return app_module.CHAT_LOG_FILE


# ===================================================================
# /chat endpoint
# ===================================================================

class TestChatEndpoint:

    def test_valid_message(self, client):
        """POST /chat with a valid message returns a response and chat_id."""
        resp = client.post("/chat", json={"message": "Hello"})
        assert resp.status_code == 200
        data = resp.get_json()
        assert "response" in data
        assert "chat_id" in data
        assert "chat_count" in data
        assert data["chat_count"] >= 1

    def test_missing_message(self, client):
        """POST /chat without a message returns 400."""
        resp = client.post("/chat", json={"message": ""})
        assert resp.status_code == 400
        assert "error" in resp.get_json()

    def test_whitespace_only_message(self, client):
        """POST /chat with whitespace-only message returns 400."""
        resp = client.post("/chat", json={"message": "   "})
        assert resp.status_code == 400

    def test_non_json_content_type(self, client):
        """POST /chat with non-JSON content type returns 400."""
        resp = client.post("/chat", data="hello", content_type="text/plain")
        assert resp.status_code == 400

    def test_message_too_long(self, client):
        """POST /chat with >1000 char message returns 400."""
        resp = client.post("/chat", json={"message": "x" * 1001})
        assert resp.status_code == 400
        assert "too long" in resp.get_json()["error"].lower()

    def test_non_string_message(self, client):
        """POST /chat with non-string message returns 400."""
        resp = client.post("/chat", json={"message": 123})
        assert resp.status_code == 400

    def test_invalid_json_structure(self, client):
        """POST /chat with a JSON list instead of object returns 400."""
        resp = client.post(
            "/chat",
            data=json.dumps([1, 2, 3]),
            content_type="application/json",
        )
        assert resp.status_code == 400


# ===================================================================
# /stats endpoint
# ===================================================================

class TestStatsEndpoint:

    def test_stats_empty(self, client):
        """GET /stats when no chats exist returns count 0."""
        resp = client.get("/stats")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["chat_count"] == 0

    def test_stats_after_chats(self, client):
        """GET /stats reflects the number of saved chats."""
        for _ in range(3):
            client.post("/chat", json={"message": "hi"})
        resp = client.get("/stats")
        assert resp.get_json()["chat_count"] == 3

    def test_stats_ready_for_finetuning(self, client):
        """GET /stats reports ready_for_finetuning once >= 10 chats."""
        for i in range(10):
            client.post("/chat", json={"message": f"msg {i}"})
        data = client.get("/stats").get_json()
        assert data["ready_for_finetuning"] is True


# ===================================================================
# /rate endpoint
# ===================================================================

class TestRateEndpoint:

    def _create_chat(self, client):
        """Helper: create a chat and return the chat_id."""
        resp = client.post("/chat", json={"message": "rate me"})
        return resp.get_json()["chat_id"]

    def test_rate_positive(self, client, chat_log):
        """Rate a chat positively and verify the log is updated."""
        chat_id = self._create_chat(client)
        resp = client.post("/rate", json={"chat_id": chat_id, "rating": 1})
        assert resp.status_code == 200
        assert resp.get_json()["rating"] == 1

        # Verify the log file was actually updated
        with open(chat_log) as f:
            entries = [json.loads(line) for line in f if line.strip()]
        matched = [e for e in entries if e["id"] == chat_id]
        assert len(matched) == 1
        assert matched[0]["rating"] == 1

    def test_rate_negative(self, client):
        chat_id = self._create_chat(client)
        resp = client.post("/rate", json={"chat_id": chat_id, "rating": 0})
        assert resp.status_code == 200
        assert resp.get_json()["rating"] == 0

    def test_rate_invalid_rating(self, client):
        chat_id = self._create_chat(client)
        resp = client.post("/rate", json={"chat_id": chat_id, "rating": 5})
        assert resp.status_code == 400
        assert "rating must be 0 or 1" in resp.get_json()["error"]

    def test_rate_missing_chat_id(self, client):
        resp = client.post("/rate", json={"rating": 1})
        assert resp.status_code == 400

    def test_rate_nonexistent_chat_id(self, client):
        self._create_chat(client)  # ensure the log file exists
        resp = client.post("/rate", json={"chat_id": "no-such-id", "rating": 1})
        assert resp.status_code == 404

    def test_rate_no_chat_history(self, client):
        """Rating when no chat log exists returns 404."""
        resp = client.post("/rate", json={"chat_id": "fake", "rating": 1})
        assert resp.status_code == 404

    def test_rate_non_json(self, client):
        resp = client.post("/rate", data="hello", content_type="text/plain")
        assert resp.status_code == 400


# ===================================================================
# Security headers
# ===================================================================

class TestSecurityHeaders:

    def test_security_headers_present(self, client):
        """Every response includes the expected security headers."""
        resp = client.get("/stats")
        assert resp.headers["X-Content-Type-Options"] == "nosniff"
        assert resp.headers["X-Frame-Options"] == "DENY"
        assert "default-src 'self'" in resp.headers["Content-Security-Policy"]
        assert resp.headers["Referrer-Policy"] == "strict-origin-when-cross-origin"


# ===================================================================
# save_chat helper
# ===================================================================

class TestSaveChat:

    def test_save_chat_creates_entry(self, app, chat_log):
        import app as app_module
        chat_id = app_module.save_chat("hello", "world")
        assert chat_id is not None
        with open(chat_log) as f:
            entry = json.loads(f.readline())
        assert entry["user"] == "hello"
        assert entry["assistant"] == "world"
        assert entry["rating"] is None

    def test_save_chat_rotation(self, app, chat_log):
        """When the log exceeds MAX_CHAT_LOG_BYTES, it rotates."""
        import app as app_module
        app_module.MAX_CHAT_LOG_BYTES = 50  # tiny limit to trigger rotation

        # Write enough data to exceed the limit
        for i in range(10):
            app_module.save_chat(f"user_{i}", f"assistant_{i}")

        # The rotated backup should exist
        assert os.path.exists(f"{chat_log}.1")
