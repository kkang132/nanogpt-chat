from __future__ import annotations

from flask import Flask, Response, render_template, request, jsonify
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
import torch
import json
import os
import uuid
from datetime import datetime
from logging.handlers import RotatingFileHandler
import logging
from model import GPT, GPTConfig

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Server
SERVER_HOST = "127.0.0.1"
SERVER_PORT = 5000
CORS_ORIGINS = [f"http://127.0.0.1:{SERVER_PORT}", f"http://localhost:{SERVER_PORT}"]

# Rate limiting
DEFAULT_RATE_LIMIT = "100 per minute"
CHAT_RATE_LIMIT = "20 per minute"
RATE_RATE_LIMIT = "30 per minute"

# Generation defaults
DEFAULT_MAX_TOKENS = 150
DEFAULT_TEMPERATURE = 0.8
DEFAULT_TOP_K = 200

# Validation
MAX_MESSAGE_LENGTH = 1000
MIN_CHATS_FOR_FINETUNING = 10

# GPT-2 base model config
BASE_BLOCK_SIZE = 1024
BASE_VOCAB_SIZE = 50257
BASE_N_LAYER = 12
BASE_N_HEAD = 12
BASE_N_EMBD = 768
BASE_DROPOUT = 0.0
BASE_BIAS = True

# ---------------------------------------------------------------------------

app = Flask(__name__)
CORS(app, origins=CORS_ORIGINS)

limiter = Limiter(
    get_remote_address,
    app=app,
    default_limits=[DEFAULT_RATE_LIMIT],
    storage_uri="memory://"
)

# Configuration — use absolute paths anchored to the app directory
_APP_DIR = os.path.abspath(os.path.dirname(__file__))
CHAT_LOG_FILE = os.path.join(_APP_DIR, 'chat_history.jsonl')
MODEL_DIR = os.path.join(_APP_DIR, 'models')
LOG_DIR = os.path.join(_APP_DIR, 'logs')
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# Configure rotating file handler for chat logs
# Max 10MB per file, keep 5 backup files (total ~50MB max)
MAX_CHAT_LOG_BYTES = 10 * 1024 * 1024  # 10MB
BACKUP_COUNT = 5

# Initialize with a small GPT model
device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
print(f"Using device: {device}")

# Load fine-tuned model (or fall back to base model)
# Auto-detect the latest checkpoint by timestamp in filename
# Checks both supervised (finetuned_*) and RL (ppo_*) checkpoints
import glob
finetuned_checkpoints = sorted(
    glob.glob(os.path.join(MODEL_DIR, 'finetuned_*.pt'))
    + glob.glob(os.path.join(MODEL_DIR, 'ppo_*.pt'))
)
finetuned_path = finetuned_checkpoints[-1] if finetuned_checkpoints else None

if finetuned_path and os.path.exists(finetuned_path):
    print("Loading fine-tuned model...")
    # Ensure weights_only=True can safely resolve model.GPTConfig
    torch.serialization.add_safe_globals([GPTConfig])
    checkpoint = torch.load(finetuned_path, map_location=device, weights_only=True)
    config = checkpoint['config']
    model = GPT(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Fine-tuned model loaded from {os.path.basename(finetuned_path)} (train loss: {checkpoint['train_loss']:.4f})")
else:
    print("Loading base GPT-2 model...")
    nano_model_path = os.path.join(MODEL_DIR, 'gpt2_nano.pt')
    config = GPTConfig(
        block_size=BASE_BLOCK_SIZE,
        vocab_size=BASE_VOCAB_SIZE,
        n_layer=BASE_N_LAYER,
        n_head=BASE_N_HEAD,
        n_embd=BASE_N_EMBD,
        dropout=BASE_DROPOUT,
        bias=BASE_BIAS,
    )
    model = GPT(config)
    model.load_state_dict(torch.load(nano_model_path, map_location=device, weights_only=True))
    print("Base model loaded")

model.eval()
model.to(device)
print(f"Model loaded with {sum(p.numel() for p in model.parameters())/1e6:.2f}M parameters")

import tiktoken
enc = tiktoken.get_encoding("gpt2")

def save_chat(user_message: str, assistant_response: str) -> str:
    """Save chat interaction to JSONL file with rotation for future fine-tuning.

    Returns the chat entry ID so callers can reference it (e.g. for rating).
    """
    chat_id = str(uuid.uuid4())
    chat_entry = {
        'id': chat_id,
        'timestamp': datetime.now().isoformat(),
        'user': user_message,
        'assistant': assistant_response,
        'rating': None,
    }

    # Check file size and rotate if needed
    if os.path.exists(CHAT_LOG_FILE):
        file_size = os.path.getsize(CHAT_LOG_FILE)
        if file_size >= MAX_CHAT_LOG_BYTES:
            # Rotate: move current to .1, shift others up, delete oldest
            for i in range(BACKUP_COUNT - 1, 0, -1):
                old_file = f"{CHAT_LOG_FILE}.{i}"
                new_file = f"{CHAT_LOG_FILE}.{i + 1}"
                if os.path.exists(old_file):
                    if i == BACKUP_COUNT - 1:
                        os.remove(old_file)  # Remove oldest
                    else:
                        os.rename(old_file, new_file)
            # Move current to .1
            os.rename(CHAT_LOG_FILE, f"{CHAT_LOG_FILE}.1")

    with open(CHAT_LOG_FILE, 'a') as f:
        f.write(json.dumps(chat_entry) + '\n')

    return chat_id

def generate_response(
    prompt: str,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    temperature: float = DEFAULT_TEMPERATURE,
    top_k: int = DEFAULT_TOP_K,
) -> str:
    """Generate response using the model"""
    model.eval()

    # Encode the prompt
    encode = lambda s: enc.encode(s, allowed_special={'<|endoftext|>'})
    decode = lambda l: enc.decode(l)

    tokens = encode(prompt)
    tokens = torch.tensor(tokens, dtype=torch.long, device=device)[None, ...]

    # Generate
    with torch.no_grad():
        for _ in range(max_tokens):
            # Crop to block_size if needed
            idx_cond = tokens if tokens.size(1) <= model.config.block_size else tokens[:, -model.config.block_size:]
            # Forward pass
            logits, _ = model(idx_cond)
            logits = logits[:, -1, :] / temperature

            # Top-k sampling
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')

            probs = torch.nn.functional.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            tokens = torch.cat((tokens, idx_next), dim=1)

            # Stop at newline or end token
            if idx_next[0].item() == enc.encode('\n')[0]:
                break

    generated_text = decode(tokens[0].tolist())
    # Extract only the generated part (after the prompt)
    response = generated_text[len(prompt):].strip()
    return response if response else "I'm thinking..."

@app.after_request
def set_security_headers(response: Response) -> Response:
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['X-Frame-Options'] = 'DENY'
    response.headers['Content-Security-Policy'] = "default-src 'self'; style-src 'self' 'unsafe-inline'; script-src 'self'"
    response.headers['Referrer-Policy'] = 'strict-origin-when-cross-origin'
    return response

@app.errorhandler(Exception)
def handle_exception(e: Exception) -> tuple[Response, int]:
    """Catch-all error handler to prevent leaking internal details."""
    app.logger.error(f"Unhandled exception: {e}")
    return jsonify({'error': 'Internal server error'}), 500

@app.route('/')
def index() -> str:
    """Render the chat UI (templates/index.html)."""
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
@limiter.limit(CHAT_RATE_LIMIT)
def chat() -> Response:
    """Chat endpoint. Expects JSON {"message": str}. Returns model response and chat count."""
    # Validate request content type
    if not request.is_json:
        return jsonify({'error': 'Content-Type must be application/json'}), 400

    data = request.json

    # Validate data is a dictionary
    if not isinstance(data, dict):
        return jsonify({'error': 'Invalid JSON structure'}), 400

    user_message = data.get('message', '')

    # Validate message exists and is a string
    if not user_message:
        return jsonify({'error': 'No message provided'}), 400

    if not isinstance(user_message, str):
        return jsonify({'error': 'Message must be a string'}), 400

    # Validate message length (prevent DoS and excessive token usage)
    if len(user_message) > MAX_MESSAGE_LENGTH:
        return jsonify({'error': f'Message too long (max {MAX_MESSAGE_LENGTH} characters)'}), 400

    # Strip and validate non-empty after stripping
    user_message = user_message.strip()
    if not user_message:
        return jsonify({'error': 'Message cannot be empty or whitespace only'}), 400

    # Generate response
    prompt = f"Human: {user_message}\nAssistant:"
    response = generate_response(prompt, max_tokens=DEFAULT_MAX_TOKENS, temperature=DEFAULT_TEMPERATURE)

    # Save the interaction (non-critical — don't fail the response on write errors)
    chat_id = None
    try:
        chat_id = save_chat(user_message, response)
    except OSError:
        app.logger.warning("Failed to save chat interaction to log file")

    chat_count = 0
    if os.path.exists(CHAT_LOG_FILE):
        with open(CHAT_LOG_FILE) as f:
            chat_count = sum(1 for _ in f)

    return jsonify({
        'response': response,
        'chat_count': chat_count,
        'chat_id': chat_id,
    })

@app.route('/stats', methods=['GET'])
def stats() -> Response:
    """Get statistics about collected chat data"""
    if not os.path.exists(CHAT_LOG_FILE):
        return jsonify({'chat_count': 0})

    with open(CHAT_LOG_FILE) as f:
        chat_count = sum(1 for _ in f)
    return jsonify({
        'chat_count': chat_count,
        'ready_for_finetuning': chat_count >= MIN_CHATS_FOR_FINETUNING
    })

@app.route('/rate', methods=['POST'])
@limiter.limit(RATE_RATE_LIMIT)
def rate() -> Response:
    """Rate a chat response. Expects {"chat_id": str, "rating": 0 or 1}."""
    if not request.is_json:
        return jsonify({'error': 'Content-Type must be application/json'}), 400

    data = request.json
    if not isinstance(data, dict):
        return jsonify({'error': 'Invalid JSON structure'}), 400

    chat_id = data.get('chat_id')
    rating = data.get('rating')

    if not chat_id or not isinstance(chat_id, str):
        return jsonify({'error': 'chat_id is required'}), 400
    if rating not in (0, 1):
        return jsonify({'error': 'rating must be 0 or 1'}), 400

    if not os.path.exists(CHAT_LOG_FILE):
        return jsonify({'error': 'No chat history found'}), 404

    lines = []
    found = False
    with open(CHAT_LOG_FILE, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            if entry.get('id') == chat_id:
                entry['rating'] = rating
                found = True
            lines.append(json.dumps(entry))

    if not found:
        return jsonify({'error': 'Chat ID not found'}), 404

    with open(CHAT_LOG_FILE, 'w') as f:
        f.write('\n'.join(lines) + '\n')

    return jsonify({'status': 'ok', 'chat_id': chat_id, 'rating': rating})


if __name__ == '__main__':
    print(f"\n{'='*60}")
    print(f"🚀 NanoGPT Chat Server Starting")
    print(f"{'='*60}")
    print(f"Device: {device}")
    print(f"Chat logs: {CHAT_LOG_FILE}")
    print(f"Access the app at: http://{SERVER_HOST}:{SERVER_PORT}")
    print(f"{'='*60}\n")

    # Security: disable debug mode and bind to localhost only in production
    # debug=True enables interactive debugger with arbitrary code execution
    # host='0.0.0.0' exposes the service to external networks
    app.run(debug=False, host=SERVER_HOST, port=SERVER_PORT)
