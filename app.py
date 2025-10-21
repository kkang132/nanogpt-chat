from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import torch
import json
import os
from datetime import datetime
from nanoGPT.model import GPT, GPTConfig

app = Flask(__name__)
CORS(app)

# Configuration
CHAT_LOG_FILE = 'chat_history.jsonl'
MODEL_DIR = 'models'
os.makedirs(MODEL_DIR, exist_ok=True)

# Initialize with a small GPT model
device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
print(f"Using device: {device}")

# Load fine-tuned model (or fall back to base model)
finetuned_path = os.path.join(MODEL_DIR, 'finetuned_20251020_100757.pt')

if os.path.exists(finetuned_path):
    print("Loading fine-tuned model...")
    # Need to add nanoGPT.model to sys.modules for unpickling
    import sys
    sys.modules['model'] = sys.modules['nanoGPT.model']

    checkpoint = torch.load(finetuned_path, map_location=device, weights_only=False)
    config = checkpoint['config']
    model = GPT(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Fine-tuned model loaded (train loss: {checkpoint['train_loss']:.4f})")
else:
    print("Loading base GPT-2 model...")
    nano_model_path = os.path.join(MODEL_DIR, 'gpt2_nano.pt')
    config = GPTConfig(
        block_size=1024,
        vocab_size=50257,
        n_layer=12,
        n_head=12,
        n_embd=768,
        dropout=0.0,
        bias=True
    )
    model = GPT(config)
    model.load_state_dict(torch.load(nano_model_path, map_location=device, weights_only=True))
    print("Base model loaded")

model.eval()
model.to(device)
print(f"Model loaded with {sum(p.numel() for p in model.parameters())/1e6:.2f}M parameters")

import tiktoken
enc = tiktoken.get_encoding("gpt2")

def save_chat(user_message, assistant_response):
    """Save chat interaction to JSONL file for future fine-tuning"""
    chat_entry = {
        'timestamp': datetime.now().isoformat(),
        'user': user_message,
        'assistant': assistant_response
    }
    with open(CHAT_LOG_FILE, 'a') as f:
        f.write(json.dumps(chat_entry) + '\n')

def generate_response(prompt, max_tokens=100, temperature=0.8, top_k=200):
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

@app.route('/')
def index():
    """Render the chat UI (templates/index.html)."""
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    """Chat endpoint. Expects JSON {"message": str}. Returns model response and chat count."""
    data = request.json
    user_message = data.get('message', '')

    if not user_message:
        return jsonify({'error': 'No message provided'}), 400

    # Generate response
    prompt = f"Human: {user_message}\nAssistant:"
    response = generate_response(prompt, max_tokens=150, temperature=0.8)

    # Save the interaction
    save_chat(user_message, response)

    return jsonify({
        'response': response,
        'chat_count': sum(1 for _ in open(CHAT_LOG_FILE)) if os.path.exists(CHAT_LOG_FILE) else 0
    })

@app.route('/stats', methods=['GET'])
def stats():
    """Get statistics about collected chat data"""
    if not os.path.exists(CHAT_LOG_FILE):
        return jsonify({'chat_count': 0})

    chat_count = sum(1 for _ in open(CHAT_LOG_FILE))
    return jsonify({
        'chat_count': chat_count,
        'ready_for_finetuning': chat_count >= 10
    })

if __name__ == '__main__':
    print(f"\n{'='*60}")
    print(f"🚀 NanoGPT Chat Server Starting")
    print(f"{'='*60}")
    print(f"Device: {device}")
    print(f"Chat logs: {CHAT_LOG_FILE}")
    print(f"Access the app at: http://localhost:5000")
    print(f"{'='*60}\n")

    app.run(debug=True, host='0.0.0.0', port=5000)
