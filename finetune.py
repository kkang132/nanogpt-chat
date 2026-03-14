"""
Fine-tuning script for nanoGPT on collected chat data
Run this after collecting sufficient chat interactions
"""
from __future__ import annotations

import json
import os
import sys
import torch
import numpy as np
from datetime import datetime

# Add nanoGPT to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'nanoGPT'))
from model import GPT, GPTConfig
import tiktoken

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Paths — anchored to the script directory
_SCRIPT_DIR = os.path.abspath(os.path.dirname(__file__))
CHAT_LOG_FILE = os.path.join(_SCRIPT_DIR, 'chat_history.jsonl')
DATA_DIR = os.path.join(_SCRIPT_DIR, 'data')
MODEL_DIR = os.path.join(_SCRIPT_DIR, 'models')

# Training hyperparameters
BATCH_SIZE = 2
BLOCK_SIZE = 128              # Reduced to handle smaller datasets
MAX_ITERS = 1000
EVAL_INTERVAL = 50
LEARNING_RATE = 3e-4
EVAL_ITERS = 10               # Reduced for small dataset
FINETUNE_DROPOUT = 0.1
WARMUP_ITERS = 100
MIN_LR_RATIO = 0.1            # Min LR is this fraction of LEARNING_RATE

# Data
MIN_CONVERSATIONS = 10
TRAIN_SPLIT = 0.9

# Early stopping
PATIENCE = 5                  # Stop if val loss doesn't improve for this many eval intervals
MIN_DELTA = 0.001             # Minimum change in val loss to count as improvement

# GPT-2 base model config
BASE_BLOCK_SIZE = 1024
BASE_VOCAB_SIZE = 50257
BASE_N_LAYER = 12
BASE_N_HEAD = 12
BASE_N_EMBD = 768
BASE_BIAS = True

# ---------------------------------------------------------------------------

device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

print(f"Using device: {device}")

def prepare_training_data() -> tuple[str, str, int, int] | None:
    """Convert chat history to training format."""
    if not os.path.exists(CHAT_LOG_FILE):
        print(f"Error: {CHAT_LOG_FILE} not found. Collect some chats first!")
        return None

    # Read all chat interactions, filtering out negatively-rated responses
    conversations: list[str] = []
    skipped_negative = 0
    with open(CHAT_LOG_FILE, 'r') as f:
        for line in f:
            chat = json.loads(line)
            # Skip negatively-rated responses; keep unrated (None) and positive (1)
            if chat.get('rating') == 0:
                skipped_negative += 1
                continue
            # Format as conversational training data
            text = f"Human: {chat['user']}\nAssistant: {chat['assistant']}\n\n"
            conversations.append(text)

    if skipped_negative > 0:
        print(f"Filtered out {skipped_negative} negatively-rated conversations")

    if len(conversations) < MIN_CONVERSATIONS:
        print(f"Warning: Only {len(conversations)} conversations found. Recommend collecting at least 50 for meaningful fine-tuning.")

    # Combine all conversations
    full_text = ''.join(conversations)

    # Create data directory
    os.makedirs(DATA_DIR, exist_ok=True)

    # Tokenize using GPT-2 tokenizer
    enc = tiktoken.get_encoding("gpt2")
    tokens = enc.encode(full_text)
    tokens_array = np.array(tokens, dtype=np.uint16)

    # Split into train and validation
    split_idx = int(len(tokens_array) * TRAIN_SPLIT)
    train_tokens = tokens_array[:split_idx]
    val_tokens = tokens_array[split_idx:]

    # Save as binary files
    train_file = os.path.join(DATA_DIR, 'train.bin')
    val_file = os.path.join(DATA_DIR, 'val.bin')

    train_tokens.tofile(train_file)
    val_tokens.tofile(val_file)

    print(f"Training data prepared:")
    print(f"  - Total conversations: {len(conversations)}")
    print(f"  - Total tokens: {len(tokens_array):,}")
    print(f"  - Training tokens: {len(train_tokens):,}")
    print(f"  - Validation tokens: {len(val_tokens):,}")
    print(f"  - Files saved to {DATA_DIR}/")

    return train_file, val_file, len(train_tokens), len(val_tokens)

def get_batch(
    split: str, train_data: np.memmap, val_data: np.memmap
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate a batch of training data."""
    data = train_data if split == 'train' else val_data
    max_start = len(data) - BLOCK_SIZE - 1
    if max_start <= 0:
        raise ValueError(f"Dataset too small! Need at least {BLOCK_SIZE + 1} tokens, but {split} set has only {len(data)} tokens. Collect more conversations or reduce block_size.")
    ix = torch.randint(max_start, (BATCH_SIZE,))
    x = torch.stack([torch.from_numpy((data[i:i+BLOCK_SIZE]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+BLOCK_SIZE]).astype(np.int64)) for i in ix])
    if device != 'cpu':
        x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss(
    model: GPT, train_data: np.memmap, val_data: np.memmap
) -> dict[str, torch.Tensor]:
    """Estimate loss on train and val sets."""
    out: dict[str, torch.Tensor] = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(EVAL_ITERS)
        for k in range(EVAL_ITERS):
            X, Y = get_batch(split, train_data, val_data)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

def finetune() -> None:
    """Main fine-tuning function."""
    print("\n" + "="*60)
    print("NanoGPT Fine-tuning")
    print("="*60 + "\n")

    # Prepare data
    result = prepare_training_data()
    if result is None:
        return

    train_file, val_file, train_len, val_len = result

    # Load data
    train_data = np.memmap(train_file, dtype=np.uint16, mode='r')
    val_data = np.memmap(val_file, dtype=np.uint16, mode='r')

    # Initialize model - load from local checkpoint
    print("\nLoading pretrained GPT-2 model from local file...")
    nano_model_path = os.path.join(MODEL_DIR, 'gpt2_nano.pt')

    config = GPTConfig(
        block_size=BASE_BLOCK_SIZE,
        vocab_size=BASE_VOCAB_SIZE,
        n_layer=BASE_N_LAYER,
        n_head=BASE_N_HEAD,
        n_embd=BASE_N_EMBD,
        dropout=FINETUNE_DROPOUT,
        bias=BASE_BIAS,
    )
    model = GPT(config)
    model.load_state_dict(torch.load(nano_model_path, map_location=device, weights_only=True))
    model.to(device)

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    # Cosine learning rate scheduler with warmup
    lr_decay_iters = MAX_ITERS
    min_lr = LEARNING_RATE * MIN_LR_RATIO

    def get_lr(it: int) -> float:
        # Linear warmup for WARMUP_ITERS steps
        if it < WARMUP_ITERS:
            return LEARNING_RATE * it / WARMUP_ITERS
        # Cosine decay down to min learning rate after warmup
        if it > lr_decay_iters:
            return min_lr
        decay_ratio = (it - WARMUP_ITERS) / (lr_decay_iters - WARMUP_ITERS)
        coeff = 0.5 * (1.0 + np.cos(np.pi * decay_ratio))  # coeff ranges 0..1
        return min_lr + coeff * (LEARNING_RATE - min_lr)

    print(f"\nStarting fine-tuning...")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Block size: {BLOCK_SIZE}")
    print(f"Max iterations: {MAX_ITERS}")
    print(f"Learning rate: {LEARNING_RATE} (with cosine decay to {min_lr})")
    print(f"Warmup iterations: {WARMUP_ITERS}")
    print(f"Early stopping: patience={PATIENCE}, min_delta={MIN_DELTA}")
    print()

    # Early stopping tracking
    best_val_loss = float('inf')
    patience_counter = 0
    early_stopped = False

    # Training loop
    for iter in range(MAX_ITERS):
        # Update learning rate
        lr = get_lr(iter)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # Evaluate periodically
        if iter % EVAL_INTERVAL == 0:
            losses = estimate_loss(model, train_data, val_data)
            val_loss = losses['val'].item()
            print(f"Step {iter}/{MAX_ITERS}: train loss {losses['train']:.4f}, val loss {val_loss:.4f}, lr {lr:.6f}")

            # Early stopping check
            if val_loss < best_val_loss - MIN_DELTA:
                best_val_loss = val_loss
                patience_counter = 0
                print(f"  → New best val loss: {best_val_loss:.4f}")
            else:
                patience_counter += 1
                print(f"  → No improvement ({patience_counter}/{PATIENCE})")

                if patience_counter >= PATIENCE:
                    print(f"\nEarly stopping triggered at iteration {iter}")
                    print(f"Best validation loss: {best_val_loss:.4f}")
                    early_stopped = True
                    break

        # Show progress every 10 steps
        elif iter % 10 == 0:
            print(f"Step {iter}/{MAX_ITERS}...", end='\r')

        # Get batch and compute loss
        xb, yb = get_batch('train', train_data, val_data)
        logits, loss = model(xb, yb)

        # Backward pass
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    # Final evaluation
    if not early_stopped:
        losses = estimate_loss(model, train_data, val_data)
        print(f"\nFinal: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
    else:
        print(f"\nTraining stopped early. Best val loss: {best_val_loss:.4f}")

    # Save the fine-tuned model
    os.makedirs(MODEL_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = os.path.join(MODEL_DIR, f'finetuned_{timestamp}.pt')

    print(f"\nSaving model to {model_path}...")
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': model.config,
        'iter': MAX_ITERS,
        'train_loss': losses['train'].item(),
        'val_loss': losses['val'].item(),
    }, model_path)

    print("\n" + "="*60)
    print("Fine-tuning complete!")
    print(f"Model saved to: {model_path}")
    print("="*60 + "\n")
    print("To use the fine-tuned model, update app.py to load this checkpoint.")

if __name__ == '__main__':
    finetune()
