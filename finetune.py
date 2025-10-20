"""
Fine-tuning script for nanoGPT on collected chat data
Run this after collecting sufficient chat interactions
"""
import json
import os
import sys
import torch
import numpy as np
from datetime import datetime

# Add nanoGPT to path
sys.path.append('nanoGPT')
from model import GPT, GPTConfig
import tiktoken

# Configuration
CHAT_LOG_FILE = 'chat_history.jsonl'
DATA_DIR = 'data'
MODEL_DIR = 'models'

# Training hyperparameters
batch_size = 2
block_size = 128  # Reduced to handle smaller datasets
max_iters = 500  # Reduced iterations for small dataset
eval_interval = 50
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
eval_iters = 10  # Reduced for small dataset
dropout = 0.1

print(f"Using device: {device}")

def prepare_training_data():
    """Convert chat history to training format"""
    if not os.path.exists(CHAT_LOG_FILE):
        print(f"Error: {CHAT_LOG_FILE} not found. Collect some chats first!")
        return None

    # Read all chat interactions
    conversations = []
    with open(CHAT_LOG_FILE, 'r') as f:
        for line in f:
            chat = json.loads(line)
            # Format as conversational training data
            text = f"Human: {chat['user']}\nAssistant: {chat['assistant']}\n\n"
            conversations.append(text)

    if len(conversations) < 10:
        print(f"Warning: Only {len(conversations)} conversations found. Recommend collecting at least 50 for meaningful fine-tuning.")

    # Combine all conversations
    full_text = ''.join(conversations)

    # Create data directory
    os.makedirs(DATA_DIR, exist_ok=True)

    # Tokenize using GPT-2 tokenizer
    enc = tiktoken.get_encoding("gpt2")
    tokens = enc.encode(full_text)
    tokens_array = np.array(tokens, dtype=np.uint16)

    # Split into train and validation (90/10)
    split_idx = int(len(tokens_array) * 0.9)
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

def get_batch(split, train_data, val_data):
    """Generate a batch of training data"""
    data = train_data if split == 'train' else val_data
    max_start = len(data) - block_size - 1
    if max_start <= 0:
        raise ValueError(f"Dataset too small! Need at least {block_size + 1} tokens, but {split} set has only {len(data)} tokens. Collect more conversations or reduce block_size.")
    ix = torch.randint(max_start, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    if device != 'cpu':
        x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss(model, train_data, val_data):
    """Estimate loss on train and val sets"""
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split, train_data, val_data)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

def finetune():
    """Main fine-tuning function"""
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
        block_size=1024,
        vocab_size=50257,
        n_layer=12,
        n_head=12,
        n_embd=768,
        dropout=dropout,
        bias=True
    )
    model = GPT(config)
    model.load_state_dict(torch.load(nano_model_path, map_location=device))
    model.to(device)

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    print(f"\nStarting fine-tuning...")
    print(f"Batch size: {batch_size}")
    print(f"Block size: {block_size}")
    print(f"Max iterations: {max_iters}")
    print(f"Learning rate: {learning_rate}")
    print()

    # Training loop
    for iter in range(max_iters):
        # Evaluate periodically
        if iter % eval_interval == 0:
            losses = estimate_loss(model, train_data, val_data)
            print(f"Step {iter}/{max_iters}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

        # Show progress every 10 steps
        elif iter % 10 == 0:
            print(f"Step {iter}/{max_iters}...", end='\r')

        # Get batch and compute loss
        xb, yb = get_batch('train', train_data, val_data)
        logits, loss = model(xb, yb)

        # Backward pass
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    # Final evaluation
    losses = estimate_loss(model, train_data, val_data)
    print(f"\nFinal: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # Save the fine-tuned model
    os.makedirs(MODEL_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = os.path.join(MODEL_DIR, f'finetuned_{timestamp}.pt')

    print(f"\nSaving model to {model_path}...")
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': model.config,
        'iter': max_iters,
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
