# Model Documentation

## GPT-2 Architecture

This project uses a GPT-2 "small" architecture with the following specifications:

### Model Configuration

```python
GPTConfig(
    block_size=1024,      # Maximum sequence length
    vocab_size=50257,     # GPT-2 vocabulary size
    n_layer=12,           # Number of transformer blocks
    n_head=12,            # Number of attention heads
    n_embd=768,           # Embedding dimension
    dropout=0.0,          # Dropout rate (inference)
    bias=True             # Use bias in linear layers
)
```

**Total Parameters**: ~124 million

### Architecture Components

#### 1. Token Embeddings
- **Vocabulary**: 50,257 tokens (GPT-2 BPE vocabulary)
- **Embedding Dimension**: 768
- **Position Embeddings**: Learned positional encodings up to 1024 positions

#### 2. Transformer Blocks (×12)

Each block consists of:

**Causal Self-Attention**:
- 12 attention heads (64 dimensions per head)
- Masked attention to prevent looking at future tokens
- Flash Attention when PyTorch ≥2.0 (faster GPU computation)
- Fallback to manual attention implementation

**Feed-Forward Network**:
- Two linear layers: 768 → 3072 → 768
- GELU activation
- Dropout for regularization

**Layer Normalization**:
- Applied before attention and MLP (pre-norm architecture)
- Optional bias parameter

**Residual Connections**:
- Skip connections around attention and MLP

#### 3. Output Layer
- Final layer normalization
- Linear projection to vocabulary size (768 → 50257)
- No softmax applied (returns raw logits)

### Attention Mechanism

#### Flash Attention (Preferred)
When available, uses PyTorch's `scaled_dot_product_attention`:
```python
y = torch.nn.functional.scaled_dot_product_attention(
    q, k, v,
    attn_mask=None,
    dropout_p=dropout,
    is_causal=True
)
```

Benefits:
- Fused CUDA kernels
- Reduced memory usage
- Faster inference and training

#### Manual Attention (Fallback)
Standard scaled dot-product attention:
```python
scores = (Q @ K^T) / sqrt(d_k)
scores = masked_fill(causal_mask)
attn = softmax(scores)
output = attn @ V
```

## Tokenization

### Tiktoken (GPT-2 BPE)

Uses OpenAI's tiktoken library with GPT-2 encoding:

```python
import tiktoken
enc = tiktoken.get_encoding("gpt2")

# Encoding
tokens = enc.encode("Hello, world!")
# [15496, 11, 995, 0]

# Decoding
text = enc.decode([15496, 11, 995, 0])
# "Hello, world!"
```

**Special Tokens**:
- `<|endoftext|>`: End of document token (ID: 50256)
- Used to separate different training examples

## Inference Process

### Generation Pipeline

1. **Prompt Formatting**
   ```python
   prompt = f"Human: {user_message}\nAssistant:"
   ```

2. **Tokenization**
   ```python
   tokens = enc.encode(prompt)
   tokens = torch.tensor(tokens, device=device)[None, ...]
   ```

3. **Autoregressive Generation**
   ```python
   for _ in range(max_tokens):
       # Crop to block_size if needed
       idx_cond = tokens[:, -block_size:]

       # Forward pass
       logits, _ = model(idx_cond)
       logits = logits[:, -1, :] / temperature

       # Top-k sampling
       v, _ = torch.topk(logits, top_k)
       logits[logits < v[:, [-1]]] = -float('Inf')

       # Sample next token
       probs = F.softmax(logits, dim=-1)
       idx_next = torch.multinomial(probs, num_samples=1)

       # Append and check for stop
       tokens = torch.cat((tokens, idx_next), dim=1)
       if idx_next[0] == newline_token:
           break
   ```

4. **Decoding**
   ```python
   generated_text = enc.decode(tokens[0].tolist())
   response = generated_text[len(prompt):].strip()
   ```

### Generation Parameters

#### Temperature (`temperature=0.8`)
Controls randomness of sampling:
- **Low (0.1-0.5)**: More deterministic, focused responses
- **Medium (0.6-0.9)**: Balanced creativity and coherence
- **High (1.0+)**: More creative but potentially less coherent

Logits are divided by temperature before softmax:
```python
logits = logits / temperature
```

#### Top-k Sampling (`top_k=200`)
Limits sampling to the k most probable tokens:
- Prevents sampling from very low-probability tokens
- Reduces nonsensical outputs
- Default: 200 tokens

#### Max Tokens (`max_tokens=150`)
Maximum number of tokens to generate:
- Prevents infinite generation
- For `/chat` endpoint: 150 tokens
- Stops early if newline is generated

### Stopping Conditions

Generation stops when:
1. `max_tokens` is reached, OR
2. A newline character is generated

## Training Process

### Data Preparation

1. **Load Conversations**
   ```python
   conversations = []
   with open('chat_history.jsonl', 'r') as f:
       for line in f:
           chat = json.loads(line)
           text = f"Human: {chat['user']}\nAssistant: {chat['assistant']}\n\n"
           conversations.append(text)
   ```

2. **Tokenization**
   ```python
   full_text = ''.join(conversations)
   tokens = enc.encode(full_text)
   tokens_array = np.array(tokens, dtype=np.uint16)
   ```

3. **Train/Val Split**
   - 90% training, 10% validation
   - Saved as binary files (`.bin`)

### Training Configuration

```python
batch_size = 2           # Small batch for limited data
block_size = 128         # Sequence length per example
max_iters = 1000         # Maximum training steps
learning_rate = 3e-4     # AdamW learning rate
dropout = 0.1            # Regularization during training
```

### Learning Rate Schedule

**Warmup + Cosine Decay**:

```python
def get_lr(iter):
    if iter < 100:  # Warmup
        return learning_rate * iter / 100

    # Cosine decay
    decay_ratio = (iter - 100) / (max_iters - 100)
    coeff = 0.5 * (1.0 + cos(π * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)
```

- **Warmup**: 100 iterations (0 → 3e-4)
- **Decay**: Cosine from 3e-4 → 3e-5
- **Min LR**: 10% of max (3e-5)

### Early Stopping

Monitors validation loss to prevent overfitting:

```python
patience = 5              # Number of eval intervals
min_delta = 0.001         # Minimum improvement threshold
```

Training stops if validation loss doesn't improve by at least `min_delta` for `patience` consecutive evaluations (5 × 50 = 250 iterations).

### Training Loop

1. **Forward Pass**: Compute loss on training batch
2. **Backward Pass**: Compute gradients
3. **Optimization**: AdamW update
4. **Evaluation** (every 50 steps):
   - Compute train and validation loss
   - Check early stopping criteria
   - Print progress

### Checkpoint Format

```python
{
    'model_state_dict': model.state_dict(),
    'config': model.config,
    'iter': current_iteration,
    'train_loss': final_train_loss,
    'val_loss': final_val_loss
}
```

Saved to: `models/finetuned_YYYYMMDD_HHMMSS.pt`

## Model Loading

### Base Model
```python
config = GPTConfig(...)
model = GPT(config)
model.load_state_dict(torch.load('models/gpt2_nano.pt'))
model.eval()
model.to(device)
```

### Fine-tuned Model
```python
# Unpickling shim for compatibility
import sys
sys.modules['model'] = sys.modules['nanoGPT.model']

checkpoint = torch.load('models/finetuned_*.pt', weights_only=False)
config = checkpoint['config']
model = GPT(config)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
model.to(device)
```

**Note**: The unpickling shim is required because fine-tuned checkpoints were saved with a different module structure.

## Device Support

### Automatic Device Selection

```python
device = (
    'cuda' if torch.cuda.is_available() else
    'mps' if torch.backends.mps.is_available() else
    'cpu'
)
```

### Performance Characteristics

| Device | Inference Speed | Training Speed | Memory |
|--------|----------------|----------------|--------|
| CUDA GPU | Fast (~100ms) | Fast | 2-4 GB |
| Apple MPS | Medium (~500ms) | Medium | 2-4 GB |
| CPU | Slow (~2-5s) | Very Slow | 2 GB |

### Memory Requirements

- **Model**: ~500 MB (FP32)
- **KV Cache**: Depends on sequence length
- **Activations**: Depends on batch size and sequence length

Minimum 2 GB GPU/Unified Memory recommended for smooth inference.

## Fine-tuning Best Practices

### Data Requirements
- **Minimum**: 10 conversations (warning threshold)
- **Recommended**: 50+ conversations for meaningful learning
- **Optimal**: 200+ conversations for good adaptation

### When to Fine-tune
- After collecting sufficient diverse conversations
- When base model responses are generic or off-topic
- To adapt to specific domain or conversational style

### Expected Results
- Fine-tuning on small datasets (10-50 examples) provides mild adaptation
- Larger datasets (100+) enable stronger personalization
- Overfitting risk is high with very small datasets

### Monitoring Training
Watch for:
- **Train loss decreasing**: Model is learning
- **Val loss plateauing**: Potential overfitting
- **Early stopping**: Automatically prevents overfitting
- **Loss diverging**: Increase `min_delta` or reduce `learning_rate`

## Limitations

1. **Context Length**: Maximum 1024 tokens
2. **Vocabulary**: Fixed GPT-2 vocabulary (can't handle unknown languages)
3. **Knowledge Cutoff**: Pretrained model knowledge up to GPT-2 training date
4. **Reasoning**: Limited multi-step reasoning capabilities
5. **Factuality**: May generate plausible but incorrect information
