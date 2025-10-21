# Code Structure

## Project Layout

```
nanogpt-chat/
├── app.py                    # Flask web server and chat endpoints
├── finetune.py              # Model fine-tuning script
├── chat_history.jsonl       # Collected conversations (generated)
├── README.md                # Project documentation
├── .gitignore              # Git ignore rules
│
├── nanoGPT/                # GPT model implementation
│   └── model.py            # Transformer architecture
│
├── templates/              # Flask HTML templates
│   └── index.html          # Chat web interface
│
├── models/                 # Model checkpoints (gitignored)
│   ├── gpt2_nano.pt       # Base pretrained model
│   └── finetuned_*.pt     # Fine-tuned checkpoints
│
├── data/                   # Training data (generated, gitignored)
│   ├── train.bin          # Training tokens
│   └── val.bin            # Validation tokens
│
├── venv/                   # Python virtual environment
│
└── docs/                   # Documentation
    ├── architecture.md
    ├── code-structure.md
    ├── model.md
    └── api.md
```

## Module Descriptions

### `app.py` - Web Server & Inference

**Purpose**: Main Flask application serving chat interface and handling model inference.

**Key Components**:

#### Configuration (lines 12-20)
- `CHAT_LOG_FILE`: Path to conversation history
- `MODEL_DIR`: Directory for model checkpoints
- Device selection (CUDA → MPS → CPU)

#### Model Loading (lines 21-53)
- Attempts to load fine-tuned model first
- Falls back to base GPT-2 if not found
- Applies unpickling shim for checkpoint compatibility
- Moves model to selected device and sets to eval mode

#### Core Functions
- `save_chat(user_message, assistant_response)` (lines 58-66)
  - Appends conversation to JSONL file
  - Includes ISO timestamp

- `generate_response(prompt, max_tokens, temperature, top_k)` (lines 68-104)
  - Tokenizes input with tiktoken
  - Runs autoregressive generation
  - Applies temperature scaling and top-k sampling
  - Stops at newline or max tokens
  - Returns decoded text

#### Flask Routes
- `GET /` (lines 106-109): Serves chat UI
- `POST /chat` (lines 111-130): Handles chat messages
  - Formats prompt as "Human: ... Assistant:"
  - Generates response
  - Saves interaction
  - Returns response + chat count
- `GET /stats` (lines 132-142): Returns chat statistics
  - Total conversation count
  - Ready-for-finetuning flag (≥10 chats)

### `finetune.py` - Model Training

**Purpose**: Fine-tunes GPT-2 on collected chat data.

**Key Components**:

#### Configuration (lines 17-35)
- Training hyperparameters (batch size, learning rate, iterations)
- Device selection
- Early stopping parameters (patience, min_delta)

#### Data Pipeline
- `prepare_training_data()` (lines 38-86)
  - Loads JSONL conversations
  - Formats as "Human: ... Assistant: ..." strings
  - Tokenizes with tiktoken
  - Splits 90/10 train/val
  - Saves as uint16 binary files
  - Returns file paths and token counts

- `get_batch(split, train_data, val_data)` (lines 88-99)
  - Randomly samples batch_size sequences
  - Each sequence is block_size tokens
  - Creates input (x) and target (y) tensors
  - Moves to device

#### Training Infrastructure
- `estimate_loss(model, train_data, val_data)` (lines 101-114)
  - Evaluates on train and validation sets
  - Averages loss over eval_iters batches
  - Temporarily sets model to eval mode

- `get_lr(it)` (lines 158-167)
  - Learning rate schedule function
  - Linear warmup for first 100 iterations
  - Cosine decay to min_lr (10% of max)

#### Training Loop (lines 116-249)
- Loads base GPT-2 checkpoint
- Creates AdamW optimizer
- Iterates up to max_iters:
  - Updates learning rate
  - Evaluates every eval_interval steps
  - Checks early stopping criteria
  - Performs backward pass and optimization
- Saves final checkpoint with metadata

### `nanoGPT/model.py` - Transformer Architecture

**Purpose**: Complete GPT-2 implementation in PyTorch.

**Key Classes**:

#### `LayerNorm` (lines 18-27)
- Custom layer normalization with optional bias
- Used throughout transformer blocks

#### `CausalSelfAttention` (lines 29-76)
- Multi-head causal self-attention
- Supports Flash Attention (PyTorch ≥2.0)
- Falls back to manual attention implementation
- Projects to Q, K, V and computes scaled dot-product attention

#### `MLP` (lines 78-92)
- Feed-forward network
- Linear → GELU → Linear → Dropout
- 4x hidden dimension expansion

#### `Block` (lines 94+)
- Transformer block combining:
  - Layer norm → Attention → Residual
  - Layer norm → MLP → Residual

#### `GPTConfig` (dataclass)
- Configuration for model architecture
- Parameters: vocab_size, n_layer, n_head, n_embd, block_size, dropout, bias

#### `GPT` (main model class)
- Token and position embeddings
- Stack of transformer blocks
- Final layer norm and language model head
- Forward pass with optional targets for training
- Generation method for inference

### `templates/index.html` - Chat Interface

**Purpose**: Single-page web UI for chat interactions.

**Structure**:
- **CSS** (lines 7-164): Modern, gradient-styled chat interface
- **HTML** (lines 166-198): Chat container, message area, input box
- **JavaScript** (lines 200-268):
  - `addMessage()`: Appends messages to chat
  - `sendMessage()`: Sends POST to `/chat` endpoint
  - Event listeners for send button and Enter key
  - Loads initial stats on page load

**Features**:
- Responsive design
- Loading animation during generation
- Automatic scroll to latest message
- Displays conversation count

## Data Formats

### `chat_history.jsonl`
JSON Lines format (one JSON object per line):
```json
{"timestamp": "2025-10-21T16:30:00.123456", "user": "What is AI?", "assistant": "AI is..."}
```

### Model Checkpoints

**Base model** (`gpt2_nano.pt`):
```python
torch.load() → state_dict (OrderedDict)
```

**Fine-tuned checkpoint** (`finetuned_*.pt`):
```python
{
    'model_state_dict': OrderedDict,  # Model weights
    'config': GPTConfig,              # Architecture config
    'iter': int,                      # Training iterations
    'train_loss': float,              # Final train loss
    'val_loss': float                 # Final validation loss
}
```

### Training Data (`train.bin`, `val.bin`)
Raw binary files of uint16 token IDs, memory-mapped for efficient loading.

## Dependencies

**Core**:
- `torch`: PyTorch for model and training
- `flask`: Web framework
- `flask-cors`: CORS support
- `tiktoken`: GPT-2 tokenizer
- `numpy`: Array operations

**Standard Library**:
- `json`: JSONL parsing
- `os`: File operations
- `datetime`: Timestamps
- `math`, `inspect`, `dataclasses`: Utilities

## Entry Points

### Development Server
```bash
python app.py
# Starts Flask on http://0.0.0.0:5000
```

### Fine-tuning
```bash
python finetune.py
# Trains on chat_history.jsonl
# Saves checkpoint to models/
```

## Configuration

Most configuration is hardcoded in the respective files:
- `app.py`: Model path, generation parameters (temp, top_k, max_tokens)
- `finetune.py`: Training hyperparameters, early stopping settings

To modify behavior, edit the configuration sections at the top of each file.
