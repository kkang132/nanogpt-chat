# Architecture Overview

## System Design

NanoGPT Chat is a Flask-based web application that provides a conversational interface to a GPT-2 language model with fine-tuning capabilities. The system collects user conversations, uses them to fine-tune the model, and serves an improved chat experience.

## High-Level Components

```
┌─────────────────────────────────────────────────────────┐
│                      Web Browser                        │
│                   (templates/index.html)                │
└────────────────────┬────────────────────────────────────┘
                     │ HTTP/REST
                     ▼
┌─────────────────────────────────────────────────────────┐
│                   Flask Web Server                      │
│                      (app.py)                           │
│  ┌─────────────┐  ┌──────────────┐  ┌──────────────┐  │
│  │   Routes    │  │  Generation  │  │   Logging    │  │
│  │  /chat      │→ │   Pipeline   │→ │chat_history  │  │
│  │  /stats     │  │              │  │    .jsonl    │  │
│  └─────────────┘  └──────────────┘  └──────────────┘  │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│                 GPT-2 Model Layer                       │
│                  (nanoGPT/model.py)                     │
│  ┌──────────────────────────────────────────────────┐  │
│  │  Transformer Architecture                        │  │
│  │  • 12 layers, 12 heads, 768 embedding dim        │  │
│  │  • Causal self-attention with Flash Attention   │  │
│  │  • Layer normalization & dropout                │  │
│  └──────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│              Fine-tuning Pipeline                       │
│                   (finetune.py)                         │
│  ┌──────────────┐  ┌──────────────┐  ┌─────────────┐  │
│  │ Data Prep    │→ │  Training    │→ │   Export    │  │
│  │  JSONL→bin   │  │  AdamW+LR    │  │  Checkpoint │  │
│  │              │  │  Scheduler   │  │             │  │
│  └──────────────┘  └──────────────┘  └─────────────┘  │
└─────────────────────────────────────────────────────────┘
```

## Data Flow

### 1. Chat Interaction Flow
```
User Input → Flask /chat endpoint → Tokenization (tiktoken)
           → Model Generation (with temperature/top-k sampling)
           → Response → Save to JSONL → Return to User
```

### 2. Fine-tuning Flow
```
chat_history.jsonl → Data Preparation (tokenization)
                   → Train/Val Split (90/10)
                   → Training Loop (AdamW + Cosine LR)
                   → Early Stopping
                   → Save Checkpoint
                   → Load in app.py for inference
```

## Key Design Decisions

### Device Selection
The application automatically selects the best available compute device:
1. CUDA GPU (if available)
2. Apple Silicon MPS (if available)
3. CPU (fallback)

### Model Loading Strategy
- **Base Model**: Loads `models/gpt2_nano.pt` (standard GPT-2 small)
- **Fine-tuned Model**: Automatically detects and loads `models/finetuned_*.pt` if present
- Module shim (`sys.modules['model']`) handles unpickling compatibility

### Generation Strategy
- **Tokenization**: Uses tiktoken GPT-2 encoder
- **Sampling**: Top-k sampling with temperature control
- **Context Window**: 1024 tokens (block_size)
- **Stopping**: Generates until newline or max_tokens reached

### Training Strategy
- **Optimizer**: AdamW with learning rate 3e-4
- **LR Schedule**: Linear warmup (100 iters) + Cosine decay
- **Early Stopping**: Patience of 5 eval intervals, min delta 0.001
- **Regularization**: Dropout 0.1 during training

## Storage

### Chat History (`chat_history.jsonl`)
JSON Lines format storing all conversations:
```json
{"timestamp": "2025-10-21T10:30:00", "user": "Hello", "assistant": "Hi there!"}
```

### Model Checkpoints (`models/`)
- `gpt2_nano.pt`: Base pretrained model (state dict only)
- `finetuned_YYYYMMDD_HHMMSS.pt`: Fine-tuned checkpoints (full checkpoint with config and loss metrics)

### Training Data (`data/`)
- `train.bin`: Training tokens (uint16 binary)
- `val.bin`: Validation tokens (uint16 binary)

## Scalability Considerations

### Current Limitations
- Single-threaded Flask server (development mode)
- In-memory model loading (no model server)
- Local file-based storage
- No authentication or user management

### Production Considerations
- Use production WSGI server (gunicorn, uwsgi)
- Implement request queuing for concurrent inference
- Add caching layer for common queries
- Implement conversation history per user
- Add rate limiting and API authentication
