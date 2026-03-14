# Model

## Architecture

GPT-2 small, implemented in `model.py` (vendored from Karpathy's [nanoGPT](https://github.com/karpathy/nanoGPT)).

| Parameter | Value |
|-----------|-------|
| Layers | 12 |
| Heads | 12 (64 dims each) |
| Embedding dim | 768 |
| Context length | 1024 tokens |
| Vocab size | 50,257 (GPT-2 BPE) |
| Parameters | ~124M |
| Attention | Flash Attention (PyTorch ≥2.0) with manual fallback |
| Activation | GELU |
| Norm | Pre-norm (LayerNorm with optional bias) |

Each block: LayerNorm → CausalSelfAttention → residual → LayerNorm → MLP (768→3072→768) → residual.

## Tokenization

Tiktoken with GPT-2 encoding. Special token: `<|endoftext|>` (ID 50256).

## Inference

Prompt format: `"Human: {message}\nAssistant:"`

Autoregressive generation with:
- Temperature: 0.8
- Top-k: 200
- Max tokens: 150
- Stop on newline

## Loading

All `torch.load()` calls use `weights_only=True`.

**Base model**: `models/gpt2_nano.pt` — bare state_dict, loaded into a fresh `GPT(GPTConfig(...))`.

**Fine-tuned model**: `models/finetuned_*.pt` — contains `model_state_dict`, `config`, loss metrics. Requires `torch.serialization.add_safe_globals([GPTConfig])` for safe deserialization.

The server auto-loads the latest fine-tuned checkpoint by filename sort. Falls back to base if none exist.

## Training

| Parameter | Value |
|-----------|-------|
| Batch size | 2 |
| Block size | 128 |
| Max iterations | 1000 |
| Learning rate | 3e-4 (AdamW) |
| LR schedule | Linear warmup (100 iters) → cosine decay to 3e-5 |
| Dropout | 0.1 (training only) |
| Early stopping | Patience 5 evals, min delta 0.001 |

Data: conversations from `chat_history.jsonl`, tokenized, split 90/10, saved as uint16 binaries.

Recommended minimum: 50 conversations. Below 10 triggers a warning.

## Device Support

Auto-selected at startup: CUDA → MPS → CPU.

| Device | Inference | Memory |
|--------|-----------|--------|
| CUDA | ~100ms | 2–4 GB |
| MPS | ~500ms | 2–4 GB |
| CPU | ~2–5s | ~2 GB |
