# NanoGPT Chat

A local chat system whose policy is a GPT-2 checkpoint, originally from Karpathy's [nanoGPT](https://github.com/karpathy/nanoGPT). The model code has been vendored and the submodule dependency removed. nanoGPT is deprecated in favor of [nanochat](https://github.com/karpathy/nanochat).

## What's Different from nanoGPT

nanoGPT is a training and sampling toolkit. This project takes the model and builds a different thing around it.

| Area | What this project adds |
|------|----------------------|
| Serving | Flask chat server, web UI, conversation logging |
| Fine-tuning | Chat-aware pipeline, timestamped checkpoints, auto-reload |
| Data | Bootstrap from OpenAssistant + GSM8K |
| RL | Gymnasium environment, three reward models, PPO with KL regularisation |
| Tests | 126 across 7 files (nanoGPT has zero) |
| Security | Localhost-only, CORS, rate limiting, input validation, `weights_only=True` on all `torch.load()` |
| Code protection | `ast-grep` rules on model config, device logic, checkpoints, public API |

See [docs/changes-from-nanogpt.md](docs/changes-from-nanogpt.md).

## Quick Start

```bash
source .venv/bin/activate
pip install -r requirements.txt
python3 app.py
```

The server runs at `http://127.0.0.1:5000`. It loads the most recent `models/finetuned_*.pt`, falling back to `models/gpt2_nano.pt` if none exists.

## The Loop

The system improves through use:

1. **Bootstrap** — downloads seed data (skips if `chat_history.jsonl` exists)
2. **Fine-tune** — supervised training with early stopping → `models/finetuned_*.pt`
3. **Evaluate baseline** — scores the supervised checkpoint (val loss + generation quality)
4. **Serve & collect** — starts the chat server, sends synthetic prompts, stops the server
5. **RL fine-tune** — PPO training → `models/ppo_*.pt`
6. **Evaluate challenger** — scores the PPO checkpoint
7. **Compare & promote** — if the challenger improves on at least one metric without regressing on the other, it is copied to `models/best.pt`

`examples/full_loop.py` runs this complete cycle in one shot:

```bash
source .venv/bin/activate
pip install -r requirements.txt

python examples/full_loop.py --dry-run   # see what it does without executing
python examples/full_loop.py             # run the full loop (~3 min on MPS)
python examples/full_loop.py --full-eval # include GSM8K evaluation (slower)
python examples/full_loop.py --rl-steps 10  # fewer PPO steps for a quick test
```

## Fine-Tuning

`finetune.py` reads `chat_history.jsonl`, formats each exchange as `Human: … \n Assistant: …`, tokenises with `tiktoken`, and trains with AdamW. Batch size 2, block size 128, up to 1,000 iterations, cosine decay, early stopping after 5 non-improving checks. Checkpoints are written to `models/`.

The dataset schema is shared between bootstrap and live data.

## Agent-First Development

This project is developed in JetBrains Air (ADE) using multiple heterogeneous agents: Claude, Codex, Gemini, and Junie. They work in parallel, each in its own worktree. The project has a settled part (the chat server, supervised fine-tuning) and an exploratory part (the RL extension). The agents are arranged so that work on one does not disturb the other.

Agent instructions live in [agents.md](agents.md).

## Structure

| Path | What it does |
|------|-------------|
| `app.py` | Flask server: inference, logging, ratings |
| `finetune.py` | Supervised fine-tuning |
| `rl_finetune.py` | RL fine-tuning (PPO) entrypoint |
| `eval.py` | Evaluation pipeline: perplexity, generation quality, GSM8K |
| `download_dataset.py` | Dataset bootstrapper |
| `model.py` | GPT-2 model (vendored from nanoGPT) |
| `rl/` | Gymnasium env, reward models, PPO trainer |
| `examples/` | End-to-end example scripts |
| `tests/` | 126 tests across 7 files |
| `templates/index.html` | Chat UI |

## Docs

[architecture](docs/architecture.md) · [api](docs/api.md) · [model](docs/model.md) · [code structure](docs/code-structure.md) · [changes from nanoGPT](docs/changes-from-nanogpt.md) · [RL roadmap](docs/rl-roadmap.md) · [testing](docs/testing.md)

## Credits

GPT-2 model from [nanoGPT](https://github.com/karpathy/nanoGPT) by Andrej Karpathy (MIT). GPT-2 by OpenAI.
