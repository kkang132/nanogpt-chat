# Code Structure

## Layout

```
nanogpt-chat/
├── app.py                    # Flask server, inference, logging
├── finetune.py               # Fine-tuning pipeline
├── eval.py                   # Evaluation pipeline (perplexity, generation, GSM8K)
├── download_dataset.py       # Dataset bootstrapper
├── agents.md                 # Agent instructions (canonical)
├── CLAUDE.md                 # Delegates to agents.md
├── requirements.txt          # Dependencies
├── sgconfig.yml              # ast-grep config
├── model.py                  # GPT-2 implementation (vendored from nanoGPT)
├── templates/
│   └── index.html            # Chat UI
├── static/
│   ├── css/styles.css        # Extracted styles
│   └── js/app.js             # Extracted client JS
├── rl/
│   ├── __init__.py
│   ├── environment.py        # Gymnasium ChatEnvironment
│   ├── reward_model.py       # Reward scoring models
│   └── ppo_trainer.py        # PPO training with value head
├── tests/
│   ├── test_app.py
│   ├── test_download_dataset.py
│   ├── test_environment.py
│   ├── test_eval.py
│   ├── test_finetune.py
│   ├── test_ppo_trainer.py
│   └── test_reward_model.py
├── models/                   # Checkpoints (gitignored)
├── data/                     # Token binaries (gitignored)
├── .ast-grep/rules/          # 6 protection rules
├── pyproject.toml            # Ruff linter/formatter config
├── .pre-commit-config.yaml   # Pre-commit hooks (Ruff)
└── docs/
```

## Modules

### app.py

Flask application. Routes: `GET /`, `POST /chat`, `POST /rate`, `GET /stats`. Loads model at import time — tries latest `finetuned_*.pt` or `ppo_*.pt`, falls back to `gpt2_nano.pt`. Generation uses tiktoken, temperature sampling, top-k. Conversations appended to `chat_history.jsonl` with rotation. Security details in [api.md](api.md).

### finetune.py

Reads `chat_history.jsonl`. Formats as `"Human: ...\nAssistant: ...\n\n"`. Tokenizes, splits 90/10, saves as `data/{train,val}.bin`. Trains with AdamW, cosine LR schedule (warmup 100 iters, decay to 3e-5), early stopping (patience 5). Saves to `models/finetuned_YYYYMMDD_HHMMSS.pt`.

### eval.py

Evaluation pipeline. Discovers checkpoints in `models/`, then runs three suites: perplexity (validation loss on held-out data), generation quality (scoring length, repetition, coherence, and format of model responses), and GSM8K accuracy (math reasoning). Results are formatted as a table and appended to `eval/eval_results.jsonl`.

### download_dataset.py

Downloads OpenAssistant oasst1 (conversation, 1500 examples) and GSM8K (math, 500 examples) via Hugging Face `datasets`. Filters OpenAssistant for English top-level prompts with the highest-ranked assistant reply. Cleans, combines, shuffles, writes to `chat_history.jsonl`. Licenses: CC-BY-4.0 (OpenAssistant), MIT (GSM8K).

### model.py

GPT-2 small, vendored from Karpathy's [nanoGPT](https://github.com/karpathy/nanoGPT). Details in [model.md](model.md). Classes: `LayerNorm`, `CausalSelfAttention`, `MLP`, `Block`, `GPTConfig`, `GPT`.

### rl/environment.py

`ChatEnvironment(gymnasium.Env)`. State: token history (Box, 512). Action: next token (Discrete). Includes `MockModel` and `MockTokenizer` for testing.

### rl/reward_model.py

ABC `RewardModel` with `score_response()` and `update_from_feedback()`. Implementations: `SimpleRatingReward` (binary feedback), `MultiCriteriaReward` (weighted: relevance, helpfulness, safety, coherence), `LearnedRewardModel` (placeholder). Factory: `create_reward_model()`.

### rl/ppo_trainer.py

PPO (Proximal Policy Optimization) training loop. Uses the GPT model's logits as the policy, adds a lightweight `ValueHead` for advantage estimation, and keeps a frozen reference model for KL penalty. Classes: `PPOConfig`, `ValueHead`, `PPOTrainer`. Saves checkpoints to `models/ppo_*.pt`.

### templates/index.html

Single-page chat UI. Vanilla JS. Posts to `/chat`, displays messages, fetches `/stats` on load.

## Data Formats

**JSONL** (`chat_history.jsonl`):
```json
{"id": "uuid", "timestamp": "...", "user": "...", "assistant": "...", "rating": null}  // from app.py (live chat)
{"user": "...", "assistant": "..."}                                                     // from download_dataset.py (bootstrap)
```

`rating`: `null` = unrated, `1` = thumbs up, `0` = thumbs down. Negatively-rated entries are excluded during fine-tuning.

**Base checkpoint** (`gpt2_nano.pt`): bare `state_dict`.

**Fine-tuned checkpoint** (`finetuned_*.pt`):
```python
{"model_state_dict": ..., "config": GPTConfig, "iter": int, "train_loss": float, "val_loss": float}
```

**Token binaries** (`train.bin`, `val.bin`): raw uint16 arrays, memory-mapped.
