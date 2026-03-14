# NanoGPT Chat

This repository is best thought of as a local chat system built on top of Karpathy's `nanoGPT`, together with a disciplined route for improving it. The supervised path already works: the server runs locally, conversations are logged, ratings can be attached to responses, and those logs can be used to fine-tune a new checkpoint. The reinforcement-learning (RL) work is available, but it is not yet the engine of the project. It is an extension under construction.

It is worth being explicit at the outset about what the project is not. It is not a hosted service, not a wrapper around a remote API, and not a production RLHF system. Its present center is a local GPT-2 checkpoint and a loop of chat, log, rate, fine-tune, and reload — with a PPO training path now available alongside the supervised one.

## Running the System

The minimal path is straightforward:

```bash
source venv/bin/activate
pip install -r requirements.txt
python app.py
```

This serves the application at `http://127.0.0.1:5000`. On startup, `app.py` looks for the most recent file matching `models/finetuned_*.pt`. If one exists, it is loaded. If none exists, the application falls back to the local base checkpoint `models/gpt2_nano.pt`.

Two further commands are part of the normal working life of the repository:

```bash
python download_dataset.py
python finetune.py
```

The first constructs an initial `chat_history.jsonl`. The second turns that file into token binaries and fine-tunes a new model checkpoint. The RL-oriented tests can be run with:

```bash
pytest tests/
```

## How the Datasets Are Constructed

The script `download_dataset.py` constructs `chat_history.jsonl` from two specific Hugging Face datasets.

| Source | Count | Method | Purpose |
|------|------:|--------|---------|
| `OpenAssistant/oasst1` | 1,500 | keep English top-level prompts and pair each with the highest-ranked English assistant reply | conversational behaviour |
| `openai/gsm8k` | 500 | sample from the training split, remove calculator annotations, and normalise the final answer marker | mathematical reasoning |

The combined set is shuffled with a fixed seed and written in JSON Lines format. Bootstrap entries look like this:

```json
{"user": "...", "assistant": "..."}
```

Live conversations collected by `app.py` are written to the same file in a slightly richer form:

```json
{"id": "...", "timestamp": "...", "user": "...", "assistant": "...", "rating": null}
```

That common format is not an incidental convenience. It means that synthetic seed data and real interaction data feed the same supervised training pipeline.

## How Fine-Tuning Is Done

`finetune.py` is intentionally specific. It does not try to be a universal training script. Instead, it takes the conversation log as it exists in this project and converts it into a form that the local GPT-2 model can use directly.

The process is as follows.

1. Read `chat_history.jsonl`.
2. Discard negatively rated examples, that is, entries with `rating == 0`.
3. Render each remaining exchange as:

```text
Human: ...
Assistant: ...
```

4. Tokenise the concatenated text with the GPT-2 tokenizer from `tiktoken`.
5. Split the token stream `90/10` into training and validation segments.
6. Write `data/train.bin` and `data/val.bin` as `uint16` arrays.
7. Load the local base checkpoint `models/gpt2_nano.pt`.
8. Fine-tune with `AdamW`, a batch size of `2`, a block size of `128`, and a maximum of `1000` iterations.
9. Use linear warmup for `100` steps, then cosine learning-rate decay.
10. Evaluate every `50` iterations and stop early after `5` non-improving validation checks with `min_delta = 0.001`.
11. Save the result as `models/finetuned_YYYYMMDD_HHMMSS.pt`.

The important point is that the fine-tuning starts from a local GPT-2 base, not from an external service. The base of the current training loop is `models/gpt2_nano.pt`, loaded through `nanoGPT.model.GPT`.

## What the Training Loop Is, and Is Not

There is already a usable training loop, but it is supervised rather than reinforcement-based. In practical terms, the present loop is:

1. optionally bootstrap `chat_history.jsonl` with `python download_dataset.py`
2. run `python app.py`
3. collect conversations and ratings
4. run `python finetune.py`
5. restart the server so that it picks up the newest fine-tuned checkpoint

This loop is simple, but that simplicity is not a defect. It gives the project a stable baseline and a way of improving behaviour without confusing aspiration with implementation.

The repository also contains a PPO training path. `rl/environment.py`, `rl/reward_model.py`, and `rl/ppo_trainer.py` are all implemented and tested. The PPO trainer uses the GPT model directly as the policy — its logits define the action distribution — with a lightweight value head for advantage estimation. A frozen reference model provides a KL penalty to prevent the policy from drifting too far from the pretrained distribution. Checkpoints are saved in the same format as `finetune.py`, so the chat server can load them without modification. The supervised system remains the simpler and more reliable path; the PPO path is available for those who want to explore RL-based alignment.

## Tests and Build Status

The repository contains `82` tests under `tests/`, all passing as of 14 March 2026. The intended verification command is:

```bash
pytest tests/
```

The suite covers the Flask application endpoints and their input validation, the supervised fine-tuning pipeline, the dataset preparation utilities, the Gymnasium environment, the reward-model layer, and the PPO trainer. A full account of what is tested, what is not, and the reasoning behind those boundaries is in `docs/testing.md`.

## Agent-First Work in JetBrains Air

This repository is being worked on in an agent-first loop in JetBrains Air. That phrase can easily become vague, so it is better to make it concrete. The usual cycle is:

1. inspect the current code and documentation
2. recover context from `docs/rl-roadmap.md` and recent commits
3. make a narrow change
4. verify what can honestly be verified
5. update the documentation so that it matches the codebase

Multiple loops can be run in parallel. That way of working suits this project particularly well, because the project has a divided character. One part of it is settled enough to be used: the local chat application and supervised fine-tuning path. Another part is exploratory: the RL extension. An agent-first loop is useful precisely when one wants to improve the second without damaging the first.

## Structure

For the full layout, see `docs/code-structure.md`. The short map is below.

| Path | Purpose |
|------|---------|
| `app.py` | Flask server, inference, logging, ratings |
| `finetune.py` | supervised fine-tuning pipeline |
| `download_dataset.py` | dataset bootstrapper |
| `nanoGPT/model.py` | GPT-2 implementation from `nanoGPT` |
| `rl/` | Gymnasium environment, reward models, PPO trainer |
| `tests/` | test suite (endpoints, fine-tuning, datasets, RL) |
| `templates/index.html` | local chat interface |
| `.ast-grep/rules/` | code-protection rules |
| `pyproject.toml` | Ruff configuration |
| `.pre-commit-config.yaml` | pre-commit hooks |

## Guardrails

Critical patterns are protected with `ast-grep` rules under `.ast-grep/rules/`. In particular, the rules guard `GPTConfig` parameters, device selection, `torch.load()` calls, `gym` imports, and public API signatures. The pre-commit hook enforces those checks. If one wishes to bypass that safeguard, the bypass is explicit: `git commit --no-verify`.

## Further Reading

- `docs/changes-from-nanogpt.md`
- `docs/architecture.md`
- `docs/api.md`
- `docs/model.md`
- `docs/code-structure.md`
- `docs/rl-roadmap.md`
- `docs/testing.md`

## Credits

- `nanoGPT` by Andrej Karpathy
- GPT-2 by OpenAI

MIT license. See `nanoGPT/LICENSE`.
