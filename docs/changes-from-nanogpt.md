# Changes from Karpathy's nanoGPT

This project began as a clone of [nanoGPT](https://github.com/karpathy/nanoGPT). What follows are the material differences. Not every modification is listed — the small ones accumulate in the usual way.

## Chat interface

Karpathy's nanoGPT is a training and sampling toolkit. There is no interactive frontend. This project wraps it in a Flask server with a web UI. You type a message, the model responds, the conversation is logged. The logging exists because the logs become training data.

## Fine-tuning pipeline

`finetune.py` reads the conversation logs, tokenizes them, and fine-tunes the model with early stopping and cosine learning rate decay. Karpathy's `train.py` is a general-purpose training script. This one is narrow — it knows about the chat format and produces timestamped checkpoints that the server picks up automatically on restart.

## Dataset bootstrapping

`download_dataset.py` seeds the conversation log with Q&A pairs from two Hugging Face datasets:

| Dataset | Examples | Purpose | License |
|---------|----------|---------|---------|
| [OpenAssistant/oasst1](https://huggingface.co/datasets/OpenAssistant/oasst1) | 1,500 | Conversational Q&A (English, highest-ranked replies) | CC-BY-4.0 |
| [openai/gsm8k](https://huggingface.co/datasets/openai/gsm8k) | 500 | Math reasoning | MIT |

The mix is weighted toward conversational data so the model learns to produce natural language answers. Karpathy's nanoGPT has no dataset preparation beyond the Shakespeare and OpenWebText examples.

## Security hardening

The original nanoGPT has no server, so security is not a concern there. This project binds to localhost only, disables Flask debug mode, restricts CORS, validates and limits input, rate-limits requests, rotates logs, sets security headers, and loads all checkpoints with `weights_only=True` to prevent arbitrary code execution during deserialization.

## Automatic checkpoint loading

The server detects the latest `models/finetuned_*.pt` on startup. No configuration change is needed after fine-tuning. Karpathy's `sample.py` requires you to specify the checkpoint path explicitly.

## RL components

`rl/environment.py` and `rl/reward_model.py` are the beginning of an RLHF extension. Nothing like this exists in the original. The PPO trainer and policy networks are not yet written.

## Test suite

Karpathy's nanoGPT has no tests. This project has 71, covering the Flask endpoints and their input validation, the fine-tuning data pipeline, the dataset cleaning utilities, the RL environment, and the reward models. The suite runs in 1.38 seconds with all heavy dependencies mocked out. A pre-commit hook runs the tests alongside the linter so that broken code cannot be committed without deliberate override. The full accounting of what is and is not tested is in `docs/testing.md`.

## Code protection

ast-grep rules guard critical patterns — model config, device logic, checkpoint loading, public APIs — from accidental modification. A pre-commit hook enforces them. Karpathy's nanoGPT has no such mechanism.
