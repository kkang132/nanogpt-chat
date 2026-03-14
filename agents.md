# Agent Instructions

## Search

Use `embark search` first. Trust results ≥50%. Fall back to Glob/Grep otherwise.

```bash
embark search "query" --limit 10
```

## Protected Patterns

ast-grep rules in `.ast-grep/rules/` block changes to `GPTConfig` params, device selection, `torch.load()`, `gym` imports (use `gymnasium`), and public API signatures.

Pre-commit enforces this. Bypass: `git commit --no-verify`.

## File Map

| File | Role | Status |
|------|------|--------|
| `app.py` | Flask server, `:5000`, rate-limited | stable |
| `finetune.py` | Supervised training on `chat_history.jsonl` | stable |
| `download_dataset.py` | OpenAssistant + GSM8K → JSONL | stable |
| `rl/environment.py` | Gymnasium `ChatEnvironment` | stable |
| `rl/reward_model.py` | `SimpleRating`, `MultiCriteria`, `Learned` | stable |
| `rl/ppo_trainer.py` | PPO loop with value head | stable |

## Code Style

PEP 8, idiomatic Python. Use type hints on all function signatures. Config in `pyproject.toml` under `[tool.ruff]`.
Run `ruff check . && ruff format .` before committing. Pre-commit hooks enforce this.
Do not suppress lint warnings without a justifying comment.

## Commits

One logical change per commit. No "and" in commit messages — split instead.
Commit dependencies before dependents. Repo must build at every commit.
Run `ruff check . && ruff format .` per commit, not as a separate lint commit.

## Context Recovery

Read `docs/rl-roadmap.md`. Check `git log`. The supervised system is the fallback — it must always work.
