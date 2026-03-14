# Agent Instructions

## Search

Use `embark search` first. Trust results above 50% similarity. Fall back to Glob/Grep below that.

```bash
embark search "query" --limit 10
```

## Protected Patterns

ast-grep rules in `.ast-grep/rules/` block changes to:
- `GPTConfig` parameters
- Device selection (`device = ...`)
- `torch.load()` calls
- `gym` imports (use `gymnasium`)
- Public API exports and function signatures

Pre-commit hook enforces this. Bypass: `git commit --no-verify`.

## File Map

| File | Role | Status |
|------|------|--------|
| `app.py` | Flask server, `127.0.0.1:5000`, rate-limited | stable |
| `finetune.py` | Supervised training on `chat_history.jsonl` | stable |
| `download_dataset.py` | GSM8K + CodeAlpaca → JSONL | stable |
| `rl/environment.py` | Gymnasium `ChatEnvironment` | implemented, tested |
| `rl/reward_model.py` | `SimpleRating`, `MultiCriteria`, `Learned` | implemented, tested |
| `rl/ppo_trainer.py` | PPO loop | **planned** (not yet created) |

## Code Style

All Python code must be **PEP 8 compliant** and **idiomatic (Pythonic)**.

- Run `ruff check .` before committing. Fix all errors.
- Run `ruff format .` to auto-format.
- Config is in `pyproject.toml` under `[tool.ruff]`.
- Pre-commit hooks enforce this automatically (`pre-commit install` to set up).

Key rules enforced:
- **PEP 8**: naming, spacing, line length (88), imports (`E`, `W`, `F`, `N`, `I`)
- **Idiomatic Python**: comprehensions over loops (`C4`), pathlib over os.path (`PTH`), modern syntax (`UP`), simplified expressions (`SIM`), clean returns (`RET`)

Do not suppress lint warnings without justification in a comment.

## Commits

Keep commits atomic — one logical change per commit.

- A commit should do **one thing**: fix a bug, add a function, update a config.
- If the commit message needs "and", split it into two commits.
- Order commits so the repo builds and passes tests at every point in history.
- Commit dependencies before dependents (e.g., add the utility function, then the code that calls it).
- Run `ruff check .` and `ruff format .` before each commit, not as a separate "lint fix" commit.

Bad: `Add PPO trainer and fix reward model bug and update requirements`
Good: three separate commits in dependency order.

## Context Recovery

Read `docs/rl-roadmap.md`. Check `git log`. The supervised system is the fallback. It must always work.
