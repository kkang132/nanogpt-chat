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
| `eval.py` | Evaluation pipeline: perplexity, generation quality, GSM8K | stable |
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

## Git Worktrees

Agents often work in worktree checkouts, not the main clone.

- **Branch naming**: Keep the branch name the tool generates (e.g. `air/dependency-management-…`). Don't rename it.
- **Path-aware commands**: Use paths relative to the worktree root or anchored to `__file__`. Never assume you're in the main checkout. `os.getcwd()` is fragile; `Path(__file__).parent` is correct.
- **No shared venvs**: Each worktree needs its own virtual environment. A venv from the main clone has baked-in paths that break in a worktree.
- **Merge target**: Worktree branches merge to `main`. Run `git log main..HEAD` to see what's diverged.
- **Cleanup**: After merging, run `git worktree remove <path>` and `git worktree prune`.

## Git Config

Do not modify `.git/config` without explicit user approval. This includes adding, changing, or removing remotes, user identity, credential helpers, or any other settings. Ask first.

Do not run `gh auth login`, `gh auth switch`, or any command that changes the authenticated GitHub identity. The authenticated user is `kkang132`. Verify with `gh auth status` if uncertain.

## Tool Reliability

Never use Bash subagents to read file contents — they can hallucinate output. Use the Read tool directly for any file where accuracy matters (especially `.git/config`, credentials, configuration). Reserve Bash for commands that *do* things, not commands that *read* things.

## Context Recovery

Read `docs/rl-roadmap.md`. Check `git log`. The supervised system is the fallback — it must always work.
