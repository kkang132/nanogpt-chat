#!/usr/bin/env python3
"""
Full improvement loop for nanogpt-chat.

Demonstrates the complete cycle: bootstrap data, fine-tune a supervised
checkpoint, evaluate it, run RL to produce a challenger, evaluate again,
and promote the winner.

Usage:
    python examples/full_loop.py             # run the full loop
    python examples/full_loop.py --dry-run   # print what would happen
    python examples/full_loop.py --full-eval # include GSM8K (slow)

Each step calls the existing project scripts via subprocess so no logic
is duplicated.  Run from the project root.
"""

from __future__ import annotations

import argparse
import json
import shutil
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths (relative to project root)
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EVAL_RESULTS = PROJECT_ROOT / "eval" / "eval_results.jsonl"
MODEL_DIR = PROJECT_ROOT / "models"
BEST_CHECKPOINT = MODEL_DIR / "best.pt"

PYTHON = sys.executable  # use the same interpreter that launched us

# Synthetic prompts sent to the server during the "serve" step
SYNTHETIC_PROMPTS = [
    "What is the capital of Japan?",
    "Explain recursion in one sentence.",
    "Tell me a fun fact about octopuses.",
    "How does gravity work?",
    "Write a short poem about rain.",
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def banner(step: int, title: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  Step {step}: {title}")
    print(f"{'=' * 60}\n")


def run(cmd: list[str], *, dry_run: bool = False) -> None:
    """Run a command, printing it first.  In dry-run mode, just print."""
    display = " ".join(cmd)
    if dry_run:
        print(f"  [dry-run] {display}")
        return
    print(f"  $ {display}")
    result = subprocess.run(cmd, cwd=PROJECT_ROOT)
    if result.returncode != 0:
        print(f"  !! Command exited with code {result.returncode}")
        sys.exit(result.returncode)


def find_latest(pattern: str) -> str | None:
    """Return the latest checkpoint matching *pattern* in models/."""
    matches = sorted(MODEL_DIR.glob(pattern))
    return str(matches[-1]) if matches else None


def read_last_n_results(n: int) -> list[dict]:
    """Read the last *n* eval results from the JSONL file."""
    if not EVAL_RESULTS.exists():
        return []
    lines = EVAL_RESULTS.read_text().strip().splitlines()
    return [json.loads(line) for line in lines[-n:]]


# ---------------------------------------------------------------------------
# Steps
# ---------------------------------------------------------------------------


def step_bootstrap(dry_run: bool) -> None:
    banner(1, "Bootstrap data")
    chat_log = PROJECT_ROOT / "chat_history.jsonl"
    if chat_log.exists() and not dry_run:
        count = sum(1 for _ in chat_log.open())
        print(f"  chat_history.jsonl already exists ({count} entries), skipping.")
        return
    run([PYTHON, "download_dataset.py"], dry_run=dry_run)


def step_finetune(dry_run: bool) -> None:
    banner(2, "Supervised fine-tuning")
    run([PYTHON, "finetune.py"], dry_run=dry_run)


def step_evaluate(
    checkpoint: str | None,
    *,
    dry_run: bool,
    full_eval: bool,
    label: str,
) -> None:
    banner_num = 3 if label == "baseline" else 6
    banner(banner_num, f"Evaluate ({label})")
    cmd = [PYTHON, "eval.py"]
    if checkpoint:
        cmd += ["--checkpoint", checkpoint]
    if not full_eval:
        cmd += ["--skip-gsm8k"]
    run(cmd, dry_run=dry_run)


def step_serve_and_collect(dry_run: bool) -> None:
    banner(4, "Serve & collect synthetic chats")
    if dry_run:
        print("  [dry-run] Start app.py, send 5 synthetic chats, stop app.py")
        return

    # Start the server in the background
    print("  Starting app.py ...")
    proc = subprocess.Popen(
        [PYTHON, "app.py"],
        cwd=PROJECT_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    # Wait for server to be ready
    url = "http://127.0.0.1:5000/stats"
    ready = False
    for _ in range(20):
        try:
            urllib.request.urlopen(url, timeout=1)
            ready = True
            break
        except (urllib.error.URLError, ConnectionRefusedError):
            time.sleep(0.5)

    if not ready:
        print("  !! Server did not start. Skipping synthetic chats.")
        proc.terminate()
        proc.wait()
        return

    # Send synthetic chat messages
    chat_url = "http://127.0.0.1:5000/chat"
    for prompt in SYNTHETIC_PROMPTS:
        payload = json.dumps({"message": prompt}).encode()
        req = urllib.request.Request(
            chat_url,
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                body = json.loads(resp.read())
                snippet = body.get("response", "")[:60]
                print(f"  [{prompt[:30]}...] -> {snippet}...")
        except Exception as e:
            print(f"  !! Chat request failed: {e}")

    # Shut down
    print("  Stopping app.py ...")
    proc.send_signal(signal.SIGINT)
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()
    print(f"  Collected {len(SYNTHETIC_PROMPTS)} synthetic conversations.")


def step_rl(dry_run: bool, rollout_steps: int) -> None:
    banner(5, "RL fine-tuning (PPO)")
    run(
        [PYTHON, "rl_finetune.py", "--rollout-steps", str(rollout_steps)],
        dry_run=dry_run,
    )


def step_compare_and_promote(dry_run: bool) -> None:
    banner(7, "Compare & promote")
    if dry_run:
        print("  [dry-run] Compare last two eval results, promote winner")
        return

    results = read_last_n_results(2)
    if len(results) < 2:
        print("  Not enough eval results to compare. Skipping promotion.")
        return

    baseline, challenger = results[0], results[1]
    print(f"  Baseline:   {baseline['checkpoint']}")
    print(f"  Challenger: {challenger['checkpoint']}")

    # Compare on val loss (lower is better) and generation quality (higher)
    b_loss = baseline.get("eval_val_loss")
    c_loss = challenger.get("eval_val_loss")
    b_gen = (baseline.get("generation") or {}).get("avg_combined_score")
    c_gen = (challenger.get("generation") or {}).get("avg_combined_score")

    print(f"\n  Val loss:   baseline={b_loss}  challenger={c_loss}")
    print(f"  Gen qual:   baseline={b_gen}  challenger={c_gen}")

    # Scoring: challenger wins if it improves on at least one metric
    # without regressing on the other.
    challenger_wins = False
    if all(v is not None for v in (b_loss, c_loss, b_gen, c_gen)):
        loss_better = c_loss < b_loss
        gen_better = c_gen > b_gen

        challenger_wins = (loss_better and not (c_gen < b_gen - 0.01)) or (
            gen_better and not (c_loss > b_loss + 0.01)
        )

    if challenger_wins:
        winner = challenger["checkpoint"]
        source = MODEL_DIR / winner
        print(f"\n  Challenger wins! Promoting {winner} -> best.pt")
        if source.exists():
            shutil.copy2(source, BEST_CHECKPOINT)
            print(f"  Saved: {BEST_CHECKPOINT}")
        else:
            print(f"  !! Checkpoint file not found: {source}")
    else:
        print("\n  Baseline holds. No promotion.")
        # Still copy baseline as best if best.pt doesn't exist
        if not BEST_CHECKPOINT.exists() and b_loss is not None:
            source = MODEL_DIR / baseline["checkpoint"]
            if source.exists():
                shutil.copy2(source, BEST_CHECKPOINT)
                print(f"  Saved baseline as initial best.pt: {BEST_CHECKPOINT}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the full nanogpt-chat improvement loop.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would happen without executing",
    )
    parser.add_argument(
        "--full-eval",
        action="store_true",
        help="Include GSM8K in evaluation (slow)",
    )
    parser.add_argument(
        "--rl-steps",
        type=int,
        default=50,
        help="PPO rollout steps (default: 50, lower for demo)",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("  nanogpt-chat: Full Improvement Loop")
    print("=" * 60)
    if args.dry_run:
        print("  (dry-run mode — nothing will be executed)\n")

    # 1. Bootstrap
    step_bootstrap(args.dry_run)

    # 2. Supervised fine-tuning
    step_finetune(args.dry_run)

    # 3. Evaluate the new supervised checkpoint (baseline)
    baseline_ckpt = find_latest("finetuned_*.pt") if not args.dry_run else None
    step_evaluate(
        baseline_ckpt,
        dry_run=args.dry_run,
        full_eval=args.full_eval,
        label="baseline",
    )

    # 4. Serve and collect synthetic data
    step_serve_and_collect(args.dry_run)

    # 5. RL fine-tuning
    step_rl(args.dry_run, args.rl_steps)

    # 6. Evaluate the RL checkpoint (challenger)
    challenger_ckpt = find_latest("ppo_*.pt") if not args.dry_run else None
    step_evaluate(
        challenger_ckpt,
        dry_run=args.dry_run,
        full_eval=args.full_eval,
        label="challenger",
    )

    # 7. Compare and promote
    step_compare_and_promote(args.dry_run)

    print(f"\n{'=' * 60}")
    print("  Loop complete.")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
