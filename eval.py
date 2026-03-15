"""
Evaluation pipeline for nanogpt-chat.

Measures model quality across three dimensions:
  1. Perplexity / validation loss (cross-entropy on held-out data)
  2. Generation quality (heuristic scoring of responses to fixed prompts)
  3. GSM8K accuracy (exact-match on math problems)

Supports multi-checkpoint comparison and logs results to JSONL.

Usage:
    python eval.py                        # evaluate all checkpoints
    python eval.py --checkpoint models/X  # evaluate a single checkpoint
    python eval.py --skip-gsm8k           # skip slow GSM8K generation
    python eval.py --skip-generation      # skip generation quality eval
"""

from __future__ import annotations

import argparse
import datetime
import json
import math
import re
from pathlib import Path

import numpy as np
import tiktoken
import torch
import torch.nn.functional as f

from model import GPT, GPTConfig

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_SCRIPT_DIR = Path(__file__).resolve().parent
MODEL_DIR = _SCRIPT_DIR / "models"
DATA_DIR = _SCRIPT_DIR / "data"
EVAL_DIR = _SCRIPT_DIR / "eval"
RESULTS_FILE = EVAL_DIR / "eval_results.jsonl"

# Eval hyperparameters
EVAL_BATCH_SIZE = 2
EVAL_BLOCK_SIZE = 128
EVAL_ITERS = 20

# Generation defaults
GENERATION_MAX_TOKENS = 150
GENERATION_TEMPERATURE = 0.8
GENERATION_TOP_K = 200

# GSM8K
GSM8K_EVAL_COUNT = 50
GSM8K_MAX_TOKENS = 200
GSM8K_TEMPERATURE = 0.5

# GPT-2 base model config (matches app.py / finetune.py)
BASE_BLOCK_SIZE = 1024
BASE_VOCAB_SIZE = 50257
BASE_N_LAYER = 12
BASE_N_HEAD = 12
BASE_N_EMBD = 768
BASE_DROPOUT = 0.0
BASE_BIAS = True

# Fixed prompts for generation quality evaluation
EVAL_PROMPTS: list[str] = [
    "Human: What is the capital of France?\nAssistant:",
    "Human: How do I make scrambled eggs?\nAssistant:",
    "Human: Tell me a joke.\nAssistant:",
    "Human: What is machine learning?\nAssistant:",
    "Human: Why is the sky blue?\nAssistant:",
    "Human: Write a haiku about coding.\nAssistant:",
    "Human: What are the benefits of exercise?\nAssistant:",
    "Human: How does photosynthesis work?\nAssistant:",
]

# ---------------------------------------------------------------------------
# Device selection (same cascade as app.py / finetune.py)
# ---------------------------------------------------------------------------

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

# ---------------------------------------------------------------------------
# Checkpoint discovery and loading
# ---------------------------------------------------------------------------


def discover_checkpoints(model_dir: str) -> list[str]:
    """Find all model checkpoints in *model_dir*, sorted by filename.

    Includes the base ``gpt2_nano.pt`` (if present) followed by any
    ``finetuned_*.pt`` and ``ppo_*.pt`` checkpoints in alphabetical
    order (which is chronological because filenames contain timestamps).
    """
    d = Path(model_dir)
    paths: list[str] = []

    base = d / "gpt2_nano.pt"
    if base.exists():
        paths.append(str(base))

    finetuned = sorted(
        [str(p) for p in d.glob("finetuned_*.pt")]
        + [str(p) for p in d.glob("ppo_*.pt")]
    )
    paths.extend(finetuned)
    return paths


def load_checkpoint(path: str) -> tuple[GPT, dict[str, object]]:
    """Load a checkpoint and return ``(model, metadata)``.

    Handles two formats:
    - Finetuned/PPO checkpoints: dict with ``model_state_dict``, ``config``,
      ``iter``, ``train_loss``, ``val_loss``.
    - Base model (``gpt2_nano.pt``): a raw ``state_dict`` with no wrapper.
    """
    from rl.ppo_trainer import PPOConfig

    torch.serialization.add_safe_globals([GPTConfig, PPOConfig])
    raw = torch.load(path, map_location=device, weights_only=True)

    if isinstance(raw, dict) and "model_state_dict" in raw:
        config = raw["config"]
        model = GPT(config)
        model.load_state_dict(raw["model_state_dict"])
        metadata = {
            "iter": raw.get("iter"),
            "train_loss": raw.get("train_loss"),
            "val_loss": raw.get("val_loss"),
        }
    else:
        # Base model — raw state_dict, use default GPT-2 config
        config = GPTConfig(
            block_size=BASE_BLOCK_SIZE,
            vocab_size=BASE_VOCAB_SIZE,
            n_layer=BASE_N_LAYER,
            n_head=BASE_N_HEAD,
            n_embd=BASE_N_EMBD,
            dropout=BASE_DROPOUT,
            bias=BASE_BIAS,
        )
        model = GPT(config)
        model.load_state_dict(raw)
        metadata = {"iter": None, "train_loss": None, "val_loss": None}

    model.eval()
    model.to(device)
    return model, metadata


# ---------------------------------------------------------------------------
# Suite 1: Perplexity / validation loss
# ---------------------------------------------------------------------------


@torch.no_grad()
def compute_val_loss(
    model: GPT, val_data: np.ndarray, num_batches: int = EVAL_ITERS
) -> float:
    """Average cross-entropy loss over *num_batches* random windows of *val_data*."""
    model.eval()
    losses = torch.zeros(num_batches)
    max_start = len(val_data) - EVAL_BLOCK_SIZE - 1
    if max_start <= 0:
        raise ValueError(
            f"Validation data too small ({len(val_data)} tokens). "
            f"Need at least {EVAL_BLOCK_SIZE + 1}."
        )

    for k in range(num_batches):
        ix = torch.randint(max_start, (EVAL_BATCH_SIZE,))
        x = torch.stack(
            [
                torch.from_numpy(val_data[i : i + EVAL_BLOCK_SIZE].astype(np.int64))
                for i in ix
            ]
        )
        y = torch.stack(
            [
                torch.from_numpy(
                    val_data[i + 1 : i + 1 + EVAL_BLOCK_SIZE].astype(np.int64)
                )
                for i in ix
            ]
        )
        if device != "cpu":
            x, y = x.to(device), y.to(device)
        _, loss = model(x, y)
        losses[k] = loss.item()

    return losses.mean().item()


def compute_perplexity(loss: float) -> float:
    """Convert cross-entropy loss to perplexity: ``exp(loss)``."""
    return math.exp(loss)


# ---------------------------------------------------------------------------
# Suite 2: Generation quality heuristics
# ---------------------------------------------------------------------------


@torch.no_grad()
def generate_text(
    model: GPT,
    enc: tiktoken.Encoding,
    prompt: str,
    max_tokens: int = GENERATION_MAX_TOKENS,
    temperature: float = GENERATION_TEMPERATURE,
    top_k: int = GENERATION_TOP_K,
) -> str:
    """Generate a response from *prompt* using top-k sampling.

    Adapted from ``app.py:generate_response`` but takes model/encoder
    explicitly instead of relying on module globals.
    """
    model.eval()
    tokens = enc.encode(prompt, allowed_special={"<|endoftext|>"})
    tokens = torch.tensor(tokens, dtype=torch.long, device=device)[None, ...]

    for _ in range(max_tokens):
        idx_cond = (
            tokens
            if tokens.size(1) <= model.config.block_size
            else tokens[:, -model.config.block_size :]
        )
        logits, _ = model(idx_cond)
        logits = logits[:, -1, :] / temperature

        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float("Inf")

        probs = f.softmax(logits, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1)
        tokens = torch.cat((tokens, idx_next), dim=1)

        if idx_next[0].item() == enc.encode("\n")[0]:
            break

    generated_text = enc.decode(tokens[0].tolist())
    return generated_text[len(prompt) :].strip()


def score_length(text: str) -> float:
    """Score response length.  Ideal range: 10–500 chars.

    Returns a value in [0.0, 1.0].
    """
    n = len(text)
    if n == 0:
        return 0.0
    if n < 10:
        return n / 10.0
    if n <= 500:
        return 1.0
    if n <= 1000:
        return 1.0 - (n - 500) / 500.0
    return 0.0


def score_repetition(text: str) -> float:
    """Detect n-gram repetition.  1.0 = no repetition, 0.0 = heavily repeated.

    Uses the ratio of unique trigrams to total trigrams.
    """
    words = text.split()
    if len(words) < 3:
        return 1.0  # too short to judge
    trigrams = [tuple(words[i : i + 3]) for i in range(len(words) - 2)]
    if not trigrams:
        return 1.0
    return len(set(trigrams)) / len(trigrams)


def score_coherence(text: str) -> float:
    """Check whether *text* contains actual English-like words.

    Heuristic: a word (alpha-only, len >= 2) is "real" if it contains
    at least one vowel.  Returns the fraction of such words.
    """
    words = list(re.findall(r"[a-zA-Z]{2,}", text))
    if not words:
        return 0.0
    vowel_count = sum(1 for w in words if re.search(r"[aeiouAEIOU]", w))
    return vowel_count / len(words)


def score_format(text: str) -> float:
    """Score chat-format adherence.

    Deductions for: empty response, prompt leakage (``Human:``),
    or very short single-word answers.
    """
    if not text.strip():
        return 0.0
    score = 1.0
    if text.strip().startswith("Human:"):
        score -= 0.5
    if len(text.split()) <= 1:
        score -= 0.3
    return max(score, 0.0)


def evaluate_generation_quality(
    model: GPT, enc: tiktoken.Encoding
) -> dict[str, object]:
    """Run all ``EVAL_PROMPTS``, score each response, return aggregates."""
    length_scores: list[float] = []
    repetition_scores: list[float] = []
    coherence_scores: list[float] = []
    format_scores: list[float] = []
    samples: list[dict[str, str]] = []

    for prompt in EVAL_PROMPTS:
        response = generate_text(model, enc, prompt)
        length_scores.append(score_length(response))
        repetition_scores.append(score_repetition(response))
        coherence_scores.append(score_coherence(response))
        format_scores.append(score_format(response))
        if len(samples) < 3:
            samples.append({"prompt": prompt, "response": response})

    def _mean(xs: list[float]) -> float:
        return sum(xs) / len(xs) if xs else 0.0

    avg_length = _mean(length_scores)
    avg_repetition = _mean(repetition_scores)
    avg_coherence = _mean(coherence_scores)
    avg_format = _mean(format_scores)
    avg_combined = _mean([avg_length, avg_repetition, avg_coherence, avg_format])

    return {
        "avg_length_score": round(avg_length, 4),
        "avg_repetition_score": round(avg_repetition, 4),
        "avg_coherence_score": round(avg_coherence, 4),
        "avg_format_score": round(avg_format, 4),
        "avg_combined_score": round(avg_combined, 4),
        "num_prompts": len(EVAL_PROMPTS),
        "sample_responses": samples,
    }


# ---------------------------------------------------------------------------
# Suite 3: GSM8K exact-match accuracy
# ---------------------------------------------------------------------------


def extract_numeric_answer(text: str) -> str | None:
    """Extract the final numeric answer from generated text.

    Looks for:
    1. ``The answer is X``
    2. ``#### X``
    3. Last number in the text (fallback)

    Returns the number as a normalised string (commas stripped) or ``None``.
    """
    # Pattern 1: "The answer is <number>"
    m = re.search(r"[Tt]he answer is\s*([+-]?[\d,]+\.?\d*)", text)
    if m:
        return m.group(1).replace(",", "")

    # Pattern 2: "#### <number>"
    m = re.search(r"####\s*([+-]?[\d,]+\.?\d*)", text)
    if m:
        return m.group(1).replace(",", "")

    # Fallback: last number in text
    numbers = re.findall(r"[+-]?[\d,]+\.?\d*", text)
    if numbers:
        return numbers[-1].replace(",", "")

    return None


def load_gsm8k_eval_set(n: int = GSM8K_EVAL_COUNT) -> list[dict[str, str]]:
    """Load *n* problems from the GSM8K **test** split.

    Uses the ``datasets`` library (already a project dependency).
    Returns ``[{"question": str, "answer": str}, ...]`` where
    *answer* is the ground-truth numeric answer only.
    """
    from datasets import load_dataset

    ds = load_dataset("openai/gsm8k", "main", split="test")
    examples: list[dict[str, str]] = []
    for i in range(min(n, len(ds))):
        row = ds[i]
        # Ground truth is after "####"
        raw_answer = row["answer"]
        m = re.search(r"####\s*(.+)", raw_answer)
        answer = m.group(1).strip().replace(",", "") if m else ""
        examples.append({"question": row["question"], "answer": answer})
    return examples


def evaluate_gsm8k(
    model: GPT, enc: tiktoken.Encoding, n: int = GSM8K_EVAL_COUNT
) -> dict[str, object]:
    """Generate answers to *n* GSM8K test problems and compute exact-match."""
    examples = load_gsm8k_eval_set(n)
    correct = 0
    no_answer = 0

    for ex in examples:
        prompt = f"Human: {ex['question']}\nAssistant:"
        response = generate_text(
            model,
            enc,
            prompt,
            max_tokens=GSM8K_MAX_TOKENS,
            temperature=GSM8K_TEMPERATURE,
            top_k=GENERATION_TOP_K,
        )
        predicted = extract_numeric_answer(response)
        if predicted is None:
            no_answer += 1
        elif predicted == ex["answer"]:
            correct += 1

    total = len(examples)
    return {
        "accuracy": round(correct / total, 4) if total else 0.0,
        "correct": correct,
        "total": total,
        "no_answer": no_answer,
    }


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def evaluate_checkpoint(
    checkpoint_path: str,
    *,
    skip_generation: bool = False,
    skip_gsm8k: bool = False,
    gsm8k_count: int = GSM8K_EVAL_COUNT,
) -> dict[str, object]:
    """Run all enabled eval suites on a single checkpoint."""
    name = Path(checkpoint_path).name
    print(f"\n{'─' * 60}")
    print(f"Evaluating: {name}")
    print(f"{'─' * 60}")

    model, metadata = load_checkpoint(checkpoint_path)
    enc = tiktoken.get_encoding("gpt2")

    result: dict[str, object] = {
        "checkpoint": name,
        "timestamp": datetime.datetime.now(datetime.UTC).isoformat(),
        "stored_train_loss": metadata.get("train_loss"),
        "stored_val_loss": metadata.get("val_loss"),
    }

    # Suite 1: Validation loss
    val_path = DATA_DIR / "val.bin"
    if val_path.exists():
        val_data = np.memmap(str(val_path), dtype=np.uint16, mode="r")
        loss = compute_val_loss(model, val_data)
        ppl = compute_perplexity(loss)
        result["eval_val_loss"] = round(loss, 4)
        result["perplexity"] = round(ppl, 2)
        print(f"  Val loss: {loss:.4f}  |  Perplexity: {ppl:.2f}")
    else:
        print(f"  Skipping val loss (no {val_path})")
        result["eval_val_loss"] = None
        result["perplexity"] = None

    # Suite 2: Generation quality
    if skip_generation:
        print("  Skipping generation quality (--skip-generation)")
        result["generation"] = None
    else:
        gen = evaluate_generation_quality(model, enc)
        result["generation"] = gen
        print(
            f"  Generation quality: {gen['avg_combined_score']:.4f} "
            f"(len={gen['avg_length_score']:.2f} "
            f"rep={gen['avg_repetition_score']:.2f} "
            f"coh={gen['avg_coherence_score']:.2f} "
            f"fmt={gen['avg_format_score']:.2f})"
        )

    # Suite 3: GSM8K
    if skip_gsm8k:
        print("  Skipping GSM8K (--skip-gsm8k)")
        result["gsm8k"] = None
    else:
        gsm = evaluate_gsm8k(model, enc, n=gsm8k_count)
        result["gsm8k"] = gsm
        print(
            f"  GSM8K accuracy: {gsm['accuracy']:.4f} "
            f"({gsm['correct']}/{gsm['total']}, "
            f"{gsm['no_answer']} no-answer)"
        )

    # Free memory before next checkpoint
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return result


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------


def format_results_table(results: list[dict[str, object]]) -> str:
    """Format a comparison table for CLI output."""
    cols = ["Checkpoint", "Val Loss", "PPL", "Gen Qual", "GSM8K"]
    header = f"{cols[0]:<30} {cols[1]:>10} {cols[2]:>10} {cols[3]:>10} {cols[4]:>10}"
    sep = "─" * len(header)
    lines = [sep, header, sep]

    for r in results:
        vl = r.get("eval_val_loss")
        val_loss = f"{vl:.4f}" if vl is not None else "—"
        ppl = f"{r['perplexity']:.2f}" if r.get("perplexity") is not None else "—"

        gen = r.get("generation")
        gen_str = f"{gen['avg_combined_score']:.4f}" if gen else "—"

        gsm = r.get("gsm8k")
        gsm_str = f"{gsm['accuracy']:.4f}" if gsm else "—"

        name = r["checkpoint"]
        if len(name) > 29:
            name = name[:26] + "..."
        lines.append(f"{name:<30} {val_loss:>10} {ppl:>10} {gen_str:>10} {gsm_str:>10}")

    lines.append(sep)
    return "\n".join(lines)


def save_results(results: list[dict[str, object]], path: str) -> None:
    """Append each result as a JSON line to *path*."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("a") as f:
        for r in results:
            # Strip sample_responses for cleaner logs
            cleaned = dict(r)
            if isinstance(cleaned.get("generation"), dict):
                gen = dict(cleaned["generation"])
                gen.pop("sample_responses", None)
                cleaned["generation"] = gen
            f.write(json.dumps(cleaned) + "\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate nanogpt-chat model checkpoints."
    )
    parser.add_argument(
        "--models-dir",
        default=MODEL_DIR,
        help="Directory to scan for checkpoints (default: models/)",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Evaluate a single checkpoint instead of all",
    )
    parser.add_argument(
        "--skip-gsm8k",
        action="store_true",
        help="Skip GSM8K eval (slow due to generation)",
    )
    parser.add_argument(
        "--skip-generation",
        action="store_true",
        help="Skip generation quality eval",
    )
    parser.add_argument(
        "--gsm8k-count",
        type=int,
        default=GSM8K_EVAL_COUNT,
        help=f"Number of GSM8K problems (default: {GSM8K_EVAL_COUNT})",
    )
    parser.add_argument(
        "--output",
        default=RESULTS_FILE,
        help="JSONL output file (default: eval/eval_results.jsonl)",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("NanoGPT-Chat Evaluation Pipeline")
    print("=" * 60)
    print(f"Device: {device}")

    # Discover checkpoints
    if args.checkpoint:
        if not Path(args.checkpoint).exists():
            print(f"Error: checkpoint not found: {args.checkpoint}")
            return
        paths = [args.checkpoint]
    else:
        paths = discover_checkpoints(args.models_dir)

    if not paths:
        print("No checkpoints found. Nothing to evaluate.")
        return

    print(f"Checkpoints: {len(paths)}")
    for p in paths:
        print(f"  - {Path(p).name}")

    # Evaluate each checkpoint
    results: list[dict[str, object]] = []
    for path in paths:
        r = evaluate_checkpoint(
            path,
            skip_generation=args.skip_generation,
            skip_gsm8k=args.skip_gsm8k,
            gsm8k_count=args.gsm8k_count,
        )
        results.append(r)

    # Print comparison table
    print(f"\n{'=' * 60}")
    print("Results Summary")
    print("=" * 60)
    print(format_results_table(results))

    # Save to JSONL
    save_results(results, args.output)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
