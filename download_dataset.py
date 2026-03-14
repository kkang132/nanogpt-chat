"""
Download and prepare Q&A datasets for fine-tuning.
Combines OpenAssistant (conversation) and GSM8K (math)
into chat_history.jsonl format.
"""
import json
import random
from datasets import load_dataset

CHAT_LOG_FILE = 'chat_history.jsonl'
OASST_COUNT = 1500
MATH_COUNT = 500
SEED = 42


def clean_gsm8k_answer(answer: str) -> str:
    """Extract a clean step-by-step answer from GSM8K format.

    GSM8K answers contain calculator annotations like <<48/2=24>>
    and a final answer after ####. Clean these up for training.
    """
    import re
    # Remove calculator annotations <<...>>
    answer = re.sub(r'<<.*?>>', '', answer)
    # Replace #### with a clear "The answer is" prefix
    answer = re.sub(r'####\s*', 'The answer is ', answer)
    return answer.strip()


def download_oasst(n: int) -> list[dict]:
    """Download n conversation pairs from OpenAssistant (CC-BY-4.0).

    Filters for English, top-level user prompts paired with the
    highest-ranked assistant reply.
    """
    print("Downloading OpenAssistant Conversations (oasst1)...")
    ds = load_dataset("OpenAssistant/oasst1", split="train")
    print(f"  Total messages: {len(ds)}")

    # Index messages by parent_id for fast lookup
    by_parent = {}
    by_id = {}
    for row in ds:
        by_id[row["message_id"]] = row
        parent = row["parent_id"]
        if parent not in by_parent:
            by_parent[parent] = []
        by_parent[parent].append(row)

    # Find English top-level user prompts with at least one assistant reply
    pairs = []
    for row in ds:
        if (row["parent_id"] is None
                and row["role"] == "prompter"
                and row["lang"] == "en"):
            # Get assistant replies to this prompt
            replies = by_parent.get(row["message_id"], [])
            assistant_replies = [
                r for r in replies
                if r["role"] == "assistant" and r["lang"] == "en"
            ]
            if not assistant_replies:
                continue
            # Pick the highest-ranked reply
            best = max(assistant_replies, key=lambda r: r.get("rank", 0) or 0)
            pairs.append({
                "user": row["text"].strip(),
                "assistant": best["text"].strip(),
            })

    print(f"  English Q&A pairs found: {len(pairs)}")

    random.seed(SEED)
    random.shuffle(pairs)
    selected = pairs[:n]
    print(f"  Selected: {len(selected)} conversation examples")
    return selected


def download_gsm8k(n: int) -> list[dict]:
    """Download n examples from GSM8K (MIT license)."""
    print("Downloading GSM8K dataset...")
    ds = load_dataset("openai/gsm8k", "main", split="train")
    print(f"  Total available: {len(ds)}")

    random.seed(SEED)
    indices = random.sample(range(len(ds)), min(n, len(ds)))
    examples = []
    for i in indices:
        row = ds[i]
        examples.append({
            "user": row["question"],
            "assistant": clean_gsm8k_answer(row["answer"])
        })
    print(f"  Selected: {len(examples)} math examples")
    return examples


def main() -> None:
    print("=" * 60)
    print("Q&A Dataset Downloader")
    print("=" * 60)
    print()

    oasst_examples = download_oasst(OASST_COUNT)
    math_examples = download_gsm8k(MATH_COUNT)

    # Combine and shuffle
    all_examples = oasst_examples + math_examples
    random.seed(SEED)
    random.shuffle(all_examples)

    # Write to chat_history.jsonl
    with open(CHAT_LOG_FILE, 'w') as f:
        for ex in all_examples:
            f.write(json.dumps(ex) + '\n')

    print()
    print(f"Written {len(all_examples)} examples to {CHAT_LOG_FILE}")
    print(f"  - Conversation (OpenAssistant): {len(oasst_examples)}")
    print(f"  - Math (GSM8K):                 {len(math_examples)}")
    print()
    print("You can now run: python finetune.py")


if __name__ == '__main__':
    main()
