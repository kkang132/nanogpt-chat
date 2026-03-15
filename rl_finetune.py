"""
RL fine-tuning entrypoint for nanogpt-chat.

Wires together the PPO trainer, reward model, and a supervised checkpoint
to run reinforcement learning. Produces format-compatible checkpoints that
app.py can load without modification.

Usage:
    python rl_finetune.py                          # use latest finetuned checkpoint
    python rl_finetune.py --checkpoint models/X.pt # use a specific checkpoint
    python rl_finetune.py --rollout-steps 50       # fewer rollout steps
    python rl_finetune.py --reward-type multi_criteria
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import tiktoken
import torch

from model import GPT, GPTConfig
from rl.ppo_trainer import PPOConfig, PPOTrainer
from rl.reward_model import create_reward_model

# ---------------------------------------------------------------------------
# Constants (match eval.py / finetune.py)
# ---------------------------------------------------------------------------

_SCRIPT_DIR = Path(__file__).resolve().parent
MODEL_DIR = _SCRIPT_DIR / "models"

BASE_BLOCK_SIZE = 1024
BASE_VOCAB_SIZE = 50257
BASE_N_LAYER = 12
BASE_N_HEAD = 12
BASE_N_EMBD = 768
BASE_DROPOUT = 0.0
BASE_BIAS = True

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Device selection (same cascade as app.py / finetune.py / eval.py)
# ---------------------------------------------------------------------------

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

# ---------------------------------------------------------------------------
# Checkpoint loading
# ---------------------------------------------------------------------------


def find_latest_checkpoint() -> str | None:
    """Return the path to the most recent finetuned or PPO checkpoint."""
    candidates = sorted(
        [str(p) for p in MODEL_DIR.glob("finetuned_*.pt")]
        + [str(p) for p in MODEL_DIR.glob("ppo_*.pt")]
    )
    return candidates[-1] if candidates else None


def load_checkpoint(path: str) -> GPT:
    """Load a checkpoint and return the model ready for RL training."""
    torch.serialization.add_safe_globals([GPTConfig])
    raw = torch.load(path, map_location=device, weights_only=True)

    if isinstance(raw, dict) and "model_state_dict" in raw:
        config = raw["config"]
        model = GPT(config)
        model.load_state_dict(raw["model_state_dict"])
    else:
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

    model.to(device)
    return model


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run PPO reinforcement learning on a nanogpt-chat checkpoint."
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Checkpoint to start from (default: latest finetuned/ppo)",
    )
    parser.add_argument(
        "--rollout-steps",
        type=int,
        default=100,
        help="Total rollout-then-update cycles (default: 100)",
    )
    parser.add_argument(
        "--reward-type",
        default="simple",
        choices=["simple", "multi_criteria"],
        help="Reward model type (default: simple)",
    )
    args = parser.parse_args()

    # Resolve checkpoint
    ckpt_path = args.checkpoint or find_latest_checkpoint()
    if ckpt_path is None:
        base = MODEL_DIR / "gpt2_nano.pt"
        if base.exists():
            ckpt_path = str(base)
        else:
            print("Error: no checkpoint found. Run finetune.py first.")
            sys.exit(1)

    logger.info(f"Loading checkpoint: {ckpt_path}")
    model = load_checkpoint(ckpt_path)

    tokenizer = tiktoken.get_encoding("gpt2")
    reward_model = create_reward_model(args.reward_type)

    config = PPOConfig(
        total_rollout_steps=args.rollout_steps,
        model_dir=str(MODEL_DIR),
    )

    trainer = PPOTrainer(
        model=model,
        tokenizer=tokenizer,
        reward_model=reward_model,
        config=config,
        device=device,
    )

    stats = trainer.train()
    logger.info(f"Training complete. {len(stats)} rollout steps recorded.")


if __name__ == "__main__":
    main()
