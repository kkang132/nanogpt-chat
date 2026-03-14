# Intent

This is an educational project. It exists to teach someone who already knows Python, PyTorch, and reinforcement learning (RL) how a GPT-style chat system works end-to-end from model architecture through fine-tuning to serving, and eventually through RL-based alignment.

## What the learner should walk away understanding

1. **The supervised loop**: chat interactions are logged, filtered by user ratings, and used to fine-tune the base GPT-2 model. The fine-tuned model is reloaded into the chat server. This is the core feedback cycle.

2. **The RL loop** (in progress): the same feedback signal — thumbs up/down from users — can drive policy optimization via PPO rather than supervised fine-tuning. The reward model translates ratings into scalar signals; the environment wraps the chat interaction as a Gymnasium MDP; the PPO trainer optimizes the language model's policy against that reward.

3. **The connection between 1 and 2**: supervised fine-tuning and RLHF are not different systems. They share the same model, the same data, and the same goal. The difference is the optimization algorithm. This project makes that visible by keeping both paths in the same codebase, using the same components.

## What this is not

- A production RLHF system. The model is small, the reward signal is crude, and the PPO implementation prioritizes clarity over performance.
- A tutorial for beginners. The reader is expected to know what PPO does and to be comfortable reading PyTorch modules.
- A scaling experiment. Everything runs on a single GPU or CPU. The point is legibility, not throughput.

## PPO trainer: design intent

The `rl/ppo_trainer.py` module completes the RL side of the project. It should:

- **Use the GPT model directly as the policy**. No separate policy network — the language model _is_ the policy. Its logits over the vocabulary at each timestep define the action distribution. A thin value head is added on top.
- **Use the existing reward model** (`rl/reward_model.py`) to score generated responses.
- **Follow the standard PPO algorithm**: collect rollouts by generating responses, compute advantages using GAE, update policy and value function with clipped surrogate objective.
- **Stay readable**. Each method should do one thing. The training loop should fit on a screen. Hyperparameters should have comments explaining what they control.
- **Produce a fine-tuned checkpoint** in the same format as `finetune.py`, so the chat server can load it without modification.

### Components

| Component | Role |
|-----------|------|
| `ValueHead` | Single linear layer projecting hidden states to scalar values |
| `PPOTrainer` | Owns the policy (GPT + value head), reference model, reward model, tokenizer. Runs the collect → compute advantages → update loop. |

### Key decisions

- **Reference model**: a frozen copy of the initial policy, used for KL penalty to prevent the policy from drifting too far from the pretrained distribution.
- **KL penalty**: applied per-token as `kl = log_prob - ref_log_prob`, added to the reward. This is the standard RLHF approach.
- **Generation**: uses the existing `GPT.generate()` for rollout collection, then replays the generated sequences through the model to get log-probs and values.
- **No separate critic network**: the value head shares the transformer backbone. This is simpler and sufficient at this scale.
