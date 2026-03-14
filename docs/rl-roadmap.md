# RL Roadmap

The supervised system works. The RL architecture is partially built. Both facts matter equally.

## Current State

The supervised pipeline — chat, log, fine-tune — is stable. Security hardening is in place (see [api.md](api.md)). Dataset bootstrapping via `download_dataset.py` is functional.

## Components

| Component | Role | Status |
|-----------|------|--------|
| `rl/environment.py` | Gymnasium env wrapping chat | **implemented** (tested) |
| `rl/reward_model.py` | Score responses from feedback (simple, multi-criteria); learned model is placeholder | **implemented** (tested) |
| `rl/ppo_trainer.py` | PPO training loop with integrated value head | **implemented** (tested) |
| `rl_finetune.py` | RL training entrypoint | planned |

## Interfaces

```python
ChatEnvironment(gymnasium.Env): reset() -> (obs, info), step(action) -> (obs, reward, terminated, truncated, info), render()
RewardModel: score_response(user_message, response, feedback), update_from_feedback(feedback_data)
PPOTrainer: collect_rollouts(), update(rollout), train(), save_checkpoint(step)
```

## Multi-Criteria Scoring

`MultiCriteriaReward` scores responses on four weighted dimensions:

| Criterion | Weight | Description |
|-----------|--------|-------------|
| relevance | 0.3 | How well the response addresses the user's message |
| helpfulness | 0.3 | Practical value of the response |
| safety | 0.2 | Absence of harmful or inappropriate content |
| coherence | 0.2 | Logical consistency and readability |

The final reward is the weighted sum of per-criterion scores (each 0.0–1.0). Custom weights can be passed to the constructor; they must sum to 1.0.

## Phases

1. **Foundation** — reward model (simple + multi-criteria), env wrapper, PPO trainer with value head (**done**)
2. **Core RL** — RL training entrypoint (`rl_finetune.py`), reward shaping, experience replay
3. **Integration** — wire into chat system, checkpointing, A/B testing
4. **Advanced** — learned reward model, preference learning, safety constraints

## Risks

- Training instability → proven algorithms, careful hyperparameters
- Quality regression → A/B testing, rollback to supervised model
- Bad feedback loops → input validation, feedback filtering

## Resumption

Read this file. Check `git log`. Continue from the last completed phase. The supervised system must always remain functional as fallback.
