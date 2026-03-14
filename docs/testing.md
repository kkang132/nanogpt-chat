# Testing

## Situation

The test suite contains 126 tests across seven files. The command to run them is:

```bash
pytest tests/
```

The suite finishes in 1.38 seconds on a machine with an Apple M-series chip. No network access, no GPU, and no model checkpoint larger than a few bytes are required.

## What Is Tested

### Flask application (`tests/test_app.py`, 20 tests)

The application presents three endpoints: `/chat`, `/rate`, and `/stats`. Each is tested through Flask's own test client, which means the tests exercise the real routing, request parsing, and response serialization without starting a network server.

The central difficulty is that `app.py` loads a GPT-2 checkpoint at module level. If the tests attempted to import the module as it stands, they would need a 500 MB file and several seconds of GPU initialization. The test fixture avoids this by patching the model constructor, `torch.load`, and the tokenizer before the import occurs. The resulting mock model returns a fixed logit distribution. This is an honest trade-off: the tests cannot verify that the model produces good text, but they can verify everything else.

The `/chat` endpoint is tested for seven distinct input conditions: a valid message, an empty string, whitespace, a non-JSON content type, a message exceeding the length limit, a non-string value, and a JSON array where an object is expected. Each asserts the correct HTTP status code and, where applicable, the shape of the error message.

The `/rate` endpoint is tested for the positive and negative cases, for an invalid rating value, for a missing `chat_id`, for a nonexistent `chat_id`, for the case where no chat log file exists at all, and for a non-JSON request body. The positive-rating test goes further than checking the HTTP response: it reads the JSONL file and confirms that the rating was written through to disk.

The `/stats` endpoint is tested empty, after several chats, and at the threshold where the server reports readiness for fine-tuning.

A separate test class verifies that security headers — `X-Content-Type-Options`, `X-Frame-Options`, `Content-Security-Policy`, and `Referrer-Policy` — are present on every response.

Two tests cover the `save_chat` helper directly: one for ordinary writes, one for log rotation when the file exceeds its size limit.

### Fine-tuning pipeline (`tests/test_finetune.py`, 15 tests)

The fine-tuning script has a shape that resists easy testing. Its main function is a monolithic training loop, and its most important sub-functions — `prepare_training_data` and `get_batch` — are the ones that can be tested cleanly. The tests focus there.

`prepare_training_data` is tested with a synthetic chat log containing 25 entries: some positively rated, some unrated, and five negatively rated. The tests confirm that negatively rated entries are filtered out, that the train/val split is approximately 90/10, and that the output `.bin` files are valid `uint16` NumPy arrays.

`get_batch` is tested for correct tensor shapes, for the next-token-prediction invariant (that `y` is `x` shifted by one position), for the error raised when the dataset is too small, and for the property that the `val` split actually reads from validation data rather than training data.

The learning-rate schedule is a closure inside the training function and cannot be imported directly. Instead, the tests reproduce the schedule's formula and verify its boundary conditions: that it starts at zero, reaches the maximum at the end of warmup, decays monotonically through the cosine phase, reaches the minimum at the final decay iteration, and clamps beyond that point.

The early-stopping logic is similarly embedded in the training loop. The tests simulate sequences of validation losses and confirm that stopping occurs at the correct iteration when losses plateau, and that the patience counter resets when a genuine improvement appears.

### Evaluation pipeline (`tests/test_eval.py`, 44 tests)

The evaluation pipeline scores checkpoints on perplexity, generation quality, and GSM8K math accuracy. The tests exercise the pure-logic helpers without loading real models or running GPU inference.

Checkpoint discovery is tested for finding finetuned, PPO, and base model checkpoints, for sorting order (base model first), for empty directories, and for ignoring non-checkpoint files. The scoring functions — `score_length`, `score_repetition`, `score_coherence`, and `score_format` — are each tested at their boundary conditions: empty input, ideal input, penalized input, and edge cases. `extract_numeric_answer` is tested for "the answer is" patterns, `####` markers, comma-separated numbers, decimals, negatives, and fallback to the last number in the text. `compute_perplexity` is tested at zero loss, a known loss value, and high loss. `compute_val_loss` is tested for correct averaging and for raising on datasets too small to batch. `format_results_table` is tested for normal output, for dash placeholders on `None` values, and for truncation of long checkpoint names. `save_results` is tested for file creation, append behaviour, valid JSONL output, and stripping of sample responses.

### Dataset utilities (`tests/test_download_dataset.py`, 6 tests)

Only one function in `download_dataset.py` is a pure transformation: `clean_gsm8k_answer`, which strips calculator annotations (`<<48/2=24>>`) and replaces the final-answer marker (`#### 42`) with prose. The six tests cover each transformation in isolation, their composition, plain text that should pass through unchanged, whitespace handling, and the case of multiple annotations in a single string.

### RL environment (`tests/test_environment.py`, 7 tests)

These tests predate the current round of work. They cover the Gymnasium `ChatEnvironment`: initialization, reset, a single step, episode termination at `max_length`, an invalid action, rendering, and cleanup.

### PPO trainer (`tests/test_ppo_trainer.py`, 11 tests)

These tests use a tiny GPT model (2 layers, 2 heads, 64-dim embeddings, vocab size 256) and a fake character-level tokenizer. The model runs on CPU and the tests complete in about a second.

The `ValueHead` is tested for output shape and for the property that its initial outputs are near zero (a consequence of the small weight initialization).

The `PPOTrainer` is tested at each level of its pipeline. `_forward_full` is tested for returning logits, hidden states, and values with the correct shapes. `_get_log_probs` is tested for shape and for the invariant that log probabilities are non-positive. `collect_rollouts` is tested for returning a dict with all expected keys and correct batch dimensions. `_compute_gae` is tested with a hand-constructed example: known rewards, values, and masks, with assertions that prompt tokens (mask=0) receive zero advantage. `update` is tested for running without error and returning finite loss values. The full `train` loop is tested end-to-end with 2 rollout steps, confirming that it produces the expected number of stats entries and writes at least one checkpoint to disk. `save_checkpoint` is tested for file existence and for the presence of the keys that `app.py` needs to load the model (`model_state_dict`, `config`). A final test confirms that the reference model's parameters remain unchanged after a PPO update.

The tests do not verify that PPO improves the model's responses. With a 2-layer model, 8-token responses, and 2 rollout steps, no meaningful learning can occur. What the tests verify is that the machinery — generation, advantage computation, clipped updates, checkpointing — executes correctly and that the components compose without error.

### Reward models (`tests/test_reward_model.py`, 23 tests)

These also predate the current round. They cover `SimpleRatingReward`, `MultiCriteriaReward`, `LearnedRewardModel`, the factory function, and interface compliance for all three implementations.

## What Is Not Tested

It would be dishonest to present a test count without also presenting what the tests cannot reach. The following areas have no automated coverage.

### Model inference quality

The Flask test fixture replaces the real model with a mock that returns fixed logits. This means that no test verifies whether the model produces coherent, relevant, or safe text. The reason is straightforward: inference quality depends on the checkpoint, and the checkpoint is not part of the repository. A test that loaded the real model would be slow, brittle, and would test the checkpoint rather than the code. If inference-quality testing becomes necessary, it belongs in a separate evaluation harness with its own data and acceptance criteria, not in the unit suite.

### The training loop itself

`finetune.py` contains a training loop that loads a model, runs gradient descent for up to 1,000 iterations, and saves a checkpoint. The tests cover the data-preparation and batching functions that feed the loop, and they verify the mathematical properties of the learning-rate schedule and early-stopping logic. But no test runs actual gradient steps. Doing so would require a real model checkpoint, would take tens of seconds at minimum, and would be nondeterministic in ways that make assertions fragile. The risk this creates is real: a change to the loop's optimizer configuration, loss computation, or checkpoint-saving logic would not be caught by the suite.

### Network-dependent dataset downloads

`download_dataset.py` calls `load_dataset` from the Hugging Face `datasets` library to fetch OpenAssistant and GSM8K data. These calls are not tested because they require network access and depend on the availability of external services. The pure transformation function that processes the downloaded data is tested; the download itself is not.

### Rate limiting under load

The tests disable Flask-Limiter so that endpoint tests can run without hitting rate limits. This means the rate-limiting configuration itself — the specific thresholds and their interaction with the Flask request cycle — is not verified. A misconfigured rate limit would not be caught.

### The HTML/JavaScript frontend

The chat interface in `templates/index.html` is a single-page application that communicates with the Flask backend via `fetch`. No browser-level or DOM-level tests exist. The frontend's behavior — message rendering, feedback button state, error display — is tested only by using it.

### Multi-process and file-locking behavior

The chat log is a JSONL file written by `app.py` and read by `finetune.py`. If both were running simultaneously, the interaction between them would matter. No test exercises concurrent access to the log file.

## Design Choices

The suite is deliberately fast. 126 tests in under two seconds means there is no practical barrier to running them before every commit. That speed comes from a decision to mock the expensive parts — model loading, GPU allocation, network calls — and test the logic that surrounds them. The trade-off is that the mocked boundaries are precisely the places where integration bugs are most likely to hide.

The suite uses pytest throughout. `test_environment.py` and `test_ppo_trainer.py` use `unittest.TestCase`, which pytest runs without complaint, so there was no reason to rewrite them.

Test isolation is achieved through `tmp_path` fixtures rather than shared state. Each test that touches the filesystem gets its own temporary directory. This makes the tests safe to run in parallel, though pytest's default mode is sequential.