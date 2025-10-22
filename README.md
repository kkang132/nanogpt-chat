# NanoGPT Chat & Fine-tuning System

A web-based chat interface for interacting with GPT-2, with automatic data collection for fine-tuning on your conversations.

## Features

- 🤖 Chat with GPT-2 through a web interface
- 💾 Automatic conversation logging for fine-tuning
- 🎯 Fine-tuning pipeline on your chat data
- 📊 Real-time statistics on collected conversations

## Quick Start

### 1. Activate the virtual environment

```bash
cd /Users/.../nanogpt-chat
source venv/bin/activate
```

### 2. Start the chat server

```bash
python app.py
```

The server will start at `http://localhost:5000`

### 3. Chat with the model

Open your browser and navigate to `http://localhost:5000`. Start chatting! All conversations are automatically saved to `chat_history.jsonl`.

### 4. Fine-tune on your data

After collecting enough conversations (recommend 50+), run:

```bash
python finetune.py
```

This will:
- Prepare your chat data for training
- Fine-tune GPT-2 on your conversations
- Save the fine-tuned model in `models/`

### 5. Use your fine-tuned model

Update `app.py` to load your fine-tuned checkpoint instead of the base GPT-2 model.

## Code Protection (ast-grep)

This project uses ast-grep to protect critical code patterns from unintended modifications by AI agents or developers.

### Protected Patterns

The following patterns are locked and cannot be modified:
- **GPT model configuration** (`GPTConfig`)
- **Device selection logic** (`device = ...`)
- **Model checkpoint loading** (`torch.load()`)

### Setup (One-time)

1. **Install ast-grep:**
   ```bash
   brew install ast-grep
   ```

2. **Set up the pre-commit hook:**
   ```bash
   # Copy the hook to your local .git/hooks directory
   cat > .git/hooks/pre-commit << 'EOF'
   #!/bin/bash

   # Get list of staged Python files
   STAGED_FILES=$(git diff --cached --name-only --diff-filter=ACM | grep '\.py$')

   if [ -z "$STAGED_FILES" ]; then
       # No Python files staged, allow commit
       exit 0
   fi

   echo "🔍 Running ast-grep validation on changed lines in staged Python files..."

   # For each staged file, check only the added/modified lines
   HAS_VIOLATIONS=0

   for file in $STAGED_FILES; do
       # Get the staged version of the file
       git show ":$file" > /tmp/staged_${file##*/}

       # Run ast-grep on the staged version
       if ! ast-grep scan /tmp/staged_${file##*/} 2>&1 | grep -q "error\["; then
           continue
       fi

       # If we get here, there are violations
       echo ""
       echo "❌ Violations found in $file:"
       ast-grep scan /tmp/staged_${file##*/}
       HAS_VIOLATIONS=1

       # Clean up temp file
       rm /tmp/staged_${file##*/}
   done

   if [ $HAS_VIOLATIONS -eq 1 ]; then
       echo ""
       echo "❌ ast-grep validation failed!"
       echo "Your changes violate protected code patterns."
       echo "Please review the errors above and modify your changes."
       echo ""
       echo "To bypass this check (not recommended): git commit --no-verify"
       exit 1
   fi

   echo "✅ ast-grep validation passed!"
   exit 0
   EOF

   chmod +x .git/hooks/pre-commit
   ```

### How It Works

When you commit changes to Python files, the pre-commit hook:
1. Scans staged files for protected patterns
2. Blocks the commit if violations are found
3. Shows you which patterns were violated

To bypass (use carefully):
```bash
git commit --no-verify -m "your message"
```

### Modifying Protection Rules

Rules are defined in `.ast-grep/rules/`. To add or modify protections, edit the YAML files in that directory.

## Project Structure

```
nanogpt-chat/
├── app.py                 # Flask web server
├── finetune.py           # Fine-tuning script
├── templates/
│   └── index.html        # Web interface
├── nanoGPT/              # Karpathy's nanoGPT (cloned)
├── chat_history.jsonl    # Logged conversations
├── data/                 # Prepared training data
└── models/               # Fine-tuned model checkpoints
```

## Configuration

Edit the following parameters in `app.py` for generation:
- `max_tokens`: Length of generated responses
- `temperature`: Randomness (higher = more creative)
- `top_k`: Top-k sampling parameter

Edit `finetune.py` for training:
- `batch_size`: Batch size for training
- `max_iters`: Number of training iterations
- `learning_rate`: Learning rate for optimizer

## Requirements

- Python 3.8+
- PyTorch
- Flask
- transformers
- tiktoken

All dependencies are already installed in the virtual environment.

## Credits & Attribution

### Core Dependencies
- **[nanoGPT](https://github.com/karpathy/nanoGPT)** by [Andrej Karpathy](https://karpathy.ai/) - The foundation GPT implementation used in this project
- **[GPT-2](https://openai.com/research/better-language-models)** by OpenAI - The base language model
- **[transformers](https://huggingface.co/transformers/)** by Hugging Face - Model loading and tokenization

### Inspiration
This project builds upon Karpathy's nanoGPT implementation. This is primarily for educational purposes.

### License
This project follows the same MIT license as nanoGPT. See the [nanoGPT LICENSE](nanoGPT/LICENSE) for details.

## Acknowledgments

Special thanks to:
- **Andrej Karpathy** for creating nanoGPT and making GPT training accessible
- **OpenAI** for the GPT-2 model and research
- **Hugging Face** for the transformers library and model ecosystem
- The broader open-source ML community for tools and inspiration
