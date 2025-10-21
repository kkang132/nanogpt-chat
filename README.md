# NanoGPT Chat & Fine-tuning System

A web-based chat interface for interacting with GPT-2, with automatic data collection for fine-tuning on your conversations.

## Features

- 🤖 Chat with GPT-2 through a beautiful web interface
- 💾 Automatic conversation logging for fine-tuning
- 🎯 Easy fine-tuning pipeline on your chat data
- 📊 Real-time statistics on collected conversations

## Quick Start

### 1. Activate the virtual environment

```bash
cd /Users/kris/IdeaProjects/nanogpt-chat
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
This project builds upon Karpathy's excellent nanoGPT implementation, which provides a clean, educational, and efficient way to work with GPT models. The nanoGPT codebase prioritizes "teeth over education" and serves as the foundation for our chat system.

### License
This project follows the same MIT license as nanoGPT. See the [nanoGPT LICENSE](nanoGPT/LICENSE) for details.

## Acknowledgments

Special thanks to:
- **Andrej Karpathy** for creating nanoGPT and making GPT training accessible
- **OpenAI** for the GPT-2 model and research
- **Hugging Face** for the transformers library and model ecosystem
- The broader open-source ML community for tools and inspiration
