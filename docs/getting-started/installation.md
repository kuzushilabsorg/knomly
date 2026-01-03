# Installation

## Requirements

- Python 3.11 or higher
- pip or uv package manager

## Basic Installation

Install the core framework:

```bash
pip install knomly
```

## Installation with Extras

Knomly uses optional dependencies to keep the core package lightweight. Install what you need:

### STT Providers

```bash
# Google Gemini
pip install knomly[stt-gemini]

# OpenAI Whisper
pip install knomly[stt-whisper]

# Deepgram
pip install knomly[stt-deepgram]
```

### LLM Providers

```bash
# OpenAI
pip install knomly[llm-openai]

# Anthropic
pip install knomly[llm-anthropic]

# Google Gemini
pip install knomly[llm-gemini]
```

### Transport Adapters

```bash
# Twilio (WhatsApp)
pip install knomly[transport-twilio]

# Telegram
pip install knomly[transport-telegram]
```

### Chat Providers

```bash
# Zulip
pip install knomly[chat-zulip]
```

### Bundles

```bash
# All providers
pip install knomly[providers]

# All transports
pip install knomly[transports]

# Everything
pip install knomly[full]
```

### Development

```bash
pip install knomly[dev]
```

## Using uv

[uv](https://github.com/astral-sh/uv) is a fast Python package manager:

```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install knomly
uv pip install knomly[full]
```

## Verify Installation

```python
import knomly
print(knomly.__version__)
# Output: 0.1.0
```

## Next Steps

- [Quick Start Tutorial](quickstart.md)
- [Core Concepts](../concepts/frames.md)
- [Examples](https://github.com/kuzushi-labs/knomly/tree/main/examples)
