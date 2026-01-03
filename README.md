# Knomly

[![PyPI version](https://badge.fury.io/py/knomly.svg)](https://badge.fury.io/py/knomly)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://github.com/kuzushi-labs/knomly/actions/workflows/test.yml/badge.svg)](https://github.com/kuzushi-labs/knomly/actions/workflows/test.yml)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)

**Knomly** is a Pipecat-inspired pipeline framework for building AI-powered voice and messaging applications.

Build modular, type-safe pipelines that process audio, text, and other data through configurable processors with support for any messaging platform.

## Features

- **Modular Pipeline Architecture** - Compose complex workflows from simple processors
- **Type-Safe Frames** - Immutable data containers with full type hints
- **Flexible Routing** - Conditional, Switch, TypeRouter, Guard, and FanOut patterns
- **Transport Abstraction** - Support any messaging platform (WhatsApp, Telegram, Slack)
- **Provider System** - Pluggable STT, LLM, and Chat integrations
- **Agent Layer** - ReAct-style agent execution with tool calling
- **Multi-Tenancy** - Per-user configuration and credential management
- **Built-in Resilience** - Retry policies, circuit breakers, rate limiting
- **Observability** - Structured logging, metrics, and distributed tracing

## Installation

```bash
pip install knomly
```

With optional dependencies:

```bash
# All features
pip install knomly[full]

# Specific providers
pip install knomly[stt-whisper,llm-openai,transport-twilio]

# Development
pip install knomly[dev]
```

## Quick Start

```python
import asyncio
from knomly import Pipeline, PipelineBuilder, Processor, PipelineContext
from knomly.pipeline.frames import Frame

class UppercaseProcessor(Processor):
    @property
    def name(self) -> str:
        return "uppercase"

    async def process(self, frame: Frame, ctx: PipelineContext) -> Frame:
        return frame.derive(text=frame.text.upper())

class WordCountProcessor(Processor):
    @property
    def name(self) -> str:
        return "word_count"

    async def process(self, frame: Frame, ctx: PipelineContext) -> Frame:
        return frame.derive(count=len(frame.text.split()))

# Build pipeline
pipeline = (
    PipelineBuilder()
    .add(UppercaseProcessor())
    .add(WordCountProcessor())
    .build()
)

# Execute
async def main():
    result = await pipeline.execute(initial_frame)
    print(f"Success: {result.success}")

asyncio.run(main())
```

## Core Concepts

### Frames

Frames are immutable data containers that flow through the pipeline:

```python
from knomly.pipeline.frames import AudioInputFrame, TranscriptionFrame

# Create a frame
frame = AudioInputFrame(
    media_url="https://example.com/audio.ogg",
    mime_type="audio/ogg",
    sender_phone="919876543210",
)

# Derive a new frame (immutable)
new_frame = frame.derive(text="transcribed text")
```

### Processors

Processors transform frames and are the building blocks of pipelines:

```python
from knomly.pipeline.processor import Processor

class MyProcessor(Processor):
    @property
    def name(self) -> str:
        return "my_processor"

    async def process(self, frame: Frame, ctx: PipelineContext) -> Frame | None:
        # Transform and return frame, or None to filter
        return frame.derive(processed=True)
```

### Intent-Based Routing

Route frames based on detected intent:

```python
from knomly.pipeline.routing import Switch

pipeline = (
    PipelineBuilder()
    .add(IntentClassifier())
    .add(Switch(
        key=get_intent,
        cases={
            "greeting": GreetingHandler(),
            "question": QuestionHandler(),
            "command": CommandHandler(),
        },
        default=UnknownHandler(),
    ))
    .build()
)
```

### Transport Abstraction

Support any messaging platform with the Transport pattern:

```python
from knomly.pipeline.transports import TwilioTransport, register_transport

# Register at startup
transport = TwilioTransport(
    account_sid="...",
    auth_token="...",
    from_number="whatsapp:+1234567890",
)
register_transport(transport)

# Use in webhook handler
transport = get_transport("twilio")
frame = await transport.normalize_request(request)
```

## Architecture

```
┌────────────────────────────────────────────────────────────────────────────┐
│                              API Layer                                      │
│  Twilio Webhook → FastAPI Handler → Background Task                        │
└─────────────────────────────────┬──────────────────────────────────────────┘
                                  │
┌─────────────────────────────────▼──────────────────────────────────────────┐
│                           Pipeline Layer                                    │
│  Audio → Transcription → Intent → Router → [Processors] → Confirmation    │
└─────────────────────────────────┬──────────────────────────────────────────┘
                                  │
┌─────────────────────────────────▼──────────────────────────────────────────┐
│                            Agent Layer                                      │
│  AgentBridgeProcessor → AgentExecutor → Tool Calls → Response              │
└─────────────────────────────────┬──────────────────────────────────────────┘
                                  │
┌─────────────────────────────────▼──────────────────────────────────────────┐
│                           Provider Layer                                    │
│  STT (Gemini/Whisper) │ LLM (OpenAI/Anthropic) │ Chat (Zulip/Slack)       │
└────────────────────────────────────────────────────────────────────────────┘
```

## Providers

Knomly supports multiple providers for each service type:

### Speech-to-Text (STT)
- Google Gemini (`knomly[stt-gemini]`)
- OpenAI Whisper (`knomly[stt-whisper]`)
- Deepgram (`knomly[stt-deepgram]`)

### Large Language Models (LLM)
- OpenAI (`knomly[llm-openai]`)
- Anthropic Claude (`knomly[llm-anthropic]`)
- Google Gemini (`knomly[llm-gemini]`)

### Chat
- Zulip (`knomly[chat-zulip]`)

### Transports
- Twilio (WhatsApp, SMS) (`knomly[transport-twilio]`)
- Telegram (`knomly[transport-telegram]`)

## Configuration

Knomly uses environment variables for configuration:

```bash
# Provider API Keys
KNOMLY_GEMINI_API_KEY=your-gemini-key
KNOMLY_OPENAI_API_KEY=your-openai-key
KNOMLY_ANTHROPIC_API_KEY=your-anthropic-key

# Zulip
KNOMLY_ZULIP_SITE=https://chat.example.com
KNOMLY_ZULIP_BOT_EMAIL=bot@example.com
KNOMLY_ZULIP_API_KEY=your-zulip-key

# Twilio
KNOMLY_TWILIO_ACCOUNT_SID=your-account-sid
KNOMLY_TWILIO_AUTH_TOKEN=your-auth-token
KNOMLY_TWILIO_WHATSAPP_NUMBER=whatsapp:+1234567890

# Database
KNOMLY_MONGODB_URL=mongodb://localhost:27017
KNOMLY_MONGODB_DATABASE=knomly
```

See [.env.example](.env.example) for a complete list.

## Examples

See the [examples](examples/) directory:

- **01-simple-pipeline** - Basic pipeline with custom processors
- **02-voice-standup** - Voice-to-chat standup pipeline
- **03-intent-routing** - Intent classification and routing
- **04-custom-transport** - Creating custom transport adapters

## Development

```bash
# Clone repository
git clone https://github.com/kuzushi-labs/knomly.git
cd knomly

# Create virtual environment
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows

# Install with dev dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Run tests with coverage
pytest --cov=knomly --cov-report=html

# Run linting
ruff check .
ruff format .

# Type checking
mypy knomly
```

## Project Structure

```
knomly/
├── knomly/                 # Main package
│   ├── adapters/          # Tool adapters (OpenAPI, etc.)
│   ├── agent/             # Agent layer (executor, processor)
│   ├── app/               # FastAPI application
│   ├── config/            # Configuration schemas
│   ├── integrations/      # External integrations (Plane, etc.)
│   ├── pipeline/          # Core pipeline framework
│   │   ├── frames/        # Frame types
│   │   ├── processors/    # Built-in processors
│   │   └── transports/    # Transport adapters
│   ├── providers/         # Service providers (STT, LLM, Chat)
│   ├── runtime/           # Dynamic configuration
│   ├── tools/             # Tool system
│   └── utils/             # Utilities
├── tests/                 # Test suite
├── examples/              # Example applications
├── docs/                  # Documentation
└── configs/               # Runtime configuration files
```

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

MIT License. See [LICENSE](LICENSE) for details.

## Acknowledgments

Inspired by [Pipecat](https://github.com/pipecat-ai/pipecat) - the open-source framework for voice and multimodal AI.

## Support

- [GitHub Issues](https://github.com/kuzushi-labs/knomly/issues) - Bug reports and feature requests
- [Discussions](https://github.com/kuzushi-labs/knomly/discussions) - Questions and community
