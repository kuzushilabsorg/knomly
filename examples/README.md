# Knomly Examples

This directory contains example implementations demonstrating Knomly's capabilities.

## Examples Overview

### 01-simple-pipeline
**Difficulty: Beginner**

Basic pipeline pattern with custom processors. Learn how to:
- Create custom Frame types
- Implement Processor classes
- Build and execute a pipeline

```bash
python -m examples.01-simple-pipeline.main
```

### 02-voice-standup
**Difficulty: Intermediate**

Complete voice-to-chat pipeline for team standups. Demonstrates:
- Audio processing with transcription
- LLM-based extraction
- Chat integration (Zulip)
- WhatsApp confirmation via Twilio

```bash
# Requires: pip install knomly[full]
python -m examples.02-voice-standup.main
```

### 03-intent-routing
**Difficulty: Intermediate**

Intent-based message routing with Switch patterns. Learn:
- Intent classification
- Conditional routing
- Branch pipelines

```bash
python -m examples.03-intent-routing.main
```

### 04-custom-transport
**Difficulty: Advanced**

Creating a custom transport adapter. Shows how to:
- Implement TransportAdapter protocol
- Register custom transports
- Handle bidirectional messaging

```bash
python -m examples.04-custom-transport.main
```

## Running Examples

1. Install Knomly:
   ```bash
   pip install knomly
   # Or for all features:
   pip install knomly[full]
   ```

2. Set up environment variables (if needed):
   ```bash
   cp .env.example .env
   # Edit .env with your API keys
   ```

3. Run an example:
   ```bash
   python -m examples.<example-name>.main
   ```

## Creating Your Own Pipeline

```python
from knomly import Pipeline, PipelineBuilder, Processor

class MyProcessor(Processor):
    @property
    def name(self) -> str:
        return "my_processor"

    async def process(self, frame, ctx):
        # Transform the frame
        return frame.derive(data=transformed_data)

# Build and run
pipeline = PipelineBuilder().add(MyProcessor()).build()
result = await pipeline.execute(initial_frame)
```
