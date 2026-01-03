# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2026-01-03

### Added

- **Core Pipeline Framework**
  - `Frame` base class with UUID, timestamps, metadata, and lineage tracking
  - `Processor` abstract base class for single-responsibility transformers
  - `Pipeline` orchestrator for sequential frame processing
  - `PipelineBuilder` for fluent pipeline construction
  - `PipelineContext` for shared state and provider access
  - `PipelineResult` for execution results with timing metrics

- **Frame Types**
  - `AudioInputFrame` - Audio data from messaging platforms
  - `TextInputFrame` - Text messages
  - `TranscriptionFrame` - STT results with language detection
  - `ExtractionFrame` - LLM extraction results (standup items, blockers)
  - `ZulipMessageFrame` - Chat message actions
  - `ConfirmationFrame` - User confirmations
  - `UserResponseFrame` - Generic user responses
  - `ErrorFrame` - Error handling with recovery info

- **Routing Processors**
  - `Conditional` - If/else routing based on predicates
  - `Switch` - Multi-way routing by key function
  - `TypeRouter` - Route by frame type
  - `Guard` - Validation with error handling
  - `FanOut` - Parallel processing with multiple processors
  - `IntentClassifierProcessor` - LLM-based intent classification

- **Transport Abstraction**
  - `TransportAdapter` protocol for messaging platform abstraction
  - `TransportRegistry` for global transport management
  - `TwilioTransport` implementation for WhatsApp/SMS

- **Provider System**
  - `STTProvider` protocol with Deepgram and Gemini implementations
  - `LLMProvider` protocol with OpenAI implementation
  - `ChatProvider` protocol with Zulip implementation
  - `ProviderRegistry` for centralized provider management

- **Built-in Processors**
  - `AudioDownloadProcessor` - Download audio from URLs
  - `TranscriptionProcessor` - Audio to text via STT providers
  - `StandupExtractionProcessor` - Extract standup items via LLM
  - `ZulipProcessor` - Post messages to Zulip
  - `ConfirmationProcessor` - Send confirmations via transport
  - `UnknownIntentProcessor` - Handle unrecognized intents
  - `PassthroughProcessor` - Pass frames unchanged

- **Resilience**
  - `ResilientProcessor` wrapper with retry and circuit breaker
  - Configurable retry policies with exponential backoff
  - Circuit breaker pattern for fault tolerance

- **Configuration**
  - MongoDB-backed configuration service
  - TTL caching for performance
  - Dynamic prompt management

- **Documentation**
  - Getting started guide
  - Core concepts (Frames, Processors, Pipelines)
  - API reference
  - Example implementations

### Infrastructure

- FastAPI webhook integration
- Docker and docker-compose support
- Pre-commit hooks for code quality
- Comprehensive test suite (338+ tests)
- Type hints throughout
- BSD-2-Clause license

[Unreleased]: https://github.com/kuzushi-labs/knomly/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/kuzushi-labs/knomly/releases/tag/v0.1.0
