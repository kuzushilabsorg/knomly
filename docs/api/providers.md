# Providers API Reference

## Provider Registry

::: knomly.providers.registry.ProviderRegistry
    options:
      show_source: true
      members:
        - register_stt
        - register_llm
        - register_chat
        - get_stt
        - get_llm
        - get_chat

## LLM Providers

::: knomly.providers.llm.base.LLMProvider
    options:
      show_source: true
      members:
        - complete

::: knomly.providers.llm.base.LLMConfig
    options:
      show_source: true

::: knomly.providers.llm.base.Message
    options:
      show_source: true
      members:
        - system
        - user
        - assistant

::: knomly.providers.llm.openai.OpenAILLMProvider
    options:
      show_source: true

::: knomly.providers.llm.openai.AnthropicLLMProvider
    options:
      show_source: true

## STT Providers

::: knomly.providers.stt.base.STTProvider
    options:
      show_source: true
      members:
        - transcribe

::: knomly.providers.stt.gemini.GeminiSTTProvider
    options:
      show_source: true

## Chat Providers

::: knomly.providers.chat.base.ChatProvider
    options:
      show_source: true
      members:
        - send_message

::: knomly.providers.chat.zulip.ZulipChatProvider
    options:
      show_source: true
