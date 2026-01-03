# Frames API Reference

## Base Frame

::: knomly.pipeline.frames.base.Frame
    options:
      show_source: true
      members:
        - create
        - derive
        - with_metadata
        - id
        - frame_type
        - data
        - metadata

## Specialized Frames

::: knomly.pipeline.frames.transcription.TranscriptionFrame
    options:
      show_source: true

::: knomly.pipeline.frames.error.ErrorFrame
    options:
      show_source: true

## Agent Frames

::: knomly.agent.frames.ToolCallFrame
    options:
      show_source: true

::: knomly.agent.frames.ToolResultFrame
    options:
      show_source: true

::: knomly.agent.frames.AgentResponseFrame
    options:
      show_source: true
