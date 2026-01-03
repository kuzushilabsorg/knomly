# Processors API Reference

## Base Processor

::: knomly.pipeline.processor.Processor
    options:
      show_source: true
      members:
        - name
        - process
        - can_process

::: knomly.pipeline.processor.PassthroughProcessor
    options:
      show_source: true

## Routing Processors

::: knomly.pipeline.routing.Conditional
    options:
      show_source: true

::: knomly.pipeline.routing.Switch
    options:
      show_source: true

::: knomly.pipeline.routing.TypeRouter
    options:
      show_source: true

::: knomly.pipeline.routing.FanOut
    options:
      show_source: true

## Pipeline Processors

::: knomly.pipeline.processors.extraction.ExtractionProcessor
    options:
      show_source: true

::: knomly.pipeline.processors.context_enrichment.ContextEnrichmentProcessor
    options:
      show_source: true
