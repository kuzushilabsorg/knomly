# Tools API Reference

## Tool Base

::: knomly.tools.base.Tool
    options:
      show_source: true
      members:
        - name
        - description
        - input_schema
        - execute

::: knomly.tools.base.ToolResult
    options:
      show_source: true

## Tool Registry

::: knomly.tools.registry.ToolRegistry
    options:
      show_source: true
      members:
        - register
        - get
        - list_tools

## Tool Factory

::: knomly.tools.factory.ToolContext
    options:
      show_source: true

## Adapters

::: knomly.adapters.openapi_adapter.OpenAPIToolAdapter
    options:
      show_source: true
      members:
        - build_tool

::: knomly.adapters.base.ToolBuilder
    options:
      show_source: true
      members:
        - build_tools
