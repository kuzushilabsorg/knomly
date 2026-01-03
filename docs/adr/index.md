# Architectural Decision Records

This directory contains Architectural Decision Records (ADRs) that document key design decisions for Knomly.

## What are ADRs?

ADRs capture important architectural decisions along with their context and consequences. They provide a permanent record of why certain decisions were made.

## Index

| ADR | Title | Status |
|-----|-------|--------|
| [ADR-003](ADR-003-integration-pattern.md) | Integration Pattern | Accepted |
| [ADR-004](ADR-004-v1-v2-invariants.md) | V1/V2 Layer Invariants | Accepted |
| [ADR-005](ADR-005-agentic-layer.md) | Agentic Layer Design | Accepted |
| [ADR-006](ADR-006-architectural-invariants.md) | Architectural Invariants | Accepted |
| [ADR-007](ADR-007-multi-tenancy-design.md) | Multi-Tenancy Design | Accepted |

## Template

When creating a new ADR, use this template:

```markdown
# ADR-NNN: Title

## Status
Proposed | Accepted | Deprecated | Superseded by ADR-XXX

## Context
What is the issue that we're seeing that is motivating this decision?

## Decision
What is the change we're proposing and/or doing?

## Consequences
What becomes easier or more difficult because of this change?
```
