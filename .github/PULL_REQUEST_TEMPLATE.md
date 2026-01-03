## Description

<!-- Describe what this PR does and why -->

## Related Issues

<!-- Link related issues: Fixes #123, Closes #456 -->

## Type of Change

<!-- Check all that apply -->

- [ ] Bug fix (non-breaking change that fixes an issue)
- [ ] New feature (non-breaking change that adds functionality)
- [ ] Breaking change (fix or feature that would cause existing functionality to change)
- [ ] Documentation update
- [ ] Refactoring (no functional changes)
- [ ] Tests (adding or updating tests)

## Layer Affected

<!-- Which architectural layer does this change? -->

- [ ] v1: Pipeline Layer (frames, processors, routing, transports)
- [ ] v2: Agent Layer (tools, executor, memory)
- [ ] v3: Runtime Layer (adapters, resolver, config)
- [ ] Providers (STT, LLM, Chat)
- [ ] Integrations (Plane, etc.)
- [ ] Application (FastAPI, webhooks)
- [ ] Documentation / Examples

## Checklist

<!-- Ensure all items are completed before requesting review -->

- [ ] I have read the [CONTRIBUTING](../CONTRIBUTING.md) guidelines
- [ ] My code follows the project's style guidelines (ruff, mypy)
- [ ] I have added tests that prove my fix/feature works
- [ ] All new and existing tests pass (`pytest`)
- [ ] I have updated documentation if needed
- [ ] I have updated CHANGELOG.md if this is user-facing

## ADR Compliance

<!-- For architectural changes, confirm ADR compliance -->

- [ ] **ADR-004**: Changes preserve frame stream as source of truth
- [ ] **ADR-006**: No secrets in frame metadata
- [ ] **ADR-007**: Multi-tenancy isolation maintained (if applicable)

## Testing

<!-- Describe how you tested this change -->

```bash
# Commands to test this change
pytest tests/test_...
```

## Screenshots / Examples

<!-- If applicable, add screenshots or code examples -->
