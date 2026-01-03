# Contributing to Knomly

Thank you for your interest in contributing to Knomly! This document provides guidelines and instructions for contributing.

## Code of Conduct

Be respectful, inclusive, and constructive. We're all here to build great software together.

## Getting Started

### Prerequisites

- Python 3.11 or higher
- Git
- A GitHub account

### Development Setup

1. **Fork and clone the repository**

   ```bash
   git clone https://github.com/YOUR_USERNAME/knomly.git
   cd knomly
   ```

2. **Create a virtual environment**

   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install development dependencies**

   ```bash
   pip install -e ".[dev]"
   ```

4. **Install pre-commit hooks**

   ```bash
   pre-commit install
   ```

5. **Verify setup**

   ```bash
   pytest
   ```

## Development Workflow

### Branching

- Create a feature branch from `main`:
  ```bash
  git checkout -b feature/your-feature-name
  ```

- Use descriptive branch names:
  - `feature/` - New features
  - `fix/` - Bug fixes
  - `docs/` - Documentation changes
  - `refactor/` - Code refactoring
  - `test/` - Test additions/changes

### Making Changes

1. **Write your code** following the project style
2. **Add tests** for new functionality
3. **Update documentation** if needed
4. **Run tests locally**:
   ```bash
   pytest
   ```
5. **Run linting**:
   ```bash
   ruff check .
   ruff format .
   ```

### Commit Messages

Follow [Conventional Commits](https://www.conventionalcommits.org/):

```
type(scope): description

[optional body]

[optional footer]
```

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation
- `style`: Formatting (no code change)
- `refactor`: Code refactoring
- `test`: Adding tests
- `chore`: Maintenance

Examples:
```
feat(transport): add Telegram transport adapter
fix(pipeline): handle empty frame list correctly
docs(readme): add installation instructions
```

### Pull Requests

1. **Push your branch**:
   ```bash
   git push origin feature/your-feature-name
   ```

2. **Open a Pull Request** on GitHub

3. **Fill out the PR template** with:
   - Clear description of changes
   - Related issue numbers
   - Testing done
   - Screenshots if applicable

4. **Wait for review** - maintainers will review and provide feedback

## Code Style

### Python Style

- Follow [PEP 8](https://pep8.org/)
- Use type hints everywhere
- Maximum line length: 100 characters
- Use `ruff` for linting and formatting

### Docstrings

Use Google-style docstrings:

```python
def process_frame(frame: Frame, ctx: PipelineContext) -> Frame:
    """
    Process a frame through the pipeline.

    Args:
        frame: The input frame to process.
        ctx: Pipeline context with providers and state.

    Returns:
        The processed frame.

    Raises:
        ValueError: If frame is invalid.
    """
```

### Type Hints

Always use type hints:

```python
from typing import Sequence

async def process(
    self,
    frame: Frame,
    ctx: PipelineContext,
) -> Frame | Sequence[Frame] | None:
    ...
```

## Testing

### Running Tests

```bash
# All tests
pytest

# With coverage
pytest --cov=knomly

# Specific test file
pytest tests/test_pipeline.py

# Verbose output
pytest -v

# Stop on first failure
pytest -x
```

### Writing Tests

- Place tests in `tests/` directory
- Name test files `test_*.py`
- Use pytest fixtures for setup
- Test both success and failure cases

```python
import pytest
from knomly import Pipeline, PipelineBuilder

class TestPipeline:
    def test_empty_pipeline_raises(self):
        with pytest.raises(ValueError):
            Pipeline([])

    @pytest.mark.asyncio
    async def test_pipeline_executes_processors(self):
        pipeline = PipelineBuilder().add(MockProcessor()).build()
        result = await pipeline.execute(mock_frame)
        assert result.success
```

## Adding New Features

### New Processor

1. Create processor in `knomly/pipeline/processors/`
2. Implement `Processor` protocol
3. Add tests in `tests/test_processors.py`
4. Update `__init__.py` exports
5. Add documentation

### New Transport

1. Create transport in `knomly/pipeline/transports/`
2. Implement `TransportAdapter` protocol
3. Add tests in `tests/test_transports.py`
4. Update `__init__.py` exports
5. Add documentation and example

### New Provider

1. Create provider in `knomly/providers/`
2. Implement appropriate protocol (STTProvider, LLMProvider, etc.)
3. Add tests
4. Update `pyproject.toml` optional dependencies
5. Add documentation

## Documentation

- Keep documentation up-to-date with code changes
- Use Markdown for documentation files
- Add docstrings to all public APIs
- Include code examples where helpful

## Release Process

Releases are managed by maintainers:

1. Update version in `pyproject.toml`
2. Update `CHANGELOG.md`
3. Create release commit
4. Tag with version
5. Push to trigger release workflow

## Questions?

- Open a [GitHub Issue](https://github.com/kuzushi-labs/knomly/issues)
- Check existing issues and discussions

Thank you for contributing!
