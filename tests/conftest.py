"""
Pytest configuration and fixtures for Knomly tests.
"""

import sys
from pathlib import Path

import pytest

# Add the repository root to path for imports
# This allows `from knomly.pipeline import ...` to work
repo_root = Path(__file__).parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))


@pytest.fixture
def sample_audio_bytes():
    """Sample audio bytes for testing."""
    return b"fake audio data for testing"


@pytest.fixture
def sample_phone():
    """Sample phone number for testing."""
    return "919876543210"


@pytest.fixture
def sample_transcription():
    """Sample transcription text for testing."""
    return "Today I will work on the API integration and fix the login bug."


@pytest.fixture
def sample_extraction():
    """Sample extraction result for testing."""
    return {
        "today_items": ["Work on API integration", "Fix login bug"],
        "yesterday_items": ["Completed database migration"],
        "blockers": [],
        "summary": "Good progress on backend tasks",
    }
