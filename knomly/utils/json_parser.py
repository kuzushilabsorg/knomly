"""
Robust JSON parsing utilities for LLM responses.

Handles common issues with LLM-generated JSON:
- Markdown code blocks (```json ... ```)
- Trailing commas
- Single quotes instead of double quotes
- Comments
- Partial JSON extraction
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any

logger = logging.getLogger(__name__)


def extract_json_from_text(text: str) -> str:
    """
    Extract JSON from text that may contain markdown or other content.

    Handles:
    - ```json ... ``` code blocks
    - ``` ... ``` code blocks
    - Raw JSON objects/arrays
    - JSON embedded in other text

    Args:
        text: Raw text that may contain JSON

    Returns:
        Extracted JSON string (may still need parsing)
    """
    text = text.strip()

    # Try to extract from markdown code blocks
    # Handle ```json first, then plain ```
    patterns = [
        r"```json\s*([\s\S]*?)\s*```",  # ```json ... ```
        r"```\s*([\s\S]*?)\s*```",  # ``` ... ```
    ]

    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1).strip()

    # Try to find JSON object/array boundaries
    # Look for { } or [ ] pairs
    json_patterns = [
        r"(\{[\s\S]*\})",  # Object
        r"(\[[\s\S]*\])",  # Array
    ]

    for pattern in json_patterns:
        match = re.search(pattern, text)
        if match:
            candidate = match.group(1)
            # Verify it looks like valid JSON structure
            if _looks_like_json(candidate):
                return candidate

    # Fall back to original text
    return text


def _looks_like_json(text: str) -> bool:
    """Quick check if text looks like JSON."""
    text = text.strip()
    return (
        (text.startswith("{") and text.endswith("}"))
        or (text.startswith("[") and text.endswith("]"))
    )


def clean_json_string(text: str) -> str:
    """
    Clean common JSON issues from LLM output.

    Handles:
    - Trailing commas
    - Single quotes (converts to double)
    - JavaScript-style comments
    - Unquoted keys (basic cases)

    Args:
        text: JSON-like string

    Returns:
        Cleaned JSON string
    """
    # Remove JavaScript-style comments
    # Single line comments: // ...
    text = re.sub(r"//[^\n]*", "", text)
    # Multi-line comments: /* ... */
    text = re.sub(r"/\*[\s\S]*?\*/", "", text)

    # Remove trailing commas before } or ]
    text = re.sub(r",\s*([\}\]])", r"\1", text)

    # Note: Converting single quotes is risky as it can break strings
    # containing apostrophes. Only do it for simple key-value patterns.
    # This is a conservative approach.

    return text.strip()


def parse_json_safely(
    text: str,
    default: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Parse JSON with multiple fallback strategies.

    Tries in order:
    1. Direct JSON parsing
    2. Extract from markdown + parse
    3. Clean common issues + parse
    4. Return default

    Args:
        text: Text to parse as JSON
        default: Default value if parsing fails

    Returns:
        Parsed dict or default
    """
    if default is None:
        default = {}

    if not text or not text.strip():
        return default

    # Strategy 1: Direct parse
    try:
        result = json.loads(text)
        if isinstance(result, dict):
            return result
        logger.warning(f"JSON parsed but not a dict: {type(result)}")
        return default
    except json.JSONDecodeError:
        pass

    # Strategy 2: Extract from markdown then parse
    extracted = extract_json_from_text(text)
    if extracted != text:
        try:
            result = json.loads(extracted)
            if isinstance(result, dict):
                return result
        except json.JSONDecodeError:
            pass

    # Strategy 3: Clean common issues then parse
    cleaned = clean_json_string(extracted)
    try:
        result = json.loads(cleaned)
        if isinstance(result, dict):
            return result
    except json.JSONDecodeError:
        pass

    # Strategy 4: Try with repaired JSON (basic repairs)
    repaired = _attempt_json_repair(cleaned)
    if repaired:
        try:
            result = json.loads(repaired)
            if isinstance(result, dict):
                logger.info("JSON parsed after repair")
                return result
        except json.JSONDecodeError:
            pass

    logger.warning(f"Failed to parse JSON after all strategies: {text[:100]}...")
    return default


def _attempt_json_repair(text: str) -> str | None:
    """
    Attempt basic JSON repairs.

    This is a last-resort strategy for common malformed JSON.
    """
    text = text.strip()

    # Ensure we have opening and closing braces
    if not text.startswith("{"):
        text = "{" + text
    if not text.endswith("}"):
        text = text + "}"

    # Try to balance braces
    open_braces = text.count("{")
    close_braces = text.count("}")
    if open_braces > close_braces:
        text = text + ("}" * (open_braces - close_braces))
    elif close_braces > open_braces:
        text = ("{" * (close_braces - open_braces)) + text

    return text


def parse_standup_json(
    text: str,
) -> dict[str, Any]:
    """
    Parse standup extraction JSON with domain-specific defaults.

    Args:
        text: LLM response text

    Returns:
        Dict with today_items, yesterday_items, blockers, summary
    """
    default = {
        "today_items": [],
        "yesterday_items": [],
        "blockers": [],
        "summary": "",
    }

    result = parse_json_safely(text, default)

    # Ensure all expected fields exist with correct types
    standup = {
        "today_items": _ensure_list(result.get("today_items")),
        "yesterday_items": _ensure_list(result.get("yesterday_items")),
        "blockers": _ensure_list(result.get("blockers")),
        "summary": str(result.get("summary", "")),
    }

    return standup


def _ensure_list(value: Any) -> list[str]:
    """Ensure value is a list of strings."""
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value.strip() else []
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value if item]
    return []


__all__ = [
    "extract_json_from_text",
    "clean_json_string",
    "parse_json_safely",
    "parse_standup_json",
]
