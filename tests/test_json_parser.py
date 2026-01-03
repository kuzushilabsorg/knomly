"""
Tests for robust JSON parsing utilities.
"""

from knomly.utils.json_parser import (
    clean_json_string,
    extract_json_from_text,
    parse_json_safely,
    parse_standup_json,
)


class TestExtractJsonFromText:
    """Tests for extract_json_from_text."""

    def test_extracts_from_json_code_block(self):
        text = '```json\n{"key": "value"}\n```'
        result = extract_json_from_text(text)
        assert result == '{"key": "value"}'

    def test_extracts_from_plain_code_block(self):
        text = '```\n{"key": "value"}\n```'
        result = extract_json_from_text(text)
        assert result == '{"key": "value"}'

    def test_extracts_json_object_from_text(self):
        text = 'Here is the result: {"key": "value"} and some more text'
        result = extract_json_from_text(text)
        assert result == '{"key": "value"}'

    def test_extracts_json_array_from_text(self):
        text = 'Result: ["a", "b", "c"]'
        result = extract_json_from_text(text)
        assert result == '["a", "b", "c"]'

    def test_returns_original_when_no_json(self):
        text = "Just plain text"
        result = extract_json_from_text(text)
        assert result == "Just plain text"

    def test_handles_nested_braces(self):
        text = '{"outer": {"inner": "value"}}'
        result = extract_json_from_text(text)
        assert result == '{"outer": {"inner": "value"}}'


class TestCleanJsonString:
    """Tests for clean_json_string."""

    def test_removes_trailing_commas(self):
        text = '{"a": 1, "b": 2,}'
        result = clean_json_string(text)
        assert result == '{"a": 1, "b": 2}'

    def test_removes_trailing_comma_in_array(self):
        text = "[1, 2, 3,]"
        result = clean_json_string(text)
        assert result == "[1, 2, 3]"

    def test_removes_single_line_comments(self):
        text = '{"a": 1 // comment\n}'
        result = clean_json_string(text)
        assert "// comment" not in result

    def test_removes_multi_line_comments(self):
        text = '{"a": /* comment */ 1}'
        result = clean_json_string(text)
        assert "/* comment */" not in result


class TestParseJsonSafely:
    """Tests for parse_json_safely."""

    def test_parses_valid_json(self):
        text = '{"key": "value"}'
        result = parse_json_safely(text)
        assert result == {"key": "value"}

    def test_parses_json_from_markdown(self):
        text = '```json\n{"key": "value"}\n```'
        result = parse_json_safely(text)
        assert result == {"key": "value"}

    def test_returns_default_on_empty_input(self):
        result = parse_json_safely("")
        assert result == {}

    def test_returns_custom_default(self):
        result = parse_json_safely("invalid", {"default": True})
        assert result == {"default": True}

    def test_handles_trailing_commas(self):
        text = '{"a": 1,}'
        result = parse_json_safely(text)
        assert result == {"a": 1}

    def test_handles_complex_nested_json(self):
        text = """
        ```json
        {
            "items": ["a", "b"],
            "nested": {"key": "value"}
        }
        ```
        """
        result = parse_json_safely(text)
        assert result["items"] == ["a", "b"]
        assert result["nested"]["key"] == "value"


class TestParseStandupJson:
    """Tests for parse_standup_json."""

    def test_parses_complete_standup(self):
        text = """
        {
            "today_items": ["Task 1", "Task 2"],
            "yesterday_items": ["Done 1"],
            "blockers": ["Issue 1"],
            "summary": "Working on tasks"
        }
        """
        result = parse_standup_json(text)

        assert result["today_items"] == ["Task 1", "Task 2"]
        assert result["yesterday_items"] == ["Done 1"]
        assert result["blockers"] == ["Issue 1"]
        assert result["summary"] == "Working on tasks"

    def test_provides_defaults_for_missing_fields(self):
        text = '{"today_items": ["Task 1"]}'
        result = parse_standup_json(text)

        assert result["today_items"] == ["Task 1"]
        assert result["yesterday_items"] == []
        assert result["blockers"] == []
        assert result["summary"] == ""

    def test_handles_invalid_json(self):
        text = "not json at all"
        result = parse_standup_json(text)

        assert result["today_items"] == []
        assert result["yesterday_items"] == []
        assert result["blockers"] == []
        assert result["summary"] == ""

    def test_handles_string_as_single_item(self):
        # Sometimes LLM returns a string instead of array
        text = '{"today_items": "Single task", "blockers": ""}'
        result = parse_standup_json(text)

        assert result["today_items"] == ["Single task"]
        assert result["blockers"] == []

    def test_handles_markdown_wrapped_response(self):
        text = """```json
        {
            "today_items": ["Working on feature"],
            "blockers": [],
            "summary": "Coding"
        }
        ```"""
        result = parse_standup_json(text)

        assert result["today_items"] == ["Working on feature"]
        assert result["summary"] == "Coding"

    def test_filters_empty_items(self):
        text = '{"today_items": ["Task 1", "", null, "Task 2"]}'
        result = parse_standup_json(text)

        assert result["today_items"] == ["Task 1", "Task 2"]
