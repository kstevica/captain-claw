"""Structured output validation for orchestrator tasks.

Validates task outputs against JSON Schema definitions with automatic
retry on failure. Inspired by Open Multi-Agent's Zod-based outputSchema.

Usage:
    schema = {"type": "object", "properties": {"summary": {"type": "string"}}, "required": ["summary"]}
    valid, error, parsed = validate_task_output(output_text, schema)
    if not valid:
        retry_prompt = build_retry_prompt(original_prompt, output_text, error)
"""

from __future__ import annotations

import json
import re
from typing import Any

from captain_claw.logging import get_logger

log = get_logger(__name__)


def extract_json_from_text(text: str) -> tuple[str | None, Any | None]:
    """Extract JSON from LLM output that may contain markdown or prose.

    Tries multiple strategies in order:
    1. Parse the entire text as JSON directly.
    2. Extract from ```json ... ``` fenced code blocks.
    3. Extract from ``` ... ``` generic code blocks.
    4. Find the first { ... } or [ ... ] balanced block.

    Returns:
        Tuple of (raw_json_string, parsed_object) or (None, None) if
        no valid JSON found.
    """
    text = text.strip()

    # Strategy 1: Entire text is valid JSON.
    parsed = _try_parse(text)
    if parsed is not None:
        return text, parsed

    # Strategy 2: Fenced json code block.
    for pattern in [
        r"```json\s*\n(.*?)\n\s*```",
        r"```\s*\n(.*?)\n\s*```",
    ]:
        match = re.search(pattern, text, re.DOTALL)
        if match:
            candidate = match.group(1).strip()
            parsed = _try_parse(candidate)
            if parsed is not None:
                return candidate, parsed

    # Strategy 3: First balanced { ... } block.
    for open_char, close_char in [("{", "}"), ("[", "]")]:
        start = text.find(open_char)
        if start == -1:
            continue
        depth = 0
        in_string = False
        escape_next = False
        for i in range(start, len(text)):
            ch = text[i]
            if escape_next:
                escape_next = False
                continue
            if ch == "\\":
                escape_next = True
                continue
            if ch == '"' and not escape_next:
                in_string = not in_string
                continue
            if in_string:
                continue
            if ch == open_char:
                depth += 1
            elif ch == close_char:
                depth -= 1
                if depth == 0:
                    candidate = text[start : i + 1]
                    parsed = _try_parse(candidate)
                    if parsed is not None:
                        return candidate, parsed
                    break

    return None, None


def validate_json_against_schema(
    data: Any,
    schema: dict[str, Any],
) -> tuple[bool, str | None]:
    """Validate parsed JSON data against a JSON Schema.

    Uses a lightweight validation approach that covers the common
    cases (type, required, properties, items, enum) without requiring
    the full jsonschema library.

    Returns:
        (True, None) on success, (False, error_message) on failure.
    """
    # Try jsonschema if available (more comprehensive).
    try:
        import jsonschema  # type: ignore[import-untyped]

        jsonschema.validate(data, schema)
        return True, None
    except ImportError:
        pass
    except Exception as e:
        return False, str(e)

    # Fallback: lightweight validation.
    errors = _validate_node(data, schema, path="$")
    if errors:
        return False, "; ".join(errors[:5])  # cap at 5 errors
    return True, None


def validate_task_output(
    output_text: str,
    schema: dict[str, Any],
) -> tuple[bool, str | None, dict | list | None]:
    """Validate an LLM task output against a JSON Schema.

    1. Extracts JSON from the output text (handling markdown fences, prose).
    2. Validates against the schema.

    Returns:
        (valid, error_message, parsed_data)
        - valid: True if extraction and validation both succeeded.
        - error_message: Description of what went wrong (None on success).
        - parsed_data: The parsed JSON data (None on failure).
    """
    if not output_text or not output_text.strip():
        return False, "Output is empty — expected JSON matching the schema.", None

    raw_json, parsed = extract_json_from_text(output_text)
    if parsed is None:
        return (
            False,
            (
                "Could not extract valid JSON from the output. "
                "Please respond with a JSON object (optionally in a ```json code fence) "
                "that matches the required schema."
            ),
            None,
        )

    valid, error = validate_json_against_schema(parsed, schema)
    if not valid:
        return (
            False,
            f"JSON was extracted but does not match the required schema: {error}",
            None,
        )

    return True, None, parsed


def build_retry_prompt(
    original_prompt: str,
    output_text: str,
    validation_error: str,
    schema: dict[str, Any],
) -> str:
    """Build a follow-up prompt for one retry after validation failure.

    Includes the original task, the failed output, the validation error,
    and the expected schema so the LLM can correct its response.
    """
    schema_str = json.dumps(schema, indent=2)

    return (
        f"{original_prompt}\n\n"
        f"---\n\n"
        f"**IMPORTANT: Your previous response did not produce valid structured output.**\n\n"
        f"Validation error:\n```\n{validation_error}\n```\n\n"
        f"Your previous output (excerpt):\n```\n{output_text[:2000]}\n```\n\n"
        f"Expected JSON Schema:\n```json\n{schema_str}\n```\n\n"
        f"Please try again. Respond with ONLY a valid JSON object that matches "
        f"the schema above. You may wrap it in a ```json code fence."
    )


def schema_summary(schema: dict[str, Any]) -> str:
    """Generate a human-readable one-line summary of a JSON Schema.

    Used for display in UI badges and log messages.
    """
    schema_type = schema.get("type", "object")
    if schema_type == "object":
        props = schema.get("properties", {})
        required = schema.get("required", [])
        if props:
            fields = []
            for name in list(props.keys())[:5]:
                marker = "*" if name in required else ""
                fields.append(f"{name}{marker}")
            extra = f" +{len(props) - 5}" if len(props) > 5 else ""
            return f"{{{', '.join(fields)}{extra}}}"
        return "{...}"
    elif schema_type == "array":
        items = schema.get("items", {})
        item_type = items.get("type", "any")
        return f"[{item_type}...]"
    return schema_type


# ------------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------------


def _try_parse(text: str) -> Any | None:
    """Attempt to parse a string as JSON, returning None on failure."""
    try:
        return json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return None


def _validate_node(
    data: Any,
    schema: dict[str, Any],
    path: str = "$",
) -> list[str]:
    """Lightweight recursive JSON Schema validator (fallback)."""
    errors: list[str] = []
    schema_type = schema.get("type")

    # Type checking
    if schema_type:
        type_map = {
            "string": str,
            "number": (int, float),
            "integer": int,
            "boolean": bool,
            "array": list,
            "object": dict,
            "null": type(None),
        }
        expected = type_map.get(schema_type)
        if expected and not isinstance(data, expected):
            errors.append(
                f"{path}: expected type '{schema_type}', got '{type(data).__name__}'"
            )
            return errors  # type mismatch, no point checking deeper

    # Enum
    if "enum" in schema and data not in schema["enum"]:
        errors.append(f"{path}: value must be one of {schema['enum']}")

    # Object properties
    if schema_type == "object" and isinstance(data, dict):
        required = set(schema.get("required", []))
        for field in required:
            if field not in data:
                errors.append(f"{path}.{field}: required field missing")
        properties = schema.get("properties", {})
        for key, prop_schema in properties.items():
            if key in data:
                errors.extend(_validate_node(data[key], prop_schema, f"{path}.{key}"))

    # Array items
    if schema_type == "array" and isinstance(data, list):
        items_schema = schema.get("items")
        if items_schema:
            for i, item in enumerate(data[:20]):  # validate first 20
                errors.extend(_validate_node(item, items_schema, f"{path}[{i}]"))

    return errors
