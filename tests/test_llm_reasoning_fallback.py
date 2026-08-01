"""When a reasoning model returns empty `content` with a populated
`reasoning_content`, the LLM layer recovers a usable answer — preferring a JSON
object/array (many internal callers demand strict JSON) over the last prose
paragraph. This is what lets deepseek-style models pass FD's JSON-strict
quality checks (claim/consistency/coverage/contract, judges, routers).
"""

from __future__ import annotations

import json

import pytest

from captain_claw.llm import _extract_json_blob, _reasoning_content_fallback


def test_extract_json_blob_prefers_fenced():
    text = 'Let me think… here it is:\n```json\n{"verdict": "pass"}\n```\nDone.'
    assert json.loads(_extract_json_blob(text)) == {"verdict": "pass"}


def test_extract_json_blob_bare_trailing_object():
    text = 'Reasoning about the grants. Final answer: {"id": "c1", "verdict": "unclear"}'
    assert json.loads(_extract_json_blob(text)) == {"id": "c1", "verdict": "unclear"}


def test_extract_json_blob_trailing_array_wins_over_inline_example():
    text = ('For example the shape is {"x": 1}. After analysis the result is '
            '[{"id": "a", "ok": true}, {"id": "b", "ok": false}]')
    out = json.loads(_extract_json_blob(text))
    assert isinstance(out, list) and out[1]["id"] == "b"


def test_extract_json_blob_none_when_no_json():
    assert _extract_json_blob("just prose, no json here") is None


def test_fallback_recovers_json_from_reasoning():
    reasoning = ('I need to score each claim. Claim 1 is verifiable, claim 2 is '
                 'not. So:\n{"claims_checked": 2, "claims_confirmed": 1}')
    out = _reasoning_content_fallback(reasoning)
    assert json.loads(out) == {"claims_checked": 2, "claims_confirmed": 1}


def test_fallback_uses_last_paragraph_without_json():
    reasoning = "First I consider the scope.\n\nThe final conclusion is: proceed."
    assert _reasoning_content_fallback(reasoning) == "The final conclusion is: proceed."


def test_fallback_empty_reasoning():
    assert _reasoning_content_fallback("") == ""
