"""Mrav digestion — map-reduce big tool results into small observations.

A tool result larger than the observation budget becomes its own set of
sub-8k LLM calls: chunk → per-chunk condense → combine. Mirrors the
chunked-processing mixin's shape, sized for the micro cap.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

from captain_claw.mrav.ledger import CHARS_PER_TOKEN, estimate_tokens, truncate_tokens
from captain_claw.mrav.prompts import DIGEST_INSTRUCTION

# complete_fn(system, user, max_tokens) -> str
CompleteFn = Callable[[str, str, int], Awaitable[str]]

_MAX_CHUNKS = 8


def split_by_tokens(text: str, chunk_tokens: int) -> list[str]:
    """Split text into ~chunk_tokens pieces on line boundaries where possible."""
    if estimate_tokens(text) <= chunk_tokens:
        return [text]
    chunk_chars = max(256, int(chunk_tokens * CHARS_PER_TOKEN))
    chunks: list[str] = []
    remaining = text
    while remaining and len(chunks) < _MAX_CHUNKS:
        if len(remaining) <= chunk_chars:
            chunks.append(remaining)
            break
        cut = remaining.rfind("\n", chunk_chars // 2, chunk_chars)
        if cut < 0:
            cut = chunk_chars
        chunks.append(remaining[:cut])
        remaining = remaining[cut:]
    if remaining and len(chunks) >= _MAX_CHUNKS:
        # Content beyond the chunk cap is tail-sampled, not silently dropped.
        chunks[-1] = chunks[-1] + "\n" + truncate_tokens(remaining, chunk_tokens // 2, keep="tail")
    return chunks


async def digest_text(
    complete_fn: CompleteFn,
    task: str,
    label: str,
    text: str,
    target_tokens: int,
    chunk_tokens: int = 4500,
) -> str:
    """Condense *text* to ~target_tokens via per-chunk calls + one combine.

    Every call stays well under the input cap: instruction + task line +
    one chunk. Falls back to plain truncation if the model output is empty.
    """
    words = max(40, int(target_tokens * 0.75))
    instruction = DIGEST_INSTRUCTION.format(label=label, words=words)
    task_line = f"TASK: {truncate_tokens(task, 200, keep='head')}"

    chunks = split_by_tokens(text, chunk_tokens)
    partials: list[str] = []
    for i, chunk in enumerate(chunks):
        header = f"{task_line}\n\n{label} (part {i + 1}/{len(chunks)}):\n" if len(chunks) > 1 else f"{task_line}\n\n{label}:\n"
        out = await complete_fn(instruction, header + chunk, max(128, target_tokens + 64))
        partials.append((out or "").strip())

    combined = "\n".join(p for p in partials if p)
    if not combined:
        return truncate_tokens(text, target_tokens, keep="split")
    if len(partials) > 1 and estimate_tokens(combined) > target_tokens:
        out = await complete_fn(
            instruction,
            f"{task_line}\n\n{label} (combined notes):\n{combined}",
            max(128, target_tokens + 64),
        )
        combined = (out or "").strip() or combined
    if estimate_tokens(combined) > target_tokens:
        combined = truncate_tokens(combined, target_tokens, keep="head")
    return combined


def describe_result(result: Any) -> str:
    """Uniform text view of a ToolResult-ish object."""
    content = getattr(result, "content", "") or ""
    error = getattr(result, "error", None)
    success = getattr(result, "success", True)
    if success:
        return content if content.strip() else "(tool succeeded with empty output)"
    return f"TOOL FAILED: {error or 'unknown error'}\n{content}".strip()
