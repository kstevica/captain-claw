"""Inputs in their own shape: a table is anonymized cell by cell (its column
names say what a cell holds), JSON value by value (its keys do), and both are
written back in the same format. Everything else is one text.
"""

from __future__ import annotations

import csv
import io
import json
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from captain_claw.anonymize.detectors import hint_for
from captain_claw.anonymize.engine import Segment

TEXT_FORMATS = ("text", "md", "csv", "tsv", "json", "jsonl")


class FormatError(ValueError):
    pass


@dataclass
class Prepared:
    format: str
    segments: list[Segment]
    rebuild: Callable[[list[str]], str]


def prepare(content: str, fmt: str = "text") -> Prepared:
    fmt = (fmt or "text").lower()
    if fmt in ("csv", "tsv"):
        return _prepare_table(content, "\t" if fmt == "tsv" else None, fmt)
    if fmt == "json":
        return _prepare_json(content)
    if fmt == "jsonl":
        return _prepare_jsonl(content)
    return Prepared(fmt if fmt in TEXT_FORMATS else "text", [Segment(content)], lambda outs: outs[0])


# ── tables ──────────────────────────────────────────────────────────


def _prepare_table(content: str, delimiter: str | None, fmt: str) -> Prepared:
    sample = content[:20000]
    if delimiter is None:
        try:
            delimiter = csv.Sniffer().sniff(sample, delimiters=",;\t|").delimiter
        except csv.Error:
            delimiter = ","
    newline = "\r\n" if "\r\n" in sample else "\n"
    rows = list(csv.reader(io.StringIO(content, newline=""), delimiter=delimiter))
    if not rows:
        return Prepared(fmt, [Segment(content)], lambda outs: outs[0])
    header = rows[0]
    hints = [hint_for(h) for h in header]
    has_header = any(hints)
    if not has_header:
        try:
            has_header = csv.Sniffer().has_header(sample)
        except csv.Error:
            has_header = False
    segments: list[Segment] = []
    for r, row in enumerate(rows):
        for c, cell in enumerate(row):
            hint = hints[c] if has_header and r > 0 and c < len(hints) else ""
            segments.append(Segment(cell, hint))

    def rebuild(outs: list[str]) -> str:
        buf = io.StringIO()
        writer = csv.writer(buf, delimiter=delimiter, lineterminator=newline)
        k = 0
        for row in rows:
            writer.writerow(outs[k:k + len(row)])
            k += len(row)
        text = buf.getvalue()
        return text if content.endswith(("\n", "\r")) else text.rstrip("\r\n")

    return Prepared(fmt, segments, rebuild)


# ── JSON ────────────────────────────────────────────────────────────


def _walk(value: Any, key: str, segments: list[Segment]) -> None:
    if isinstance(value, dict):
        for k, v in value.items():
            segments.append(Segment(str(k)))
            _walk(v, str(k), segments)
    elif isinstance(value, list):
        for v in value:
            _walk(v, key, segments)
    elif isinstance(value, str):
        segments.append(Segment(value, hint_for(key)))
    elif isinstance(value, int) and not isinstance(value, bool) and abs(value) >= 100000:
        # Long numbers can be an OIB, a phone, an account stored as a number.
        segments.append(Segment(str(value), hint_for(key)))


def _rebuild(value: Any, outs: list[str], pos: list[int]) -> Any:
    if isinstance(value, dict):
        new: dict[str, Any] = {}
        for k, v in value.items():
            nk = outs[pos[0]]
            pos[0] += 1
            base, n = nk, 2
            while nk in new:  # two keys that became the same token
                nk = f"{base} ({n})"
                n += 1
            new[nk] = _rebuild(v, outs, pos)
        return new
    if isinstance(value, list):
        return [_rebuild(v, outs, pos) for v in value]
    if isinstance(value, str):
        out = outs[pos[0]]
        pos[0] += 1
        return out
    if isinstance(value, int) and not isinstance(value, bool) and abs(value) >= 100000:
        out = outs[pos[0]]
        pos[0] += 1
        return value if out == str(value) else out
    return value


def _prepare_json(content: str) -> Prepared:
    try:
        data = json.loads(content)
    except ValueError as exc:
        raise FormatError(f"not valid JSON ({exc.msg} at line {exc.lineno})") from None
    segments: list[Segment] = []
    _walk(data, "", segments)
    indent = 2 if "\n" in content.strip() else None

    def rebuild(outs: list[str]) -> str:
        text = json.dumps(_rebuild(data, outs, [0]), ensure_ascii=False, indent=indent)
        return text + ("\n" if content.endswith("\n") else "")

    return Prepared("json", segments, rebuild)


def _prepare_jsonl(content: str) -> Prepared:
    lines = content.splitlines()
    parsed: list[Any] = []
    segments: list[Segment] = []
    for n, line in enumerate(lines, start=1):
        if not line.strip():
            parsed.append(None)
            continue
        try:
            value = json.loads(line)
        except ValueError as exc:
            raise FormatError(f"line {n} is not valid JSON ({exc.msg})") from None
        parsed.append(value)
        _walk(value, "", segments)

    def rebuild(outs: list[str]) -> str:
        pos = [0]
        out_lines = []
        for line, value in zip(lines, parsed):
            out_lines.append(line if value is None else json.dumps(_rebuild(value, outs, pos), ensure_ascii=False))
        return "\n".join(out_lines) + ("\n" if content.endswith("\n") else "")

    return Prepared("jsonl", segments, rebuild)
