"""``filter_by`` validation for the deep-memory routes (A2, always on).

Every deep-memory read and delete ANDs the tenant scope (``owner_id:=…``) onto
a caller-supplied Typesense ``filter_by``. Typesense's filter language has
``||`` and parentheses, so a raw filter such as ``x:=1) || (id:!=0`` could
regroup the expression around that scope and widen it past the tenant. This
module accepts only a flat, ``&&``-joined list of simple clauses on the deep
memory schema's own fields (``captain_claw.deep_memory`` — never ``owner_id``),
and refuses everything else:

    filter  := clause ( SP* "&&" SP* clause )*        split on "&&" only outside backticks
    clause  := field ":" op? value
    op      := "!=" | ">=" | "<=" | "=" | ">" | "<"   (> >= < <= only on numeric fields)
    value   := scalar | list | range
    scalar  := "`" [^`\\x00-\\x1f]{1,500} "`"  |  bare
    bare    := [A-Za-z0-9_.\\-/:@+]{1,200}            (numeric fields: -?[0-9]{1,19})
    list    := "[" SP* scalar ( SP* "," SP* scalar ){0,49} SP* "]"   (op absent, "=" or "!=")
    range   := "[" SP* int SP* ".." SP* int SP* "]"  (numeric fields only, op absent)

A backticked value is literal to Typesense, so it may contain ``)``, ``||`` or
``&&``. A backslash is refused everywhere (bare values never allow one): the
whole guarantee rests on Typesense toggling its quoting on every backtick,
exactly as the splitter here does, and a backslash-backtick read as an escape
would move a quote boundary and hand the rest of the filter back to the
grouping grammar.
"""

from __future__ import annotations

import re


class FilterByError(ValueError):
    """A ``filter_by`` outside the accepted grammar; ``str(exc)`` says why."""


FILTER_FIELDS = frozenset({"source", "reference", "path", "tags", "doc_id", "content_hash",
                           "chunk_index", "start_line", "end_line", "updated_at"})  # no owner_id
NUMERIC_FIELDS = frozenset({"chunk_index", "start_line", "end_line", "updated_at"})
FILTER_BY_MAX_LEN = 1000
FILTER_BY_MAX_CLAUSES = 10

_MAX_LIST_ITEMS = 50
_OPS = ("!=", ">=", "<=", "=", ">", "<")
_ORDER_OPS = frozenset({">=", "<=", ">", "<"})
_BARE_RE = re.compile(r"[A-Za-z0-9_.\-/:@+]{1,200}")
_INT_RE = re.compile(r"-?[0-9]{1,19}")
_QUOTED_RE = re.compile(r"`[^`\x00-\x1f]{1,500}`")
_RANGE_RE = re.compile(r"\[ *(-?[0-9]{1,19}) *\.\. *(-?[0-9]{1,19}) *\]")
_CONTROL_RE = re.compile(r"[\x00-\x1f\x7f]")


def _split_outside_backticks(text: str, sep: str) -> list[str]:
    """Split on ``sep`` outside backtick-quoted spans; FilterByError when a
    backtick is left open."""
    parts: list[str] = []
    buf: list[str] = []
    quoted = False
    i, n, w = 0, len(text), len(sep)
    while i < n:
        ch = text[i]
        if ch == "`":
            quoted = not quoted
            buf.append(ch)
            i += 1
            continue
        if not quoted and text.startswith(sep, i):
            parts.append("".join(buf))
            buf = []
            i += w
            continue
        buf.append(ch)
        i += 1
    if quoted:
        raise FilterByError("unbalanced backtick")
    parts.append("".join(buf))
    return parts


def _check_unquoted(text: str) -> None:
    """Refuse grouping / alternation outside backticks: ``(``, ``)``, ``|``, a
    lone ``&``, and ``!`` that isn't part of ``!=``."""
    quoted = False
    for i, ch in enumerate(text):
        if ch == "`":
            quoted = not quoted
            continue
        if quoted:
            continue
        if ch in "()":
            raise FilterByError("parentheses aren't allowed")
        if ch == "|":
            raise FilterByError("'||' isn't allowed")
        if ch == "&":
            prev_amp = i > 0 and text[i - 1] == "&"
            next_amp = i + 1 < len(text) and text[i + 1] == "&"
            if not (prev_amp or next_amp) or (prev_amp and next_amp):
                raise FilterByError("use '&&' to join clauses")
        if ch == "!" and not text.startswith("!=", i):
            raise FilterByError("'!' is only allowed as '!='")
    if quoted:
        raise FilterByError("unbalanced backtick")


def _check_scalar(value: str, numeric: bool) -> None:
    if value.startswith("`"):
        if not _QUOTED_RE.fullmatch(value):
            raise FilterByError("a quoted value must be 1-500 characters between backticks")
        return
    if numeric:
        if not _INT_RE.fullmatch(value):
            raise FilterByError("a numeric field needs an integer value")
        return
    if not _BARE_RE.fullmatch(value):
        raise FilterByError(
            "an unquoted value may only use letters, digits and _ . - / : @ + "
            "(quote anything else in backticks)")


def _check_list(value: str, numeric: bool) -> None:
    inner = value[1:-1]
    items = _split_outside_backticks(inner, ",")
    if len(items) > _MAX_LIST_ITEMS:
        raise FilterByError(f"a list may hold at most {_MAX_LIST_ITEMS} values")
    for item in items:
        item = item.strip(" ")
        if not item:
            raise FilterByError("empty value in a list")
        _check_scalar(item, numeric)


def _check_clause(clause: str) -> None:
    field, sep, rest = clause.partition(":")
    if not sep:
        raise FilterByError("each clause needs 'field:value'")
    if field == "owner_id":
        raise FilterByError("owner_id can't be filtered on")
    if field not in FILTER_FIELDS:
        raise FilterByError(f"unknown field {field[:40]!r}")
    numeric = field in NUMERIC_FIELDS
    op = next((o for o in _OPS if rest.startswith(o)), "")
    value = rest[len(op):]
    if op in _ORDER_OPS and not numeric:
        raise FilterByError(f"'{op}' only works on numeric fields")
    if not value:
        raise FilterByError(f"no value for {field}")
    if value.startswith("["):
        if not value.endswith("]"):
            raise FilterByError("unclosed list")
        if _RANGE_RE.fullmatch(value):
            if not numeric:
                raise FilterByError("a range only works on numeric fields")
            if op:
                raise FilterByError("a range takes no operator")
            return
        if op not in ("", "=", "!="):
            raise FilterByError(f"a list can't be used with '{op}'")
        _check_list(value, numeric)
        return
    _check_scalar(value, numeric)


def validate_filter_by(raw: object) -> str:
    """The stripped filter ("" allowed) when it is inside the grammar above;
    FilterByError (saying why) otherwise."""
    if not isinstance(raw, str):
        raise FilterByError("must be text")
    if len(raw) > FILTER_BY_MAX_LEN:
        raise FilterByError(f"longer than {FILTER_BY_MAX_LEN} characters")
    if _CONTROL_RE.search(raw):
        raise FilterByError("control characters aren't allowed")
    if "\\" in raw:
        raise FilterByError("backslashes aren't allowed")
    text = raw.strip()
    if not text:
        return ""
    _check_unquoted(text)
    clauses = _split_outside_backticks(text, "&&")
    if len(clauses) > FILTER_BY_MAX_CLAUSES:
        raise FilterByError(f"more than {FILTER_BY_MAX_CLAUSES} clauses")
    for clause in clauses:
        clause = clause.strip(" ")
        if not clause:
            raise FilterByError("empty clause")
        _check_clause(clause)
    return text
