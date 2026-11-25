"""An account plan's usage limit, as a model's server or CLI says it.

A subscription plan (ChatGPT's through the Codex login, Claude's 5-hour or
weekly limit) runs out and comes back by itself at a known time. That is not
an ordinary rate limit, which a few seconds' wait cures, and not a refused
key, billing or an empty balance, which never come back by themselves even
when they're said with the word "limit".

Shared by the agent loop (which ends the turn and carries on after the reset,
``limit_carry_on``) and anything else that reads a provider's or a CLI's
refusal. Text only: no I/O, no clock of its own.
"""

from __future__ import annotations

import re
import time
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import datetime, timedelta, tzinfo
from typing import Any
from zoneinfo import ZoneInfo

# The plan ran out. "Rate limit reached for gpt-4o … try again in 1.2s" is an
# API key's per-minute limit, not a plan's: never matched.
_PLAN_LIMIT = re.compile(
    r"usage limit|usage_limit_reached|(?<!rate )(?<!rate_)limit reached|hit your (?:[\w-]+ )?limit"
    r"|out of (?:extra )?usage|(?:5|five)-hour limit|weekly limit", re.IGNORECASE)
# What never resets by itself, whatever else the text says.
_NOT_TIMED = re.compile(
    r"credit balance|billing|payment required|insufficient|api.?key|invalid.*key"
    r"|unauthori[sz]ed|authenticat|usage_not_included|not included in your plan", re.IGNORECASE)

LIMIT_LONGEST_S = 8 * 24 * 3600   # a reset said to be further off than this isn't believed

_RESET_EPOCH = re.compile(r"limit reached\|(\d{10})\b", re.IGNORECASE)
# ChatGPT's refusal body: {"error": {"type": "usage_limit_reached", "resets_at": 1764090000, …}}
_RESET_AT_FIELD = re.compile(r"\bresets?_at\W{0,3}(\d{10})\b", re.IGNORECASE)
_RESET_SECONDS = re.compile(r"resets?_in_seconds\W{0,3}(\d+)|retry.after\W{0,3}(\d+)\s*(?:s|secs?|seconds)\b",
                            re.IGNORECASE)
_RESET_IN = re.compile(r"(?:try again|resets?|available again|back)\s+in\s+((?:\d+\s*[a-z]+[\s,]*(?:and\s+)?)+)",
                       re.IGNORECASE)
_DURATION = re.compile(r"(\d+)\s*(days?|d|hours?|hrs?|h|minutes?|mins?|m|seconds?|secs?|s)\b", re.IGNORECASE)
_RESET_AT = re.compile(
    r"(?:resets?|try again|available again)(?:\s+at|\s+on)?\s+"
    r"(?:(?P<mon>jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s+(?P<day>\d{1,2})(?:st|nd|rd|th)?,?\s+(?:at\s+)?)?"
    r"(?P<h>\d{1,2})(?::(?P<m>\d{2}))?\s*(?P<ap>[ap]\.?\s?m\.?)?(?![\d:])"
    r"(?:\s*\((?P<tz>[A-Za-z_]+(?:/[A-Za-z0-9_+-]+)+|UTC)\))?", re.IGNORECASE)
_MONTHS = ("jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec")
_UNIT_S = {"d": 86400, "h": 3600, "m": 60, "s": 1}


def plan_limit(text: str) -> bool:
    """Does *text* say the account's usage limit is used up — one that resets by itself?"""
    return bool(_PLAN_LIMIT.search(text or "")) and not _NOT_TIMED.search(text or "")


def reset_said(text: str, now: float, local: tzinfo | None = None) -> float | None:
    """When the limit resets (epoch seconds), as *text* says it: an epoch after
    "limit reached|" or in a ``resets_at`` field, seconds to wait, "try again in
    2 days 3 hours", or a clock time ("resets 5pm (Europe/Zagreb)", "try again
    at 9:49 PM", "resets Nov 30, 5pm") in the zone it names, else *local*,
    else the computer's own. None when it doesn't say."""
    text = text or ""
    for pattern in (_RESET_EPOCH, _RESET_AT_FIELD):
        m = pattern.search(text)
        if m:
            return float(m.group(1))
    m = _RESET_SECONDS.search(text)
    if m:
        return now + float(m.group(1) or m.group(2))
    m = _RESET_IN.search(text)
    if m:
        total = sum(int(n) * _UNIT_S[unit[0].lower()] for n, unit in _DURATION.findall(m.group(1)))
        if total:
            return now + total
    for m in _RESET_AT.finditer(text):
        hour, minute, ap = int(m.group("h")), int(m.group("m") or 0), (m.group("ap") or "").lower()
        if not ap and not m.group("m") and not m.group("mon"):
            continue   # a bare number ("resets 5 …") isn't surely a time
        if ap.startswith("p") and hour < 12:
            hour += 12
        elif ap.startswith("a") and hour == 12:
            hour = 0
        if hour > 23 or minute > 59:
            continue
        try:
            zone = ZoneInfo(m.group("tz")) if m.group("tz") else local
        except Exception:  # noqa: BLE001 — a zone this computer doesn't know: the local one
            zone = local
        today = datetime.fromtimestamp(now, zone) if zone else datetime.fromtimestamp(now).astimezone()
        if m.group("mon"):
            month, day = _MONTHS.index(m.group("mon")[:3].lower()) + 1, int(m.group("day"))
            try:
                at = today.replace(month=month, day=day, hour=hour, minute=minute, second=0, microsecond=0)
                if at.timestamp() < now - 86400:
                    at = at.replace(year=at.year + 1)
            except ValueError:
                continue
        else:
            at = today.replace(hour=hour, minute=minute, second=0, microsecond=0)
            if at.timestamp() <= now:
                at += timedelta(days=1)
        return at.timestamp()
    return None


@dataclass(frozen=True)
class PlanLimit:
    """A plan's usage limit stopped a call."""

    reset_at: float | None   # epoch seconds; None when nothing said, or it's past belief
    plan: str = ""           # whose plan, for the person: "ChatGPT", "Claude", or ""


def _chain(exc: BaseException) -> Iterator[BaseException]:
    seen: set[int] = set()
    cur: BaseException | None = exc
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        yield cur
        cur = cur.__cause__ or cur.__context__


def _plan_name(text: str) -> str:
    low = text.lower()
    if "chatgpt" in low or "codex" in low:
        return "ChatGPT"
    if "claude" in low:
        return "Claude"
    return ""


def from_error(err: BaseException | Any, now: float | None = None,
               local: tzinfo | None = None) -> PlanLimit | None:
    """The plan limit *err* (or what it was raised from) reports, or None.

    An HTTP error counts only as a 429: a plan limit is never a 400 or a 500
    that happens to use the words."""
    if not isinstance(err, BaseException):
        return None
    chain = list(_chain(err))
    codes = {getattr(e, "status_code", None) for e in chain} - {None}
    if codes and 429 not in codes:
        return None
    text = "\n".join(str(e) for e in chain)
    if not plan_limit(text):
        return None
    now = time.time() if now is None else now
    at = reset_said(text, now, local)
    if at is not None:
        at = None if at - now > LIMIT_LONGEST_S else max(at, now)
    return PlanLimit(reset_at=at, plan=_plan_name(text))
