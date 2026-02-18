"""Captain Claw pseudo-cron schedule parsing and time calculations."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
import re
from typing import Any


WEEKDAY_MAP = {
    "mon": 0,
    "monday": 0,
    "tue": 1,
    "tues": 1,
    "tuesday": 1,
    "wed": 2,
    "wednesday": 2,
    "thu": 3,
    "thur": 3,
    "thurs": 3,
    "thursday": 3,
    "fri": 4,
    "friday": 4,
    "sat": 5,
    "saturday": 5,
    "sun": 6,
    "sunday": 6,
}


def now_utc() -> datetime:
    """Return timezone-aware current UTC datetime."""
    return datetime.now(UTC)


def to_utc_iso(value: datetime) -> str:
    """Serialize datetime as UTC ISO string."""
    return value.astimezone(UTC).isoformat()


def parse_iso(value: str) -> datetime:
    """Parse ISO datetime and normalize to UTC."""
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _parse_hhmm(text: str) -> tuple[int, int]:
    match = re.fullmatch(r"([01]?\d|2[0-3]):([0-5]\d)", text.strip())
    if not match:
        raise ValueError("Expected time format HH:MM")
    return int(match.group(1)), int(match.group(2))


def parse_schedule_tokens(tokens: list[str]) -> tuple[dict[str, Any], int]:
    """Parse schedule from tokenized args and return (schedule, consumed_tokens)."""
    if not tokens:
        raise ValueError("Missing schedule")

    head = tokens[0].strip().lower()
    if head == "every":
        if len(tokens) < 2:
            raise ValueError("Usage: every <Nm|Nh>")
        value = tokens[1].strip().lower()
        match = re.fullmatch(r"(\d+)([mh])", value)
        if not match:
            raise ValueError("Usage: every <Nm|Nh> (example: every 15m)")
        interval = int(match.group(1))
        unit = match.group(2)
        if interval <= 0:
            raise ValueError("Interval must be > 0")
        return {
            "type": "interval",
            "unit": "minutes" if unit == "m" else "hours",
            "interval": interval,
        }, 2

    if head == "daily":
        if len(tokens) < 2:
            raise ValueError("Usage: daily <HH:MM>")
        hour, minute = _parse_hhmm(tokens[1])
        return {"type": "daily", "hour": hour, "minute": minute}, 2

    if head == "weekly":
        if len(tokens) < 3:
            raise ValueError("Usage: weekly <day> <HH:MM>")
        day_text = tokens[1].strip().lower()
        if day_text not in WEEKDAY_MAP:
            raise ValueError("Invalid weekday; use mon..sun")
        hour, minute = _parse_hhmm(tokens[2])
        return {
            "type": "weekly",
            "weekday": WEEKDAY_MAP[day_text],
            "day": day_text[:3],
            "hour": hour,
            "minute": minute,
        }, 3

    raise ValueError("Unsupported schedule. Use: every|daily|weekly")


def schedule_to_text(schedule: dict[str, Any]) -> str:
    """Render schedule dict into compact human text."""
    typ = str(schedule.get("type", "")).strip().lower()
    if typ == "interval":
        unit = str(schedule.get("unit", "minutes")).lower()
        interval = int(schedule.get("interval", 0))
        suffix = "m" if unit.startswith("minute") else "h"
        return f"every {interval}{suffix}"
    if typ == "daily":
        return f"daily {int(schedule.get('hour', 0)):02d}:{int(schedule.get('minute', 0)):02d}"
    if typ == "weekly":
        day = str(schedule.get("day", "mon")).lower()
        return (
            f"weekly {day} "
            f"{int(schedule.get('hour', 0)):02d}:{int(schedule.get('minute', 0)):02d}"
        )
    return "unknown"


def compute_next_run(schedule: dict[str, Any], now: datetime | None = None) -> datetime:
    """Compute next run datetime in UTC from schedule."""
    current = (now or now_utc()).astimezone(UTC)
    typ = str(schedule.get("type", "")).strip().lower()

    if typ == "interval":
        interval = max(1, int(schedule.get("interval", 1)))
        unit = str(schedule.get("unit", "minutes")).lower()
        delta = timedelta(minutes=interval if unit.startswith("minute") else interval * 60)
        return current + delta

    if typ == "daily":
        hour = int(schedule.get("hour", 0))
        minute = int(schedule.get("minute", 0))
        candidate = current.replace(hour=hour, minute=minute, second=0, microsecond=0)
        if candidate <= current:
            candidate += timedelta(days=1)
        return candidate

    if typ == "weekly":
        target_day = int(schedule.get("weekday", 0))
        hour = int(schedule.get("hour", 0))
        minute = int(schedule.get("minute", 0))
        days_ahead = (target_day - current.weekday()) % 7
        candidate = current + timedelta(days=days_ahead)
        candidate = candidate.replace(hour=hour, minute=minute, second=0, microsecond=0)
        if candidate <= current:
            candidate += timedelta(days=7)
        return candidate

    raise ValueError("Unknown schedule type")

