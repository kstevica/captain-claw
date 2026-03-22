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

    # One-shot: "in 3d", "in 2h", "in 30m", "in 3 days", "in 2 hours"
    if head == "in":
        if len(tokens) < 2:
            raise ValueError("Usage: in <Nd|Nh|Nm> (e.g. in 3d, in 2h, in 30m)")
        # Handle "in 3d" or "in 3 days" (two tokens)
        value = tokens[1].strip().lower()
        match = re.fullmatch(r"(\d+)([dhm])", value)
        consumed = 2
        if not match and len(tokens) >= 3:
            # "in 3 days" / "in 2 hours" / "in 30 minutes"
            unit_word = tokens[2].strip().lower().rstrip("s")  # strip plural
            unit_map = {"day": "d", "hour": "h", "minute": "m", "min": "m"}
            unit_char = unit_map.get(unit_word)
            if unit_char and value.isdigit():
                match = re.fullmatch(r"(\d+)([dhm])", f"{value}{unit_char}")
                consumed = 3
        if not match:
            raise ValueError("Usage: in <Nd|Nh|Nm> (e.g. in 3d, in 2h, in 30m)")
        amount = int(match.group(1))
        unit = match.group(2)
        if amount <= 0:
            raise ValueError("Amount must be > 0")
        return {
            "type": "once",
            "unit": {"d": "days", "h": "hours", "m": "minutes"}[unit],
            "amount": amount,
        }, consumed

    # One-shot: "once <ISO-datetime>" or "once <YYYY-MM-DD>"
    if head == "once":
        if len(tokens) < 2:
            raise ValueError("Usage: once <YYYY-MM-DD> or once <ISO-datetime>")
        dt_str = tokens[1].strip()
        try:
            target = datetime.fromisoformat(dt_str)
            if target.tzinfo is None:
                target = target.replace(tzinfo=UTC)
        except ValueError:
            raise ValueError(f"Cannot parse datetime: {dt_str}")
        return {
            "type": "once",
            "target_iso": target.astimezone(UTC).isoformat(),
        }, 2

    raise ValueError(
        "Unsupported schedule. Use: every <Nm|Nh> | daily <HH:MM> | "
        "weekly <day> <HH:MM> | in <Nd|Nh|Nm> | once <datetime>"
    )


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
    if typ == "once":
        if "target_iso" in schedule:
            return f"once at {schedule['target_iso']}"
        amount = schedule.get("amount", "?")
        unit = schedule.get("unit", "days")
        return f"in {amount} {unit}"
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

    if typ == "once":
        # Absolute target time.
        if "target_iso" in schedule:
            return parse_iso(str(schedule["target_iso"]))
        # Relative delay (in Nd/Nh/Nm).
        amount = max(1, int(schedule.get("amount", 1)))
        unit = str(schedule.get("unit", "days")).lower()
        if unit.startswith("day"):
            return current + timedelta(days=amount)
        if unit.startswith("hour"):
            return current + timedelta(hours=amount)
        if unit.startswith("minute") or unit.startswith("min"):
            return current + timedelta(minutes=amount)
        return current + timedelta(days=amount)

    raise ValueError("Unknown schedule type")

