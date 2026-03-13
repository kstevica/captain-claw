# Summary: cron.py

# cron.py Summary

**Summary:** Captain Claw's pseudo-cron scheduling module that parses human-readable schedule specifications (interval, daily, weekly) and computes next execution times. Handles timezone-aware datetime operations with UTC normalization and provides bidirectional conversion between schedule dictionaries and human-readable text formats.

**Purpose:** Solves the problem of scheduling recurring tasks with flexible, human-friendly syntax while maintaining timezone consistency. Enables users to specify schedules like "every 15m", "daily 09:30", or "weekly monday 14:00" and calculates when tasks should next execute.

**Most Important Functions/Classes:**

1. **`parse_schedule_tokens(tokens: list[str])`** – Parses tokenized schedule arguments into a normalized dictionary structure. Supports three schedule types: interval-based (every Nm/Nh), daily (at specific HH:MM), and weekly (on specific day at HH:MM). Returns tuple of (schedule_dict, consumed_token_count) for flexible argument parsing. Validates input format and raises descriptive errors.

2. **`compute_next_run(schedule: dict[str, Any], now: datetime | None = None)`** – Core scheduling engine that calculates the next execution datetime from a schedule dictionary. Handles three schedule types with proper boundary logic: interval schedules add time delta to current time; daily schedules find next occurrence at target time (advancing to next day if already passed); weekly schedules calculate days ahead using modulo arithmetic and handle edge cases where candidate time has already passed.

3. **`schedule_to_text(schedule: dict[str, Any])`** – Inverse of parse_schedule_tokens; renders schedule dictionary back to compact human-readable format (e.g., "every 15m", "daily 09:30", "weekly mon 14:00"). Enables round-trip serialization and user-facing display.

4. **`_parse_hhmm(text: str)`** – Helper function that validates and extracts hour/minute components from HH:MM format strings using regex. Enforces valid ranges (0-23 hours, 0-59 minutes) and raises ValueError on malformed input.

5. **`now_utc()`, `to_utc_iso()`, `parse_iso()`** – Timezone utility functions ensuring all datetime operations work in UTC. `now_utc()` returns current time with UTC timezone info; `to_utc_iso()` serializes to ISO format; `parse_iso()` parses ISO strings and normalizes to UTC, handling both timezone-aware and naive inputs.

**Architecture Notes:**
- Stateless, functional design with no class definitions—all operations are pure functions
- Schedule representation uses simple dictionaries with type discriminator field, enabling extensibility
- Weekday mapping supports multiple aliases (mon/monday, tue/tues/tuesday, etc.) for user convenience
- All datetime operations normalize to UTC to prevent timezone-related bugs in distributed systems
- Token-based parsing returns consumed count, allowing integration with larger argument parsers