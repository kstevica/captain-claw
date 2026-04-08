"""Google Calendar actions for the gws tool.

Mixin used by :class:`GwsTool`. Provides calendar_list, calendar_search,
calendar_create, and calendar_agenda.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from typing import Any

from captain_claw.tools.registry import ToolResult


class GwsCalendarMixin:
    """Google Calendar actions."""

    async def _calendar_list(
        self,
        binary: str,
        calendar_id: str = "primary",
        max_results: int | float | None = None,
        days: int | float | None = None,
        **kw: Any,
    ) -> ToolResult:
        """List upcoming calendar events."""
        limit = min(int(max_results or 10), 100)
        look_ahead = int(days or 7)

        now = datetime.now(timezone.utc)
        time_max = now + timedelta(days=look_ahead)

        params: dict[str, Any] = {
            "calendarId": calendar_id,
            "maxResults": limit,
            "singleEvents": True,
            "orderBy": "startTime",
            "timeMin": now.isoformat(),
            "timeMax": time_max.isoformat(),
        }
        args = ["calendar", "events", "list", "--params", json.dumps(params)]
        return await self._run_gws(binary, args)

    async def _calendar_search(
        self,
        binary: str,
        query: str = "",
        calendar_id: str = "primary",
        max_results: int | float | None = None,
        days: int | float | None = None,
        **kw: Any,
    ) -> ToolResult:
        """Search calendar events by text."""
        if not query:
            return ToolResult(success=False, error="query is required for calendar_search.")

        limit = min(int(max_results or 10), 100)
        look_ahead = int(days or 30)

        now = datetime.now(timezone.utc)
        time_max = now + timedelta(days=look_ahead)

        params: dict[str, Any] = {
            "calendarId": calendar_id,
            "maxResults": limit,
            "singleEvents": True,
            "orderBy": "startTime",
            "timeMin": now.isoformat(),
            "timeMax": time_max.isoformat(),
            "q": query,
        }
        args = ["calendar", "events", "list", "--params", json.dumps(params)]
        return await self._run_gws(binary, args)

    async def _calendar_create(
        self,
        binary: str,
        summary: str = "",
        start: str = "",
        end: str = "",
        attendees: str = "",
        calendar_id: str = "primary",
        **kw: Any,
    ) -> ToolResult:
        """Create a new calendar event."""
        if not summary:
            return ToolResult(success=False, error="summary is required for calendar_create.")
        if not start:
            return ToolResult(success=False, error="start is required for calendar_create.")

        args = ["calendar", "+insert", "--summary", summary, "--start", start]
        if end:
            args.extend(["--end", end])
        if attendees:
            args.extend(["--attendees", attendees])
        if calendar_id and calendar_id != "primary":
            args.extend(["--calendar-id", calendar_id])

        return await self._run_gws(binary, args)

    async def _calendar_agenda(
        self, binary: str, days: int | float | None = None, calendar_id: str = "primary", **kw: Any,
    ) -> ToolResult:
        """Show calendar agenda."""
        look_ahead = int(days or 7)
        args = ["calendar", "+agenda", "--days", str(look_ahead)]
        if calendar_id and calendar_id != "primary":
            args.extend(["--calendar-id", calendar_id])
        return await self._run_gws(binary, args, json_output=False)
