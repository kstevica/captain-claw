"""Google Calendar tool for creating, listing, updating, and deleting events.

Uses the Google Calendar REST API v3 via httpx with OAuth2 Bearer tokens
managed by :class:`~captain_claw.google_oauth_manager.GoogleOAuthManager`.
No additional Google SDK dependencies are required.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from typing import Any

import httpx

from captain_claw.config import get_config
from captain_claw.logging import get_logger
from captain_claw.tools.registry import Tool, ToolResult

log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_CALENDAR_API = "https://www.googleapis.com/calendar/v3"
_CALENDAR_SCOPE = "https://www.googleapis.com/auth/calendar"

# Fields to request from the events endpoint.
_EVENT_FIELDS = (
    "id,summary,description,location,start,end,status,"
    "htmlLink,created,updated,recurrence,attendees,reminders,"
    "creator,organizer"
)
_LIST_FIELDS = f"nextPageToken,items({_EVENT_FIELDS})"


class GoogleCalendarTool(Tool):
    """Interact with Google Calendar: list, search, create, update, and delete events."""

    name = "google_calendar"
    description = (
        "Interact with Google Calendar. Actions: list_events (browse upcoming events), "
        "search_events (find events by text), get_event (get event details), "
        "create_event (create a new calendar event), update_event (modify an existing event), "
        "delete_event (remove an event), list_calendars (list available calendars)."
    )
    timeout_seconds = 120.0
    parameters = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": [
                    "list_events",
                    "search_events",
                    "get_event",
                    "create_event",
                    "update_event",
                    "delete_event",
                    "list_calendars",
                ],
                "description": "The action to perform.",
            },
            "calendar_id": {
                "type": "string",
                "description": (
                    "Calendar ID to operate on. Defaults to 'primary'. "
                    "Use list_calendars to find other calendar IDs."
                ),
            },
            "event_id": {
                "type": "string",
                "description": "Event ID (for get_event, update_event, delete_event).",
            },
            "query": {
                "type": "string",
                "description": "Free-text search query (for search_events).",
            },
            "summary": {
                "type": "string",
                "description": "Event title/summary (for create_event, update_event).",
            },
            "description": {
                "type": "string",
                "description": "Event description/notes (for create_event, update_event).",
            },
            "location": {
                "type": "string",
                "description": "Event location (for create_event, update_event).",
            },
            "start": {
                "type": "string",
                "description": (
                    "Event start. For timed events use ISO 8601 datetime "
                    "(e.g. '2026-02-24T10:00:00+01:00'). "
                    "For all-day events use date only (e.g. '2026-02-24')."
                ),
            },
            "end": {
                "type": "string",
                "description": (
                    "Event end. For timed events use ISO 8601 datetime. "
                    "For all-day events use date only (the exclusive end date, "
                    "e.g. '2026-02-25' for a single-day event on 2026-02-24). "
                    "If omitted for create_event, defaults to start + 1 hour "
                    "(timed) or start + 1 day (all-day)."
                ),
            },
            "timezone": {
                "type": "string",
                "description": (
                    "IANA timezone for start/end (e.g. 'Europe/Berlin', 'America/New_York'). "
                    "Only used when start/end don't include timezone offset. "
                    "Defaults to the calendar's timezone."
                ),
            },
            "attendees": {
                "type": "array",
                "items": {"type": "string"},
                "description": "List of attendee email addresses (for create_event, update_event).",
            },
            "reminders": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "method": {
                            "type": "string",
                            "enum": ["email", "popup"],
                            "description": "Reminder method.",
                        },
                        "minutes": {
                            "type": "number",
                            "description": "Minutes before the event to trigger the reminder.",
                        },
                    },
                },
                "description": (
                    "Custom reminders. E.g. [{'method': 'popup', 'minutes': 30}]. "
                    "If omitted, calendar defaults are used."
                ),
            },
            "recurrence": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "RRULE strings for recurring events. "
                    "E.g. ['RRULE:FREQ=WEEKLY;COUNT=10']."
                ),
            },
            "max_results": {
                "type": "number",
                "description": "Maximum number of results to return (default 10, max 100).",
            },
            "time_min": {
                "type": "string",
                "description": (
                    "Lower bound (inclusive) for event start time (ISO 8601). "
                    "Defaults to now for list_events."
                ),
            },
            "time_max": {
                "type": "string",
                "description": "Upper bound (exclusive) for event end time (ISO 8601).",
            },
            "color_id": {
                "type": "string",
                "description": "Event color ID (1-11). See Google Calendar color definitions.",
            },
        },
        "required": ["action"],
    }

    def __init__(self) -> None:
        self._client = httpx.AsyncClient(
            timeout=120.0,
            follow_redirects=True,
            headers={"User-Agent": "Captain Claw/0.1.0 (Google Calendar Tool)"},
        )

    async def execute(self, action: str, **kwargs: Any) -> ToolResult:
        """Dispatch to the appropriate action handler."""
        # Pop injected kwargs that tools receive from the registry.
        kwargs.pop("_runtime_base_path", None)
        kwargs.pop("_saved_base_path", None)
        kwargs.pop("_session_id", None)
        kwargs.pop("_abort_event", None)
        kwargs.pop("_file_registry", None)
        kwargs.pop("_task_id", None)

        try:
            token = await self._get_access_token()
        except RuntimeError as e:
            return ToolResult(success=False, error=str(e))

        handlers = {
            "list_events": self._action_list_events,
            "search_events": self._action_search_events,
            "get_event": self._action_get_event,
            "create_event": self._action_create_event,
            "update_event": self._action_update_event,
            "delete_event": self._action_delete_event,
            "list_calendars": self._action_list_calendars,
        }
        handler = handlers.get(action)
        if handler is None:
            return ToolResult(
                success=False,
                error=f"Unknown action '{action}'. Use one of: {', '.join(handlers)}",
            )

        try:
            return await handler(token, **kwargs)
        except httpx.HTTPStatusError as exc:
            return self._handle_http_error(exc)
        except httpx.HTTPError as exc:
            log.error("Google Calendar HTTP error", action=action, error=str(exc))
            return ToolResult(success=False, error=f"HTTP error: {exc}")
        except Exception as exc:
            log.error("Google Calendar tool error", action=action, error=str(exc))
            return ToolResult(success=False, error=str(exc))

    # ------------------------------------------------------------------
    # Token access
    # ------------------------------------------------------------------

    async def _get_access_token(self) -> str:
        """Retrieve a valid Google OAuth access token.

        Raises RuntimeError if Google is not connected or tokens are
        expired and cannot be refreshed.
        """
        from captain_claw.google_oauth_manager import GoogleOAuthManager
        from captain_claw.session import get_session_manager

        mgr = GoogleOAuthManager(get_session_manager())
        tokens = await mgr.get_tokens()
        if not tokens:
            raise RuntimeError(
                "Google account is not connected. "
                "Please connect via the web UI (Settings > Google OAuth) or "
                "navigate to /auth/google/login in your browser."
            )

        # Check if the Calendar scope is present.
        granted = set(tokens.scope.split()) if tokens.scope else set()
        if _CALENDAR_SCOPE not in granted:
            raise RuntimeError(
                "Google Calendar scope not granted. Your current OAuth connection "
                "does not include Calendar access. Please disconnect and reconnect "
                "your Google account to grant Calendar permissions."
            )

        return tokens.access_token

    def _auth_headers(self, token: str) -> dict[str, str]:
        """Build authorization headers."""
        return {"Authorization": f"Bearer {token}"}

    # ------------------------------------------------------------------
    # Error handling
    # ------------------------------------------------------------------

    @staticmethod
    def _handle_http_error(exc: httpx.HTTPStatusError) -> ToolResult:
        """Convert HTTP status errors into user-friendly messages."""
        status = exc.response.status_code
        try:
            body = exc.response.json()
            message = body.get("error", {}).get("message", str(exc))
        except Exception:
            message = str(exc)

        if status == 401:
            return ToolResult(
                success=False,
                error="Google authentication expired. Please reconnect your Google account.",
            )
        elif status == 403:
            return ToolResult(
                success=False,
                error=f"Permission denied: {message}",
            )
        elif status == 404:
            return ToolResult(
                success=False,
                error="Event or calendar not found. Please check the ID.",
            )
        elif status == 409:
            return ToolResult(
                success=False,
                error=f"Conflict: {message}",
            )
        elif status == 429:
            return ToolResult(
                success=False,
                error="Google Calendar rate limit exceeded. Please try again in a moment.",
            )
        else:
            return ToolResult(
                success=False,
                error=f"Google Calendar API error ({status}): {message}",
            )

    # ------------------------------------------------------------------
    # Action: list_calendars
    # ------------------------------------------------------------------

    async def _action_list_calendars(
        self,
        token: str,
        max_results: int | float | None = None,
        **kwargs: Any,
    ) -> ToolResult:
        """List all calendars accessible by the user."""
        limit = min(int(max_results or 50), 250)

        resp = await self._client.get(
            f"{_CALENDAR_API}/users/me/calendarList",
            params={
                "maxResults": limit,
                "fields": "items(id,summary,description,primary,backgroundColor,timeZone,accessRole)",
            },
            headers=self._auth_headers(token),
        )
        resp.raise_for_status()
        data = resp.json()

        calendars = data.get("items", [])
        if not calendars:
            return ToolResult(success=True, content="No calendars found.")

        lines = [f"Calendars ({len(calendars)} found):\n"]
        for cal in calendars:
            primary = " [PRIMARY]" if cal.get("primary") else ""
            lines.append(
                f"  {cal.get('summary', '(no name)')}{primary}"
                f"\n    ID: {cal['id']}"
                f"  |  Timezone: {cal.get('timeZone', '?')}"
                f"  |  Access: {cal.get('accessRole', '?')}"
            )
            if cal.get("description"):
                lines.append(f"    Description: {cal['description']}")

        return ToolResult(success=True, content="\n".join(lines))

    # ------------------------------------------------------------------
    # Action: list_events
    # ------------------------------------------------------------------

    async def _action_list_events(
        self,
        token: str,
        calendar_id: str = "primary",
        max_results: int | float | None = None,
        time_min: str | None = None,
        time_max: str | None = None,
        **kwargs: Any,
    ) -> ToolResult:
        """List upcoming events from a calendar."""
        limit = min(int(max_results or 10), 100)

        params: dict[str, Any] = {
            "maxResults": limit,
            "singleEvents": "true",
            "orderBy": "startTime",
            "fields": _LIST_FIELDS,
        }
        if time_min:
            params["timeMin"] = self._ensure_rfc3339(time_min)
        else:
            params["timeMin"] = datetime.now(timezone.utc).isoformat()
        if time_max:
            params["timeMax"] = self._ensure_rfc3339(time_max)

        resp = await self._client.get(
            f"{_CALENDAR_API}/calendars/{calendar_id}/events",
            params=params,
            headers=self._auth_headers(token),
        )
        resp.raise_for_status()
        data = resp.json()

        events = data.get("items", [])
        if not events:
            return ToolResult(success=True, content="No upcoming events found.")

        lines = [f"Upcoming events ({len(events)}):\n"]
        for ev in events:
            lines.append(self._format_event(ev))

        return ToolResult(success=True, content="\n".join(lines))

    # ------------------------------------------------------------------
    # Action: search_events
    # ------------------------------------------------------------------

    async def _action_search_events(
        self,
        token: str,
        query: str = "",
        calendar_id: str = "primary",
        max_results: int | float | None = None,
        time_min: str | None = None,
        time_max: str | None = None,
        **kwargs: Any,
    ) -> ToolResult:
        """Search events by free-text query."""
        if not query:
            return ToolResult(success=False, error="Search query is required.")

        limit = min(int(max_results or 10), 100)

        params: dict[str, Any] = {
            "q": query,
            "maxResults": limit,
            "singleEvents": "true",
            "orderBy": "startTime",
            "fields": _LIST_FIELDS,
        }
        if time_min:
            params["timeMin"] = self._ensure_rfc3339(time_min)
        if time_max:
            params["timeMax"] = self._ensure_rfc3339(time_max)

        resp = await self._client.get(
            f"{_CALENDAR_API}/calendars/{calendar_id}/events",
            params=params,
            headers=self._auth_headers(token),
        )
        resp.raise_for_status()
        data = resp.json()

        events = data.get("items", [])
        if not events:
            return ToolResult(success=True, content=f"No events found matching '{query}'.")

        lines = [f"Search results for '{query}' ({len(events)} found):\n"]
        for ev in events:
            lines.append(self._format_event(ev))

        return ToolResult(success=True, content="\n".join(lines))

    # ------------------------------------------------------------------
    # Action: get_event
    # ------------------------------------------------------------------

    async def _action_get_event(
        self,
        token: str,
        event_id: str = "",
        calendar_id: str = "primary",
        **kwargs: Any,
    ) -> ToolResult:
        """Get detailed information about a specific event."""
        if not event_id:
            return ToolResult(success=False, error="event_id is required for get_event action.")

        resp = await self._client.get(
            f"{_CALENDAR_API}/calendars/{calendar_id}/events/{event_id}",
            params={"fields": _EVENT_FIELDS},
            headers=self._auth_headers(token),
        )
        resp.raise_for_status()
        ev = resp.json()

        return ToolResult(success=True, content=self._format_event_detail(ev))

    # ------------------------------------------------------------------
    # Action: create_event
    # ------------------------------------------------------------------

    async def _action_create_event(
        self,
        token: str,
        summary: str = "",
        calendar_id: str = "primary",
        start: str | None = None,
        end: str | None = None,
        timezone: str | None = None,
        description: str | None = None,
        location: str | None = None,
        attendees: list[str] | None = None,
        reminders: list[dict[str, Any]] | None = None,
        recurrence: list[str] | None = None,
        color_id: str | None = None,
        **kwargs: Any,
    ) -> ToolResult:
        """Create a new calendar event."""
        if not summary:
            return ToolResult(success=False, error="summary is required for create_event action.")
        if not start:
            return ToolResult(success=False, error="start is required for create_event action.")

        body = self._build_event_body(
            summary=summary,
            start=start,
            end=end,
            timezone=timezone,
            description=description,
            location=location,
            attendees=attendees,
            reminders=reminders,
            recurrence=recurrence,
            color_id=color_id,
        )

        resp = await self._client.post(
            f"{_CALENDAR_API}/calendars/{calendar_id}/events",
            json=body,
            params={"fields": _EVENT_FIELDS},
            headers={
                **self._auth_headers(token),
                "Content-Type": "application/json",
            },
        )
        resp.raise_for_status()
        ev = resp.json()

        event_id = ev.get("id", "?")
        event_summary = ev.get("summary", summary)
        link = ev.get("htmlLink", "")

        msg = f"Created event '{event_summary}' on Google Calendar.\n  ID: {event_id}"
        start_info = self._format_event_time(ev)
        if start_info:
            msg += f"\n  When: {start_info}"
        if link:
            msg += f"\n  Link: {link}"

        return ToolResult(success=True, content=msg)

    # ------------------------------------------------------------------
    # Action: update_event
    # ------------------------------------------------------------------

    async def _action_update_event(
        self,
        token: str,
        event_id: str = "",
        calendar_id: str = "primary",
        summary: str | None = None,
        start: str | None = None,
        end: str | None = None,
        timezone: str | None = None,
        description: str | None = None,
        location: str | None = None,
        attendees: list[str] | None = None,
        reminders: list[dict[str, Any]] | None = None,
        recurrence: list[str] | None = None,
        color_id: str | None = None,
        **kwargs: Any,
    ) -> ToolResult:
        """Update an existing calendar event (partial patch)."""
        if not event_id:
            return ToolResult(success=False, error="event_id is required for update_event action.")

        body: dict[str, Any] = {}

        if summary is not None:
            body["summary"] = summary
        if description is not None:
            body["description"] = description
        if location is not None:
            body["location"] = location
        if color_id is not None:
            body["colorId"] = color_id

        if start is not None:
            is_all_day = len(start) <= 10  # date-only
            if is_all_day:
                body["start"] = {"date": start}
                if end:
                    body["end"] = {"date": end}
                else:
                    body["end"] = {"date": self._next_day(start)}
            else:
                start_obj: dict[str, str] = {"dateTime": self._ensure_rfc3339(start)}
                end_obj: dict[str, str] = {}
                if timezone:
                    start_obj["timeZone"] = timezone
                body["start"] = start_obj
                if end:
                    end_obj = {"dateTime": self._ensure_rfc3339(end)}
                    if timezone:
                        end_obj["timeZone"] = timezone
                    body["end"] = end_obj
                else:
                    body["end"] = {"dateTime": self._ensure_rfc3339(start, offset_hours=1)}
                    if timezone:
                        body["end"]["timeZone"] = timezone

        if attendees is not None:
            body["attendees"] = [{"email": email} for email in attendees]

        if reminders is not None:
            body["reminders"] = {
                "useDefault": False,
                "overrides": reminders,
            }

        if recurrence is not None:
            body["recurrence"] = recurrence

        if not body:
            return ToolResult(success=False, error="No fields to update. Provide at least one field.")

        resp = await self._client.patch(
            f"{_CALENDAR_API}/calendars/{calendar_id}/events/{event_id}",
            json=body,
            params={"fields": _EVENT_FIELDS},
            headers={
                **self._auth_headers(token),
                "Content-Type": "application/json",
            },
        )
        resp.raise_for_status()
        ev = resp.json()

        event_summary = ev.get("summary", event_id)
        return ToolResult(
            success=True,
            content=f"Updated event '{event_summary}'.\n  ID: {event_id}\n  Link: {ev.get('htmlLink', '')}",
        )

    # ------------------------------------------------------------------
    # Action: delete_event
    # ------------------------------------------------------------------

    async def _action_delete_event(
        self,
        token: str,
        event_id: str = "",
        calendar_id: str = "primary",
        **kwargs: Any,
    ) -> ToolResult:
        """Delete a calendar event."""
        if not event_id:
            return ToolResult(success=False, error="event_id is required for delete_event action.")

        resp = await self._client.delete(
            f"{_CALENDAR_API}/calendars/{calendar_id}/events/{event_id}",
            headers=self._auth_headers(token),
        )
        resp.raise_for_status()

        return ToolResult(
            success=True,
            content=f"Deleted event {event_id} from calendar '{calendar_id}'.",
        )

    # ------------------------------------------------------------------
    # Event body builder
    # ------------------------------------------------------------------

    def _build_event_body(
        self,
        summary: str,
        start: str,
        end: str | None,
        timezone: str | None,
        description: str | None,
        location: str | None,
        attendees: list[str] | None,
        reminders: list[dict[str, Any]] | None,
        recurrence: list[str] | None,
        color_id: str | None,
    ) -> dict[str, Any]:
        """Build the event JSON body for create/update."""
        body: dict[str, Any] = {"summary": summary}

        # Determine if all-day or timed event.
        is_all_day = len(start) <= 10  # date-only format like '2026-02-24'

        if is_all_day:
            body["start"] = {"date": start}
            if end:
                body["end"] = {"date": end}
            else:
                # All-day events: end is exclusive, so single day = start + 1.
                body["end"] = {"date": self._next_day(start)}
        else:
            start_obj: dict[str, str] = {"dateTime": self._ensure_rfc3339(start)}
            if timezone:
                start_obj["timeZone"] = timezone
            body["start"] = start_obj

            if end:
                end_obj: dict[str, str] = {"dateTime": self._ensure_rfc3339(end)}
                if timezone:
                    end_obj["timeZone"] = timezone
                body["end"] = end_obj
            else:
                # Default to 1 hour duration.
                end_obj = {"dateTime": self._ensure_rfc3339(start, offset_hours=1)}
                if timezone:
                    end_obj["timeZone"] = timezone
                body["end"] = end_obj

        if description:
            body["description"] = description
        if location:
            body["location"] = location
        if color_id:
            body["colorId"] = color_id

        if attendees:
            body["attendees"] = [{"email": email} for email in attendees]

        if reminders is not None:
            body["reminders"] = {
                "useDefault": False,
                "overrides": reminders,
            }

        if recurrence:
            body["recurrence"] = recurrence

        return body

    # ------------------------------------------------------------------
    # Formatting helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _format_event(ev: dict[str, Any]) -> str:
        """Format an event for list/search output (compact)."""
        summary = ev.get("summary", "(no title)")
        event_id = ev.get("id", "?")
        status = ev.get("status", "")

        # Start/end
        start = ev.get("start", {})
        end = ev.get("end", {})
        if "date" in start:
            start_str = start["date"]
            end_str = end.get("date", "")
            time_str = f"{start_str}" + (f" → {end_str}" if end_str and end_str != start_str else " (all day)")
        elif "dateTime" in start:
            start_str = start["dateTime"][:19].replace("T", " ")
            end_str = end.get("dateTime", "")[:19].replace("T", "") if end.get("dateTime") else ""
            # If same day, show only the time for end.
            if end_str and start_str[:10] == end.get("dateTime", "")[:10]:
                end_time = end.get("dateTime", "")[11:16]
                time_str = f"{start_str} → {end_time}"
            else:
                time_str = f"{start_str}" + (f" → {end_str}" if end_str else "")
        else:
            time_str = "unknown time"

        location = ev.get("location", "")
        loc_str = f"  |  📍 {location}" if location else ""
        status_str = f"  [{status}]" if status and status != "confirmed" else ""

        return (
            f"  📅 {summary}{status_str}"
            f"\n    When: {time_str}{loc_str}"
            f"\n    ID: {event_id}"
        )

    @staticmethod
    def _format_event_detail(ev: dict[str, Any]) -> str:
        """Format an event with full details."""
        lines = [f"Event: {ev.get('summary', '(no title)')}"]
        lines.append(f"  ID: {ev.get('id', '?')}")
        lines.append(f"  Status: {ev.get('status', '?')}")

        # Time
        start = ev.get("start", {})
        end = ev.get("end", {})
        if "date" in start:
            lines.append(f"  Start: {start['date']} (all day)")
            if end.get("date"):
                lines.append(f"  End: {end['date']} (exclusive)")
        elif "dateTime" in start:
            lines.append(f"  Start: {start['dateTime']}")
            if end.get("dateTime"):
                lines.append(f"  End: {end['dateTime']}")
            tz = start.get("timeZone")
            if tz:
                lines.append(f"  Timezone: {tz}")

        if ev.get("location"):
            lines.append(f"  Location: {ev['location']}")
        if ev.get("description"):
            lines.append(f"  Description: {ev['description']}")

        if ev.get("creator"):
            creator = ev["creator"]
            lines.append(f"  Creator: {creator.get('displayName', creator.get('email', '?'))}")
        if ev.get("organizer"):
            org = ev["organizer"]
            lines.append(f"  Organizer: {org.get('displayName', org.get('email', '?'))}")

        if ev.get("attendees"):
            att_lines = []
            for att in ev["attendees"]:
                email = att.get("email", "?")
                status = att.get("responseStatus", "?")
                name = att.get("displayName", "")
                label = f"{name} <{email}>" if name else email
                att_lines.append(f"    - {label} ({status})")
            lines.append(f"  Attendees ({len(ev['attendees'])}):")
            lines.extend(att_lines)

        if ev.get("recurrence"):
            lines.append(f"  Recurrence: {', '.join(ev['recurrence'])}")

        reminders = ev.get("reminders", {})
        if reminders.get("useDefault"):
            lines.append("  Reminders: calendar default")
        elif reminders.get("overrides"):
            rem_parts = []
            for r in reminders["overrides"]:
                rem_parts.append(f"{r.get('method', '?')} {r.get('minutes', '?')}min before")
            lines.append(f"  Reminders: {', '.join(rem_parts)}")

        if ev.get("htmlLink"):
            lines.append(f"  Link: {ev['htmlLink']}")
        lines.append(f"  Created: {ev.get('created', '?')}")
        lines.append(f"  Updated: {ev.get('updated', '?')}")

        return "\n".join(lines)

    @staticmethod
    def _format_event_time(ev: dict[str, Any]) -> str:
        """Extract a human-readable time string from an event."""
        start = ev.get("start", {})
        end = ev.get("end", {})
        if "date" in start:
            s = start["date"]
            e = end.get("date", "")
            return f"{s} (all day)" if not e or e == s else f"{s} → {e}"
        elif "dateTime" in start:
            s = start["dateTime"]
            e = end.get("dateTime", "")
            return f"{s}" + (f" → {e}" if e else "")
        return ""

    # ------------------------------------------------------------------
    # Time helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _ensure_rfc3339(dt_str: str, offset_hours: int = 0) -> str:
        """Ensure a datetime string is RFC 3339 compliant.

        If offset_hours is given, add that many hours to the parsed time.
        """
        try:
            dt = datetime.fromisoformat(dt_str)
        except ValueError:
            # Already RFC 3339 or unparseable — return as-is.
            if offset_hours:
                return dt_str  # Can't offset what we can't parse.
            return dt_str

        if offset_hours:
            dt = dt + timedelta(hours=offset_hours)

        # If no timezone info, assume UTC.
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)

        return dt.isoformat()

    @staticmethod
    def _next_day(date_str: str) -> str:
        """Return the next day for an all-day event end date."""
        try:
            dt = datetime.strptime(date_str[:10], "%Y-%m-%d")
            return (dt + timedelta(days=1)).strftime("%Y-%m-%d")
        except ValueError:
            return date_str

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    async def close(self) -> None:
        """Close the HTTP client."""
        await self._client.aclose()
