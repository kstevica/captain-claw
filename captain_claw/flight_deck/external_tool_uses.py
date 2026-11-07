"""Custom tools, Phase 7: what a tool may ask the agent that ran it to do.

A tool lists its ``uses`` in tool.json (docs/external-tools-plan.md, Phase 7).
Publishing refuses a use no custom tool may have; every request a running tool
makes is checked here against what it declared and against the risk of the
real call before the agent sees it. The agent then checks it again with its
own rules.

- read and write: allowed;
- send: only when the tool's owner is the one running it;
- run code, money, and anything unclassified: never;
- never, whatever the class: tools that change the agent itself or set up
  later work (a tool must not leave a scheduled job or a habit behind);
- web reads only from the hosts the tool names, since a URL can carry data out;
- writes that reach other people (calendar invitations, a response to an
  organiser) count as send here, stricter than the tool-call register's classes.
"""

from __future__ import annotations

import re
from typing import Any
from urllib.parse import urlsplit

from captain_claw.flight_deck import tool_risk

MAX_USES = 20
MAX_ACTIONS = 20
MAX_HOSTS = 10
MAX_REQUESTS = 50          # per top-level run, nested tools included

ALLOWED = frozenset({"read", "write", "send"})
OWNER_ONLY = frozenset({"send"})

# Tools that change the agent itself or set up later work.
NEVER = frozenset({
    "cron", "personality", "playbooks", "insights", "facts", "intentions", "project_memory",
    "reasoning_effort", "apis", "synthesize_flow", "todo", "topics",
    # A tool reaches other custom tools with a custom_tool entry, never through this one.
    "custom_tools",
})

# Tools that fetch a URL: a tool using them names the hosts it reads from.
WEB_FETCHERS = frozenset({"web_fetch", "web_fetch_batch", "web_get"})

# Writes that reach other people: send, for custom tools.
_OUTWARD: dict[str, frozenset[str]] = {
    "google_calendar": frozenset({"create_event", "update_event", "delete_event"}),
    "microsoft_calendar": frozenset({"create_event", "update_event", "delete_event", "respond_event"}),
}

_HOST_RE = re.compile(r"^(?=.{1,253}$)(?!-)[a-z0-9-]{1,63}(?<!-)(\.(?!-)[a-z0-9-]{1,63}(?<!-))+$")
_ACTION_RE = re.compile(r"^[a-z][a-z0-9_]{0,39}$")
_CLASS_WORDS = {"exec": "it runs code", "money": "it moves money", "other": "Flight Deck doesn't know what it does"}


class UsesError(ValueError):
    """A ``uses`` list publishing refuses; the message says why."""


def _action_of(params: Any) -> str:
    if not isinstance(params, dict):
        return ""
    value = params.get("action")
    return value.strip().lower() if isinstance(value, str) else ""


def risk(tool: str, params: Any = None) -> str:
    """The risk class of a call a custom tool asks for (``tool_risk.risk_for``,
    with writes that reach other people as send)."""
    name = str(tool or "").strip().lower()
    cls = tool_risk.risk_for(name, params)
    if cls == "write" and _action_of(params) in _OUTWARD.get(name, frozenset()):
        return "send"
    return cls


def known(tool: str) -> bool:
    return tool in tool_risk._TOOL_RISK


def _why(raw: dict, what: str) -> str:
    why = raw.get("why")
    if not isinstance(why, str) or not 1 <= len(why.strip()) <= 200:
        raise UsesError(f"{what}: 'why' must say in 1–200 characters why the tool needs it.")
    return why.strip()


def _tool_entry(raw: dict) -> dict[str, Any]:
    name = raw.get("tool")
    if not isinstance(name, str) or not name.strip():
        raise UsesError("each use names a 'tool' (one of the agent's tools).")
    name = name.strip().lower()
    what = f"use of {name!r}"
    if name in NEVER:
        raise UsesError(f"custom tools can't use {name!r}: it changes the agent itself or sets up later work.")
    if name.startswith("mcp_") or not known(name):
        raise UsesError(f"custom tools can't use {name!r}: Flight Deck doesn't know that tool.")
    actions_raw = raw.get("actions")
    actions: list[str] = []
    if actions_raw is not None:
        if (not isinstance(actions_raw, list) or not 1 <= len(actions_raw) <= MAX_ACTIONS
                or not all(isinstance(a, str) and _ACTION_RE.match(a.strip().lower()) for a in actions_raw)):
            raise UsesError(f"{what}: 'actions' must list 1–{MAX_ACTIONS} of the tool's actions, like [\"search\", \"read\"].")
        actions = list(dict.fromkeys(a.strip().lower() for a in actions_raw))
    for action in actions or [""]:
        cls = risk(name, {"action": action} if action else None)
        if cls not in ALLOWED:
            label = f"{name} {action}".strip()
            raise UsesError(f"custom tools can't use {label!r}: {_CLASS_WORDS.get(cls, cls)}.")
    hosts: list[str] = []
    hosts_raw = raw.get("hosts")
    if name in WEB_FETCHERS:
        if (not isinstance(hosts_raw, list) or not 1 <= len(hosts_raw) <= MAX_HOSTS
                or not all(isinstance(h, str) for h in hosts_raw)):
            raise UsesError(f"{what}: list the 'hosts' it reads from (1–{MAX_HOSTS}), like [\"api.frankfurter.app\"]: "
                            "a URL can carry data out.")
        for h in hosts_raw:
            host = h.strip().lower().rstrip(".")
            if not _HOST_RE.match(host):
                raise UsesError(f"{what}: {h!r} isn't a host name (like api.example.com).")
            hosts.append(host)
        hosts = list(dict.fromkeys(hosts))
    elif hosts_raw is not None:
        raise UsesError(f"{what}: 'hosts' is only for the tools that fetch web pages.")
    entry: dict[str, Any] = {"tool": name, "why": _why(raw, what)}
    if actions:
        entry["actions"] = actions
    if hosts:
        entry["hosts"] = hosts
    return entry


def parse_uses(raw: Any, parse_other: Any = None) -> tuple[dict[str, Any], ...]:
    """Check a tool.json ``uses`` list (absent means none). *parse_other* reads
    entries that aren't one of the agent's tools (``custom_tool``); without it
    they are refused. Raises :class:`UsesError`."""
    if raw is None:
        return ()
    if not isinstance(raw, list) or len(raw) > MAX_USES:
        raise UsesError(f"'uses' must be a list of at most {MAX_USES}.")
    out: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for item in raw:
        if not isinstance(item, dict):
            raise UsesError("each use is an object, like {\"tool\": \"web_fetch\", \"hosts\": [...], \"why\": \"...\"}.")
        if "tool" in item:
            entry = _tool_entry(item)
            key = ("tool", entry["tool"])
        elif parse_other is not None:
            entry = parse_other(item)
            key = next(((k, str(entry[k])) for k in ("custom_tool",) if k in entry), ("?", repr(entry)))
        else:
            raise UsesError("each use names a 'tool' (one of the agent's tools).")
        if key in seen:
            raise UsesError(f"{key[1]!r} is in 'uses' twice.")
        seen.add(key)
        out.append(entry)
    return tuple(out)


def urls_of(tool: str, params: Any) -> list[str]:
    """The URLs a web-fetching call would read."""
    if not isinstance(params, dict):
        return []
    found: list[str] = []
    for key in ("url", "urls", "link", "links"):
        value = params.get(key)
        if isinstance(value, str):
            found.append(value)
        elif isinstance(value, list):
            found.extend(v for v in value if isinstance(v, str))
    return found


def host_allowed(url: str, hosts: list[str]) -> bool:
    try:
        parts = urlsplit(str(url).strip())
    except ValueError:
        return False
    if parts.scheme not in ("http", "https") or not parts.hostname or parts.username or parts.password:
        return False
    host = parts.hostname.lower().rstrip(".")
    return any(host == h or host.endswith("." + h) for h in hosts)


def check_request(uses: tuple[dict[str, Any], ...], tool: str, params: Any, *, owner_runs: bool) -> str | None:
    """Why this request is refused, or None when it may go to the agent."""
    name = str(tool or "").strip().lower()
    if name in NEVER:
        return f"Custom tools can't use {name}."
    entry = next((u for u in uses if u.get("tool") == name), None)
    if entry is None:
        return f"{name} isn't in this tool's uses (tool.json): it can only ask for what it declared."
    if not isinstance(params, dict):
        return f"The input for {name} must be an object."
    action = _action_of(params)
    if entry.get("actions") and action not in entry["actions"]:
        allowed = ", ".join(entry["actions"])
        return f"{name} {action or '(no action)'} isn't declared: this tool may only use {name} {allowed}."
    cls = risk(name, params)
    if cls not in ALLOWED:
        return f"Custom tools can't use {name} {action}".rstrip() + f": {_CLASS_WORDS.get(cls, cls)}."
    if cls in OWNER_ONLY and not owner_runs:
        return (f"{name} {action}".rstrip() + " reaches other people, so a custom tool may use it only when its "
                "owner runs it.")
    if name in WEB_FETCHERS:
        urls = urls_of(name, params)
        if not urls:
            return f"{name}: no URL to check against this tool's hosts."
        for url in urls:
            if not host_allowed(url, entry.get("hosts") or []):
                return (f"{name}: {str(url)[:200]} isn't on this tool's hosts "
                        f"({', '.join(entry.get('hosts') or [])}).")
    return None


def describe(uses: tuple[dict[str, Any], ...]) -> list[str]:
    """Each use in a few words, for the page and the agent."""
    out = []
    for u in uses:
        if "tool" in u:
            detail = u.get("actions") or u.get("hosts") or []
            out.append(f"{u['tool']} ({', '.join(detail)})" if detail else u["tool"])
        elif "custom_tool" in u:
            out.append(f"custom tool {u['custom_tool']}")
    return out
