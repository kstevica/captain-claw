"""Before an agent is exported for someone else: what it learned, to look over,
and what it was connected to.

What an agent learned while it was being trained is about two things: the work
itself, and the person who trained it (their preferences, their contacts,
their other clients' deadlines). Its insights have a category, so the review
shows them per category, the ones about the work ticked by default and the
ones usually about a person not. Its contacts are a choice of their own.
"""

from __future__ import annotations

import sqlite3
import tempfile
from contextlib import closing
from pathlib import Path
from typing import Any

from captain_claw.flight_deck.agent_bundle import catalog, stores

INSIGHT_CATEGORIES: tuple[tuple[str, str, bool], ...] = (
    # (category, label, ticked by default when the agent goes to someone else)
    ("fact", "Facts", True),
    ("workflow", "How the work is done", True),
    ("decision", "Decisions", True),
    ("reference", "References", True),
    ("feedback", "Feedback on how it works", False),
    ("preference", "Preferences of the person it worked with", False),
    ("project", "Projects", False),
    ("deadline", "Deadlines", False),
    ("contact", "People", False),
)
DEFAULT_INSIGHTS = frozenset(c for c, _, on in INSIGHT_CATEGORIES if on)
_LABELS = {c: label for c, label, _ in INSIGHT_CATEGORIES}
_ITEMS_PER_CATEGORY = 500


def review(agent_dir: Path) -> dict[str, Any]:
    """Its insights per category (with the items, to untick one by one) and how
    many contacts it keeps."""
    out: dict[str, Any] = {"insights": [], "contacts": 0}
    with tempfile.TemporaryDirectory() as tmp:
        src = agent_dir / catalog.BY_KEY["insights"].path
        if src.is_file():
            snap = Path(tmp) / "insights.db"
            stores.snapshot(src, snap)
            with closing(sqlite3.connect(snap)) as c:
                c.row_factory = sqlite3.Row
                if _has(c, "insights"):
                    rows = c.execute("SELECT id, content, category, importance FROM insights "
                                     "ORDER BY importance DESC, updated_at DESC").fetchall()
                    by: dict[str, list[dict[str, Any]]] = {}
                    for r in rows:
                        by.setdefault(str(r["category"] or "fact"), []).append(
                            {"id": str(r["id"]), "content": str(r["content"] or "")[:400],
                             "importance": int(r["importance"] or 0)})
                    order = [c for c, _, _ in INSIGHT_CATEGORIES] + sorted(set(by) - set(_LABELS))
                    out["insights"] = [{"id": cat, "label": _LABELS.get(cat, cat.title()), "count": len(by[cat]),
                                        "default": cat in DEFAULT_INSIGHTS, "items": by[cat][:_ITEMS_PER_CATEGORY]}
                                       for cat in order if by.get(cat)]
        src = agent_dir / catalog.BY_KEY["sessions"].path
        if src.is_file():
            snap = Path(tmp) / "sessions.db"
            stores.snapshot(src, snap)
            with closing(sqlite3.connect(snap)) as c:
                if _has(c, "contacts"):
                    out["contacts"] = int(c.execute("SELECT count(*) FROM contacts").fetchone()[0])
    return out


def _has(c: sqlite3.Connection, table: str) -> bool:
    return c.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)).fetchone() is not None


def filter_insights(db: Path, *, categories: set[str] | None, exclude: set[str]) -> int:
    """Drop from the snapshot *db* the insights not in *categories* (None: every
    category) and those in *exclude*, and its review queue; how many stay."""
    with closing(sqlite3.connect(db)) as c:
        if not _has(c, "insights"):
            return 0
        if categories is not None:
            marks = ",".join("?" * len(categories)) or "''"
            c.execute(f"DELETE FROM insights WHERE COALESCE(category, 'fact') NOT IN ({marks})", tuple(categories))
        for chunk in _chunks(sorted(exclude), 500):
            c.execute(f"DELETE FROM insights WHERE id IN ({','.join('?' * len(chunk))})", tuple(chunk))
        if _has(c, "insights_pending"):
            c.execute("DELETE FROM insights_pending")
        c.commit()
        return int(c.execute("SELECT count(*) FROM insights").fetchone()[0])


def _chunks(items: list[str], n: int):
    for i in range(0, len(items), n):
        yield items[i:i + n]


# ── what it was connected to ────────────────────────────────────────────────

_TOOL_CONNECTIONS: tuple[tuple[tuple[str, ...], str, str], ...] = (
    (("google_mail", "google_calendar", "google_drive"), "google", "Google account (mail, calendar, Drive)"),
    (("microsoft_mail", "microsoft_calendar", "microsoft_onedrive"), "microsoft", "Microsoft account (Outlook, OneDrive)"),
    (("imap_mail", "send_mail"), "imap", "An email inbox (IMAP)"),
    (("telegram_inbox",), "telegram", "Telegram"),
    (("whatsapp_web", "whatsapp_send_file"), "whatsapp", "WhatsApp"),
    (("jev",), "jev", "Jev"),
    (("external_memory",), "external_memory", "External memory"),
    (("payments",), "payments", "Payments"),
    (("custom_tools",), "custom_tools", "Custom tools"),
)


def connections(slug: str, owner_id: str, settings: dict[str, Any], config: dict[str, Any]) -> list[dict[str, str]]:
    """What the agent was connected to, by name only — none of it travels."""
    tools = {str(t) for t in settings.get("tools") or []}
    out: list[dict[str, str]] = []
    for names, kind, label in _TOOL_CONNECTIONS:
        if tools & set(names):
            out.append({"kind": kind, "label": label})
    tg = config.get("telegram") if isinstance(config.get("telegram"), dict) else {}
    if tg.get("enabled"):
        out.append({"kind": "telegram_bot", "label": "Its own Telegram bot"})
    try:
        from captain_claw.flight_deck import mcp_storage
        for rec in mcp_storage.load_servers():
            allowed = rec.get("allowed_agents") or []
            if not allowed or slug in allowed:
                out.append({"kind": "mcp", "label": f"MCP server “{rec.get('name')}”"})
    except Exception:
        pass
    return out
