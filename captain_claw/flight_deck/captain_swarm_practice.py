"""Captain Swarm — practice sets from the owner's own history.

docs/captain-swarm-plan.md §5.2. Recurring real work has already left a trail;
a practice set is a frozen sample of it, each case labelled by what the owner
actually did, so the labels cost the owner no time.

The first one is **inbox triage**: received mail from a window of days, each
message labelled by whether the owner wrote in its thread within a few days
of it arriving (Gmail's ``SENT`` label, or mail from one of the owner's own
addresses). Mail too new for that to be known yet is left out. The most
recent weeks are held out: an attempt never sees them, and they decide
whether a gain on the open part was real.

The second is **calendar invitations** (:func:`build_calendar_invites`): past
events on the owner's Google Calendar that someone else organized, each
labelled by the owner's own answer — accepted is yes, declined is no. Ones
left unanswered or tentative aren't a decision and are left out. The newest
invitations are held out. It needs the calendar permission on the owner's
Google connection.

Sealed, per owner, in the deck's data dir — never in a repo, never in a lane::

    FD_DATA_DIR/captain_swarm/sealed/practice/<owner>/<set_id>/
        meta.json        source, window, counts, base rates, the split date
        open.jsonl       the cases a run may learn from
        holdout.jsonl    the most recent weeks
        measurements/    what each measurement found (case ids, no mail text)

Folders 0700, files 0600. The owner can delete a set whole.

Labelling (:func:`label_messages`), sampling (:func:`sample_evenly`), the
split (:func:`split_by_time`) and the case (:func:`case_of`) are pure; the
Gmail reads take an injectable fetch so tests run without Google.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import secrets
import shutil
import time
from collections.abc import Awaitable, Callable, Iterable, Mapping
from datetime import UTC, datetime, timezone
from pathlib import Path
from typing import Any

from captain_claw import gmail_inbound
from captain_claw.logging import get_logger

log = get_logger(__name__)

KIND_MAIL_TRIAGE = "mail_triage"
KIND_CALENDAR = "calendar_invites"
KINDS = {
    KIND_MAIL_TRIAGE: {"title": "Inbox triage: did you reply?", "family": "mail-triage", "id": "mt"},
    KIND_CALENDAR: {"title": "Calendar invitations: did you accept?", "family": "calendar-invites", "id": "ci"},
}
CALENDAR_API = "https://www.googleapis.com/calendar/v3"
_EVENTS_CAP = 3000
_DESC_CHARS = 1500

_DAY_MS = 86_400_000
_BODY_CHARS = 3000
_LIST_CAP = 5000             # message ids read from Gmail at most
_MAX_CASES = 1000
_FETCH_PARALLEL = 5
# Gmail's own sorting, set when mail arrives — what a person triaging sees too.
# Labels set later (starred, important, the owner's own) could leak the answer.
_CATEGORY_LABELS = ("CATEGORY_PERSONAL", "CATEGORY_SOCIAL", "CATEGORY_PROMOTIONS",
                    "CATEGORY_UPDATES", "CATEGORY_FORUMS")
_NOT_RECEIVED = frozenset({"SENT", "DRAFT", "SPAM", "TRASH", "CHAT"})
_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")

Fetch = Callable[[str, Mapping[str, Any]], Awaitable[tuple[int, dict]]]


# ── where it lives ───────────────────────────────────────────────────


def sealed_root() -> Path:
    from captain_claw.flight_deck import server as srv
    return srv.DATA_DIR / "captain_swarm" / "sealed"


def _owner_dir(owner_id: str) -> Path:
    key = re.sub(r"[^A-Za-z0-9_.-]", "_", owner_id or "local")[:80] or "local"
    return sealed_root() / "practice" / key


def set_dir(owner_id: str, set_id: str) -> Path:
    if not _ID_RE.match(set_id or ""):
        raise ValueError("bad set id")
    return _owner_dir(owner_id) / set_id


def _private_dir(path: Path) -> None:
    """*path* made, and it and every folder up to the sealed root closed to
    anyone but the deck's user. Nothing above the sealed root is touched."""
    path.mkdir(parents=True, exist_ok=True)
    root = sealed_root()
    for p in [path, *path.parents]:
        try:
            p.relative_to(root)
        except ValueError:
            break
        os.chmod(p, 0o700)


def _write_private(path: Path, text: str) -> None:
    _private_dir(path.parent)
    tmp = path.with_name(f".{path.name}.tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        fh.write(text)
    tmp.replace(path)


def write_json(path: Path, data: Any) -> None:
    _write_private(path, json.dumps(data, ensure_ascii=False, indent=1))


def _read_json(path: Path) -> dict | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


# ── pure parts ───────────────────────────────────────────────────────


def _own(addr: str, own_keys: frozenset[str]) -> bool:
    return bool(addr) and gmail_inbound.own_address_key(addr) in own_keys


def label_messages(messages: Iterable[Mapping[str, Any]], own_addresses: Iterable[str], *,
                   reply_ms: int, now_ms: int) -> dict[str, dict[str, Any]]:
    """Every received message of one thread → what the owner did about it.

    *messages* are ``{"id", "internal_ms", "labels", "from"}``. A message is
    the owner's when Gmail labels it ``SENT`` or its From is an own address.
    A received message is ``replied`` when the owner wrote in the thread
    after it and within *reply_ms*; one too new for that to be settled
    (``now_ms - reply_ms`` or later) gets ``mature: False``.
    """
    own_keys = frozenset(gmail_inbound.own_address_key(a) for a in own_addresses if a)
    rows = sorted(messages, key=lambda m: int(m.get("internal_ms") or 0))
    mine = [int(m.get("internal_ms") or 0) for m in rows
            if "SENT" in set(m.get("labels") or ()) or _own(str(m.get("from") or ""), own_keys)]
    out: dict[str, dict[str, Any]] = {}
    seen = 0
    for m in rows:
        labels = set(m.get("labels") or ())
        t = int(m.get("internal_ms") or 0)
        if "SENT" in labels or _own(str(m.get("from") or ""), own_keys):
            continue
        seen += 1
        if labels & _NOT_RECEIVED:
            continue
        out[str(m["id"])] = {
            "replied": any(t < s <= t + reply_ms for s in mine),
            "mature": t + reply_ms <= now_ms,
            "owner_wrote_before": any(s < t for s in mine),
            "position": seen,
        }
    return out


def sample_evenly(items: list[Any], limit: int) -> list[Any]:
    """At most *limit* of *items*, spread evenly over the list (Gmail lists
    newest first, so this spreads over time)."""
    if limit <= 0:
        return []
    if len(items) <= limit:
        return list(items)
    step = len(items) / limit
    return [items[int(i * step)] for i in range(limit)]


def split_by_time(cases: list[dict], holdout_from_ms: int) -> tuple[list[dict], list[dict]]:
    """``(open, holdout)``: cases that arrived at or after *holdout_from_ms*
    are held out. Each list oldest first."""
    ordered = sorted(cases, key=lambda c: int(c.get("received_ms") or 0))
    return ([c for c in ordered if int(c.get("received_ms") or 0) < holdout_from_ms],
            [c for c in ordered if int(c.get("received_ms") or 0) >= holdout_from_ms])


def _iso(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).isoformat(timespec="minutes")


def case_of(inb: gmail_inbound.Inbound, label: Mapping[str, Any], own_addresses: Iterable[str]) -> dict[str, Any]:
    """One email as a practice case: what a person triaging would see, and
    the owner's actual decision. The quoted part of the body is cut and the
    rest clipped; nothing set after the mail arrived is kept."""
    blockers = gmail_inbound.auto_reply_blockers(inb, own_addresses=own_addresses)
    from_addr = inb.from_addrs[0] if inb.from_addrs else ""
    return {
        "id": inb.message_id,
        "thread_id": inb.thread_id,
        "received_ms": inb.internal_ms,
        "received": _iso(inb.internal_ms) if inb.internal_ms else "",
        "from": f"{inb.from_name} <{from_addr}>" if inb.from_name and from_addr else (from_addr or inb.from_name),
        "to": list(inb.to_addrs)[:20],
        "cc": list(inb.cc_addrs)[:20],
        "subject": inb.subject[:300],
        "body": gmail_inbound.strip_quoted(inb.text)[:_BODY_CHARS],
        "attachments": list(inb.attachment_names)[:20],
        "categories": [c.removeprefix("CATEGORY_").lower() for c in _CATEGORY_LABELS if c in inb.label_ids],
        "bulk": "auto_or_list" in blockers or "automated_sender" in blockers,
        "owner_wrote_before": bool(label.get("owner_wrote_before")),
        "position": int(label.get("position") or 1),
        "label": bool(label.get("replied")),
    }


def _summary(cases: list[dict]) -> dict[str, Any]:
    n = len(cases)
    yes = sum(1 for c in cases if c.get("label"))
    return {"cases": n, "replied": yes, "base_rate": round(yes / n, 4) if n else 0.0,
            "first": cases[0]["received"] if cases else "", "last": cases[-1]["received"] if cases else ""}


# ── reading Gmail ────────────────────────────────────────────────────


def gmail_fetch(client: Any, token: str) -> Fetch:
    """A :data:`Fetch` over the owner's Gmail, through the auto-reply's reader."""
    from captain_claw.flight_deck import gmail_autoreply

    async def fetch(path: str, params: Mapping[str, Any]) -> tuple[int, dict]:
        return await gmail_autoreply._gmail_get(client, token, path, params)
    return fetch


async def own_addresses(fetch: Fetch) -> list[str]:
    """The mailbox's address and its verified send-as aliases."""
    st, profile = await fetch("/profile", {})
    address = str(profile.get("emailAddress") or "").strip().lower()
    if st != 200 or not address:
        raise RuntimeError("Gmail didn't say whose mailbox this is.")
    own = [address]
    st, data = await fetch("/settings/sendAs", {})
    if st == 200:
        for alias in data.get("sendAs") or []:
            if isinstance(alias, dict) and (alias.get("isPrimary") is True
                                            or alias.get("verificationStatus") == "accepted"):
                addr = str(alias.get("sendAsEmail") or "").strip().lower()
                if addr:
                    own.append(addr)
    return list(dict.fromkeys(own))


async def _list_received(fetch: Fetch, after_s: int, before_s: int) -> list[dict]:
    """Received mail in the window, newest first: ``[{"id", "threadId"}]``."""
    q = f"-in:sent -in:drafts -in:chats -in:spam -in:trash after:{after_s} before:{before_s}"
    out: list[dict] = []
    token = ""
    while len(out) < _LIST_CAP:
        params: dict[str, Any] = {"q": q, "maxResults": 500}
        if token:
            params["pageToken"] = token
        st, data = await fetch("/messages", params)
        if st != 200:
            raise RuntimeError(f"Gmail wouldn't list the mail (HTTP {st}).")
        out += [m for m in data.get("messages") or [] if isinstance(m, dict) and m.get("id")]
        token = str(data.get("nextPageToken") or "")
        if not token:
            break
    return out[:_LIST_CAP]


async def build_mail_triage(fetch: Fetch, *, now_ms: int, window_days: int = 90, reply_days: int = 3,
                            holdout_days: int = 21, max_cases: int = 400,
                            progress: Callable[[str, int, int], None] | None = None) -> dict[str, Any]:
    """Read the owner's mail and build the cases: ``{"own", "open", "holdout",
    "skipped"}``. The window ends *reply_days* before now, so every label is
    settled; the last *holdout_days* of it are held out."""
    report = progress or (lambda _stage, _done, _total: None)
    reply_ms = reply_days * _DAY_MS
    end_ms = now_ms - reply_ms
    start_ms = end_ms - window_days * _DAY_MS
    own = await own_addresses(fetch)
    report("listing", 0, 0)
    listed = await _list_received(fetch, start_ms // 1000, end_ms // 1000)
    chosen = sample_evenly(listed, min(max(1, max_cases), _MAX_CASES))
    labels: dict[str, dict[str, dict]] = {}
    cases: list[dict] = []
    skipped = {"unreadable": 0, "own": 0, "unsettled": 0}
    lock = asyncio.Lock()
    sem = asyncio.Semaphore(_FETCH_PARALLEL)
    done = 0

    async def thread_labels(tid: str) -> dict[str, dict]:
        async with lock:
            if tid in labels:
                return labels[tid]
        st, data = await fetch(f"/threads/{tid}", {"format": "metadata", "metadataHeaders": ["From"]})
        rows = []
        if st == 200:
            for m in data.get("messages") or []:
                hdrs = {h.get("name", "").lower(): h.get("value", "") for h in (m.get("payload") or {}).get("headers") or []}
                rows.append({"id": m.get("id"), "internal_ms": int(m.get("internalDate") or 0),
                             "labels": m.get("labelIds") or [], "from": _addr(hdrs.get("from", ""))})
        got = label_messages(rows, own, reply_ms=reply_ms, now_ms=now_ms)
        async with lock:
            labels[tid] = got
        return got

    async def one(ref: dict) -> None:
        nonlocal done
        async with sem:
            mid, tid = str(ref["id"]), str(ref.get("threadId") or ref["id"])
            try:
                label = (await thread_labels(tid)).get(mid)
                if label is None:
                    skipped["own"] += 1
                    return
                if not label["mature"]:
                    skipped["unsettled"] += 1
                    return
                st, msg = await fetch(f"/messages/{mid}", {"format": "full"})
                if st != 200:
                    skipped["unreadable"] += 1
                    return
                cases.append(case_of(gmail_inbound.parse_message(msg), label, own))
            finally:
                done += 1
                report("reading", done, len(chosen))

    await asyncio.gather(*(one(ref) for ref in chosen))
    open_cases, holdout = split_by_time(cases, end_ms - holdout_days * _DAY_MS)
    return {"own": own, "open": open_cases, "holdout": holdout, "skipped": skipped,
            "window": {"from": _iso(start_ms), "to": _iso(end_ms), "holdout_from": _iso(end_ms - holdout_days * _DAY_MS)}}


# ── reading Google Calendar ──────────────────────────────────────────


def calendar_fetch(client: Any, token: str) -> Fetch:
    """A :data:`Fetch` over the owner's Google Calendar (read-only calls)."""

    async def fetch(path: str, params: Mapping[str, Any]) -> tuple[int, dict]:
        try:
            r = await client.get(f"{CALENDAR_API}{path}", params=dict(params),
                                 headers={"Authorization": f"Bearer {token}"})
            return r.status_code, (r.json() if r.content else {})
        except Exception as exc:  # noqa: BLE001 — a failed read is a status, like Gmail's reader
            return 0, {"error": str(exc)[:200]}
    return fetch


def _event_ms(value: Any) -> int:
    """An event's time (``{"dateTime"}`` or ``{"date"}``, or an RFC 3339 string) in ms; 0 if unreadable."""
    raw = value.get("dateTime") or value.get("date") if isinstance(value, Mapping) else value
    try:
        dt = datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return 0
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return int(dt.timestamp() * 1000)


def invite_case(event: Mapping[str, Any]) -> dict[str, Any] | None:
    """One invitation the owner answered, as a case; None for anything else:
    their own events, ones without guests, ones never or only tentatively answered."""
    if (event.get("organizer") or {}).get("self") or event.get("status") == "cancelled":
        return None
    attendees = [a for a in event.get("attendees") or [] if isinstance(a, Mapping)]
    me = next((a for a in attendees if a.get("self")), None)
    answer = str((me or {}).get("responseStatus") or "")
    if answer not in ("accepted", "declined"):
        return None
    start, end = _event_ms(event.get("start") or {}), _event_ms(event.get("end") or {})
    asked = _event_ms(event.get("created")) or start
    if not start or not asked:
        return None
    others = [str(a.get("displayName") or a.get("email") or "") for a in attendees if not a.get("self")]
    organizer = event.get("organizer") or {}
    return {
        "id": str(event.get("id") or "")[:120], "received_ms": asked, "received": _iso(asked),
        "title": str(event.get("summary") or "(no title)")[:300],
        "organizer": (f"{organizer.get('displayName')} <{organizer.get('email')}>" if organizer.get("displayName")
                      else str(organizer.get("email") or "(unknown)")),
        "start": _iso(start), "duration_min": max(0, (end - start) // 60_000) if end else 0,
        "all_day": "date" in (event.get("start") or {}) and "dateTime" not in (event.get("start") or {}),
        "recurring": bool(event.get("recurringEventId")),
        "guests": len(others), "others": [o for o in others if o][:8],
        "location": str(event.get("location") or "")[:200],
        "description": " ".join(str(event.get("description") or "").split())[:_DESC_CHARS],
        "label": answer == "accepted",
    }


async def build_calendar_invites(fetch: Fetch, *, now_ms: int, window_days: int = 180, holdout_days: int = 30,
                                 max_cases: int = 400,
                                 progress: Callable[[str, int, int], None] | None = None) -> dict[str, Any]:
    """Past invitations, each labelled by the owner's own answer; the newest held out."""
    st, cal = await fetch("/calendars/primary", {})
    if st in (401, 403):
        raise RuntimeError("Your Google connection doesn't include the calendar: an admin turns on the calendar "
                           "permission for Google, then you reconnect Google.")
    if st != 200:
        raise RuntimeError(f"Google Calendar didn't answer (HTTP {st}).")
    own = str(cal.get("id") or "").lower()
    params: dict[str, Any] = {"timeMin": _iso(now_ms - window_days * _DAY_MS), "timeMax": _iso(now_ms),
                              "singleEvents": "true", "orderBy": "startTime", "maxResults": 250,
                              "showDeleted": "false"}
    events: list[dict] = []
    while len(events) < _EVENTS_CAP:
        st, data = await fetch("/calendars/primary/events", params)
        if st != 200:
            raise RuntimeError(f"Google Calendar wouldn't list the events (HTTP {st}).")
        events += [e for e in data.get("items") or [] if isinstance(e, dict)]
        if progress:
            progress("listing", len(events), 0)
        if not data.get("nextPageToken"):
            break
        params = {**params, "pageToken": data["nextPageToken"]}
    skipped = {"own": 0, "not answered": 0}
    # One case per series: the latest answer to a recurring meeting stands for it.
    by_key: dict[str, dict] = {}
    for e in events:
        case = invite_case(e)
        if case is None:
            skipped["own" if (e.get("organizer") or {}).get("self") else "not answered"] += 1
            continue
        by_key[str(e.get("recurringEventId") or case["id"])] = case
    cases = sorted(by_key.values(), key=lambda c: c["received_ms"])
    cases = sample_evenly(cases, max_cases) if len(cases) > max_cases else cases
    cut = now_ms - holdout_days * _DAY_MS
    open_, hold = split_by_time(cases, cut)
    return {"own": [own] if own else [], "open": open_, "holdout": hold, "skipped": skipped,
            "window": {"from": _iso(now_ms - window_days * _DAY_MS), "to": _iso(now_ms), "holdout_from": _iso(cut)}}


def _addr(header: str) -> str:
    m = re.search(r"<([^>]+)>", header or "")
    return (m.group(1) if m else (header or "").split(",")[0]).strip().strip('"').lower()


# ── the store ────────────────────────────────────────────────────────


def new_set_id(kind: str = KIND_MAIL_TRIAGE) -> str:
    return f"{KINDS[kind]['id']}-{time.strftime('%Y%m%d')}-{secrets.token_hex(3)}"


def save_set(owner_id: str, set_id: str, built: dict[str, Any], params: dict[str, Any], *,
             kind: str = KIND_MAIL_TRIAGE) -> dict[str, Any]:
    """Freeze *built* as practice set *set_id*; its meta."""
    d = set_dir(owner_id, set_id)
    for split in ("open", "holdout"):
        _write_private(d / f"{split}.jsonl",
                       "".join(json.dumps(c, ensure_ascii=False) + "\n" for c in built[split]))
    meta = {
        "id": set_id, "kind": kind, "state": "ready", "created": time.time(),
        "title": KINDS[kind]["title"],
        "source": {"gmail" if kind == KIND_MAIL_TRIAGE else "calendar": built["own"][0] if built["own"] else "",
                   "params": params, "window": built["window"]},
        "label": (f"you wrote in the thread within {params.get('reply_days', 3)} days" if kind == KIND_MAIL_TRIAGE
                  else "you accepted the invitation"),
        "open": _summary(built["open"]), "holdout": _summary(built["holdout"]),
        "skipped": built["skipped"],
    }
    write_json(d / "meta.json", meta)
    return meta


def mark(owner_id: str, set_id: str, **fields: Any) -> None:
    """Change a set's meta (its state while it is being built)."""
    d = set_dir(owner_id, set_id)
    meta = _read_json(d / "meta.json") or {"id": set_id, "kind": fields.get("kind") or KIND_MAIL_TRIAGE,
                                           "created": time.time()}
    meta.update(fields)
    write_json(d / "meta.json", meta)


def get_meta(owner_id: str, set_id: str) -> dict | None:
    try:
        return _read_json(set_dir(owner_id, set_id) / "meta.json")
    except ValueError:
        return None


def list_sets(owner_id: str) -> list[dict]:
    base = _owner_dir(owner_id)
    if not base.is_dir():
        return []
    out = [m for d in base.iterdir() if d.is_dir() and (m := _read_json(d / "meta.json"))]
    return sorted(out, key=lambda m: -float(m.get("created") or 0))


def load_cases(owner_id: str, set_id: str, split: str) -> list[dict]:
    if split not in ("open", "holdout"):
        raise ValueError("split is open or holdout")
    path = set_dir(owner_id, set_id) / f"{split}.jsonl"
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    return [json.loads(line) for line in lines if line.strip()]


def delete_set(owner_id: str, set_id: str) -> bool:
    d = set_dir(owner_id, set_id)
    if not d.is_dir():
        return False
    shutil.rmtree(d)
    return True
