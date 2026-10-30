"""Telling a user something is waiting for them, on the channels they chose.

Autonomous Work reaches out two ways: a nudge (what the agent said) and a
proposal waiting for the user's approval. Besides the agent's own chat, each
goes to every channel the user switched on in their autonomy config:

  * WhatsApp — ``nudge_to_whatsapp``: the text, to ``notify_waid`` (the
    bridge allowlist and /mute still apply, see ``push_to_waid``).
  * Telegram — ``notify_telegram``: the text, to the user's own chat with
    the deck's shared bot.
  * Devices  — ``notify_devices``: the user's connected handhelds beep and
    show how many proposals wait; the text itself stays in Flight Deck. A
    device that was asleep beeps for what came meanwhile when it next
    connects (``device_connected``), once.

And always the bell — Flight Deck's own notifications, in the full view and
the kiosk alike: where the text is read, and a proposal approved or rejected.

The devices' count follows the ledger: every status change calls
``waiting_may_have_changed`` (autonomy.py), which sends a silent update
when the number moved.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from typing import Any

from captain_claw.flight_deck.autonomy import _norm_user, get_store, resolve_config

_log = logging.getLogger(__name__)

# A beep a device missed while asleep is still worth one when it connects
# for this long; after that it only shows the count.
UNSEEN_KEEP_S = 24 * 3600.0
DEVICE_LINE_MAX = 120
# A bell notification's ref type, by its type (the action's id is the ref):
# the bell offers Approve / Reject on a proposal while it waits.
BELL_REF = "autonomy_action"
_BELL_REFS = {"autonomy_proposal": BELL_REF, "autonomy_nudge": "autonomy_nudge"}

# The newest thing that beeped for each user (wall time, its line), when each
# device last beeped (keyed by user + device id), and the waiting count each
# user's devices last got. In memory: after a restart a device shows the
# count when it connects, without a beep for what came before.
_last_event: dict[str, tuple[float, str]] = {}
_device_heard: dict[str, float] = {}
_sent_count: dict[str, int] = {}


def waiting_count(user_id: str) -> int:
    """How many of this user's proposals wait for their approval (a
    Taskboard next step that waited a week expires first)."""
    get_store().expire_taskboard_proposals(user_id)
    return get_store().count_status(user_id, "awaiting_approval")


def _quiet(cfg: dict[str, Any]) -> bool:
    from captain_claw.flight_deck.arbiter import _in_quiet_hours

    return _in_quiet_hours(int(cfg.get("quiet_hours_start", 22)), int(cfg.get("quiet_hours_end", 8)))


def device_line(text: str) -> str:
    """The text as a device shows it: its first line, plain, no emoji."""
    from captain_claw.flight_deck.bus_client import activity_text
    from captain_claw.flight_deck.meta_webhook_bridge import strip_markdown

    first = next((ln for ln in strip_markdown(str(text or "")).splitlines() if ln.strip()), "")
    line = activity_text(first, DEVICE_LINE_MAX + 1)
    return line if len(line) <= DEVICE_LINE_MAX else line[:DEVICE_LINE_MAX - 1].rstrip() + "…"


def taskboard_link(task_id: str, base: str = "") -> str:
    """The link that opens the Taskboard at a task: ``?taskboard=<id>`` on
    ``base`` (the deck's public URL), or on the page the bell is read on."""
    if not task_id:
        return ""
    if not base:
        return f"?taskboard={task_id}"
    sep = "&" if "?" in base else ("" if base.endswith("/") else "/") + "?"
    return f"{base}{sep}taskboard={task_id}"


def task_of(action: dict[str, Any]) -> str:
    """The Taskboard task an action is about ('' when none): a next step's
    own, else the one the board keeps it on (``autonomy:<id>``, or its
    follow-up's ``fu:<id>``) — on the action's user's board only."""
    from captain_claw.flight_deck.autonomy import TASKBOARD_SOURCE

    payload = action.get("payload") or {}
    if action.get("source") == TASKBOARD_SOURCE and payload.get("task_id"):
        return str(payload["task_id"])
    try:
        from captain_claw.flight_deck.taskboard_nextsteps import open_board

        board, uid = open_board(), _norm_user(str(action.get("user_id") or ""))
        if board is None:
            return ""
        keys = [f"autonomy:{action.get('id')}"] + ([f"fu:{payload['follow_up_id']}"]
                                                   if payload.get("follow_up_id") else [])
        return next((t for t in (board.task_for_key(uid, k) for k in keys) if t), "")
    except Exception as exc:  # noqa: BLE001 — a link is a courtesy, never a failure
        _log.debug("no Taskboard task for %s: %s", action.get("id"), exc)
        return ""


def proposal_text(action: dict[str, Any], waiting: int) -> str:
    """What WhatsApp and Telegram say about a new proposal — the link opens
    its task on the Taskboard when it is about one."""
    title = str(action.get("title") or "").strip() or "a new proposal"
    more = f" ({waiting} waiting)" if waiting > 1 else ""
    url = os.environ.get("FD_PUBLIC_URL", "").strip()
    task_id = task_of(action) if url else ""
    if task_id:
        url = taskboard_link(task_id, url)
    return f"Waiting for your approval in Flight Deck{more}: {title}" + (f"\n{url}" if url else "")


def nudge_texts(action: dict[str, Any], text: str) -> tuple[str, str]:
    """A nudge as WhatsApp / Telegram say it and as the bell does: with the
    link to its task on the Taskboard when it is about one (WhatsApp and
    Telegram only on a deck with a public URL; the bell's link opens in
    place). Devices show the first line, never the link."""
    task_id = task_of(action)
    if not task_id:
        return text, text
    url = os.environ.get("FD_PUBLIC_URL", "").strip()
    out = f"{text}\n{taskboard_link(task_id, url)}" if url else text
    return out, f"{text}\n[Open in Taskboard]({taskboard_link(task_id)})"


def _device_key(user_id: str, device_id: str) -> str:
    return f"{_norm_user(user_id)}:{device_id}"


async def notify_owner(
    user_id: str, kind: str, text: str, *, device_text: str = "", cfg: dict[str, Any] | None = None,
    bell: tuple[str, str, str] | None = None, action_id: str = "",
) -> dict[str, Any]:
    """Send ``text`` to each channel ``user_id`` switched on. ``kind`` ("nudge",
    "proposal", "test") names the log lines; ``device_text`` is the devices'
    line (default: the text's first); ``bell`` = (type, title, body) goes to
    the bell first, about ``action_id``. Returns what each channel did, None
    where it is off: WhatsApp ``[sent, total]``, Telegram ``True/False``,
    devices how many beeped, the bell ``True/False``. Never raises."""
    out: dict[str, Any] = {"whatsapp": None, "telegram": None, "devices": None, "bell": None}
    text = str(text or "").strip()
    if not text:
        return out
    if bell:
        out["bell"] = await _to_bell(user_id, *bell, action_id=action_id)
    try:
        cfg = cfg or resolve_config(user_id)
    except Exception as exc:  # noqa: BLE001
        _log.debug("notify: no config for %s: %s", user_id, exc)
        return out
    store = get_store()

    if cfg.get("nudge_to_whatsapp", True):
        try:
            from captain_claw.flight_deck.fd_dispatch import _nudge_waids
            from captain_claw.flight_deck.whatsapp_bridge import push_to_waid

            waids, skipped = _nudge_waids(cfg)
            sent = 0
            for waid in waids:
                if await push_to_waid(waid, text):
                    sent += 1
            out["whatsapp"] = [sent, len(waids)]
            if waids:
                store.log(user_id, f"{kind} → whatsapp", f"sent to {sent}/{len(waids)} recipient(s)")
            elif skipped:
                store.log(user_id, f"{kind} → whatsapp skipped", skipped)
        except Exception as exc:  # noqa: BLE001
            _log.debug("%s whatsapp delivery failed: %s", kind, exc)

    if cfg.get("notify_telegram", False):
        try:
            from captain_claw.flight_deck.fd_scheduler import _deliver_telegram_owner

            res = await _deliver_telegram_owner(user_id, text)
            out["telegram"] = bool(res.get("ok"))
            store.log(user_id, f"{kind} → telegram" + ("" if res.get("ok") else " skipped"),
                      str(res.get("note") or ""))
        except Exception as exc:  # noqa: BLE001
            _log.debug("%s telegram delivery failed: %s", kind, exc)

    if cfg.get("notify_devices", True):
        try:
            line = device_line(device_text or text)
            count = waiting_count(user_id)
            beep = not _quiet(cfg)
            now = time.time()
            if kind != "test":
                _last_event[_norm_user(user_id)] = (now, line)
            reached = await _push_devices(user_id, count, line, beep)
            for device_id in reached:
                _device_heard[_device_key(user_id, device_id)] = now
            out["devices"] = len(reached)
            if reached:
                store.log(user_id, f"{kind} → devices",
                          f"{'beeped' if beep else 'shown quietly (quiet hours) on'} {len(reached)} device(s)")
        except Exception as exc:  # noqa: BLE001
            _log.debug("%s device delivery failed: %s", kind, exc)
    return out


async def _to_bell(user_id: str, bell_type: str, title: str, body: str, *, action_id: str = "") -> bool:
    """One bell notification for the user (``local`` with sign-in off)."""
    try:
        from captain_claw.flight_deck.auth import get_db

        await get_db().add_notification(
            _norm_user(user_id), bell_type, title.strip()[:200], body.strip()[:2000],
            ref_type=_BELL_REFS.get(bell_type, "") if action_id else "", ref_id=action_id,
        )
        return True
    except Exception as exc:  # noqa: BLE001
        _log.debug("bell notification for %s failed: %s", user_id, exc)
        return False


async def _push_devices(user_id: str, count: int, text: str, beep: bool) -> list[str]:
    from captain_claw.flight_deck.device_bridge import push_waiting

    reached = await push_waiting(user_id, count, text, beep)
    _sent_count[_norm_user(user_id)] = count
    return reached


async def notify_proposal(user_id: str, action: dict[str, Any]) -> dict[str, Any]:
    """A proposal now waits for this user's approval: tell them. Never raises."""
    try:
        waiting = waiting_count(user_id)
    except Exception as exc:  # noqa: BLE001
        _log.debug("proposal notify: no count for %s: %s", user_id, exc)
        waiting = 1
    title = str(action.get("title") or "").strip()
    return await notify_owner(
        user_id, "proposal", proposal_text(action, waiting),
        device_text=f"Waiting for your approval: {title}" if title else "A proposal waits in Flight Deck",
        bell=_proposal_bell(action), action_id=str(action.get("id") or ""),
    )


def _proposal_bell(action: dict[str, Any]) -> tuple[str, str, str]:
    """A proposal's bell notification: (type, title, body) — its last line
    the link to its task on the Taskboard when it is about one."""
    title = str(action.get("title") or "").strip()
    rationale = str(action.get("rationale") or "").strip()
    task_id = task_of(action)
    return ("autonomy_proposal", "Waiting for your approval",
            (title or "A new proposal") + (f"\n{rationale[:600]}" if rationale else "")
            + (f"\n[Open in Taskboard]({taskboard_link(task_id)})" if task_id else ""))


async def bell_backfill(user_id: str) -> int:
    """Every proposal waiting for this user gets a bell notification if it has
    none yet (made before the bell held them, or when it couldn't be
    written): the bell lists what the devices count. How many were added;
    never raises. Read or dismissed ones are never added again."""
    try:
        from captain_claw.flight_deck.auth import get_db

        db = get_db()
        added = 0
        for action in get_store().list_actions(user_id, status="awaiting_approval", limit=50):
            aid = str(action.get("id") or "")
            if not aid or await db.count_notifications_since(
                    _norm_user(user_id), "", "autonomy_proposal", ref_type=BELL_REF, ref_id=aid):
                continue
            if await _to_bell(user_id, *_proposal_bell(action), action_id=aid):
                added += 1
        return added
    except Exception as exc:  # noqa: BLE001
        _log.debug("bell backfill for %s failed: %s", user_id, exc)
        return 0


async def waiting_bells(user_id: str, have: set[str], *, unread_only: bool = False) -> list[dict[str, Any]]:
    """The bell notifications of proposals still waiting for this user that
    aren't in `have` (notification ids): older than the bell's page, yet
    still to be answered there. Never raises."""
    try:
        from captain_claw.flight_deck.auth import get_db

        aids = [str(a.get("id") or "") for a in
                get_store().list_actions(user_id, status="awaiting_approval", limit=200)]
        rows = await get_db().list_notifications_by_ref(user_id, BELL_REF, aids)
        return [r for r in rows if r.get("id") not in have and not (unread_only and r.get("read"))]
    except Exception as exc:  # noqa: BLE001
        _log.debug("waiting bells for %s failed: %s", user_id, exc)
        return []


def waiting_may_have_changed(user_id: str) -> None:
    """From the ledger (sync): bring the devices' count up to date, later on
    this event loop. No loop (a script, a test): nothing to do."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return
    loop.create_task(_sync_count(user_id))


async def _sync_count(user_id: str) -> None:
    try:
        cfg = resolve_config(user_id)
        if not cfg.get("notify_devices", True):
            return
        count = waiting_count(user_id)
        if _sent_count.get(_norm_user(user_id)) == count:
            return
        await _push_devices(user_id, count, "", False)
    except Exception as exc:  # noqa: BLE001
        _log.debug("waiting count sync failed for %s: %s", user_id, exc)


async def devices_switched_off(user_id: str) -> None:
    """Notifications to devices went off: clear the count they show."""
    try:
        await _push_devices(user_id, 0, "", False)
    except Exception as exc:  # noqa: BLE001
        _log.debug("clearing device counts failed for %s: %s", user_id, exc)


def device_connected(user_id: str, device_id: str) -> dict[str, Any] | None:
    """The ``waiting`` frame for a device that just connected: the count (0
    clears one it kept), and a beep for what came while it was away — once,
    and only while recent. None when its owner has devices switched off."""
    try:
        cfg = resolve_config(user_id)
        if not cfg.get("notify_devices", True):
            return None
        count = waiting_count(user_id)
    except Exception as exc:  # noqa: BLE001
        _log.debug("waiting frame for %s failed: %s", device_id, exc)
        return None
    now = time.time()
    key = _device_key(user_id, device_id)
    at, line = _last_event.get(_norm_user(user_id), (0.0, ""))
    unseen = at > _device_heard.get(key, 0.0) and now - at < UNSEEN_KEEP_S
    if unseen:
        _device_heard[key] = now
    _sent_count[_norm_user(user_id)] = count
    return {"type": "waiting", "count": count, "text": line if unseen else "",
            "beep": bool(unseen and not _quiet(cfg))}
