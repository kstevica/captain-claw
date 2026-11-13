"""Tasks the person leaves for Claude in a data room, asked from the Inbox.

Claude Desktop can't be sent work: it acts when the person prompts it in a chat.
So a task waits in the room until Claude, asked to check, lists it through the
connector (``room_tasks``), does it, and delivers with ``task=<n>``. The answer
lands in the Inbox like any delivery, restored there for the owner alone.

The person's words may hold real names. They are anonymized with the room's
own vault and settings first, and the person sees exactly what Claude will read
before it is queued. A known real value still in it afterwards refuses the task.
Both versions are kept encrypted with the room vault's key.
"""

from __future__ import annotations

import re
import time
from typing import Any

from captain_claw.flight_deck import rooms_store as rs
from captain_claw.logging import get_logger

log = get_logger(__name__)

MAX_PROMPT = 8_000
ABOUT_CHARS = 20_000   # the delivery a task is about goes along in tokens up to this; longer → room_get_delivery


class TaskError(ValueError):
    """Told to the person (or to Claude) — never with a real value in it."""


def _clean(prompt: Any) -> str:
    text = re.sub(r"[ \t]+\n", "\n", str(prompt or "")).strip()
    if not text:
        raise TaskError("Write what Claude should do.")
    if len(text) > MAX_PROMPT:
        raise TaskError(f"A task takes at most {MAX_PROMPT:,} characters.")
    return text


async def anonymize_prompt(room: dict[str, Any], prompt: Any) -> dict[str, Any]:
    """The person's words as Claude will read them (tokens), through the room's
    own vault and settings; ``still_real`` lists tokens whose real value is
    still there (nothing is queued then)."""
    from captain_claw.flight_deck import anonymize_routes as ar
    from captain_claw.flight_deck import rooms_inbox as inbox
    from captain_claw.flight_deck import rooms_job as rj

    text = _clean(prompt)
    try:
        await inbox._vault(room)
    except inbox.DeliveryError:
        raise TaskError("This room's vault is gone — it can't take tasks.") from None
    data = rj._data(room, room["vault_id"])
    try:
        out, info = await ar.anonymize_blob(room["owner_id"], "task.md", text.encode("utf-8"), data,
                                            actor=f"room:{room['id']}", source="data-room-task")
    except Exception as exc:  # noqa: BLE001 — HTTPException (trusted model down …): its detail has no value in it
        detail = getattr(exc, "detail", "") or type(exc).__name__
        raise TaskError(f"The task couldn't be anonymized: {detail}") from None
    tokens = out.decode("utf-8").strip()
    _row, _cipher, state = await inbox._vault(room)  # what the vault knows now, new names included
    return {"prompt_tokens": tokens, "still_real": rj.leaks(tokens, rj.needles(state)),
            "found": info.get("summary") or {}}


async def queue(room: dict[str, Any], prompt: Any, about: Any = None) -> dict[str, Any]:
    from captain_claw.flight_deck import rooms_inbox as inbox

    text = _clean(prompt)
    about_seq = None
    if about not in (None, "", 0):
        try:
            about_seq = int(about)
        except (TypeError, ValueError):
            raise TaskError("about is a delivery's number.") from None
        if await rs.store().delivery(room["id"], about_seq) is None:
            raise TaskError("No such delivery in this room.")
    res = await anonymize_prompt(room, text)
    if res["still_real"]:
        raise TaskError("A real name was still in the task after anonymizing it — reword it, or add the name to the "
                        "room's names to hide, and try again.")
    _row, cipher, _state = await inbox._vault(room)
    seq = await rs.store().add_task(room["id"], room["owner_id"],
                                    cipher.encrypt(text.encode("utf-8")).decode("ascii"),
                                    cipher.encrypt(res["prompt_tokens"].encode("utf-8")).decode("ascii"), about_seq)
    task = await rs.store().task(room["id"], seq)
    assert task is not None
    return view(task, cipher)


def _open(cipher: Any, value: str) -> str:
    try:
        return cipher.decrypt(value.encode("ascii")).decode("utf-8")
    except Exception:  # noqa: BLE001 — a task this key can't open is shown as such, never guessed
        return ""


def view(task: dict[str, Any], cipher: Any) -> dict[str, Any]:
    """For the owner: their words and what Claude reads."""
    return {"seq": task["seq"], "created_at": task["created_at"], "about": task["about"], "state": task["state"],
            "prompt": _open(cipher, task["prompt"]), "prompt_tokens": _open(cipher, task["prompt_tokens"]),
            "taken_at": task["taken_at"], "done_at": task["done_at"], "delivery": task["delivery"]}


async def owner_list(room: dict[str, Any]) -> list[dict[str, Any]]:
    from captain_claw.flight_deck import rooms_inbox as inbox

    items = await rs.store().tasks(room["id"])
    if not items:
        return []
    _row, cipher, _state = await inbox._vault(room)
    return [view(t, cipher) for t in items]


async def cancel(room: dict[str, Any], seq: int) -> bool:
    task = await rs.store().task(room["id"], seq)
    if task is None:
        raise TaskError("No such task.")
    if task["state"] not in ("waiting", "taken"):
        raise TaskError("This task is already answered or cancelled.")
    return await rs.store().set_task(room["id"], seq, state="cancelled", done_at=time.time())


async def for_claude(room: dict[str, Any], client_id: str) -> list[dict[str, Any]]:
    """The room's open tasks as Claude reads them (tokens only); listing one
    marks it taken, so the person sees Claude has it."""
    from captain_claw.flight_deck import rooms_inbox as inbox

    items = await rs.store().open_tasks(room["id"])
    if not items:
        return []
    _row, cipher, _state = await inbox._vault(room)
    out = []
    for t in items:
        entry: dict[str, Any] = {"task": t["seq"], "asked": time.strftime("%Y-%m-%d %H:%M", time.gmtime(t["created_at"])),
                                 "prompt": _open(cipher, t["prompt_tokens"])}
        if t["about"]:
            d = await rs.store().delivery(room["id"], int(t["about"]))
            if d is not None:
                try:
                    p = inbox.open_payload(cipher, d)
                except Exception:  # noqa: BLE001
                    p = None
                if p is not None:
                    body = str(p.get("body_tokens") or "")
                    entry["about"] = {"delivery": t["about"], "title": p.get("title_tokens") or "",
                                      "content": body[:ABOUT_CHARS]}
                    if len(body) > ABOUT_CHARS:
                        entry["about"]["note"] = (f"Cut at {ABOUT_CHARS:,} characters — room_get_delivery gives "
                                                  "the rest, page by page.")
        if t["state"] == "waiting":
            await rs.store().set_task(room["id"], t["seq"], state="taken", client_id=client_id, taken_at=time.time())
        out.append(entry)
    return out


async def close(room: dict[str, Any], seq: Any, delivery: int, client_id: str) -> str:
    """Mark a task answered by *delivery*; a note for the receipt."""
    try:
        seq = int(seq)
    except (TypeError, ValueError):
        return "task is a task's number (from room_tasks); the delivery is in, without a task."
    task = await rs.store().task(room["id"], seq)
    if task is None:
        return f"There is no task {seq} in this room; the delivery is in, without a task."
    if task["state"] not in ("waiting", "taken"):
        return f"Task {seq} was already answered or cancelled; the delivery is in as a new one."
    await rs.store().set_task(room["id"], seq, state="done", done_at=time.time(), delivery=delivery,
                              client_id=client_id or task["client_id"])
    d = await rs.store().delivery(room["id"], delivery)
    if d is not None:
        await rs.store().update_delivery(room["id"], delivery, report={**d["report"], "task": seq})
    return f"Task {seq} is answered with this delivery."
