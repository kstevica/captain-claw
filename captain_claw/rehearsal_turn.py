"""A rehearsal — a Captain Swarm lane acts on nothing in the world.

docs/captain-swarm-plan.md §4.1–4.2. A worker spawned for a Captain Swarm
attempt (a Bat run in rehearsal) carries ``CLAW_REHEARSAL=<run>:<experiment>``
in its environment, and every turn it takes is a rehearsal:

* **Reads run:** files, the web, mailboxes and calendars, the browser looking
  at pages.
* **Its own files run:** writing and editing in its workspace, documents it
  makes there.
* **Everything that reaches outside is recorded as an intent and not done:**
  mail sent or drafted, a mailbox or calendar changed, Drive, messages, posts,
  other agents, payments, a spend settled (authorizing one, which only holds
  it, runs), an outside API called. Flight Deck keeps the intent with the
  experiment, so the winner can be carried out once, later, through Bat's
  gate.
* **Refused:** running code and driving the screen (computer use gets a
  desktop of its own, later), starting another run, and memory writes — a
  rehearsal keeps nothing.
* **The browser stops at the commit point.** It navigates, reads and fills in
  fields, but a click or key that would submit, buy, pay, send, delete,
  confirm, sign, publish, book or accept is recorded and not done. When it
  can't tell, it's a commit point.
* **One attempt per site and per account.** Before the browser opens a page or
  signs in with stored credentials, Flight Deck is asked whether another
  attempt is on that site or account right now; if one is, or if it can't be
  asked, the browser doesn't go there.

Fails closed: a tool this module doesn't know is an intent, never a call. The
decision is made here, from the tool's name and arguments, never from what the
worker says about them; afterwards Flight Deck checks the tool-call register
for anything outward that ran anyway (``captain_swarm_rehearsal``).
"""

from __future__ import annotations

import os
import re
from typing import Any

ENV = "CLAW_REHEARSAL"
_TAG_RE = re.compile(r"^cs_[0-9a-f]{6,32}:exp_[0-9a-f]{6,32}$")

# Writes that stay in the worker's own workspace.
LOCAL_WRITES = frozenset({"write", "edit", "workspace_write", "docx", "xlsx", "pptx", "visualize", "image_gen",
                          "image_manipulation", "pocket_tts"})
# Memory: a rehearsal keeps nothing.
MEMORY_WRITES = frozenset({"insights", "topics", "intentions", "facts", "personality", "playbooks",
                           "project_memory", "todo", "reasoning_effort"})
# What only asks the owner, or only judges text: no world changes.
SEND_ALLOWED = frozenset({"ask_human", "jev"})
# Code, the screen, and other runs.
EXEC_REFUSED_TOOLS = frozenset({"shell", "terminal", "desktop_action", "scripts", "app_runner", "code", "termux",
                                "pinchtab", "vatra", "basna", "bat", "crew", "captain_swarm", "screen_capture",
                                "custom_tools"})

_COMMIT_RE = re.compile(
    r"\b(submit|buy|purchase|order|check ?out|pay|payment|place (my |the )?order|send|post|publish|delete|remove|"
    r"confirm|book|reserve|sign ?up|register|subscribe|accept|agree|transfer|donate|finish|complete|"
    r"continue to payment|save card|unsubscribe|cancel (my )?(order|subscription|account))\b",
    re.IGNORECASE)
_SAFE_BROWSER = frozenset({"open", "navigate", "screenshot", "scroll", "wait", "status", "close", "observe", "act",
                           "accessibility_tree", "find_element", "type", "login", "credentials_list",
                           "network_start", "network_stop", "network_list", "network_capture", "network_clear",
                           "switch_app", "list_sessions", "workflow_record_start", "workflow_record_stop",
                           "workflow_list", "workflow_show"})
_SECRET_KEYS = re.compile(r"pass|secret|token|key|auth|cookie|card|cvv|iban", re.IGNORECASE)

INTENT = ("Recorded as an intent, not done: {what}. This is a rehearsal — nothing outside is changed. If this "
          "attempt wins, the owner can have it carried out once. Carry on as if it will be, and say in your "
          "result what you would do.")
COMMIT = ("Stopped at the commit point, not done: {what}. This is a rehearsal — a click or key that would submit, "
          "buy, pay, send, book or confirm is recorded instead. Note the page and what you filled in, and finish "
          "your result: the owner can have it carried out once if this attempt wins.")
REFUSED = "{what} isn't available in a rehearsal: {why}"
SITE_BUSY = ("Not now: {why}. Attempts run at the same time, and only one at a time may use a site or an account "
             "in its browser, so the site doesn't see a crowd. Use another source for now, or read the page with "
             "web_fetch; come back to it later if you still need its browser.")


def active() -> str:
    """The ``<run>:<experiment>`` this process rehearses for, or "" when it doesn't."""
    tag = os.environ.get(ENV, "").strip()
    return tag if _TAG_RE.match(tag) else ""


def _browser(args: dict) -> tuple[str, str]:
    action = str(args.get("action") or "").strip().lower()
    target = " ".join(str(args.get(k) or "") for k in ("text", "selector", "role", "goal")).strip()
    if action in _SAFE_BROWSER:
        return "allow", ""
    if action == "click":
        if _COMMIT_RE.search(target):
            return "commit", f"click {target[:120]!r}"
        role = str(args.get("role") or "").lower()
        selector = str(args.get("selector") or "").lower()
        if not str(args.get("text") or "").strip() and (role == "button" or re.search(r"button|btn|submit", selector)):
            return "commit", f"click a button it can't read ({target[:120]!r})"
        return "allow", ""
    if action == "press_key":
        key = str(args.get("key") or "").strip().lower()
        return ("commit", "press Enter (it can submit a form)") if key in ("enter", "return") else ("allow", "")
    if action in ("api_replay", "api_test"):
        method = str(args.get("method") or "GET").upper()
        return ("allow", "") if method == "GET" else ("intent", f"{method} {str(args.get('endpoint') or '')[:120]}")
    if action == "workflow_run":
        return "intent", f"replay the workflow {str(args.get('workflow_id') or args.get('workflow_name') or '')[:80]}"
    return "refuse", f"browser {action}"


def classify(name: str, arguments: Any) -> tuple[str, str]:
    """``(verdict, what)``: ``allow``, ``intent``, ``commit`` or ``refuse``, and
    the call in words."""
    from captain_claw.flight_deck.tool_risk import risk_for
    tool = str(name or "").strip()
    args = arguments if isinstance(arguments, dict) else {}
    action = str(args.get("action") or "").strip().lower()
    what = f"{tool} {action}".strip()
    if tool == "browser":
        return _browser(args)
    if tool in EXEC_REFUSED_TOOLS:
        return "refuse", what
    if tool == "spend":
        return ("allow", "") if action in ("authorize", "status") else ("intent", what)
    risk = risk_for(tool, args)
    if risk == "read":
        return "allow", ""
    if tool in LOCAL_WRITES:
        return "allow", ""
    if tool in MEMORY_WRITES:
        return "refuse", what
    if tool in SEND_ALLOWED:
        return "allow", ""
    return "intent", what


def _why(verdict_what: str) -> str:
    tool = verdict_what.split(" ", 1)[0]
    if tool in MEMORY_WRITES:
        return "a rehearsal keeps nothing."
    if tool in ("vatra", "basna", "bat", "crew", "captain_swarm"):
        return "a rehearsal starts no other runs."
    return "no code runs and no screen is driven in a rehearsal. Use the web, the browser and your own files."


def safe_args(arguments: Any) -> dict:
    """What an intent keeps of a call's arguments: plain values only, secrets left out."""
    args = arguments if isinstance(arguments, dict) else {}
    out = {}
    for k, v in args.items():
        if str(k).startswith("_") or _SECRET_KEYS.search(str(k)):
            continue
        if isinstance(v, (str, int, float, bool)):
            out[str(k)] = v[:2000] if isinstance(v, str) else v
        elif isinstance(v, list) and all(isinstance(x, (str, int, float)) for x in v):
            out[str(k)] = [x[:300] if isinstance(x, str) else x for x in v[:30]]
    return out


async def _post(path: str, body: dict, agent: Any) -> dict | None:
    """POST *body* to Flight Deck's Captain Swarm agent route: its JSON answer, or
    None when it can't be asked."""
    try:
        import httpx

        from captain_claw.fd_client import resolve_flight_deck_url
        from captain_claw.tools.bat import _agent_secret_headers, _fd_url_pinned
        fd_url = resolve_flight_deck_url(getattr(agent, "_fd_url", "") or "")
        if not fd_url:
            return None
        try:
            from captain_claw.config import get_config
            web = get_config().web
            port, auth = int(web.port or 0), web.auth_token or ""
        except Exception:
            port, auth = 0, ""
        body = {"web_auth": auth if _fd_url_pinned(fd_url) else "", "source_port": port,
                "owner_id": os.environ.get("FD_OWNER_ID", ""), **body}
        async with httpx.AsyncClient(timeout=10.0) as client:
            r = await client.post(f"{fd_url}/fd/captain-swarm/agent/{path}", json=body,
                                  headers=_agent_secret_headers(fd_url))
        return r.json() if r.status_code == 200 else None
    except Exception:
        return None


async def _record(tag: str, kind: str, name: str, arguments: Any, what: str, agent: Any) -> bool:
    """Hand the intent to Flight Deck. Best effort: not recorded is still not done."""
    got = await _post("intent", {"tag": tag, "kind": kind, "tool": str(name), "args": safe_args(arguments),
                                 "what": what[:500]}, agent)
    return got is not None


def browser_place(name: str, arguments: Any) -> dict:
    """The site or account a browser call is about to use: ``{"url": …}``,
    ``{"account": …}``, or ``{}`` for a call that goes nowhere new."""
    args = arguments if isinstance(arguments, dict) else {}
    if str(name or "") != "browser":
        return {}
    action = str(args.get("action") or "").strip().lower()
    if action == "login" and str(args.get("app_name") or "").strip():
        return {"account": str(args["app_name"]).strip()[:80]}
    url = str(args.get("url") or "").strip()
    return {"url": url[:2000]} if url.lower().startswith(("http://", "https://")) else {}


async def _site_check(tag: str, place: dict, agent: Any) -> str:
    """"" when this attempt may use *place* now, else why not. Fails closed."""
    got = await _post("site", {"tag": tag, **place}, agent)
    if got is None:
        return "Flight Deck couldn't be asked whether another attempt is on that site"
    return "" if got.get("ok") else str(got.get("why") or "another attempt is on that site")


async def tool_block(name: str, arguments: Any, agent: Any) -> str | None:
    """The answer to a tool call in a rehearsal when it must not run, else None."""
    tag = active()
    if not tag:
        return None
    verdict, what = classify(name, arguments)
    if verdict == "allow":
        place = browser_place(name, arguments)
        if place:
            why = await _site_check(tag, place, agent)
            if why:
                return SITE_BUSY.format(why=why)
        return None
    if verdict == "refuse":
        return REFUSED.format(what=what, why=_why(what))
    await _record(tag, verdict, name, arguments, what, agent)
    return (COMMIT if verdict == "commit" else INTENT).format(what=what)
