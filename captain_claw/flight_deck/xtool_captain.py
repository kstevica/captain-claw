"""captain: Flight Deck's helper for Captain custom tools.

Don't change this file: publishing puts Flight Deck's own copy into every
version. Standard library only.

While a tool runs, Flight Deck keeps a private line open to it (the file
descriptor in $XT_CAPTAIN_FD). Through it the tool can ask the agent that ran
it to use one of the agent's tools, run another custom tool, or bring in a file
from the person's folders. Each must be declared in tool.json's "uses";
anything else is refused.

    import captain
    page = captain.tool("web_fetch", url="https://api.frankfurter.app/latest")
    # {"ok": True, "content": "...", "error": None}
    total = captain.custom_tool("csv-total", {"file": captain.file(path), "column": "amount"})
    # the other tool's output; its files are in captain.last_files
    local = captain.vfs_file("vfs:reports/march.xlsx")   # a path this tool can read

    account = captain.ask("Which account?", choices=["Business", "Personal"])   # needs "asks": true

Tests: set XT_FAKE_CALLS to a JSON file listing the replies, in order, as
tool.json's examples record them under "calls":
    [{"tool": "web_fetch", "reply": {"ok": true, "content": "..."}},
     {"custom_tool": "csv-total", "reply": {"ok": true, "output": {...}}},
     {"ask": ["answer"], "reply": {"answers": {"answer": "Business"}}}]
"""

from __future__ import annotations

import json
import os
import socket
from typing import Any

__all__ = ["CaptainError", "Declined", "NoAgent", "NoAnswer", "Refused", "ask", "custom_tool", "file", "last_files",
           "tool", "vfs_file"]


class CaptainError(Exception):
    """Something Flight Deck or the agent said no to, or a call that failed."""


class Refused(CaptainError):  # noqa: N818 — the name tool authors write
    """Not allowed: not declared in "uses", or not something a custom tool may do."""


class NoAgent(CaptainError):  # noqa: N818 — the name tool authors write
    """This run has no line to an agent (and XT_FAKE_CALLS isn't set)."""


class Declined(CaptainError):  # noqa: N818 — the name tool authors write
    """The person chose "Don't answer"."""


class NoAnswer(CaptainError):  # noqa: N818 — the name tool authors write
    """No answer in time, or nobody to ask (the run started on its own)."""


last_files: list[str] = []

_line: dict[str, Any] = {"sock": None, "reader": None, "n": 0, "fake": None}


def _fake_calls() -> list[dict] | None:
    """The recorded replies (XT_FAKE_CALLS), or None when the run is real."""
    if _line["fake"] is None:
        path = os.environ.get("XT_FAKE_CALLS", "")
        if not path:
            _line["fake"] = False
        else:
            with open(path, encoding="utf-8") as fh:
                calls = json.load(fh)
            if not isinstance(calls, list):
                raise CaptainError("XT_FAKE_CALLS must name a JSON list of recorded calls.")
            _line["fake"] = list(calls)
    return None if _line["fake"] is False else _line["fake"]


def _what(req: dict) -> str:
    """What a request names, as a recorded call does: the tool, or the file."""
    return str(req.get("ref") if req["op"] == "file" else req.get("name") or "")


def _from_fake(calls: list[dict], req: dict) -> dict:
    op = req["op"]
    if not calls:
        raise CaptainError(f"No recorded reply left for {op} {_what(req)}".rstrip() + ".")
    nxt = calls.pop(0)
    want = nxt.get(op) if isinstance(nxt, dict) else None
    if want is None or (op != "ask" and str(want) != _what(req)):
        shown = json.dumps({k: v for k, v in nxt.items() if k != "reply"}) if isinstance(nxt, dict) else repr(nxt)
        raise CaptainError(f"The next recorded call is {shown}, but the tool asked for {op} {_what(req)}".rstrip() + ".")
    reply = nxt.get("reply")
    if not isinstance(reply, dict):
        reply = {"ok": True, "output": reply}
    if reply.get("refused"):
        raise Refused(str(reply.get("error") or "Refused."))
    return reply


def _send(req: dict) -> dict:
    calls = _fake_calls()
    if calls is not None:
        return _from_fake(calls, req)
    if _line["sock"] is None:
        fd = os.environ.get("XT_CAPTAIN_FD", "")
        if not fd.isdigit():
            raise NoAgent("This run has no line to an agent: run the tool from an agent (or, in tests, set "
                          "XT_FAKE_CALLS).")
        sock = socket.socket(fileno=int(fd))
        _line["sock"] = sock
        _line["reader"] = sock.makefile("rb")
    _line["n"] += 1
    req = {**req, "id": _line["n"]}
    _line["sock"].sendall((json.dumps(req, ensure_ascii=False) + "\n").encode("utf-8"))
    raw = _line["reader"].readline()
    if not raw:
        raise CaptainError("Flight Deck closed the line.")
    reply = json.loads(raw)
    if not isinstance(reply, dict):
        raise CaptainError("Flight Deck sent something that isn't an answer.")
    if reply.get("refused"):
        raise Refused(str(reply.get("error") or "Refused."))
    return reply


def file(path: str | os.PathLike) -> dict:
    """Mark a file this tool wrote (in $XT_OUT or $TMPDIR) to hand to a tool:
    Flight Deck copies it where the agent can read it and puts that path in."""
    return {"$file": os.path.abspath(os.fspath(path))}


def tool(name: str, **input: Any) -> dict:
    """Ask the agent to use one of its tools. Returns {"ok", "content", "error"};
    a tool that ran and failed is ok False. Raises Refused when it isn't allowed."""
    reply = _send({"op": "tool", "name": name, "input": input})
    return {"ok": bool(reply.get("ok")), "content": str(reply.get("content") or ""),
            "error": reply.get("error") or None}


def custom_tool(name: str, input: dict | None = None) -> Any:
    """Run another custom tool through the agent. Returns its output; its files
    are in captain.last_files. Raises CaptainError when it failed."""
    reply = _send({"op": "custom_tool", "name": name, "input": input or {}})
    last_files[:] = [str(f) for f in reply.get("files") or []]
    if not reply.get("ok"):
        raise CaptainError(str(reply.get("error") or f"{name} failed."))
    return reply.get("output")


def ask(question: str | list[dict], *, choices: list[str] | None = None, multiple: bool = False,
        confirm: bool = False, lines: int = 1, help: str | None = None, default: Any = None,
        optional: bool = False, wait_s: float = 600) -> Any:
    """Ask the person running the tool, and wait for the answer (at most
    *wait_s*, an hour at most); the tool's time stops meanwhile. Needs
    "asks": true in tool.json.

    One question (a string) returns its answer: text, one of *choices*, a list
    of them (*multiple*), or True/False (*confirm*). A list of question objects
    ({"id", "question", and "choices"/"multiple"/"min"/"max"/"confirm"/"lines"/
    "help"/"default"/"optional"}) shows them in one dialog and returns a dict
    by id. Raises Declined ("Don't answer") or NoAnswer (no answer in time, or
    nobody to ask)."""
    single = isinstance(question, str)
    if single:
        q: dict[str, Any] = {"id": "answer", "question": question, "optional": optional}
        if choices is not None:
            q["choices"] = list(choices)
            if multiple:
                q["multiple"] = True
        elif confirm:
            q["confirm"] = True
        else:
            q["lines"] = lines
        if help:
            q["help"] = help
        if default is not None:
            q["default"] = default
        questions: list[dict] = [q]
    else:
        questions = list(question)
    reply = _send({"op": "ask", "questions": questions, "wait_s": wait_s})
    if reply.get("declined"):
        raise Declined(str(reply.get("error") or "They chose not to answer."))
    if not reply.get("ok", "answers" in reply):  # a recorded reply may give only the answers
        raise NoAnswer(str(reply.get("error") or "No answer."))
    answers = reply.get("answers") or {}
    return answers.get("answer") if single else answers


def vfs_file(ref: str) -> str:
    """Bring a file from the person's folders (vfs:<folder>/<path>) into this
    run; returns a path this tool can read. Needs {"tool": "read"} in "uses"."""
    reply = _send({"op": "file", "name": "read", "ref": str(ref)})
    if not reply.get("ok"):
        raise CaptainError(str(reply.get("error") or f"{ref} couldn't be brought in."))
    return str(reply.get("path"))
