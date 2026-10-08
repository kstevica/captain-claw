#!/usr/bin/env python3
"""Context replay — what each turn's LLM call carried, rebuilt offline.

Rebuilds every human turn of a stored session and runs the REAL
``Agent._build_messages`` on a stub agent (no LLM, no network), then prints
per-section tokens from the context trace, the tool-schema size, how many
synthetic messages were replayed as history, and how much of each turn's
first request repeats the previous turn's last request — the prefix a
provider's prompt cache can reuse.

The database is opened read-only. Run it from a neutral directory: config is
merged from ``./config.yaml`` and ``./.env`` in the working directory, and
the system prompt also reads ``~/.captain-claw`` (point ``HOME`` elsewhere to
leave out this machine's reflections and profile).

Usage:
  cd /tmp/neutral && PYTHONPATH=<repo> <repo>/.venv/bin/python \\
      <repo>/scripts/context_replay.py --db /path/to/sessions.db --list
  ... --db /path/to/sessions.db --session <id> --last 12
  ... --db /path/to/sessions.db --session <id> --json > replay.json
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from captain_claw.agent import Agent  # noqa: E402
from captain_claw.config import get_config, set_config  # noqa: E402
from captain_claw.llm import LLMProvider, LLMResponse  # noqa: E402
from captain_claw.session import Session  # noqa: E402

try:
    import tiktoken

    _ENC = tiktoken.get_encoding("cl100k_base")

    def count_tokens(text: str) -> int:
        return len(_ENC.encode(text or "", disallowed_special=()))
except Exception:  # pragma: no cover - fallback when tiktoken is absent
    def count_tokens(text: str) -> int:
        return max(1, len(text or "") // 4)


class _StubProvider(LLMProvider):
    provider = "openai"
    model = "replay-stub"

    async def complete(self, messages, tools=None, temperature=None, max_tokens=None):
        return LLMResponse(content="")

    async def complete_streaming(self, messages, tools=None, temperature=None, max_tokens=None):
        if False:
            yield ""

    def count_tokens(self, text: str) -> int:
        return count_tokens(text)


def _origin(msg: dict) -> str:
    try:
        from captain_claw import msg_origin

        origin_of = getattr(msg_origin, "origin_of", None)
        if origin_of is not None:
            return str(origin_of(msg))
    except Exception:
        pass
    return "human" if msg.get("role") == "user" else str(msg.get("role", ""))


def _load(db: str, session_id: str) -> Session:
    conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    row = conn.execute(
        "SELECT id, name, messages, created_at, updated_at, metadata FROM sessions WHERE id = ?",
        (session_id,),
    ).fetchone()
    if row is None:
        raise SystemExit(f"session {session_id!r} not found")
    return Session.from_dict({
        "id": row[0], "name": row[1], "messages": json.loads(row[2] or "[]"),
        "created_at": row[3], "updated_at": row[4], "metadata": json.loads(row[5] or "{}"),
    })


def _list(db: str) -> None:
    conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    rows = conn.execute(
        "SELECT id, name, length(messages), updated_at FROM sessions ORDER BY length(messages) DESC"
    ).fetchall()
    for sid, name, size, updated in rows:
        if size and size > 2:
            print(f"{sid}  {size:>10,} chars  {updated}  {name}")


def _serialize(messages) -> str:
    """The request body as the prefix cache sees it (order matters)."""
    parts = []
    for m in messages:
        parts.append(json.dumps({
            "role": m.role, "content": m.content, "tool_call_id": m.tool_call_id,
            "tool_calls": m.tool_calls, "reasoning_content": m.reasoning_content,
        }, ensure_ascii=False, sort_keys=True))
    return "\n".join(parts)


def _common_prefix_chars(a: str, b: str) -> int:
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


def _build(agent: Agent, stored: Session, upto: int, turn_start: int, text: str):
    msgs = copy.deepcopy(stored.messages[:upto])
    for m in msgs:   # recount with this tokenizer
        for key in ("token_count", "_tc_counted", "_rc_counted", "reasoning_token_count"):
            m.pop(key, None)
    agent.session = Session(id=stored.id, name=stored.name, messages=msgs,
                            metadata=copy.deepcopy(stored.metadata))
    agent._turn_system_prompt = None
    agent._turn_context_notes = None
    agent._turn_user_text = text
    out = agent._build_messages(tool_messages_from_index=turn_start, query=text)
    return out, dict(agent.last_context_window)


def replay(db: str, session_id: str, last: int) -> list[dict]:
    stored = _load(db, session_id)
    cfg = get_config().model_copy(deep=True)
    budget = (stored.metadata.get("context_window") or {}).get("context_budget_tokens")
    if budget:
        cfg.context.max_tokens = int(budget)
    cfg.logging.level = "WARNING"
    set_config(cfg)
    try:
        from captain_claw.logging import configure_logging

        configure_logging()
    except Exception:
        pass

    agent = Agent(provider=_StubProvider())
    agent.memory = None
    agent._build_env_now_text = lambda: "System environment:\n- (replayed turn)"
    try:
        agent._register_default_tools()
        tool_defs = agent.tools.get_definitions()
    except Exception:
        tool_defs = []

    turns = [i for i, m in enumerate(stored.messages)
             if m.get("role") == "user" and _origin(m) == "human"]
    report: list[dict] = []
    prev_last_request = ""
    for n, t in enumerate(turns):
        text = str(stored.messages[t].get("content", ""))
        first, window = _build(agent, stored, t + 1, t, text)
        agent._note_tool_schema_tokens(tool_defs)
        window = dict(agent.last_context_window)
        first_request = _serialize(first)
        reused = _common_prefix_chars(prev_last_request, first_request) if prev_last_request else 0
        next_t = turns[n + 1] if n + 1 < len(turns) else len(stored.messages)
        last_request, _ = _build(agent, stored, next_t, t, text)
        prev_last_request = _serialize(last_request)
        report.append({
            "turn": n,
            "index": t,
            "sections": window.get("sections", {}),
            "prompt_tokens": window.get("prompt_tokens"),
            "prompt_tokens_with_tools": window.get("prompt_tokens_with_tools"),
            "tool_schema_tokens": window.get("tool_schema_tokens"),
            "context_notes_used": window.get("context_notes_used"),
            "dropped_messages": window.get("dropped_messages"),
            "synthetic_suppressed": window.get("synthetic_suppressed", 0),
            "prefix_reused_share": round(reused / len(first_request), 3) if first_request else 0.0,
        })
    return report[-last:] if last else report


def _print_table(rows: list[dict]) -> None:
    cols = ("system_static", "system_dynamic", "prior_history", "current_chain", "turn_message",
            "context_block", "env_note", "reasoning_prior")
    header = "turn  idx " + " ".join(f"{c[:13]:>13}" for c in cols) + "   tools  suppr  reuse"
    print(header)
    for r in rows:
        s = r["sections"]
        print(f"{r['turn']:4d} {r['index']:4d} "
              + " ".join(f"{int(s.get(c, 0) or 0):13d}" for c in cols)
              + f"  {int(r.get('tool_schema_tokens') or 0):6d}  {int(r.get('synthetic_suppressed') or 0):5d}"
              + f"  {r['prefix_reused_share']:5.2f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--db", required=True, help="path to a sessions.db (opened read-only)")
    parser.add_argument("--session", help="session id to replay")
    parser.add_argument("--last", type=int, default=0, help="only the last N human turns")
    parser.add_argument("--list", action="store_true", help="list sessions by size")
    parser.add_argument("--json", action="store_true", help="print JSON instead of a table")
    args = parser.parse_args()
    try:
        if args.list or not args.session:
            _list(args.db)
            return
        rows = replay(args.db, args.session, args.last)
        if args.json:
            print(json.dumps(rows, indent=1))
        else:
            _print_table(rows)
    finally:
        sys.stdout.flush()
        # Agent construction leaves non-daemon worker threads behind.
        os._exit(0)


if __name__ == "__main__":
    main()
