"""Tool search: full-mode agents past a token budget of tool definitions.

Every tool an agent may call went out with its full definition on every model
call: about 39k tokens for a default agent, 42k under Flight Deck and 59k with
every account connected. Eco mode already sends only a core set to small
models; this is the same idea for full mode, with a way back to the rest.

When the definitions come to more than ``tools.tool_search.budget_tokens``:

- always in full: the file and web basics, MCP tools, every tool used in the
  session and every tool ``search_tools`` loaded;
- then, while the budget lasts: the tools the first message's wording points
  at (eco mode's intent patterns), eco mode's core set, then the rest,
  smallest first;
- the others stay named, with a one-line summary, in the system prompt's
  "Available tools" list, and ``search_tools`` (sent in their place) loads
  the matches: their full definitions go out with every call from the next
  step on. The result itself only names them — repeating the definitions
  there would send each one twice on every later call (once in the tools,
  once in the history).

The registry still executes any tool by name: this only decides which
definitions the model is given.

The selection is made once per session, kept in the session's metadata, and
only grows (a search or a call adds a tool). Tool definitions sit at the front
of every request, so a set that changed with each message would break the
provider's prompt cache for the whole conversation.

Flight Deck's agent settings write ``~/.captain-claw/tool_search.json``
(``{"enabled": …, "budget_tokens": …}``), read on every call, so a change
applies from the agent's next message without a restart. With no file the
config's ``tools.tool_search`` decides.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Callable, Iterable

from captain_claw.logging import get_logger

log = get_logger(__name__)

SEARCH_TOOLS_NAME = "search_tools"
STATE_KEY = "tool_search"

# Never deferred: the file and web basics a strong model reaches for first.
ESSENTIAL_TOOLS: frozenset[str] = frozenset({
    "shell", "read", "write", "edit", "glob", "grep", "web_fetch", "web_search",
})

DEFAULT_LIMIT = 3
MAX_LIMIT = 10


# ── Settings: the switch file, else the config ──────────────────────────

SWITCH_FILENAME = "tool_search.json"


def _switch_path() -> Path:
    return Path("~/.captain-claw").expanduser() / SWITCH_FILENAME


def parse_switch(text: object) -> dict[str, Any] | None:
    """``{"enabled": bool, "budget_tokens": int}`` from the switch file's
    text; None when it isn't one."""
    try:
        data = json.loads(str(text or ""))
    except (TypeError, ValueError):
        return None
    if not isinstance(data, dict) or not isinstance(data.get("enabled"), bool):
        return None
    try:
        budget = int(data.get("budget_tokens"))
    except (TypeError, ValueError):
        return None
    if budget <= 0:
        return {"enabled": False, "budget_tokens": _default_budget()}
    return {"enabled": data["enabled"], "budget_tokens": budget}


def render_switch(enabled: bool, budget: int) -> str:
    """What the switch file holds."""
    return json.dumps({"enabled": bool(enabled), "budget_tokens": int(budget)})


def _default_budget() -> int:
    from captain_claw.config import ToolSearchConfig

    return ToolSearchConfig().budget_tokens


def settings() -> dict[str, Any]:
    """``{"enabled", "budget_tokens"}`` for this agent right now."""
    try:
        switch = parse_switch(_switch_path().read_text(encoding="utf-8"))
    except OSError:
        switch = None
    if switch is not None:
        return switch
    try:
        from captain_claw.config import get_config

        cfg = get_config().tools.tool_search
        budget = int(cfg.budget_tokens or 0)
        return {"enabled": bool(cfg.enabled) and budget > 0, "budget_tokens": budget or _default_budget()}
    except Exception:
        return {"enabled": False, "budget_tokens": 0}


def clear() -> None:
    """Drop the switch file so the config decides again (the agent's own
    settings page changed ``tools.tool_search``)."""
    try:
        _switch_path().unlink()
    except OSError:
        pass


def budget_tokens() -> int:
    """The budget, or 0 when tool search is off."""
    s = settings()
    return max(0, int(s["budget_tokens"])) if s["enabled"] else 0


# ── Sizing ──────────────────────────────────────────────────────────────


def _wire(defn: dict[str, Any]) -> str:
    return json.dumps(defn, sort_keys=True, separators=(",", ":"), default=str)


def definition_tokens(defn: dict[str, Any], count: Callable[[str], int],
                      cache: dict[str, int] | None = None) -> int:
    """Tokens of one definition as sent (the same wire form the context
    trace counts)."""
    wire = _wire(defn)
    if cache is None:
        return int(count(wire))
    key = hashlib.sha1(wire.encode("utf-8")).hexdigest()[:16]
    tokens = cache.get(key)
    if tokens is None:
        if len(cache) >= 512:
            cache.clear()
        tokens = cache[key] = int(count(wire))
    return tokens


# ── The search_tools definition ─────────────────────────────────────────


def search_definition(deferred: Iterable[str]) -> dict[str, Any]:
    """``search_tools``, naming the tools not loaded yet (in a stable order,
    so the definition changes only when the loaded set does)."""
    names = ", ".join(deferred)
    return {
        "name": SEARCH_TOOLS_NAME,
        "description": (
            "Load more of your tools. To keep each request small, some tools in "
            "your 'Available tools' list are not in your function list yet: "
            f"{names}. Search by a tool's name or by what you need to do (for "
            "example 'pptx', 'resize an image', 'send an email'): the matching "
            "tools are added to your function list from your next step and stay "
            "loaded for the rest of the session."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Tool names, or words for what you need to do.",
                },
                "limit": {
                    "type": "integer",
                    "description": f"How many tools to load at most (default {DEFAULT_LIMIT}, "
                                   f"up to {MAX_LIMIT}).",
                },
            },
            "required": ["query"],
        },
    }


# ── Selection ───────────────────────────────────────────────────────────


def select(
    defs: list[dict[str, Any]],
    sizes: dict[str, int],
    *,
    budget: int,
    state: dict[str, Any] | None,
    used: Iterable[str] = (),
    intent: Iterable[str] = (),
    core: Iterable[str] = (),
    count: Callable[[str], int] | None = None,
) -> tuple[list[dict[str, Any]], list[str], dict[str, Any] | None]:
    """Which definitions go in full.

    Returns ``(sent, deferred, state)``: the definitions to send (registry
    order, ``search_tools`` last when anything is deferred), the deferred
    names (registry order) and the session state to keep. Everything goes out
    unchanged when it fits the budget.

    *state* is what the session kept from an earlier call. While its budget
    matches, its loaded set is reused and only grows by *used* — *intent* and
    the budget fill are decided once, when the session is first over budget.
    """
    names = [str(d.get("name", "")) for d in defs]
    total = sum(sizes.get(n, 0) for n in names)
    if budget <= 0 or total <= budget:
        return defs, [], state

    present = set(names)
    used_set = set(used) & present
    fixed = (ESSENTIAL_TOOLS | used_set | {n for n in names if n.startswith("mcp_")}) & present

    if state is not None and int(state.get("budget", -1)) == budget:
        loaded = (set(state.get("loaded") or []) | used_set) & present
        loaded |= fixed
        # Kept tools that are hidden right now (an account disconnected) stay
        # in the state; they come back when the gate opens again.
        keep = set(state.get("loaded") or []) | loaded
    else:
        loaded = set(fixed)
        intent_set = set(intent)
        core_set = set(core)
        reserve = 0
        if count is not None:
            reserve = int(count(_wire(search_definition(n for n in names if n not in loaded))))
        spent = reserve + sum(sizes.get(n, 0) for n in loaded)
        rest = sorted(
            (n for n in names if n not in intent_set and n not in core_set),
            key=lambda n: (sizes.get(n, 0), n),
        )
        order = ([n for n in names if n in intent_set]
                 + [n for n in names if n in core_set and n not in intent_set]
                 + rest)
        for n in order:
            if n in loaded:
                continue
            size = sizes.get(n, 0)
            if spent + size <= budget:
                loaded.add(n)
                spent += size
        keep = loaded

    sent = [d for d in defs if d.get("name") in loaded]
    deferred = [n for n in names if n not in loaded]
    new_state = {"budget": budget, "loaded": sorted(keep)}
    if deferred:
        sent = sent + [search_definition(deferred)]
    return sent, deferred, new_state


# ── Agent side ──────────────────────────────────────────────────────────


def _get_state(agent: Any) -> dict[str, Any] | None:
    meta = getattr(getattr(agent, "session", None), "metadata", None)
    state = meta.get(STATE_KEY) if isinstance(meta, dict) else getattr(agent, "_tool_search_state", None)
    return state if isinstance(state, dict) else None


def _set_state(agent: Any, state: dict[str, Any]) -> None:
    meta = getattr(getattr(agent, "session", None), "metadata", None)
    if isinstance(meta, dict):
        meta[STATE_KEY] = state
    else:
        agent._tool_search_state = state


def _used_tools(agent: Any) -> set[str]:
    """Tools called this turn or in the session's recent history."""
    used: set[str] = set()
    for key in getattr(agent, "_turn_tool_call_counts", None) or {}:
        used.add(str(key).split("|", 1)[0])
    session = getattr(agent, "session", None)
    for msg in (getattr(session, "messages", None) or [])[-40:]:
        for tc in msg.get("tool_calls") or []:
            name = tc.get("name") or (tc.get("function") or {}).get("name", "")
            if name:
                used.add(str(name))
    used.discard(SEARCH_TOOLS_NAME)
    return used


def _forget(agent: Any) -> None:
    agent._tool_search_pool = {}
    agent._tool_search_sent = frozenset()


def apply(agent: Any, defs: list[dict[str, Any]], user_input: str = "") -> list[dict[str, Any]]:
    """The definitions a full-mode call sends (see the module docstring)."""
    budget = budget_tokens()
    if not budget or not defs or any(d.get("name") == SEARCH_TOOLS_NAME for d in defs):
        # Off, nothing to send, or a registered tool owns the name.
        _forget(agent)
        return defs
    from captain_claw.agent_orchestration_mixin import _ECO_CORE_TOOLS, _eco_select_tools_by_intent

    count = agent._count_tokens
    cache = getattr(agent, "_tool_search_token_cache", None)
    if not isinstance(cache, dict):
        cache = agent._tool_search_token_cache = {}
    sizes = {str(d.get("name", "")): definition_tokens(d, count, cache) for d in defs}
    state = _get_state(agent)
    sent, deferred, new_state = select(
        defs, sizes, budget=budget, state=state,
        used=_used_tools(agent),
        intent=_eco_select_tools_by_intent(user_input or ""),
        core=_ECO_CORE_TOOLS,
        count=count,
    )
    if not deferred:
        _forget(agent)
        return defs
    if new_state != state:
        _set_state(agent, new_state)
    agent._tool_search_pool = {str(d.get("name", "")): d for d in defs}
    agent._tool_search_sent = frozenset(str(d.get("name", "")) for d in sent)
    sig = (len(sent), tuple(deferred))
    if getattr(agent, "_tool_search_logged", None) != sig:
        agent._tool_search_logged = sig
        log.info(
            "Tool search: deferred tool definitions",
            budget=budget,
            full_tokens=sum(sizes.values()),
            sent_tokens=sum(sizes.get(str(d.get("name", "")), 0) for d in sent[:-1]),
            sent=len(sent) - 1,
            deferred=len(deferred),
        )
    return sent


# ── Search ──────────────────────────────────────────────────────────────

_STOP_WORDS = frozenset(
    "a an and or the to of for in on at by with from into my me i we you it its this that "
    "is are be can could do does please some any all use using need want wants tool tools "
    "how what which".split()
)
_WORD_RE = re.compile(r"[a-z0-9]+")


def _stem(word: str) -> str:
    for suffix in ("ing", "ies", "es", "ed", "s"):
        if len(word) > len(suffix) + 2 and word.endswith(suffix):
            return word[: -len(suffix)]
    return word


def _terms(text: str) -> list[str]:
    return [_stem(w) for w in _WORD_RE.findall(text.lower()) if w not in _STOP_WORDS]


def _matches(term: str, word: str) -> bool:
    if term == word:
        return True
    short, long_ = (term, word) if len(term) <= len(word) else (word, term)
    return len(short) >= 4 and long_.startswith(short)


def search(defs: list[dict[str, Any]], query: str, limit: int = DEFAULT_LIMIT) -> list[dict[str, Any]]:
    """The definitions that best match *query*: a tool's exact name always
    matches; otherwise words in the name count most, then eco mode's intent
    patterns, then words in the description."""
    from captain_claw.agent_orchestration_mixin import _eco_select_tools_by_intent

    query = str(query or "")
    lowered = query.lower()
    exact = {n for n in re.findall(r"[a-z0-9_]+", lowered)}
    terms = _terms(query.replace("_", " "))
    intent = _eco_select_tools_by_intent(query)
    scored: list[tuple[int, int, dict[str, Any]]] = []
    for idx, defn in enumerate(defs):
        name = str(defn.get("name", ""))
        if name == SEARCH_TOOLS_NAME:
            continue
        score = 100 if name.lower() in exact else 0
        name_words = [_stem(p) for p in name.lower().split("_") if p]
        desc_words = {_stem(w) for w in _WORD_RE.findall(str(defn.get("description", "")).lower())}
        for term in terms:
            if any(_matches(term, w) for w in name_words):
                score += 10
            elif any(_matches(term, w) for w in desc_words):
                score += 2
        if name in intent:
            score += 6
        if score:
            scored.append((score, idx, defn))
    scored.sort(key=lambda row: (-row[0], row[1]))
    n_exact = sum(1 for s, _, _ in scored if s >= 100)
    return [d for _, _, d in scored[: max(limit, n_exact)]]


def _summary(defn: dict[str, Any]) -> str:
    """A definition's description, cut to its first sentence."""
    from captain_claw.agent_context_mixin import _short_tool_desc

    return _short_tool_desc(str(defn.get("description", ""))) or str(defn.get("name", ""))


def run(agent: Any, arguments: dict[str, Any]) -> Any:
    """Execute ``search_tools``: load the matching tools for this session.
    The result names them with a one-line summary; their definitions arrive
    in the function list of the next call."""
    from captain_claw.tools.registry import ToolResult

    pool: dict[str, dict[str, Any]] = getattr(agent, "_tool_search_pool", None) or {}
    query = str((arguments or {}).get("query") or "").strip()
    if not pool:
        return ToolResult(success=True, content=(
            "Every tool you have is already in your function list; call it directly."))
    if not query:
        return ToolResult(success=False, error="Give a query: a tool's name or what you need to do.")
    try:
        limit = int((arguments or {}).get("limit") or DEFAULT_LIMIT)
    except (TypeError, ValueError):
        limit = DEFAULT_LIMIT
    limit = max(1, min(MAX_LIMIT, limit))

    sent: frozenset[str] = getattr(agent, "_tool_search_sent", None) or frozenset()
    hits = search(list(pool.values()), query, limit)
    deferred = [n for n in pool if n not in sent]
    if not hits:
        return ToolResult(success=True, content=(
            f"No tool matches \"{query}\". Not loaded yet: {', '.join(deferred)}. "
            "Try a tool's name or other words."))

    new = [d for d in hits if d.get("name") not in sent]
    already = [str(d.get("name")) for d in hits if d.get("name") in sent]
    if new:
        state = _get_state(agent) or {"budget": budget_tokens(), "loaded": sorted(sent)}
        loaded = set(state.get("loaded") or []) | {str(d.get("name")) for d in new}
        _set_state(agent, {**state, "loaded": sorted(loaded)})
        agent._tool_search_sent = sent | {str(d.get("name")) for d in new}
        log.info("Tool search: loaded tools", query=query[:80], loaded=[d.get("name") for d in new])
    lines: list[str] = []
    if new:
        lines.append(f"Loaded {len(new)} tool{'s' if len(new) != 1 else ''} into your function list; "
                     f"call {'them' if len(new) != 1 else 'it'} from your next step:")
        lines.extend(f"- {d.get('name')}: {_summary(d)}" for d in new)
    if already:
        lines.append("Already in your function list: " + ", ".join(already) + ".")
    return ToolResult(success=True, content="\n".join(lines))
