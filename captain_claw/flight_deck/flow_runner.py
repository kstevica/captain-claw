"""FlowRunner — executes a Flow inside Flight Deck, dispatching steps to the pool.

Deterministic loop owned by FD; the work happens on pooled agents:
  * ``tool``  step → direct RPC to an agent's ``/api/tool`` (no LLM), or an
                     FD-internal tool (e.g. face_identify) when ``on: fd``.
  * ``agent`` step → ``/fd/consult-peer`` (scoped prompt; reuses busy-retry).
  * ``branch``     → conditional jump (deterministic).
  * ``emit``       → push to a channel (WhatsApp / log).

Dependencies are injected by the FD server to avoid import cycles.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import time
from datetime import UTC, datetime
from typing import Any, Awaitable, Callable

from captain_claw.logging import get_logger
from captain_claw.flight_deck.flows_store import FlowStore

log = get_logger(__name__)

_TEMPLATE_RE = re.compile(r"\{\{\s*([a-zA-Z0-9_.\[\]]+)\s*\}\}")
_VISION_HINTS = ("vision", "image", "multimodal", "minimax", "llava", "vl")
_DEFAULT_MAX_STEPS = 20            # per-frame local sanity cap
_DEFAULT_MAX_TOTAL_STEPS = 100     # shared budget across the whole call tree
_DEFAULT_MAX_DEPTH = 8             # gosub recursion depth cap
_DEFAULT_JOIN_TIMEOUT = 300.0      # seconds to wait on a spawned future

# Sentinel goto target that ends the flow (used by branch `goto`/`else`).
_STOP_TARGET = "__stop__"


class _FlowStopped(Exception):
    """Raised at a step boundary to unwind the whole frame stack on stop."""


class _Root:
    """State shared across every frame of one logical run (the call tree).

    Frames push/pop on `control.frames` for status/control; the budget and trace
    are shared so recursion can't multiply per-flow limits and the run log sees
    one ordered, depth-tagged timeline."""

    __slots__ = ("run_id", "control", "trace", "budget", "depth_cap", "dry",
                 "arch_agents", "arch_slugs", "arch_lock")

    def __init__(self, run_id: str, control: "_RunControl | None", dry: bool,
                 budget: dict[str, int], depth_cap: int) -> None:
        self.run_id = run_id
        self.control = control
        self.trace: list[dict[str, Any]] = []
        self.budget = budget          # mutable {"steps_left": int}
        self.depth_cap = depth_cap
        self.dry = dry
        # Ephemeral archetype agents spawned by `on archetype:<id>` selectors,
        # cached for the whole run (shared across gosub frames) and disposed in
        # run()'s finally. `arch_agents`: archetype id → resolved agent dict;
        # `arch_slugs`: spawn slugs to stop; `arch_lock` serialises lazy spawns
        # so two concurrent steps on the same archetype don't double-spawn.
        self.arch_agents: dict[str, dict[str, Any]] = {}
        self.arch_slugs: list[str] = []
        self.arch_lock = asyncio.Lock()


class _RunControl:
    """In-memory control handle for a live run — pause / resume / stop.

    A run registers one of these while executing; the loop checks it at each
    step boundary. ``resume`` is set while running and cleared to pause;
    ``stopped`` ends the run at the next boundary (optionally after delivering
    ``stop_message`` on the originating channel)."""

    __slots__ = ("resume", "stopped", "stop_message", "owner", "name", "frames", "handle")

    def __init__(self, owner: str = "", name: str = "", handle: str = "") -> None:
        self.resume = asyncio.Event()
        self.resume.set()  # set = running, cleared = paused
        self.stopped = False
        self.stop_message = ""
        self.owner = owner  # caller identity, for '/flow stop|pause|resume'
        self.name = name    # flow name, for '/flow status'
        self.handle = handle  # short stable tag, for '/flow stop <handle>'
        self.frames: list[dict[str, Any]] = []  # live call stack (gosub frames)


_WORLD_ACTING_STEPS = {"emit", "tool", "input"}


def _is_world_acting(flow: dict[str, Any]) -> bool:
    """A flow that touches the outside world — messages the user, runs a tool, or
    asks for input. Synthesized (agent-authored) flows may not call a *permanent*
    world-acting flow without human approval (no borrowing of vetted authority)."""
    if str((flow.get("output") or {}).get("channel") or "") in ("whatsapp", "same", "glasses", "web"):
        return True
    return any(str(s.get("type")) in _WORLD_ACTING_STEPS for s in (flow.get("steps") or []))


def _slug(name: str) -> str:
    """A short base handle from a flow name: initials for multi-word, first two
    chars for a single word."""
    words = re.findall(r"[A-Za-z0-9]+", name or "")
    if len(words) >= 2:
        return "".join(w[0] for w in words[:3]).lower()
    if words:
        return words[0][:2].lower()
    return "fl"


def _assign_handle(name: str) -> str:
    """A handle unique among currently-live runs (base slug, then base2, base3…)."""
    taken = {c.handle for c in _RUN_CONTROL.values() if c.handle}
    base = _slug(name)
    h, n = base, 2
    while h in taken:
        h = f"{base}{n}"
        n += 1
    return h


def resolve_runs(owner: str, target: str) -> list[str]:
    """Resolve a '/flow <cmd> [target]' selector to run ids for *owner*.

    target: '' → the most-recently-started run; 'all' → every live run; a handle
    (exact) or a flow-name fragment (substring) → matching runs. Returns [] when a
    non-empty target matches nothing (caller may then treat it as a stop message)."""
    active = [(rid, c) for rid, c in _RUN_CONTROL.items() if c.owner == owner and not c.stopped]
    if not active:
        return []
    t = (target or "").strip().lower()
    if not t:
        return [active[-1][0]]  # most-recent
    if t == "all":
        return [rid for rid, _ in active]
    by_handle = [rid for rid, c in active if (c.handle or "").lower() == t]
    if by_handle:
        return by_handle
    by_name = [rid for rid, c in active if t in (c.name or "").lower()]
    return by_name


_RUN_CONTROL: dict[str, _RunControl] = {}


def _owner_key(payload: dict[str, Any]) -> str:
    """Caller identity for run control — mirrors flow_router.input_key so a
    user's '/flow stop' reaches the flow they triggered."""
    waid = str(payload.get("waid") or payload.get("whatsapp_waid") or "")
    if waid:
        return f"waid:{waid}"
    return f"chan:{str(payload.get('channel') or '')}:{int(payload.get('origin_port') or 0)}"


def runs_for_owner(owner: str) -> list[str]:
    """Active (not-yet-stopped) run ids controllable by *owner*."""
    return [rid for rid, c in _RUN_CONTROL.items() if c.owner == owner and not c.stopped]


def owner_is_paused(owner: str) -> bool:
    """True if *owner* has a live run that is currently paused."""
    return any(
        c.owner == owner and not c.stopped and not c.resume.is_set()
        for c in _RUN_CONTROL.values()
    )


def owner_run_states(owner: str) -> list[dict[str, Any]]:
    """Snapshot of *owner*'s live runs for '/flow status' (with call-stack crumb)."""
    out: list[dict[str, Any]] = []
    for c in _RUN_CONTROL.values():
        if c.owner != owner or c.stopped:
            continue
        crumb = " › ".join(str(f.get("flow") or "?") for f in c.frames)
        active = str(c.frames[-1].get("step") or "") if c.frames else ""
        out.append({
            "name": c.name,
            "handle": c.handle,
            "paused": not c.resume.is_set(),
            "crumb": crumb,
            "step": active,
        })
    return out


def request_pause(run_id: str) -> bool:
    """Pause a live run at its next step boundary. False if not controllable."""
    c = _RUN_CONTROL.get(run_id)
    if not c or c.stopped:
        return False
    c.resume.clear()
    return True


def request_resume(run_id: str) -> bool:
    """Resume a paused run. False if the run is not controllable."""
    c = _RUN_CONTROL.get(run_id)
    if not c:
        return False
    c.resume.set()
    return True


def request_stop(run_id: str, message: str = "") -> bool:
    """Stop a live run at its next boundary, optionally sending *message* first.
    False if the run is not controllable."""
    c = _RUN_CONTROL.get(run_id)
    if not c:
        return False
    c.stopped = True
    c.stop_message = str(message or "")
    c.resume.set()  # unblock a paused run so it can observe the stop
    return True


def run_is_controllable(run_id: str) -> bool:
    """True while a run is registered and can accept pause/resume/stop."""
    return run_id in _RUN_CONTROL


def _dig(ctx: dict[str, Any], path: str) -> Any:
    """Resolve a dotted path like 'steps.analyze.output' against ctx."""
    cur: Any = ctx
    for part in path.split("."):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            return ""
    return cur


def _disp(v: Any) -> str:
    """Render a value for display/templating. Lists become newline-joined text
    (readable, and re-splittable by `foreach`); dicts stay JSON."""
    if isinstance(v, list):
        return "\n".join(_disp(x) for x in v)
    if isinstance(v, dict):
        return json.dumps(v, default=str)
    if isinstance(v, bool):
        return "true" if v else "false"
    if v is None:
        return ""
    return str(v)


def _render(value: Any, ctx: dict[str, Any]) -> Any:
    """Substitute {{path}} in strings; recurse into dicts/lists."""
    if isinstance(value, str):
        def _sub(m: re.Match) -> str:
            got = _dig(ctx, m.group(1))
            return got if isinstance(got, str) else _disp(got)
        return _TEMPLATE_RE.sub(_sub, value)
    if isinstance(value, dict):
        return {k: _render(v, ctx) for k, v in value.items()}
    if isinstance(value, list):
        return [_render(v, ctx) for v in value]
    return value


# ── value-expression evaluator (for the `set` step) ────────────────────
#
# A small, SAFE evaluator (no eval()) over scalars and lists. Supports:
#   • literals: 42, 3.14, "str", 'str', [a, b, c]
#   • operators: + - * /  (+ is also string-concat / list-concat)
#   • {{path}} operands (resolved against ctx)
#   • functions: split, join, len, upper, lower, trim, first, last, append,
#                int, str, contains
# Used by `set <name> = <expr>` to compute counters, build strings, split text
# into lists, accumulate results, etc. Arbitrary computation still belongs in a
# tool/agent step — this is just enough to orchestrate.

_VAL_TOKEN_RE = re.compile(
    r"""\{\{.*?\}\}                         # {{template}}
      | "(?:[^"\\]|\\.)*" | '(?:[^'\\]|\\.)*'  # quoted string
      | \d+\.\d+ | \d+                      # number
      | [A-Za-z_][A-Za-z0-9_]*              # name (function)
      | [+\-*/(),\[\]]                       # operators / punctuation
    """,
    re.X,
)

_VAL_ESC = {"n": "\n", "t": "\t", "r": "\r", "\\": "\\", '"': '"', "'": "'"}


def _val_unescape(s: str) -> str:
    out, i, n = [], 0, len(s)
    while i < n:
        if s[i] == "\\" and i + 1 < n and s[i + 1] in _VAL_ESC:
            out.append(_VAL_ESC[s[i + 1]]); i += 2; continue
        out.append(s[i]); i += 1
    return "".join(out)


def _to_num(x: Any) -> Any:
    if isinstance(x, bool):
        return int(x)
    if isinstance(x, (int, float)):
        return x
    s = str(x).strip()
    try:
        return int(s)
    except ValueError:
        try:
            return float(s)
        except ValueError:
            return 0


def _is_num(x: Any) -> bool:
    if isinstance(x, bool):
        return False
    if isinstance(x, (int, float)):
        return True
    s = str(x).strip()
    if not s:
        return False
    try:
        int(s); return True
    except ValueError:
        try:
            float(s); return True
        except ValueError:
            return False


_DUR_RE = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*([smhd]?)\s*$", re.I)
_DUR_UNIT = {"s": 1.0, "m": 60.0, "h": 3600.0, "d": 86400.0}


def _parse_duration(v: Any) -> float:
    """'30s' / '5m' / '2h' / '1d' / a bare number (seconds) → seconds."""
    if isinstance(v, (int, float)):
        return float(v)
    m = _DUR_RE.match(str(v or ""))
    if not m:
        return 0.0
    return float(m.group(1)) * _DUR_UNIT[(m.group(2) or "s").lower()]


def _as_list(v: Any) -> list[Any]:
    """Coerce a value to a list: a real list stays; a JSON array string parses;
    any other string splits on newlines (empties dropped)."""
    if isinstance(v, list):
        return v
    if v is None:
        return []
    s = str(v).strip()
    if not s:
        return []
    if s[0] == "[":
        try:
            j = json.loads(s)
            if isinstance(j, list):
                return j
        except (ValueError, TypeError):
            pass
    return [p.strip() for p in s.split("\n") if p.strip()]


def _apply_arith(op: str, a: Any, b: Any) -> Any:
    if op == "+":
        if isinstance(a, list) or isinstance(b, list):
            la = a if isinstance(a, list) else [a]
            lb = b if isinstance(b, list) else [b]
            return list(la) + list(lb)
        a_empty = isinstance(a, str) and not a.strip()
        b_empty = isinstance(b, str) and not b.strip()
        if (_is_num(a) or a_empty) and (_is_num(b) or b_empty) and not (a_empty and b_empty):
            return _to_num(a) + _to_num(b)
        return _disp(a) + _disp(b)
    x, y = _to_num(a), _to_num(b)
    if op == "-":
        return x - y
    if op == "*":
        return x * y
    if op == "/":
        return (x / y) if y else 0
    return ""


def _apply_func(name: str, args: list[Any]) -> Any:
    name = name.lower()
    a0 = args[0] if args else ""
    if name == "split":
        sep = _disp(args[1]) if len(args) > 1 else "\n"
        return [p.strip() for p in _disp(a0).split(sep) if p.strip()]
    if name == "join":
        sep = _disp(args[1]) if len(args) > 1 else "\n"
        return sep.join(_disp(x) for x in _as_list(a0))
    if name == "len":
        return len(a0) if isinstance(a0, (list, str)) else len(_disp(a0))
    if name == "upper":
        return _disp(a0).upper()
    if name == "lower":
        return _disp(a0).lower()
    if name == "trim":
        return _disp(a0).strip()
    if name == "first":
        lst = _as_list(a0)
        return lst[0] if lst else ""
    if name == "last":
        lst = _as_list(a0)
        return lst[-1] if lst else ""
    if name == "append":
        return _as_list(a0) + [args[1] if len(args) > 1 else ""]
    if name == "int":
        return _to_num(a0)
    if name == "str":
        return _disp(a0)
    if name == "contains":
        hay = a0 if isinstance(a0, list) else _disp(a0)
        needle = args[1] if len(args) > 1 else ""
        return (needle in hay) if isinstance(hay, list) else (_disp(needle) in hay)
    return ""


def _eval_value(expr: str, ctx: dict[str, Any]) -> Any:
    """Evaluate a value expression. Returns a scalar or list (never raises —
    a parse problem yields the best-effort partial value or '')."""
    toks = _VAL_TOKEN_RE.findall(expr or "")
    pos = [0]

    def peek() -> Any:
        return toks[pos[0]] if pos[0] < len(toks) else None

    def nxt() -> Any:
        t = peek(); pos[0] += 1; return t

    def parse_expr() -> Any:
        v = parse_term()
        while peek() in ("+", "-"):
            v = _apply_arith(nxt(), v, parse_term())
        return v

    def parse_term() -> Any:
        v = parse_factor()
        while peek() in ("*", "/"):
            v = _apply_arith(nxt(), v, parse_factor())
        return v

    def parse_args() -> list[Any]:
        args: list[Any] = []
        if peek() not in (")", None):
            args.append(parse_expr())
            while peek() == ",":
                nxt(); args.append(parse_expr())
        return args

    def parse_factor() -> Any:
        t = nxt()
        if t is None:
            return ""
        if t == "(":
            v = parse_expr()
            if peek() == ")":
                nxt()
            return v
        if t == "[":
            items: list[Any] = []
            if peek() not in ("]", None):
                items.append(parse_expr())
                while peek() == ",":
                    nxt(); items.append(parse_expr())
            if peek() == "]":
                nxt()
            return items
        if t.startswith("{{"):
            return _dig(ctx, t[2:-2].strip())
        if t[0] in "\"'":
            return _val_unescape(t[1:-1])
        if t[0].isdigit():
            return float(t) if "." in t else int(t)
        # identifier → function call if followed by '(', else a bare word string
        if peek() == "(":
            nxt()
            args = parse_args()
            if peek() == ")":
                nxt()
            return _apply_func(t, args)
        return t

    try:
        return parse_expr()
    except Exception:
        return ""


class FlowRunner:
    def __init__(
        self,
        store: FlowStore,
        *,
        get_agents: Callable[[], list[dict[str, Any]]],
        resolve_auth: Callable[[int], str],
        fd_self_base: str,
        fd_tools: dict[str, Callable[[dict[str, Any]], Awaitable[str]]] | None = None,
        whatsapp_send: Callable[[str, str], Awaitable[Any]] | None = None,
        transfer_file: Callable[[str, int, str], Awaitable[tuple[list[str], list[str]]]] | None = None,
        load_archetype: Callable[[dict[str, Any], str], Awaitable[dict[str, Any] | None]] | None = None,
        spawn_archetype: Callable[..., Awaitable[tuple[int, str, str]]] | None = None,
        stop_archetype: Callable[[str], Awaitable[None]] | None = None,
        resolve_tier_cfg: Callable[[dict[str, Any], str, str], Awaitable[dict[str, Any]]] | None = None,
    ) -> None:
        self.store = store
        self.get_agents = get_agents
        self.resolve_auth = resolve_auth
        self.fd_self_base = fd_self_base.rstrip("/")
        self.fd_tools = fd_tools or {}        # FD-internal tools (e.g. face_identify)
        self.whatsapp_send = whatsapp_send
        # Uploads a file to a target agent, returning (image_paths, file_paths)
        # ON THE TARGET. Lets the runner verify delivery + use the target's path.
        self.transfer_file = transfer_file
        # ── archetype selector seams (`agent on archetype:<id>`) ──
        # All optional: if any is None the selector is unavailable and a step
        # using it fails cleanly with a clear message (never a crash). Injected
        # by the FD server, which owns the registry + spawn path.
        #   load_archetype(payload, arch_id) -> archetype dict | None
        #     (resolves the owning user from the payload, merges user archetypes)
        #   spawn_archetype(archetype, tier, tcfg, payload) -> (port, token, slug)
        #   stop_archetype(slug) -> None
        #   resolve_tier_cfg(payload, arch_id, tier) -> tier-config dict ({} ok)
        self.load_archetype = load_archetype
        self.spawn_archetype = spawn_archetype
        self.stop_archetype = stop_archetype
        self.resolve_tier_cfg = resolve_tier_cfg

    # ── agent pool selection ───────────────────────────────────────────

    def _select_agent(self, selector: str, payload: dict[str, Any]) -> dict[str, Any] | None:
        sel = (selector or "origin").strip()
        agents = [a for a in self.get_agents() if str(a.get("status", "")).lower() in ("running", "")]
        if sel == "origin" or not sel:
            host = payload.get("origin_host") or "localhost"
            port = payload.get("origin_port")
            name = str(payload.get("origin_name") or "").strip()
            # Resolve to the LIVE registry entry so we get the current port AND
            # the matching auth token. Prefer name (the origin may have drifted
            # ports since the message arrived), then port.
            if name:
                for a in agents:
                    if str(a.get("name", "")).lower() == name.lower():
                        return a
            if port:
                for a in agents:
                    if int(a.get("port") or 0) == int(port):
                        return a
                # Fallback: hand-built target with a resolved token.
                return {"name": name or "origin", "host": host, "port": int(port),
                        "auth": self.resolve_auth(int(port)) if self.resolve_auth else ""}
            return agents[0] if agents else None
        if sel.startswith("name:"):
            want = sel.split(":", 1)[1].strip().lower()
            for a in agents:
                if str(a.get("name", "")).lower() == want:
                    return a
            return None
        if sel.startswith("capability:"):
            cap = sel.split(":", 1)[1].strip().lower()
            hints = _VISION_HINTS if cap in ("vision", "image", "multimodal") else (cap,)
            for a in agents:
                blob = f"{a.get('name','')} {a.get('description','')}".lower()
                if any(h in blob for h in hints):
                    return a
            return None
        # "any"
        return agents[0] if agents else None

    # ── ephemeral archetype agents (`on archetype:<id>[@tier]`) ─────────

    @staticmethod
    def _parse_archetype_selector(selector: str) -> tuple[str, str]:
        """`archetype:fact-checker@reason` → ('fact-checker', 'reason'); the tier
        is optional and defaults to '' (the archetype's own tier resolves)."""
        rest = selector.split(":", 1)[1].strip()
        if "@" in rest:
            aid, _, tier = rest.partition("@")
            return aid.strip(), tier.strip()
        return rest, ""

    async def _ensure_archetype_agent(
        self, selector: str, root: "_Root", payload: dict[str, Any],
    ) -> tuple[dict[str, Any] | None, str]:
        """Resolve `archetype:<id>[@tier]` to a spawned agent dict, spawning once
        per run and caching on `root`. Returns (agent_dict | None, error). The
        agent dict carries the usual {name,host,port,auth} plus `fleet_instructions`
        (the archetype's SOP, prepended to the step prompt since the spawn config
        sets tools+mode but not the system prompt)."""
        if not (self.load_archetype and self.spawn_archetype):
            return None, ("archetype selectors are not available in this deployment "
                          "(no spawn seam configured)")
        aid, tier = self._parse_archetype_selector(selector)
        if not aid:
            return None, "archetype selector needs an id, e.g. archetype:fact-checker"
        cache_key = f"{aid}@{tier}"
        # Serialise lazy spawns so two concurrent steps on the same archetype
        # don't each spawn an agent (the second would wastefully orphan one).
        async with root.arch_lock:
            cached = root.arch_agents.get(cache_key)
            if cached is not None:
                return cached, ""
            try:
                arch = await self.load_archetype(payload, aid)
            except Exception as exc:  # noqa: BLE001
                return None, f"archetype lookup failed: {exc}"
            if not arch:
                return None, f"no archetype '{aid}' (check the id / your archetype library)"
            tcfg: dict[str, Any] = {}
            if tier and self.resolve_tier_cfg:
                try:
                    tcfg = await self.resolve_tier_cfg(payload, aid, tier) or {}
                except Exception as exc:  # noqa: BLE001
                    return None, f"tier '{tier}' could not be resolved: {exc}"
            try:
                port, token, slug = await self.spawn_archetype(arch, tier, tcfg, payload)
            except Exception as exc:  # noqa: BLE001
                return None, f"could not spawn archetype '{aid}': {exc}"
            # Track the slug for disposal BEFORE the readiness wait, so a spawn
            # that comes up too slowly is still cleaned up by run()'s finally.
            if slug:
                root.arch_slugs.append(slug)
            # `spawn_archetype` returns after only a short settle (~0.3s), but a
            # fresh agent's HTTP server takes seconds to boot — dispatching now
            # would hit connection-refused. Wait until it actually serves before
            # handing the agent to the step (consult-peer handles busy/retry from
            # there, but not the initial not-listening-yet window).
            ready = await self._wait_agent_ready("localhost", int(port), token or "")
            if not ready:
                return None, (f"archetype '{aid}' spawned (port {port}) but did not become "
                              f"reachable in time — it may be slow to boot or failed to start")
            agent = {
                "name": f"archetype:{aid}",
                "host": "localhost",
                "port": int(port),
                "auth": token or "",
                "fleet_instructions": str(arch.get("fleet_instructions") or ""),
            }
            root.arch_agents[cache_key] = agent
            log.info("flow spawned ephemeral archetype", archetype=aid, tier=tier or "(default)", slug=slug)
            return agent, ""

    async def _wait_agent_ready(self, host: str, port: int, token: str,
                                timeout: float | None = None) -> bool:
        """Poll a freshly spawned agent until its HTTP app serves (any non-5xx
        response on a cheap endpoint) or `timeout` elapses. A FastAPI app mounts
        all routes at once, so a served `/api/files` means `/ws` (the dispatch
        surface) is up too. Connection-refused → keep waiting; the agent is still
        booting (model init, session create)."""
        import httpx
        if timeout is None:
            try:
                timeout = float(os.environ.get("FD_FLOW_ARCH_READY_S", "60"))
            except (TypeError, ValueError):
                timeout = 60.0
        url = f"http://{host}:{port}/api/files" + (f"?token={token}" if token else "")
        loop = asyncio.get_event_loop()
        deadline = loop.time() + max(1.0, timeout)
        delay = 0.4
        async with httpx.AsyncClient(timeout=5.0) as client:
            while loop.time() < deadline:
                try:
                    resp = await client.get(url)
                    if resp.status_code < 500:
                        return True
                except Exception:  # noqa: BLE001 — connection refused while booting
                    pass
                await asyncio.sleep(delay)
                delay = min(1.5, delay * 1.3)
        return False

    async def _dispose_archetypes(self, root: "_Root") -> None:
        """Stop every archetype agent spawned during the run (best-effort)."""
        if not (self.stop_archetype and root.arch_slugs):
            return
        for slug in root.arch_slugs:
            try:
                await self.stop_archetype(slug)
            except Exception as exc:  # noqa: BLE001
                log.warning("flow archetype dispose failed", slug=slug, error=str(exc))
        root.arch_slugs.clear()
        root.arch_agents.clear()

    async def _resolve_step_agent(
        self, selector: str, root: "_Root", payload: dict[str, Any],
    ) -> tuple[dict[str, Any] | None, str]:
        """Resolve a step's `on` selector to a target agent dict. For
        `archetype:<id>` this spawns (or reuses) an ephemeral archetype agent;
        every other selector resolves synchronously against the live pool."""
        sel = (selector or "").strip()
        if sel.startswith("archetype:"):
            return await self._ensure_archetype_agent(sel, root, payload)
        return self._select_agent(sel, payload), ""

    # ── step dispatch ──────────────────────────────────────────────────

    async def _run_tool(self, step: dict[str, Any], ctx: dict[str, Any], payload: dict[str, Any], root: "_Root") -> tuple[str, str]:
        """Return (output, agent_label)."""
        tool = str(step.get("tool") or "")
        args = _render(step.get("args") or {}, ctx)
        selector = str(step.get("on") or "origin")

        # FD-internal tool (e.g. face_identify) — runs in-process, no agent.
        if selector == "fd" or tool in self.fd_tools:
            fn = self.fd_tools.get(tool)
            if not fn:
                return f"(no FD-internal tool '{tool}')", "fd"
            return (await fn(args)), "fd"

        agent, aerr = await self._resolve_step_agent(selector, root, payload)
        if not agent:
            return f"(no agent available for selector '{selector}'{': ' + aerr if aerr else ''})", ""
        import httpx
        url = f"http://{agent['host']}:{agent['port']}/api/tool"
        token = self.resolve_auth(int(agent["port"]))
        try:
            async with httpx.AsyncClient(timeout=600.0) as client:
                resp = await client.post(url, json={"tool": tool, "args": args, "token": token})
            if resp.status_code != 200:
                return f"(tool {tool} failed: HTTP {resp.status_code} {resp.text[:200]})", agent.get("name", "")
            data = resp.json() or {}
            out = data.get("content") if data.get("success") else f"(error: {data.get('error')})"
            return str(out or ""), agent.get("name", "")
        except Exception as exc:
            return f"(tool {tool} dispatch failed: {exc})", agent.get("name", "")

    async def _run_agent_step(self, step: dict[str, Any], ctx: dict[str, Any], payload: dict[str, Any], root: "_Root") -> tuple[str, str]:
        guard = step.get("guardrails") or {}
        deny = guard.get("deny") or []
        attach = _render(str(step.get("attach") or ""), ctx)
        selector = str(step.get("on") or "capability:vision")
        agent, aerr = await self._resolve_step_agent(selector, root, payload)
        if not agent:
            return f"(no agent for selector '{selector}'{': ' + aerr if aerr else ''})", ""

        # When a file/image is attached, UPLOAD it to the target FIRST and use
        # the TARGET-local path. `attach`/{{trigger.image_path}} is the ORIGIN
        # agent's path (where the message arrived) — the target can't read that
        # path, so we transfer the bytes and reference the copy on the target.
        # Verify it actually landed, then render the prompt with the TARGET path.
        target_images: list[str] = []
        target_files: list[str] = []
        render_ctx = ctx
        if attach:
            if not self.transfer_file:
                return "(cannot attach: file transfer unavailable)", agent.get("name", "")
            try:
                target_images, target_files = await self.transfer_file(
                    agent["host"], int(agent["port"]), attach,
                )
            except Exception as exc:
                return f"(failed to send the attachment to {agent.get('name','')}: {exc})", agent.get("name", "")
            if not target_images and not target_files:
                return (f"(attachment was NOT received by {agent.get('name','')} — "
                        f"upload failed or source file missing: {attach})"), agent.get("name", "")
            # Make {{trigger.image_path}} / {{attached_image_path}} resolve to the
            # TARGET copy for THIS step's prompt (origin path is meaningless there).
            _tpath = target_images[0] if target_images else target_files[0]
            render_ctx = dict(ctx)
            render_ctx["trigger"] = {**(ctx.get("trigger") or {}), "image_path": _tpath}
            render_ctx["attached_image_path"] = _tpath

        prompt = _render(str(step.get("prompt") or ""), render_ctx)

        # Ephemeral archetype agents are spawned with the archetype's tools +
        # cognitive_mode but NOT its SOP (the spawn config has no system-prompt
        # slot), so fold the archetype's `fleet_instructions` in as a preamble —
        # this is what makes `on archetype:fact-checker` actually behave like a
        # fact-checker rather than a generic agent.
        _fleet = str(agent.get("fleet_instructions") or "")
        if _fleet:
            prompt = f"{_fleet}\n\n---\n\n{prompt}"

        # FD-only mitigation (works even if the target runs older code): when a
        # file/image is attached, prepend a hard instruction to ignore memory and
        # describe ONLY the attachment. The deterministic version (suppressing the
        # target's memory injection) needs the target on current code.
        if attach:
            prompt = (
                "A file/image is attached to THIS message. Answer using ONLY what you "
                "actually see in the attachment. IGNORE everything else in your context "
                "— do NOT use 'Persistent insights from memory', remembered facts, fleet "
                "ports, or earlier descriptions. Describe only the attached content.\n\n"
                + prompt
            )

        # Deterministic tool denials for the target. Start with the step's own
        # deny list; when a file/image is attached, also block shell/scripts/read
        # so the model uses the ATTACHED content instead of operating on the path.
        _deny = list(deny)
        if attach:
            for t in ("shell", "scripts", "read"):
                if t not in _deny:
                    _deny.append(t)
        if deny:
            prompt = (
                f"Constraints: do NOT use these tools this turn: {', '.join(deny)}. "
                f"Do not write or run scripts. Answer directly.\n\n{prompt}"
            )
        import httpx
        body = {
            "host": agent["host"], "port": int(agent["port"]),
            "auth": str(agent.get("auth") or ""),  # token from the same entry as the port
            "message": prompt, "source_name": "FlowEngine", "timeout": 480.0,
            "no_flow": True,         # loop guard
            "no_broadcast": True,    # FlowRunner is the sole deliverer (no channel leak)
            "deny_tools": _deny,
            "image_paths": target_images,   # already on the TARGET (verified)
            "file_paths": target_files,
        }
        # Progress breadcrumb: name the specialist working this step so a long /
        # multi-stage flow isn't a silent "thinking" spinner in the origin UI.
        who = self._who(agent)
        await self._push_progress(payload, "narration", {"text": f"▶ {who} working…"})

        final, err = "", ""
        _last_note = ""  # de-dupe identical consecutive progress lines
        try:
            async with httpx.AsyncClient(timeout=600.0) as client:
                async with client.stream("POST", f"{self.fd_self_base}/fd/consult-peer", json=body) as resp:
                    if resp.status_code != 200:
                        return f"(agent step failed: HTTP {resp.status_code})", agent.get("name", "")
                    async for line in resp.aiter_lines():
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            evt = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        if evt.get("done") and evt.get("ok"):
                            final = str(evt.get("response") or "")
                            # Surface the peer's LLM token usage as an activity
                            # entry, attributed to the specialist (no clobbering of
                            # the origin agent's own context meter).
                            _usage = evt.get("usage")
                            if isinstance(_usage, dict):
                                _udet = self._usage_detail(_usage)
                                if _udet:
                                    await self._push_progress(payload, "monitor", {
                                        "tool_name": f"{who}:llm",
                                        "arguments": {},
                                        "output": _udet,
                                    })
                            break
                        if evt.get("ok") is False:
                            err = str(evt.get("error") or "agent error")
                            break
                        # Forward the peer's intermediate activity to the origin UI
                        # as TRANSIENT events (no chat-bubble spam): its narration /
                        # status update the activity line, its tool calls land in the
                        # monitor panel. The final answer still returns below.
                        ev = str(evt.get("event") or "")
                        data = evt.get("data") or {}
                        if ev in ("narration", "status", "thinking"):
                            note = str(data.get("text") or data.get("status") or "").strip()
                            if note and note != _last_note:
                                _last_note = note
                                await self._push_progress(payload, "thinking", {"text": f"{who}: {note[:200]}"})
                        elif ev == "monitor":
                            tool = str(data.get("tool_name") or data.get("tool") or "").strip()
                            if tool:
                                await self._push_progress(payload, "monitor", {
                                    "tool_name": f"{who}:{tool}",
                                    "arguments": data.get("arguments") or {},
                                    "output": str(data.get("output") or "")[:400],
                                })
        except Exception as exc:
            err = str(exc)
        return (final or f"(no result: {err})"), agent.get("name", "")

    async def _run_vision_step(self, step: dict[str, Any], ctx: dict[str, Any], payload: dict[str, Any], root: "_Root") -> tuple[str, str]:
        """Lean image describe: upload the image to a vision agent and call its
        /api/vision (raw model call) — NO agent loop, memory, tools, or history."""
        image_src = _render(str(step.get("attach") or step.get("image") or "{{trigger.image_path}}"), ctx)
        prompt = _render(str(step.get("prompt") or "Describe this image in detail."), ctx)
        selector = str(step.get("on") or "capability:vision")
        agent, aerr = await self._resolve_step_agent(selector, root, payload)
        if not agent:
            return f"(no vision agent for selector '{selector}'{': ' + aerr if aerr else ''})", ""
        if not image_src:
            return "(no image to describe)", agent.get("name", "")
        if not self.transfer_file:
            return "(file transfer unavailable)", agent.get("name", "")
        # Upload to the target and verify it landed; use the TARGET-local path.
        try:
            imgs, files = await self.transfer_file(agent["host"], int(agent["port"]), image_src)
        except Exception as exc:
            return f"(failed to send image to {agent.get('name','')}: {exc})", agent.get("name", "")
        target = imgs[0] if imgs else (files[0] if files else "")
        if not target:
            return f"(image NOT received by {agent.get('name','')}: {image_src})", agent.get("name", "")
        import httpx
        url = f"http://{agent['host']}:{int(agent['port'])}/api/vision"
        token = str(agent.get("auth") or "")
        try:
            async with httpx.AsyncClient(timeout=600.0) as client:
                resp = await client.post(
                    url, params=({"token": token} if token else {}),
                    json={"image": target, "prompt": prompt},
                )
            if resp.status_code != 200:
                return f"(vision failed: HTTP {resp.status_code} {resp.text[:200]})", agent.get("name", "")
            data = resp.json() or {}
            out = data.get("content") if data.get("success") else f"(vision error: {data.get('error')})"
            return str(out or "(empty vision result)"), agent.get("name", "")
        except Exception as exc:
            return f"(vision dispatch failed: {exc})", agent.get("name", "")

    async def _deliver(self, payload: dict[str, Any], text: str, *, role: str = "assistant") -> bool:
        """Send a message to the user on the originating channel.

        WhatsApp → whatsapp_send. Agent-handled channels (web/glasses) → push
        into the origin agent's chat UI via /api/chat/push (the flow runs in FD,
        so it can't rely on the agent relaying inside a blocking turn)."""
        text = str(text or "")
        if not text:
            return False
        waid = str(payload.get("waid") or payload.get("whatsapp_waid") or "")
        if waid and self.whatsapp_send:
            try:
                await self.whatsapp_send(waid, text)
                return True
            except Exception as exc:
                log.warning("flow deliver (whatsapp) failed: %s", exc)
                return False
        agent = self._select_agent("origin", payload)
        if not agent:
            return False
        import httpx
        url = f"http://{agent['host']}:{agent['port']}/api/chat/push"
        token = agent.get("auth") or (self.resolve_auth(int(agent["port"])) if self.resolve_auth else "")
        params = {"token": token} if token else {}
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.post(url, params=params, json={"text": text, "role": role})
            return resp.status_code == 200
        except Exception as exc:
            log.warning("flow deliver (agent push) failed: %s", exc)
            return False

    async def _push_progress(self, payload: dict[str, Any], kind: str, fields: dict[str, Any]) -> None:
        """Push a live-progress event into the origin agent's chat UI so the user
        sees what a flow step (or its spawned specialist) is doing in real time.

        `kind` is a transient UI event the agent already renders: `thinking`/
        `status` update the activity line the user is watching, `monitor` lands in
        the tool panel, `narration` is a one-line breadcrumb bubble. Skipped on
        WhatsApp (per-event progress would spam a messaging channel) and when there
        is no origin agent (scheduled runs). Best-effort — never raises."""
        if str(payload.get("waid") or payload.get("whatsapp_waid") or ""):
            return
        if not (payload.get("origin_port") or payload.get("origin_name")):
            return  # no origin chat UI to render into (e.g. scheduler runs)
        agent = self._select_agent("origin", payload)
        if not agent:
            return
        import httpx
        url = f"http://{agent['host']}:{agent['port']}/api/chat/push"
        token = agent.get("auth") or (self.resolve_auth(int(agent["port"])) if self.resolve_auth else "")
        params = {"token": token} if token else {}
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                await client.post(url, params=params, json={"kind": kind, "event": fields})
        except Exception:
            pass  # progress is advisory; never fail a step over it

    @staticmethod
    def _who(agent: dict[str, Any]) -> str:
        """Friendly label for progress lines — strips the `archetype:` prefix."""
        name = str(agent.get("name") or "agent")
        return name.split(":", 1)[1] if name.startswith("archetype:") else name

    @staticmethod
    def _usage_detail(usage: dict[str, Any]) -> str:
        """Render a peer's end-of-turn LLM usage as one line: 'model · in→out tok ·
        N total'. Tolerates the agent's field shape (input_tokens|prompt_tokens)."""
        last = usage.get("last") if isinstance(usage.get("last"), dict) else usage
        model = str(last.get("model") or usage.get("model") or "").strip()
        it = last.get("input_tokens")
        if it is None:
            it = last.get("prompt_tokens")
        ot = last.get("output_tokens")
        if ot is None:
            ot = last.get("completion_tokens")
        tot = None
        total = usage.get("total")
        if isinstance(total, dict):
            tot = total.get("total_tokens") or total.get("total")
        parts: list[str] = []
        if model:
            parts.append(model)
        if it is not None or ot is not None:
            parts.append(f"{int(it or 0)}→{int(ot or 0)} tok")
        if tot:
            parts.append(f"{int(tot)} total")
        return " · ".join(parts)

    async def _run_input_step(
        self, step: dict[str, Any], ctx: dict[str, Any],
        payload: dict[str, Any], flow: dict[str, Any], *, dry: bool = False,
    ) -> tuple[str, str]:
        """Pause the run, prompt the user (naming the flow), and resume with
        their reply as this step's output. Returns (user_text, "")."""
        prompt = _render(str(step.get("prompt") or "Please reply with your input."), ctx)
        flow_name = str(flow.get("name") or "Flow")

        if dry:
            return f"(dry-run: would pause and ask: {prompt})", ""

        waid = str(payload.get("waid") or payload.get("whatsapp_waid") or "")
        channel = str(payload.get("channel") or "")
        origin_port = int(payload.get("origin_port") or 0)
        # Announce that input is needed — the flow name + handle let the user know
        # which automation is asking (and how to address it: '/flow pause <handle>').
        handle = str((ctx.get("system") or {}).get("handle") or "")
        tag = f" `[{handle}]`" if handle else ""
        announce = f"⏳ *{flow_name}*{tag} needs your input:\n\n{prompt}"

        if not await self._deliver(payload, announce):
            # No reachable channel to ask on — fail clearly instead of hanging.
            return "(input step: no channel available to prompt the user)", ""

        from captain_claw.flight_deck import flow_router
        timeout = float(step.get("timeout") or 3600.0)
        key = flow_router.input_key(waid=waid, channel=channel, origin_port=origin_port)
        # Remember the prompt so '/flow resume' can re-show the question.
        flow_router.set_input_prompt(key, announce)
        try:
            text = await flow_router.wait_for_input(key, timeout=timeout)
        except asyncio.TimeoutError as exc:
            raise RuntimeError(f"timed out waiting for input ({flow_name})") from exc
        except asyncio.CancelledError as exc:
            raise RuntimeError(f"input wait superseded ({flow_name})") from exc
        finally:
            flow_router.clear_input_prompt(key)
        return text, ""

    async def _emit(self, step: dict[str, Any], ctx: dict[str, Any], payload: dict[str, Any]) -> str:
        channel = str(step.get("channel") or "log")
        body = _render(str(step.get("body") or "{{steps}}"), ctx)
        # 'same'/'whatsapp'/'glasses'/'web' all deliver on the originating
        # channel (whatsapp_send or agent chat-push); 'log' just records.
        if channel in ("whatsapp", "same", "glasses", "web"):
            if await self._deliver(payload, body):
                return f"(emitted to {channel})"
        return body  # 'log' / fallthrough — captured in the run record

    # ── flow resolution (for gosub) ────────────────────────────────────

    async def _resolve_flow_by_name(self, name: str) -> dict[str, Any] | None:
        """Resolve a flow by name for `gosub` (Phase 1: permanent space only)."""
        name_l = name.strip().lower()
        if not name_l:
            return None
        try:
            getter = getattr(self.store, "get_flow_by_name", None)
            if getter:
                return await getter(name)
            for f in await self.store.list_flows():
                if str(f.get("name", "")).lower() == name_l:
                    return f
        except Exception as exc:
            log.warning("gosub resolve failed: %s", exc)
        return None

    async def _run_gosub(
        self, step: dict[str, Any], ctx: dict[str, Any],
        payload: dict[str, Any], root: "_Root", depth: int, caller_origin: str = "user",
    ) -> tuple[str, str, str]:
        """Synchronously call another flow. Returns (return_value, label, status)."""
        name = _render(str(step.get("flow") or ""), ctx).strip()
        if not name:
            return "(gosub: no flow name)", "gosub", "error"
        # Depth cap is a hard runaway guard — it raises (fails the run) rather than
        # returning a catchable status, so infinite recursion can't be swallowed.
        if depth + 1 > root.depth_cap:
            raise RuntimeError(f"max flow recursion depth ({root.depth_cap}) exceeded")
        target = await self._resolve_flow_by_name(name)
        if not target:
            return f"(gosub: no flow named '{name}')", "gosub", "error"
        blocked = self._guard_cross_space(caller_origin, target, name, "gosub")
        if blocked:
            return blocked, "gosub", "error"
        args = _render(step.get("args") or {}, ctx)
        if not isinstance(args, dict):
            args = {}
        # The child carries the parent's identity (so delivery + input keying still
        # resolve) plus the call args. Args merge into the trigger so a reused flow
        # that reads {{trigger.<k>}} works, and {{args.<k>}} is always available.
        child_payload = dict(payload)
        child_payload["args"] = args
        for k, v in args.items():
            child_payload[k] = v
        result = await self._run_frame(target, child_payload, root=root, depth=depth + 1)
        st = str(result.get("status") or "done")
        return str(result.get("value", "")), f"flow:{name}", ("done" if st in ("done", "returned") else st)

    def _guard_cross_space(self, caller_origin: str, target: dict[str, Any], name: str, verb: str) -> str:
        """Return a block message if a synthesized (agent) flow tries to call a
        permanent world-acting flow, else ''."""
        if (caller_origin == "agent"
                and str(target.get("space") or "user") == "user"
                and _is_world_acting(target)):
            return (f"({verb} blocked: a synthesized flow may not call the world-acting "
                    f"flow '{name}' without approval — promote it first)")
        return ""

    async def _run_spawn(
        self, step: dict[str, Any], ctx: dict[str, Any], payload: dict[str, Any],
        caller_origin: str = "user",
    ) -> tuple[str, str, str]:
        """Launch another flow as an INDEPENDENT background root run and stash its
        task as a future. Returns immediately; the parent continues. Returns
        (note, label, status) — status 'error' only if the flow can't be found."""
        name = _render(str(step.get("flow") or ""), ctx).strip()
        if not name:
            return "(spawn: no flow name)", "spawn", "error"
        target = await self._resolve_flow_by_name(name)
        if not target:
            return f"(spawn: no flow named '{name}')", "spawn", "error"
        blocked = self._guard_cross_space(caller_origin, target, name, "spawn")
        if blocked:
            return blocked, "spawn", "error"
        args = _render(step.get("args") or {}, ctx)
        if not isinstance(args, dict):
            args = {}
        child_payload = dict(payload)
        child_payload["args"] = args
        for k, v in args.items():
            child_payload[k] = v
        # New, independent root run (its own run_id + control handle): not killed
        # by the parent's `flow stop`, reachable by `flow stop all`, joinable.
        task = asyncio.create_task(self.run(target, child_payload))
        ctx.setdefault("_spawns", {})[str(step.get("id"))] = task
        return f"(spawned '{name}')", f"flow:{name}", "done"

    async def _run_join(
        self, step: dict[str, Any], ctx: dict[str, Any], root: "_Root",
    ) -> tuple[str, str, str]:
        """Wait for a spawned future and collect its result. Returns
        (output, label, status) where status is done/error/stopped/timeout.
        Honors a stop on THIS run by aborting the wait (the spawned run, being
        independent, keeps going unless separately stopped)."""
        jid = _render(str(step.get("join") or ""), ctx).strip()
        task = (ctx.get("_spawns") or {}).get(jid)
        if task is None:
            return f"(join: no spawn '{jid}' in this flow)", f"join:{jid}", "error"
        _to = step.get("timeout")
        timeout = float(_to) if _to is not None else _DEFAULT_JOIN_TIMEOUT
        poll = min(0.2, timeout) if timeout > 0 else 0.05
        ctrl = root.control
        loop = asyncio.get_event_loop()
        deadline = loop.time() + timeout
        while True:
            if ctrl is not None and ctrl.stopped:
                raise _FlowStopped()  # unwind THIS run; spawned task survives
            try:
                # shield: a join timeout/stop must not cancel the spawned run.
                result = await asyncio.wait_for(asyncio.shield(task), timeout=poll)
                st = str((result or {}).get("status") or "done")
                out = str((result or {}).get("output") or "")
                return out, f"join:{jid}", ("done" if st in ("done", "returned") else st)
            except asyncio.TimeoutError:
                if loop.time() >= deadline:
                    return f"(join '{jid}' timed out after {timeout:g}s)", f"join:{jid}", "timeout"
                continue
            except Exception as exc:
                return f"(join '{jid}' failed: {exc})", f"join:{jid}", "error"

    async def _run_foreach(
        self, step: dict[str, Any], ctx: dict[str, Any], payload: dict[str, Any],
        root: "_Root", depth: int, caller_origin: str,
    ) -> tuple[Any, str, str]:
        """Run a flow once per item of a list. mode 'gosub' = sequential, 'spawn'
        = parallel map. Returns (list-of-results, label, status). The loop
        variable (`var`, default `item`) is bound for each iteration."""
        var = str(step.get("var") or "item")
        items = _as_list(_eval_value(str(step.get("in") or ""), ctx))
        fname = str(step.get("flow") or "")
        mode = str(step.get("mode") or "gosub")
        if not fname:
            return [], "foreach", "error"
        results: list[Any] = []
        if mode == "spawn":
            tasks: list[Any] = []
            for it in items:
                ctx[var] = it
                args = _render(step.get("args") or {}, ctx)
                if not isinstance(args, dict):
                    args = {}
                target = await self._resolve_flow_by_name(_render(fname, ctx))
                if not target or self._guard_cross_space(caller_origin, target, fname, "foreach"):
                    tasks.append(None)
                    continue
                cp = dict(payload)
                cp["args"] = args
                for k, v in args.items():
                    cp[k] = v
                tasks.append(asyncio.create_task(self.run(target, cp)))
            timeout = float(step.get("timeout") or _DEFAULT_JOIN_TIMEOUT)
            for tk in tasks:
                if tk is None:
                    results.append("")
                    continue
                try:
                    r = await asyncio.wait_for(asyncio.shield(tk), timeout=timeout)
                    results.append(str((r or {}).get("output") or ""))
                except Exception:
                    results.append("")
        else:
            for it in items:
                ctx[var] = it
                sub = {"flow": fname, "args": step.get("args") or {}}
                o, _a, _st = await self._run_gosub(sub, ctx, payload, root, depth, caller_origin)
                results.append(o)
        ctx.pop(var, None)
        return results, "foreach", "done"

    async def _interruptible_sleep(self, secs: float, ctrl: "_RunControl | None") -> None:
        """Sleep, but wake immediately if the run is stopped."""
        loop = asyncio.get_event_loop()
        deadline = loop.time() + max(0.0, secs)
        while True:
            if ctrl is not None and ctrl.stopped:
                raise _FlowStopped()
            rem = deadline - loop.time()
            if rem <= 0:
                return
            await asyncio.sleep(min(0.5, rem))

    async def _run_wait_step(
        self, step: dict[str, Any], ctx: dict[str, Any],
        payload: dict[str, Any], flow: dict[str, Any], *, dry: bool = False,
    ) -> tuple[str, str]:
        """Park until an inbound message satisfies the `until` condition. The
        matching message's text becomes this step's output; non-matching messages
        fall through to the agent (the flow stays paused)."""
        cond = str(step.get("until") or "").strip()
        if not cond:
            return "(wait: no condition)", ""
        if dry:
            return f"(dry-run: would wait until {cond})", ""
        # shorthand: "contains X" / "matches X" → test against {{trigger.text}}
        low = cond.lower()
        if (low.startswith("contains ") or low.startswith("matches ")) and "{{" not in cond:
            cond = "{{trigger.text}} " + cond
        waid = str(payload.get("waid") or payload.get("whatsapp_waid") or "")
        channel = str(payload.get("channel") or "")
        origin_port = int(payload.get("origin_port") or 0)
        from captain_claw.flight_deck import flow_router
        timeout = float(step.get("timeout") or 86400.0)
        key = flow_router.input_key(waid=waid, channel=channel, origin_port=origin_port)
        try:
            text = await flow_router.wait_for_match(key, cond, timeout=timeout)
        except asyncio.TimeoutError as exc:
            raise RuntimeError("timed out waiting for condition") from exc
        except asyncio.CancelledError as exc:
            raise RuntimeError("wait superseded") from exc
        return text, ""

    # ── frame loop (one flow's steps) ──────────────────────────────────

    async def _run_frame(
        self, flow: dict[str, Any], payload: dict[str, Any], *,
        root: "_Root", depth: int,
    ) -> dict[str, Any]:
        """Execute one flow's steps and return {value, status}. `gosub` recurses
        here at depth+1; the parent captures the return value. Frame-local ctx
        (steps/calls) is fresh; budget/trace/control are shared via `root`."""
        steps = list(flow.get("steps") or [])
        guard = flow.get("guardrails") or {}
        max_steps = int(guard.get("max_steps", _DEFAULT_MAX_STEPS))
        flow_name = str(flow.get("name") or "Flow")
        frame_origin = str(flow.get("origin") or "user")  # 'agent' = synthesized
        _now = datetime.now(UTC)
        ctx: dict[str, Any] = {
            "trigger": payload,
            "args": payload.get("args") or {},
            "steps": {},
            "calls": {},
            "vars": {},
            "system": {
                "now": _now.isoformat(),
                "date": _now.strftime("%Y-%m-%d"),
                "time": _now.strftime("%H:%M"),
                "agent": str(payload.get("origin_name") or ""),
                "channel": str(payload.get("channel") or ""),
                "handle": (root.control.handle if root.control else ""),
            },
        }
        ctrl = root.control
        run_id = root.run_id
        dry = root.dry
        frame: dict[str, Any] | None = None
        if ctrl is not None:
            frame = {"depth": depth, "flow": flow_name, "step": ""}
            ctrl.frames.append(frame)
        paused_marked = False
        by_id = {s.get("id"): i for i, s in enumerate(steps)}
        i = 0
        executed = 0
        frame_value = ""
        frame_status = "done"
        self_delivered = False  # did the last executed step already message the user?
        retry_used: dict[str, int] = {}  # per-step retry counters
        try:
            while i < len(steps):
                # ── pause / stop control (checked at each step boundary) ──
                if ctrl is not None:
                    if not ctrl.stopped and not ctrl.resume.is_set():
                        if not paused_marked:
                            paused_marked = True
                            try:
                                await self.store.set_run_status(run_id, "paused")
                            except Exception:
                                pass
                        await ctrl.resume.wait()
                        if not ctrl.stopped:
                            paused_marked = False
                            try:
                                await self.store.set_run_status(run_id, "running")
                            except Exception:
                                pass
                    if ctrl.stopped:
                        raise _FlowStopped()  # unwind the whole stack to run()
                # ── budgets ──
                if root.budget["steps_left"] <= 0:
                    raise RuntimeError("flow step budget exhausted")
                if executed >= max_steps:
                    raise RuntimeError(f"max_steps ({max_steps}) exceeded")
                step = steps[i]
                sid = str(step.get("id") or f"step{i}")
                stype = str(step.get("type") or "tool")
                executed += 1
                root.budget["steps_left"] -= 1
                if frame is not None:
                    frame["step"] = sid
                t0 = time.monotonic()

                if stype == "branch":
                    cases = step.get("cases")
                    if not cases:
                        cases = [{"when": step.get("when"), "goto": step.get("goto")}]
                    target = None
                    label = "not taken"
                    for c in cases:
                        if _eval_expr(str(c.get("when") or ""), ctx):
                            target = c.get("goto")
                            label = f"taken→{target}"
                            break
                    if target is None:
                        els = step.get("else") or step.get("default")
                        if els:
                            target = els
                            label = f"else→{target}"
                    out = f"branch {label}"
                    _record(root.trace, sid, stype, "done", "", out, t0, depth)
                    if not dry:
                        await self.store.add_step_result(run_id, sid, executed, type=stype, status="done", output_text=out, ms=_ms(t0), depth=depth, frame=flow_name)
                    if target and str(target) == _STOP_TARGET:
                        break  # branch ends the flow (no value)
                    if target and target in by_id:
                        i = by_id[target]
                        continue
                    i += 1
                    continue

                if stype == "while":
                    # Loop: while the condition holds, jump to the target (whose
                    # path loops back here); otherwise fall through.
                    taken = _eval_expr(str(step.get("when") or ""), ctx)
                    tgt = step.get("goto")
                    out = f"while {('→' + str(tgt)) if taken else 'exit'}"
                    _record(root.trace, sid, stype, "done", "", out, t0, depth)
                    if not dry:
                        await self.store.add_step_result(run_id, sid, executed, type=stype, status="done", output_text=out, ms=_ms(t0), depth=depth, frame=flow_name)
                    if taken and tgt and str(tgt) == _STOP_TARGET:
                        break
                    if taken and tgt and tgt in by_id:
                        i = by_id[tgt]
                        continue
                    i += 1
                    continue

                if stype == "return":
                    # Explicit exit. `return <expr>` returns the expr; bare `return`
                    # hands back the last step's output (frame_value so far).
                    val = step.get("value")
                    if val:
                        frame_value = _render(str(val), ctx)
                        self_delivered = False  # a returned value isn't auto-delivered
                    _record(root.trace, sid, "return", "done", "", frame_value, t0, depth)
                    if not dry:
                        await self.store.add_step_result(run_id, sid, executed, type="return", status="done", output_text=str(frame_value), ms=_ms(t0), depth=depth, frame=flow_name)
                    break

                call_status: str | None = None  # set by call steps (gosub/join/spawn)
                self_delivered = False
                if stype == "gosub":
                    out, agent, call_status = await self._run_gosub(step, ctx, payload, root, depth, frame_origin)
                    ctx["calls"][sid] = {"output": out, "status": call_status}
                elif stype == "spawn":
                    out, agent, call_status = await self._run_spawn(step, ctx, payload, frame_origin)
                    ctx.setdefault("spawns", {})[sid] = {"status": call_status}
                elif stype == "join":
                    out, agent, call_status = await self._run_join(step, ctx, root)
                    jid = _render(str(step.get("join") or ""), ctx).strip()
                    ctx.setdefault("joins", {})[jid] = {"output": out, "status": call_status}
                elif stype == "error":
                    # Handler step: report the error on the user channel (the
                    # message may reference {{error.message}}) and continue.
                    msg = _render(str(step.get("message") or ""), ctx)
                    if msg:
                        self_delivered = await self._deliver(payload, msg)
                    out, agent = (msg or "(error handler)"), ""
                elif stype == "set":
                    name = str(step.get("var") or "")
                    val = _eval_value(str(step.get("expr") or ""), ctx)
                    if name:
                        ctx.setdefault("vars", {})[name] = val
                    out, agent = val, ""
                elif stype == "foreach":
                    out, agent, call_status = await self._run_foreach(step, ctx, payload, root, depth, frame_origin)
                elif stype == "sleep":
                    secs = _parse_duration(step.get("duration"))
                    if not dry:
                        try:
                            await self.store.set_run_status(run_id, "sleeping")
                        except Exception:
                            pass
                        await self._interruptible_sleep(secs, ctrl)
                        try:
                            await self.store.set_run_status(run_id, "running")
                        except Exception:
                            pass
                    out, agent = f"(slept {secs:g}s)", ""
                elif stype == "wait":
                    out, agent = await self._run_wait_step(step, ctx, payload, flow, dry=dry)
                elif stype == "tool":
                    out, agent = await self._run_tool(step, ctx, payload, root)
                elif stype == "agent":
                    out, agent = await self._run_agent_step(step, ctx, payload, root)
                elif stype == "vision":
                    out, agent = await self._run_vision_step(step, ctx, payload, root)
                elif stype == "input":
                    out, agent = await self._run_input_step(step, ctx, payload, flow, dry=dry)
                elif stype == "emit":
                    out, agent = await self._emit(step, ctx, payload), ""
                else:
                    out, agent = f"(unknown step type '{stype}')", ""

                # An emit to a user channel already delivered — so the root must
                # not re-deliver this as the flow's final output (would leak the
                # "(emitted to …)" sentinel as a second message).
                if stype == "emit":
                    self_delivered = str(step.get("channel") or "log") in ("whatsapp", "same", "glasses", "web")

                ctx["steps"][sid] = {"output": out}
                _maybe_attach_fields(ctx["steps"][sid], out)
                rec_status = call_status if call_status in ("error", "stopped", "timeout") else "done"
                _disp_out = _disp(out)
                _record(root.trace, sid, stype, rec_status, agent, _disp_out, t0, depth)
                if not dry:
                    await self.store.add_step_result(
                        run_id, sid, executed, type=stype, status=rec_status,
                        agent=agent, output_text=_disp_out, ms=_ms(t0),
                        depth=depth, frame=flow_name,
                    )
                frame_value = _disp_out

                # Error routing: a failed call exposes {{error.*}}.
                if call_status in ("error", "stopped", "timeout"):
                    # `retry: N` — re-run the failing step up to N times before
                    # giving up (not for a stop, which is intentional).
                    rmax = int(step.get("retry") or 0)
                    if rmax and call_status != "stopped" and retry_used.get(sid, 0) < rmax:
                        retry_used[sid] = retry_used.get(sid, 0) + 1
                        continue  # re-execute the same step
                    ctx["error"] = {"message": _disp_out, "status": call_status, "step": sid}
                    # `on error -> <target>` jumps to a handler; else fall through
                    # (the author can still branch on {{calls|joins.<id>.status}}).
                    tgt = step.get("on_error")
                    if tgt:
                        if str(tgt) == _STOP_TARGET:
                            break
                        if tgt in by_id:
                            i = by_id[tgt]
                            continue
                elif call_status == "done":
                    retry_used.pop(sid, None)  # reset the retry budget on success

                i += 1
                # `return` directive (supersedes the bare `stop` flag): exit with
                # an optional value.
                if "return" in step:
                    rexpr = step.get("return")
                    if rexpr:
                        frame_value = _render(str(rexpr), ctx)
                        self_delivered = False  # returned value differs from what was emitted
                    break
                if step.get("stop"):
                    break  # "Stop after this step" → end the frame here
        finally:
            if ctrl is not None and frame is not None:
                try:
                    ctrl.frames.remove(frame)
                except ValueError:
                    pass
        return {"value": frame_value, "status": frame_status, "self_delivered": self_delivered}

    # ── run (root concerns: control, budget, delivery, persistence) ─────

    async def run(self, flow: dict[str, Any], payload: dict[str, Any] | None = None, *, dry: bool = False, run_id: str | None = None) -> dict[str, Any]:
        payload = payload or {}
        if dry:
            run_id = ""
        elif not run_id:
            run_id = await self.store.start_run(flow["id"], flow.get("name", ""), payload)
        # Register a control handle so the run can be paused/resumed/stopped.
        ctrl: _RunControl | None = None
        if not dry and run_id:
            _name = str(flow.get("name") or "")
            ctrl = _RunControl(owner=_owner_key(payload), name=_name, handle=_assign_handle(_name))
            _RUN_CONTROL[run_id] = ctrl
        guard = flow.get("guardrails") or {}
        budget = {"steps_left": int(guard.get("max_total_steps", _DEFAULT_MAX_TOTAL_STEPS))}
        depth_cap = int(guard.get("max_depth", _DEFAULT_MAX_DEPTH))
        root = _Root(run_id=run_id, control=ctrl, dry=dry, budget=budget, depth_cap=depth_cap)

        status = "done"
        error = ""
        final_text = ""
        try:
            result = await self._run_frame(flow, payload, root=root, depth=0)
            final_text = str(result.get("value") or "")
            # Deliver the root flow's output to the user channel (children never
            # do this — their value returns to the caller; only an explicit `emit`
            # in a child reaches the user). Skip if the last step was itself an
            # emit to a user channel — it already delivered.
            output = flow.get("output") or {}
            out_channel = str(output.get("channel") or "log")
            if not result.get("self_delivered") and not dry and out_channel in ("whatsapp", "same", "glasses", "web") and final_text:
                try:
                    await self._deliver(payload, str(final_text))
                except Exception as exc:
                    log.warning("flow output delivery failed: %s", exc)
        except _FlowStopped:
            status = "stopped"
            if ctrl is not None and ctrl.stop_message:
                try:
                    await self._deliver(payload, ctrl.stop_message)
                except Exception as exc:
                    log.warning("flow stop message delivery failed: %s", exc)
        except Exception as exc:
            # A stop requested while a frame was blocked off the control loop
            # (e.g. waiting on `input`) surfaces here as the cancelled wait —
            # treat it as a clean stop, not a failure.
            if ctrl is not None and ctrl.stopped:
                status = "stopped"
                if ctrl.stop_message:
                    try:
                        await self._deliver(payload, ctrl.stop_message)
                    except Exception as dexc:
                        log.warning("flow stop message delivery failed: %s", dexc)
            else:
                status = "error"
                error = str(exc)
                log.warning("flow run failed", flow=flow.get("name"), error=error)
        finally:
            # Always dispose ephemeral archetype agents spawned by this run,
            # whatever the outcome (done / stopped / error) — no orphaned agents.
            await self._dispose_archetypes(root)

        if not dry:
            await self.store.finish_run(run_id, status, error)
            _RUN_CONTROL.pop(run_id, None)
            # Lifecycle: record the outcome on a synthesized (scratch) flow so it
            # earns its way to promotion (or to quarantine). No-op otherwise.
            if str(flow.get("space") or "user") == "scratch" and flow.get("id"):
                rec = getattr(self.store, "record_outcome", None)
                if rec:
                    try:
                        await rec(flow["id"], status == "done")
                    except Exception as exc:
                        log.warning("scratch outcome record failed: %s", exc)
        return {"run_id": run_id, "status": status, "error": error, "steps": root.trace, "output": final_text}


# ── Branch condition evaluator ─────────────────────────────────────────
#
# A small, SAFE boolean expression evaluator (no eval()). Supports:
#   • logicals:    and / or / not   (also && / || / !)
#   • grouping:    ( ... )
#   • comparisons: ==  !=  >  <  >=  <=  contains  matches (~)
#   • operands:    {{path}} (resolved against ctx), "quoted", or bare words
#   • truthiness:  a lone operand is true unless it's empty/false/0/none/no
#
# {{...}} tokens are resolved at eval time (not string-substituted first), so a
# value containing spaces/quotes/operators can't corrupt the parse.

_FALSY = {"", "false", "0", "none", "no", "null"}

_TOKEN_RE = re.compile(
    r"""\{\{.*?\}\}            # {{var}}
      | "[^"]*" | '[^']*'      # quoted string
      | >= | <= | == | !=      # 2-char comparisons
      | && | \|\|              # logical and / or
      | [()<>~!]               # parens & single-char ops
      | [^\s()<>~!&|"'=]+       # bare operand / keyword
    """,
    re.X,
)

_CMP = {"==", "!=", ">", "<", ">=", "<=", "contains", "matches", "~"}


def _val_to_str(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, bool):
        return "true" if v else "false"
    return str(v)


def _resolve_operand(tok: str, ctx: dict[str, Any]) -> str:
    if tok.startswith("{{") and tok.endswith("}}"):
        return _val_to_str(_dig(ctx, tok[2:-2].strip()))
    if len(tok) >= 2 and tok[0] in "\"'" and tok[-1] == tok[0]:
        return tok[1:-1]
    return tok


def _apply_cmp(op: str, lhs: str, rhs: str) -> bool:
    if op == "==":
        return lhs == rhs
    if op == "!=":
        return lhs != rhs
    if op == "contains":
        return rhs.lower() in lhs.lower()
    if op in ("matches", "~"):
        try:
            return bool(re.search(rhs, lhs, re.I))
        except re.error:
            return False
    # Numeric comparisons, with lexicographic fallback.
    try:
        a: Any = float(lhs)
        b: Any = float(rhs)
    except ValueError:
        a, b = lhs, rhs
    if op == ">":
        return a > b
    if op == "<":
        return a < b
    if op == ">=":
        return a >= b
    if op == "<=":
        return a <= b
    return False


def _eval_expr(expr: str, ctx: dict[str, Any]) -> bool:
    """Evaluate a boolean branch condition against the run context. Safe: a
    hand-written recursive-descent parser, never eval(). Returns False on any
    parse/eval error so a malformed condition can't crash the run."""
    toks = [m.group(0) for m in _TOKEN_RE.finditer(expr or "")]
    if not toks:
        return False
    i = 0

    def low() -> str | None:
        return toks[i].lower() if i < len(toks) else None

    def parse_or() -> bool:
        nonlocal i
        v = parse_and()
        while low() in ("or", "||"):
            i += 1
            r = parse_and()
            v = v or r
        return v

    def parse_and() -> bool:
        nonlocal i
        v = parse_not()
        while low() in ("and", "&&"):
            i += 1
            r = parse_not()
            v = v and r
        return v

    def parse_not() -> bool:
        nonlocal i
        if low() in ("not", "!"):
            i += 1
            return not parse_not()
        return parse_atom()

    def parse_atom() -> bool:
        nonlocal i
        if i >= len(toks):
            return False
        if toks[i] == "(":
            i += 1
            v = parse_or()
            if i < len(toks) and toks[i] == ")":
                i += 1
            return v
        lhs = _resolve_operand(toks[i], ctx)
        i += 1
        op = low()
        if op in _CMP:
            i += 1
            rhs = _resolve_operand(toks[i], ctx) if i < len(toks) else ""
            if i < len(toks):
                i += 1
            return _apply_cmp(op, lhs, rhs)
        return lhs.strip().lower() not in _FALSY

    try:
        return bool(parse_or())
    except Exception:
        return False


def _maybe_attach_fields(slot: dict[str, Any], out: Any) -> None:
    """If a tool returned a JSON object string, expose its keys (e.g. .label)."""
    if not isinstance(out, str):
        return
    s = out.strip()
    if s.startswith("{") and s.endswith("}"):
        try:
            obj = json.loads(s)
            if isinstance(obj, dict):
                for k, v in obj.items():
                    slot.setdefault(k, v)
        except (ValueError, TypeError):
            pass


def _ms(t0: float) -> int:
    return int((time.monotonic() - t0) * 1000)


def _record(trace: list, sid: str, stype: str, status: str, agent: str, out: Any, t0: float, depth: int = 0) -> None:
    trace.append({"step_id": sid, "type": stype, "status": status, "agent": agent,
                  "output": str(out)[:4000], "ms": _ms(t0), "depth": depth})
