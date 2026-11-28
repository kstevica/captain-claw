"""Team up — an agent splits a task among helpers: copies of itself in this process.

docs/team-up-plan.md. In the agent's web chat, the agent (the *lead*) sees a
task with 2–4 independent parts and proposes a team. With the agent's setting
"Ask me" (the default) the person taps **Team up** on a card; with "On" the
team starts at once; "Off" offers nothing.

* **Helpers are forks of the lead**, built like a lane (``_build_scoped_agent``)
  in this same process: the same memory, tools, connections and model. Each
  starts from a copy of the lead's conversation in a child session of its own,
  so on a provider with prompt caching most of its input is cached and it
  knows everything the person said. Only its last message differs: its part.
* **The team runs in the background.** The lead stays free: the person keeps
  talking to it, and every lead call carries a live "Team now" note (the
  ``_team_note_fn`` hook beside ``/btw`` notes in ``_build_messages``). The
  lead's ``team`` tool can read the status, tell a helper something (it
  arrives at that helper's next step) and stop helpers.
* **Every step is visible.** Helpers' frames become ``{"type": "team", ...}``
  events on the lead's lane, which the Flight Deck chat shows as live cards.
* **Helpers never act outward.** Sending, publishing, payments, calendar and
  mailbox changes, schedules, memory and settings writes and other runs are
  refused at call time (:func:`tool_block`), so the tool list — and the
  prompt cache — stays the same as the lead's. The lead does such things once,
  after the team is done.
* **One browser holder at a time** (the lead included while a team runs).
* **Budget: a multiple of a normal turn** — the median tokens of this agent's
  recent chat turns, times the setting (6x by default). At the budget each
  working helper is told to finish with what it has; past 1.15x it is stopped.
* **When every helper is done** the lead gets a turn with their results (once
  it is free) and writes the person one reply.

Helpers can't team up (depth 1), one team per lead at a time, 2–4 helpers.
"""

from __future__ import annotations

import asyncio
import copy
import json
import re
import secrets
import statistics
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from captain_claw.logging import get_logger

log = get_logger(__name__)

# ── Settings ──────────────────────────────────────────────────────────────

SETTINGS_FILENAME = "team_up.json"
TURNS_FILENAME = "team_up_turns.json"
_HOME = Path("~/.captain-claw").expanduser()

MODES = ("off", "ask", "on")
DEFAULT_SETTINGS: dict[str, Any] = {"mode": "ask", "max_helpers": 3, "budget_x": 6}
MIN_HELPERS = 2
MAX_HELPERS = 4
BUDGET_X_MIN = 3
BUDGET_X_MAX = 12

# A normal turn: the median of the last TURNS_KEPT owner chat turns, never
# below TURN_FLOOR; TURN_DEFAULT until MIN_TURNS_KNOWN turns are recorded.
TURNS_KEPT = 50
MIN_TURNS_KNOWN = 3
TURN_FLOOR = 30_000
TURN_DEFAULT = 60_000

STEPS_KEPT = 200          # per helper, for the live cards and status
TELLS_KEPT = 8            # lead messages a helper keeps in view
RESULT_CHARS = 14_000     # one helper's result in the lead's finishing turn
RESULTS_CHARS = 42_000    # all of them
TEAMS_KEPT = 12           # finished teams kept in memory for the cards
RECORDS_KEPT = 10         # team records kept on the lead's session
WRAP_UP_AT = 1.0          # x budget: helpers are told to finish
HARD_STOP_AT = 1.15       # x budget: helpers still working are stopped
WARN_AT = 0.8             # x budget: the lead and the person are told
STOP_GRACE_SECONDS = 25.0 # after a clean stop request, the task is cancelled
LEAD_WAIT_SECONDS = 3600  # how long a finished team waits for the lead to be free

RESULT_HEAD = "[Team '{title}' finished]"
_RESULT_HEAD_RE = re.compile(r"^\[Team '.*?' (?:finished|stopped)\]")

_NAME_RE = re.compile(r"[^\w .&+\-/']+", re.UNICODE)


def _settings_path() -> Path:
    return _HOME / SETTINGS_FILENAME


def _turns_path() -> Path:
    return _HOME / TURNS_FILENAME


def normalize_settings(raw: Any) -> dict[str, Any]:
    """The settings with every value in range (unknown keys dropped)."""
    data = raw if isinstance(raw, dict) else {}
    mode = str(data.get("mode", DEFAULT_SETTINGS["mode"]) or "").strip().lower()
    if mode not in MODES:
        mode = DEFAULT_SETTINGS["mode"]
    try:
        helpers = int(data.get("max_helpers", DEFAULT_SETTINGS["max_helpers"]))
    except (TypeError, ValueError):
        helpers = DEFAULT_SETTINGS["max_helpers"]
    try:
        budget_x = int(data.get("budget_x", DEFAULT_SETTINGS["budget_x"]))
    except (TypeError, ValueError):
        budget_x = DEFAULT_SETTINGS["budget_x"]
    return {
        "mode": mode,
        "max_helpers": max(MIN_HELPERS, min(MAX_HELPERS, helpers)),
        "budget_x": max(BUDGET_X_MIN, min(BUDGET_X_MAX, budget_x)),
    }


def read_settings(path: Path | None = None) -> dict[str, Any]:
    """This agent's Team up settings (Flight Deck writes the file)."""
    try:
        raw = json.loads((path or _settings_path()).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        raw = {}
    return normalize_settings(raw)


def render_settings(data: Any) -> str:
    return json.dumps(normalize_settings(data), indent=2) + "\n"


# ── A normal turn ─────────────────────────────────────────────────────────

def turn_tokens(usage: Any) -> int:
    """A turn's size: fresh input plus output (cache reads left out, as they
    cost little and every fork re-reads the lead's history from the cache)."""
    if not isinstance(usage, dict):
        return 0

    def _n(key: str) -> int:
        try:
            return max(0, int(usage.get(key) or 0))
        except (TypeError, ValueError):
            return 0

    fresh = _n("prompt_tokens") + _n("completion_tokens")
    return max(fresh, _n("total_tokens")) + _n("cache_creation_input_tokens")


def _read_turns() -> list[int]:
    try:
        raw = json.loads(_turns_path().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    out: list[int] = []
    for item in raw if isinstance(raw, list) else []:
        try:
            value = int(item)
        except (TypeError, ValueError):
            continue
        if value > 0:
            out.append(value)
    return out[-TURNS_KEPT:]


def note_turn(usage: Any) -> None:
    """Record one owner chat turn's size (the base of the team budget)."""
    tokens = turn_tokens(usage)
    if tokens <= 0:
        return
    turns = (_read_turns() + [tokens])[-TURNS_KEPT:]
    try:
        path = _turns_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(turns), encoding="utf-8")
    except OSError as exc:
        log.debug("Team up: couldn't record the turn size", error=str(exc))


def normal_turn() -> int:
    turns = _read_turns()
    if len(turns) < MIN_TURNS_KNOWN:
        return TURN_DEFAULT
    return max(TURN_FLOOR, int(statistics.median(turns)))


def team_budget(settings: dict[str, Any] | None = None) -> int:
    cfg = normalize_settings(settings) if settings is not None else read_settings()
    return normal_turn() * int(cfg["budget_x"])


# ── State ─────────────────────────────────────────────────────────────────

HELPER_STATES = ("waiting", "working", "stopping", "done", "failed", "stopped")
TEAM_STATES = ("proposed", "running", "finishing", "done", "stopped", "dismissed", "failed")
_LIVE_HELPER = ("waiting", "working", "stopping")


def _now() -> float:
    return time.time()


@dataclass
class Helper:
    id: str
    name: str
    goal: str
    produces: str = ""
    state: str = "waiting"
    session_id: str = ""
    agent: Any = None
    task: asyncio.Task | None = None
    steps: deque = field(default_factory=lambda: deque(maxlen=STEPS_KEPT))
    step_count: int = 0
    tool_count: int = 0
    tokens: int = 0
    usage: dict[str, int] = field(default_factory=dict)
    result: str = ""
    error: str = ""
    draft: str = ""
    tells: list[str] = field(default_factory=list)
    files: list[str] = field(default_factory=list)
    started_at: float = 0.0
    ended_at: float = 0.0
    wrap_up: bool = False
    stopped_by: str = ""
    model: str = ""

    def last_step(self) -> str:
        return self.steps[-1]["text"] if self.steps else ""

    def snapshot(self, *, steps: int = 0) -> dict[str, Any]:
        data: dict[str, Any] = {
            "id": self.id,
            "name": self.name,
            "goal": self.goal,
            "produces": self.produces,
            "state": self.state,
            "session_id": self.session_id,
            "step_count": self.step_count,
            "tool_count": self.tool_count,
            "tokens": self.tokens,
            "usage": dict(self.usage),
            "last": self.last_step()[:240],
            "error": self.error[:400],
            "draft": self.draft[-600:] if self.state in _LIVE_HELPER else "",
            "result": self.result[:4000] if self.state not in _LIVE_HELPER else "",
            "files": list(self.files[-12:]),
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "wrap_up": self.wrap_up,
            "stopped_by": self.stopped_by,
            "model": self.model,
        }
        if steps:
            data["steps"] = list(self.steps)[-steps:]
        return data


@dataclass
class Team:
    id: str
    title: str
    lead: Any
    lane: str
    lead_session_id: str
    helpers: list[Helper]
    mode: str = "ask"
    state: str = "proposed"
    created_at: float = field(default_factory=_now)
    started_at: float = 0.0
    ended_at: float = 0.0
    budget: int = 0
    warned: bool = False
    browser_holder: str = ""
    delivered: bool = False
    note: str = ""
    watch: asyncio.Task | None = None

    def used(self) -> int:
        return sum(h.tokens for h in self.helpers)

    def helper(self, ref: str) -> Helper | None:
        key = str(ref or "").strip().lower()
        if not key:
            return None
        for h in self.helpers:
            if key in (h.id.lower(), h.name.lower()):
                return h
        for h in self.helpers:  # a unique prefix of the name
            if h.name.lower().startswith(key):
                return h
        return None

    def live(self) -> bool:
        return self.state in ("proposed", "running", "finishing")

    def snapshot(self, *, steps: int = 0) -> dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "lane": self.lane,
            "mode": self.mode,
            "state": self.state,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "budget": self.budget,
            "used": self.used(),
            "warned": self.warned,
            "browser_holder": self.browser_holder,
            "note": self.note,
            "helpers": [h.snapshot(steps=steps) for h in self.helpers],
        }


_TEAMS: dict[str, Team] = {}
_SERVER: Any = None


def bind_server(server: Any) -> None:
    """The web server this agent's chat runs on (helpers are built there)."""
    global _SERVER
    _SERVER = server


def listed() -> bool:
    """Is the `team` tool in the tool list? Only where helpers can run (a web
    chat in this process) and while the setting isn't Off."""
    return _SERVER is not None and read_settings()["mode"] != "off"


def find(team_id: str) -> Team | None:
    return _TEAMS.get(str(team_id or "").strip())


def teams_for_lane(lane: str) -> list[Team]:
    return sorted((t for t in _TEAMS.values() if t.lane == lane), key=lambda t: t.created_at)


def active_for(agent: Any) -> Team | None:
    """The live team this agent leads (proposed, running or finishing)."""
    if agent is None:
        return None
    for team in _TEAMS.values():
        if team.lead is agent and team.live():
            return team
    return None


def helper_of(agent: Any) -> tuple[Team, Helper] | None:
    marker = getattr(agent, "_team_helper", None)
    if not (isinstance(marker, tuple) and len(marker) == 2):
        return None
    team = find(marker[0])
    if team is None:
        return None
    for h in team.helpers:
        if h.id == marker[1]:
            return team, h
    return None


def is_helper(agent: Any) -> bool:
    return isinstance(getattr(agent, "_team_helper", None), tuple)


def _prune() -> None:
    done = sorted((t for t in _TEAMS.values() if not t.live()), key=lambda t: t.ended_at or t.created_at)
    for team in done[:-TEAMS_KEPT] if len(done) > TEAMS_KEPT else []:
        _TEAMS.pop(team.id, None)


# ── Where the lead is ─────────────────────────────────────────────────────

def lane_of(agent: Any, server: Any = None) -> str:
    """The web-chat lane *agent* drives ("" when it isn't a chat lane of its
    owner: a public visitor, a member, Telegram …)."""
    server = server or _SERVER
    if server is None or agent is None:
        return ""
    if agent is getattr(server, "agent", None):
        return getattr(server, "LANE_MAIN", "A")
    for lane, lane_agent in (getattr(server, "_lane_agents", None) or {}).items():
        if lane_agent is agent:
            return str(lane)
    return ""


def _lane_refused(server: Any, lane: str) -> bool:
    """Lanes Flight Deck opens for others or for automation: no teams there."""
    if lane.startswith(getattr(server, "REMOTE_LANE_PREFIX", "BP")):
        return True
    try:
        from captain_claw import exam_turn
        from captain_claw.config import get_config

        auto = server.normalize_lane(get_config().session.automation_lane or "")
        if auto and auto != server.LANE_MAIN and lane == auto:
            return True
        if lane == server.normalize_lane(exam_turn.LANE):
            return True
    except Exception:
        pass
    return False


_LOCAL_PROVIDERS = ("ollama", "lmstudio", "llamacpp", "llama_cpp", "llama.cpp", "mlx", "mtplx", "vllm-local")


def _local_single_slot(agent: Any) -> bool:
    """A model served from this machine: parallel helpers would push each
    other's cache out and run slower than one agent."""
    provider = getattr(agent, "provider", None)
    name = str(getattr(provider, "provider", "") or "").strip().lower()
    base = str(getattr(provider, "base_url", "") or getattr(provider, "api_base", "") or "").lower()
    if any(name.startswith(p) for p in _LOCAL_PROVIDERS):
        return True
    return bool(re.search(r"//(localhost|127\.0\.0\.1|0\.0\.0\.0|\[::1\])[:/]", base))


def why_unavailable(agent: Any) -> str:
    """"" when *agent* may team up on this turn, else why not (in words)."""
    if is_helper(agent):
        return "You're one of the helpers of a team: helpers don't team up. Do your part yourself."
    server = _SERVER
    if server is None:
        return "Team up works in your web chat, which isn't running here."
    settings = read_settings()
    if settings["mode"] == "off":
        return ("Team up is off for you. Do the task yourself; your owner can turn it on in "
                "Flight Deck → Agent settings.")
    if getattr(agent, "_public_scoped", False) or getattr(agent, "_speaker_principal", None) is not None:
        return "Team up isn't available in this conversation. Do the task yourself."
    try:
        from captain_claw import speaker as _speaker

        if _speaker.current() is not None or _speaker.is_speaker_agent(agent):
            return "Team up is off on a shared agent. Do the task yourself."
    except Exception:
        pass
    lane = lane_of(agent, server)
    if not lane or _lane_refused(server, lane):
        return "Team up works only in your owner's web chat. Do the task yourself."
    try:
        channel = agent._turn_channel()
    except Exception:
        channel = ""
    if channel and channel not in ("web", "flight_deck", "fd", "kiosk", "app"):
        return f"Team up works in the web chat, not on {channel}. Do the task yourself."
    if getattr(agent, "_tenant_hidden", False):
        return "Team up isn't available in this conversation. Do the task yourself."
    if _local_single_slot(agent):
        return ("Team up is off while you run on a model served from this machine: helpers would "
                "take turns on it and finish later than you alone. Do the task yourself.")
    live = active_for(agent)
    if live is not None and live.state in ("running", "finishing"):
        return (f"A team is already working on '{live.title}'. Check it with team status, or stop it "
                "with team stop, before starting another.")
    return ""


# ── Events to the lead's chat ────────────────────────────────────────────

def _lane_send(team: Team) -> Callable[[dict], None] | None:
    server = _SERVER
    if server is None:
        return None
    if team.lane == getattr(server, "LANE_MAIN", "A"):
        return lambda msg: server._broadcast(msg)
    return server._lane_send(team.lane)


def _emit(target: Team, event: str, **payload: Any) -> None:
    send = _lane_send(target)
    if send is None:
        return
    try:
        send({"type": "team", "event": event, "team_id": target.id, **payload})
    except Exception as exc:  # noqa: BLE001 — a card that misses one frame recovers on sync
        log.debug("Team up: event not sent", team=target.id, event_name=event, error=str(exc))


def _emit_team(team: Team, event: str = "team") -> None:
    _emit(team, event, team=team.snapshot())


def _emit_helper(team: Team, h: Helper) -> None:
    _emit(team, "helper", helper=h.snapshot(), used=team.used(), budget=team.budget,
          browser_holder=team.browser_holder)


def sync_frames(lane: str, *, steps: int = 60) -> list[dict[str, Any]]:
    """What a chat that just (re)connected to *lane* needs to rebuild its cards."""
    return [{"type": "team", "event": "snapshot", "team_id": t.id, "team": t.snapshot(steps=steps)}
            for t in teams_for_lane(lane)]


# ── Proposing and starting ───────────────────────────────────────────────

def _clean(text: Any, limit: int) -> str:
    value = re.sub(r"\s+", " ", str(text or "")).strip()
    return value[:limit].rstrip()


def _clean_name(text: Any, index: int) -> str:
    value = _NAME_RE.sub("", _clean(text, 40)).strip(" .-/")
    return value[:32].strip() or f"Helper {index + 1}"


def parse_parts(parts: Any, max_helpers: int) -> tuple[list[dict[str, str]], str]:
    """(parts, error): each part is {name, goal, produces}."""
    if isinstance(parts, str):
        try:
            parts = json.loads(parts)
        except ValueError:
            return [], "parts must be a list of {name, goal, produces}."
    if not isinstance(parts, list):
        return [], "parts must be a list of {name, goal, produces}."
    out: list[dict[str, str]] = []
    seen: set[str] = set()
    for i, raw in enumerate(parts):
        if isinstance(raw, str):
            raw = {"goal": raw}
        if not isinstance(raw, dict):
            continue
        goal = _clean(raw.get("goal") or raw.get("task") or raw.get("part"), 900)
        if not goal:
            continue
        name = _clean_name(raw.get("name") or raw.get("title"), i)
        base, n = name, 2
        while name.lower() in seen:
            name = f"{base} {n}"[:32]
            n += 1
        seen.add(name.lower())
        out.append({"name": name, "goal": goal, "produces": _clean(raw.get("produces") or raw.get("output"), 400)})
    if len(out) < MIN_HELPERS:
        return [], (f"A team needs at least {MIN_HELPERS} independent parts. If the task doesn't split, "
                    "do it yourself.")
    if len(out) > max_helpers:
        return [], (f"At most {max_helpers} helpers for you. Merge parts, or do the smallest ones "
                    "yourself after the team.")
    return out, ""


async def propose(agent: Any, title: str, parts: Any) -> tuple[Team | None, str]:
    """Offer a team. Under "Ask me" the person's tap starts it; under "On" it
    starts now. (team, "") or (None, why not)."""
    why = why_unavailable(agent)
    if why:
        return None, why
    settings = read_settings()
    cleaned, error = parse_parts(parts, int(settings["max_helpers"]))
    if error:
        return None, error
    old = active_for(agent)
    if old is not None and old.state == "proposed":
        old.state = "dismissed"
        old.ended_at = _now()
        _emit_team(old)
    team = Team(
        id=secrets.token_hex(4),
        title=_clean(title, 120) or "Team",
        lead=agent,
        lane=lane_of(agent),
        lead_session_id=str(getattr(getattr(agent, "session", None), "id", "") or ""),
        helpers=[Helper(id=f"h{i + 1}", name=p["name"], goal=p["goal"], produces=p["produces"])
                 for i, p in enumerate(cleaned)],
        mode=settings["mode"],
    )
    team.budget = team_budget(settings)
    _TEAMS[team.id] = team
    _prune()
    agent._team_note_fn = (lambda t=team: lead_note(t))  # type: ignore[attr-defined]
    _emit_team(team, "proposed")
    if team.mode == "on":
        error = await start(team, by="agent")
        if error:
            return None, error
    return team, ""


def _fork_messages(lead: Any) -> list[dict[str, Any]]:
    """The lead's conversation as the helpers inherit it. A turn still in
    progress (a team started from inside it) is cut after its opening message,
    so no helper inherits an unanswered tool call."""
    from captain_claw import msg_origin

    messages = list(getattr(getattr(lead, "session", None), "messages", None) or [])
    if not messages:
        return []
    busy = bool(getattr(lead, "_lane_busy", False))
    server = _SERVER
    if server is not None and lead is getattr(server, "agent", None):
        busy = bool(getattr(server, "_busy", False))
    if busy:
        for i in range(len(messages) - 1, -1, -1):
            msg = messages[i]
            if not isinstance(msg, dict) or msg.get("role") != "user":
                continue
            if str(msg.get("origin") or "") in msg_origin.USER_INJECTED_ORIGINS:
                continue
            messages = messages[: i + 1]
            break
    return copy.deepcopy(messages)


# Session metadata a helper never inherits: per-chat bookkeeping that would
# make it deliver to a channel, replay automation results or claim the team.
_FORK_METADATA_DROP = (
    "team_runs", "team_helper", "subagent", "pipeline", "automation_results", "whatsapp_waid",
    "origin", "fleet_events", "pending_session_change", "speaker_id", "speaker_lane", "speaker_name",
)


def _fork_metadata(meta: Any) -> dict[str, Any]:
    data = copy.deepcopy(meta) if isinstance(meta, dict) else {}
    for key in _FORK_METADATA_DROP:
        data.pop(key, None)
    return data


async def start(team: Team, *, by: str = "person") -> str:
    """Fork the helpers and run them in the background. "" or why not."""
    if team.state != "proposed":
        return f"That team is {team.state}, not waiting to start."
    server = _SERVER
    if server is None:
        return "Team up works in the web chat, which isn't running here."
    lead = team.lead
    if active_for(lead) not in (None, team):
        return "Another team of yours is already working."
    from captain_claw.session import get_session_manager

    sm = get_session_manager()
    team.state = "running"
    team.started_at = _now()
    team.note = f"started by {'you' if by == 'person' else 'the agent'}"
    base = _fork_messages(lead)
    meta = _fork_metadata(getattr(getattr(lead, "session", None), "metadata", None))
    # Helpers keep their files in the lead's session folders: the finishing
    # turn hands them over with the reply, and the system prompt (which names
    # that folder) stays byte-identical to the lead's for the prompt cache.
    meta["files_session_id"] = str(meta.get("files_session_id") or team.lead_session_id or "")
    lead_name = str(getattr(getattr(lead, "session", None), "name", "") or "chat")
    model = ""
    provider = getattr(lead, "provider", None)
    if provider is not None:
        model = str(getattr(provider, "model", "") or "")
    try:
        for h in team.helpers:
            session = await sm.create_session(
                name=f"{lead_name[:60]} · team {team.id} · {h.name}"[:120],
                metadata={
                    **meta,
                    "subagent": {"spawned_by": team.lead_session_id, "spawn_depth": 1,
                                 "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())},
                    "team_helper": {"team": team.id, "helper": h.id, "name": h.name,
                                    "lead_session": team.lead_session_id, "title": team.title},
                },
            )
            session.messages = copy.deepcopy(base)
            await sm.save_session(session)
            agent = await server._build_scoped_agent(
                session, _helper_send(team, h), register_tools=False, warm_owner_caches=True)
            if provider is not None:
                try:
                    agent.provider = copy.copy(provider)
                except Exception:
                    agent.provider = provider
            agent._team_helper = (team.id, h.id)  # type: ignore[attr-defined]
            agent._team_note_fn = (lambda t=team, hh=h: helper_note(t, hh))  # type: ignore[attr-defined]
            agent._suppress_rephrase = True  # type: ignore[attr-defined]
            h.agent = agent
            h.session_id = session.id
            h.model = model
    except Exception as exc:  # noqa: BLE001 — nothing started: tell the lead, keep the card honest
        log.warning("Team up: couldn't start the helpers", team=team.id, error=str(exc))
        team.state = "failed"
        team.ended_at = _now()
        team.note = f"couldn't start: {exc}"[:300]
        _emit_team(team)
        return f"The team couldn't start: {exc}"
    lead._team_note_fn = (lambda t=team: lead_note(t))  # type: ignore[attr-defined]
    for h in team.helpers:
        h.state = "working"
        h.started_at = _now()
        h.task = asyncio.create_task(_run_helper(team, h), name=f"team-{team.id}-{h.id}")
    team.watch = asyncio.create_task(_watch(team), name=f"team-{team.id}-watch")
    _emit_team(team, "started")
    _save_record(team)
    log.info("Team up: started", team=team.id, helpers=len(team.helpers), budget=team.budget, lane=team.lane)
    return ""


def dismiss(team: Team) -> str:
    if team.state != "proposed":
        return f"That team is {team.state}."
    team.state = "dismissed"
    team.ended_at = _now()
    _clear_lead_note(team)
    _emit_team(team)
    return ""


# ── A helper's run ────────────────────────────────────────────────────────

def helper_brief(team: Team, h: Helper) -> str:
    others = [f"{o.name} — {o.goal}" for o in team.helpers if o is not h]
    folder = f"team-{team.id}/{re.sub(r'[^a-z0-9]+', '-', h.name.lower()).strip('-') or h.id}"
    lines = [
        f"[Team up — a note from the runner, not a message from the user] You are \"{h.name}\", one of "
        f"{len(team.helpers)} helpers. Every helper is a copy of you, this same agent: the conversation "
        "above is the one you were having with the user. You split the user's request into parts, and "
        "this copy takes one of them. The user watches every helper's progress live.",
        "",
        f"The whole task: {team.title}",
        f"Your part: {h.goal}",
        f"Hand back: {h.produces or 'a complete, concrete result for your part'}",
    ]
    if others:
        lines.append("The other helpers: " + "; ".join(others))
    lines += [
        "",
        "How to work:",
        "- Do only your part, thoroughly, with your usual tools. Don't redo the others' parts.",
        "- You can't send, post, publish, pay, change the calendar or mailbox, schedule anything, or "
        "change your memory or settings here: the lead does such things once, after the team is done. "
        "If your part calls for one, write in your result exactly what should be done. Something worth "
        "remembering goes in your result as a line starting \"remember:\".",
        f"- Files you make go under {folder}/ so they don't collide with the others'.",
        "- The browser is shared: only one helper holds it at a time. Prefer web_search and web_fetch.",
        "- `team status` shows what the other helpers are doing.",
        "- Messages from the lead may appear while you work. Follow them.",
        "- Ask the user (ask_user) only if you truly can't go on without them.",
        "- When you're done, reply with your result. It goes to the lead, not straight to the user: be "
        "complete and concrete (facts, numbers, sources, file paths), no greeting, no offer of more help.",
    ]
    return "\n".join(lines)


def helper_note(team: Team, h: Helper) -> str:
    """The live "New for you" block on each of a helper's calls."""
    parts: list[str] = []
    if h.tells:
        parts.append("[From your lead — added while you work. Follow these.]\n"
                     + "\n".join(f"- {t}" for t in h.tells[-TELLS_KEPT:]))
    if h.wrap_up:
        parts.append("[Team budget reached — stop researching now. Write your final result with what you "
                     "have, and say what is left undone.]")
    return "\n\n".join(parts)


def lead_note(team: Team) -> str:
    """The live "Team now" block on each of the lead's calls while a team is live."""
    if team.state == "proposed":
        names = ", ".join(h.name for h in team.helpers)
        return (f"[Team now — you offered the user a team for '{team.title}' ({names}); they haven't "
                "answered the card yet. If their message says go ahead, call team start. If they want it "
                "done without a team, do it yourself and don't propose it again.]")
    if team.state not in ("running", "finishing"):
        return ""
    lines = [f"[Team now — your helpers are working in the background on '{team.title}'. The user sees "
             "their progress in the Team panel. This note is live; don't recite it.]"]
    for h in team.helpers:
        bits = [h.state]
        if h.state in _LIVE_HELPER:
            bits.append(f"{h.step_count} steps")
            last = h.last_step()
            if last:
                bits.append(f"last: {last[:120]}")
        bits.append(f"{_k(h.tokens)} tokens")
        lines.append(f"- {h.name}: " + " · ".join(bits))
    used = team.used()
    lines.append(f"Budget: {_k(used)} of {_k(team.budget)} tokens.")
    if team.state == "finishing":
        lines.append("All helpers are done; their results reach you in a moment.")
    else:
        lines.append("Use team status for details, team tell to message a helper, team stop to stop one or "
                     "all. When they finish you get their results and write the user one reply.")
    return "\n".join(lines)


def _k(n: int) -> str:
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if n >= 1000:
        return f"{round(n / 1000)}k"
    return str(n)


def _step(team: Team, h: Helper, kind: str, text: str, **extra: Any) -> None:
    text = re.sub(r"\s+", " ", str(text or "")).strip()
    if not text:
        return
    if h.steps and h.steps[-1]["text"] == text[:500] and h.steps[-1]["kind"] == kind:
        return
    step = {"t": round(_now(), 2), "kind": kind, "text": text[:500], **extra}
    h.steps.append(step)
    h.step_count += 1
    if kind == "tool":
        h.tool_count += 1
    _emit(team, "step", helper_id=h.id, step=step, step_count=h.step_count, tool_count=h.tool_count)


def _helper_send(team: Team, h: Helper) -> Callable[[dict], None]:
    """The scoped agent's frames, turned into team events on the lead's lane."""
    last_draft = [0.0]

    def send(msg: dict) -> None:
        try:
            kind = str(msg.get("type") or "")
            if kind == "narration":
                _step(team, h, "narration", msg.get("text", ""))
            elif kind == "tool_output_inline":
                summary = str(msg.get("summary") or msg.get("tool") or "")
                _step(team, h, "tool", summary, tool=str(msg.get("tool") or ""))
            elif kind == "thinking":
                text = str(msg.get("text") or "")
                if text.startswith("Calling LLM"):
                    _step(team, h, "llm", text)
            elif kind == "approval_notice":
                _step(team, h, "note", msg.get("message", ""))
            elif kind == "response_stream":
                h.draft = (h.draft + str(msg.get("text") or ""))[-2000:]
                now = time.monotonic()
                if now - last_draft[0] > 0.8:
                    last_draft[0] = now
                    _emit(team, "draft", helper_id=h.id, draft=h.draft[-600:])
            elif kind == "turn_usage":
                h.usage = {k: int(msg.get(k) or 0) for k in (
                    "prompt_tokens", "completion_tokens", "cache_read_input_tokens",
                    "cache_creation_input_tokens", "total_tokens")}
                h.tokens = turn_tokens(h.usage)
                _check_budget(team)
                _emit_helper(team, h)
            elif kind == "chat_message" and msg.get("role") in ("image", "html_file", "file"):
                path = str(msg.get("content") or "").strip()
                if path and path not in h.files:
                    h.files.append(path)
                    _step(team, h, "file", path)
        except Exception as exc:  # noqa: BLE001 — a frame never breaks the helper
            log.debug("Team up: frame skipped", team=team.id, helper=h.id, error=str(exc))

    return send


def _check_budget(team: Team) -> None:
    if team.budget <= 0 or team.state != "running":
        return
    used = team.used()
    if not team.warned and used >= team.budget * WARN_AT:
        team.warned = True
        _emit_team(team)
    half = team.budget / 2
    for h in team.helpers:
        if h.state != "working":
            continue
        if not h.wrap_up and (used >= team.budget * WRAP_UP_AT or h.tokens >= half):
            h.wrap_up = True
            _step(team, h, "note", "Budget reached — asked to finish with what it has")
        if used >= team.budget * HARD_STOP_AT or h.tokens >= half * HARD_STOP_AT:
            _request_stop(team, h, by="budget")


async def _run_helper(team: Team, h: Helper) -> None:
    from captain_claw import mail_authority, msg_origin
    from captain_claw.session import get_session_manager

    agent = h.agent
    brief = helper_brief(team, h)
    tok = mail_authority.bind(mail_authority.automated("fd_worker", "", "deny"))
    try:
        msg_origin.hint_turn_provenance(agent, channel="web", turn_origin="worker_task")
        _step(team, h, "note", "Started")
        reply = await agent.complete(brief)
        h.result = str(reply or "").strip()
        if h.state == "stopping" or getattr(agent, "_last_complete_cancelled", False):
            h.state = "stopped"
            h.stopped_by = h.stopped_by or "you"
        elif not h.result:
            h.state = "failed"
            h.error = h.error or "finished without a result"
        else:
            h.state = "done"
    except asyncio.CancelledError:
        h.state = "stopped"
        h.stopped_by = h.stopped_by or "you"
    except Exception as exc:  # noqa: BLE001 — one helper's failure is reported, the team goes on
        h.state = "failed"
        h.error = f"{type(exc).__name__}: {exc}"[:400]
        log.warning("Team up: helper failed", team=team.id, helper=h.id, error=h.error)
    finally:
        mail_authority.reset(tok)
        h.ended_at = _now()
        try:
            final = turn_tokens(getattr(agent, "last_usage", None))
            if final > h.tokens:
                h.tokens = final
        except Exception:
            pass
        if team.browser_holder == h.id:
            team.browser_holder = ""
        if agent is not None and getattr(agent, "session", None) is not None:
            try:
                await get_session_manager().save_session(agent.session)
            except Exception as exc:  # noqa: BLE001
                log.debug("Team up: helper session not saved", helper=h.id, error=str(exc))
        label = {"done": "Done", "stopped": "Stopped", "failed": "Failed"}.get(h.state, h.state)
        _step(team, h, "note", label + (f" — {h.error}" if h.state == "failed" and h.error else ""))
        _emit_helper(team, h)


def _request_stop(team: Team, h: Helper, *, by: str) -> None:
    if h.state not in ("working", "waiting"):
        return
    h.state = "stopping"
    h.stopped_by = by
    agent = h.agent
    if agent is not None and hasattr(agent, "cancel_event"):
        agent.cancel_event.set()
    task = h.task

    async def _hard_stop() -> None:
        await asyncio.sleep(STOP_GRACE_SECONDS)
        if task is not None and not task.done():
            task.cancel()

    if task is not None and not task.done():
        asyncio.create_task(_hard_stop())
    _emit_helper(team, h)


def stop(team: Team, helper_ref: str = "", *, by: str = "you") -> str:
    """Stop one helper, or the whole team ("" / "all")."""
    if team.state == "proposed":
        return dismiss(team)
    if team.state != "running":
        return f"That team is {team.state}."
    ref = str(helper_ref or "").strip().lower()
    if ref and ref != "all":
        h = team.helper(ref)
        if h is None:
            return f"No helper called {helper_ref!r}. Helpers: " + ", ".join(x.name for x in team.helpers)
        if h.state not in ("working", "waiting"):
            return f"{h.name} is already {h.state}."
        _request_stop(team, h, by=by)
        return ""
    team.note = f"stopped by {by}"
    for h in team.helpers:
        _request_stop(team, h, by=by)
    return ""


def tell(team: Team, helper_ref: str, text: str) -> str:
    """A message from the lead, shown at the helper's next step."""
    if team.state != "running":
        return f"That team is {team.state}."
    message = _clean(text, 2000)
    if not message:
        return "Give the text to tell."
    ref = str(helper_ref or "").strip().lower()
    targets = [h for h in team.helpers if h.state == "working"] if ref in ("", "all") else []
    if ref not in ("", "all"):
        h = team.helper(ref)
        if h is None:
            return f"No helper called {helper_ref!r}. Helpers: " + ", ".join(x.name for x in team.helpers)
        if h.state != "working":
            return f"{h.name} is {h.state}: it can't take messages now."
        targets = [h]
    if not targets:
        return "No helper is working now."
    for h in targets:
        h.tells.append(message)
        _step(team, h, "tell", message)
    return ""


async def _watch(team: Team) -> None:
    """Wait for every helper, then hand the results to the lead."""
    tasks = [h.task for h in team.helpers if h.task is not None]
    try:
        await asyncio.gather(*tasks, return_exceptions=True)
    except asyncio.CancelledError:
        return
    if team.state != "running":
        return
    lead = team.lead
    stopped_by_person = all(h.state == "stopped" and h.stopped_by == "you" for h in team.helpers)
    if stopped_by_person:
        team.state = "stopped"
        team.ended_at = _now()
        _clear_lead_note(team)
        _emit_team(team)
        _save_record(team)
        return
    team.state = "finishing"
    _emit_team(team)
    delivered = False
    try:
        delivered = await _deliver(team)
    except Exception as exc:  # noqa: BLE001
        log.warning("Team up: results not delivered", team=team.id, error=str(exc))
    team.delivered = delivered
    team.state = "done" if any(h.state == "done" for h in team.helpers) else "failed"
    team.ended_at = _now()
    if not delivered:
        team.note = "the results couldn't reach the chat; ask the agent for team status"
    if getattr(lead, "_team_note_fn", None) is not None:
        _clear_lead_note(team)
    _emit_team(team)
    _save_record(team)


def _clear_lead_note(team: Team) -> None:
    lead = team.lead
    if active_for(lead) is None:
        try:
            lead._team_note_fn = None  # type: ignore[attr-defined]
        except Exception:
            pass


def result_message(team: Team) -> str:
    """The finishing turn's opening message: every helper's result."""
    elapsed = (max(h.ended_at for h in team.helpers) or _now()) - (team.started_at or _now())
    lines = [
        RESULT_HEAD.format(title=team.title.replace("'", "’"))
        + " A note from the Team up runner, not a message from the user. Your helpers are done. "
        "Write the user ONE reply that brings their work together: lead with the answer, keep what "
        "matters from each part, and say plainly what's missing or failed. The user saw the progress "
        "but not these results. Helpers changed nothing outside: if their work calls for sending, "
        "booking, paying or remembering something, offer it — don't do it unasked.",
        "",
        f"Team: {len(team.helpers)} helpers · {_duration(elapsed)} · {_k(team.used())} tokens "
        f"(budget {_k(team.budget)})",
    ]
    budget_left = RESULTS_CHARS
    for h in team.helpers:
        status = {"done": "done", "failed": "failed", "stopped": f"stopped by {h.stopped_by or 'you'}"}.get(
            h.state, h.state)
        lines += ["", f"## {h.name} — {status}", f"Part: {h.goal}"]
        if h.result:
            room = max(1500, min(RESULT_CHARS, budget_left))
            text = h.result if len(h.result) <= room else h.result[:room] + "\n… [cut; the full result is in the helper's session]"
            budget_left -= len(text)
            lines += ["Result:", text]
        elif h.error:
            lines.append(f"Error: {h.error}")
        else:
            lines.append("No result.")
        if h.files:
            lines.append("Files: " + ", ".join(h.files[-12:]))
    return "\n".join(lines)


def _duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    if seconds < 60:
        return f"{seconds}s"
    minutes, sec = divmod(seconds, 60)
    if minutes < 60:
        return f"{minutes}m {sec:02d}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h {minutes:02d}m"


def is_result_message(text: str) -> bool:
    return bool(_RESULT_HEAD_RE.match(str(text or "")))


class _LaneSocket:
    """A stand-in socket on the lead's lane, for the finishing turn when no
    browser is connected there (the turn still runs and lands in the session)."""

    closed = False

    def __init__(self, lane: str) -> None:
        self._lane = lane

    async def send_str(self, data: str) -> None:  # noqa: D401 — the frames go to the lane's sockets
        return None

    async def send_json(self, data: Any) -> None:
        return None


def _lead_busy(server: Any, team: Team) -> bool:
    if team.lane == server.LANE_MAIN:
        return bool(getattr(server, "_busy", False))
    return bool(getattr(team.lead, "_lane_busy", False))


def _lane_socket(server: Any, lane: str) -> Any:
    if lane == server.LANE_MAIN:
        for ws in list(getattr(server, "clients", ()) or ()):
            if not getattr(ws, "closed", False) and not getattr(ws, "_public_session_id", None) \
                    and server.normalize_lane(getattr(ws, "_lane", "")) == lane:
                return ws
        return _LaneSocket(lane)
    for ws in list((getattr(server, "_lane_sockets", None) or {}).get(lane, ())):
        if not getattr(ws, "closed", False):
            return ws
    return _LaneSocket(lane)


async def _deliver(team: Team) -> bool:
    """The lead's finishing turn, once it is free."""
    server = _SERVER
    if server is None:
        return False
    from captain_claw import mail_authority
    from captain_claw.web.chat_handler import handle_chat

    content = result_message(team)
    deadline = time.monotonic() + LEAD_WAIT_SECONDS
    while time.monotonic() < deadline:
        if _lead_busy(server, team):
            await asyncio.sleep(0.5)
            continue
        ws = _lane_socket(server, team.lane)
        launched = await handle_chat(
            server, ws, content,
            automation=mail_authority.automated("peer_relay", "", "deny"),
            no_rephrase=True,
            source="team",
            source_label="Team up",
        )
        if launched:
            return True
        await asyncio.sleep(1.0)
    return False


# ── The record on the lead's session ─────────────────────────────────────

def _save_record(team: Team) -> None:
    session = getattr(team.lead, "session", None)
    if session is None or str(getattr(session, "id", "")) != team.lead_session_id:
        return
    meta = getattr(session, "metadata", None)
    if not isinstance(meta, dict):
        return
    record = {
        "id": team.id, "title": team.title, "state": team.state,
        "started_at": team.started_at, "ended_at": team.ended_at,
        "budget": team.budget, "used": team.used(),
        "helpers": [{"id": h.id, "name": h.name, "goal": h.goal, "state": h.state,
                     "session_id": h.session_id, "tokens": h.tokens,
                     "result": h.result[:1500], "files": h.files[-12:]} for h in team.helpers],
    }
    runs = [r for r in meta.get("team_runs", []) if isinstance(r, dict) and r.get("id") != team.id]
    meta["team_runs"] = (runs + [record])[-RECORDS_KEPT:]


# ── The guard: what a helper may not do ──────────────────────────────────

# Other runs, agents and the screen.
_HELPER_REFUSED_TOOLS = frozenset({
    "basna", "vatra", "bat", "crew", "captain_swarm", "first_mate", "colleagues", "consult_peer",
    "botport", "flight_deck", "terminal", "desktop_action", "hosting", "public_link", "app_runner",
    "synthesize_flow", "custom_tools", "linked_system", "direct_api", "termux",
})
# The agent's memory and its own settings: the lead's, not a copy's.
_HELPER_KEEP_TOOLS = frozenset({
    "insights", "topics", "intentions", "facts", "personality", "playbooks", "project_memory",
    "typesense", "external_memory", "model_switch", "reasoning_effort", "drive_level", "cron", "todo",
})
# Changes on the person's accounts: calendars, contacts, mailboxes, drives.
_HELPER_ACCOUNT_TOOLS = frozenset({
    "google_calendar", "microsoft_calendar", "contacts", "google_mail", "microsoft_mail", "imap_mail",
    "google_drive", "microsoft_onedrive", "apis",
})
# Shared screens: one holder at a time.
_BROWSER_TOOLS = frozenset({"browser", "pinchtab", "web_chrome", "browser_batch", "web_playbook"})
# Unknown to the risk table but safe for a copy.
_HELPER_OK_OTHER = frozenset({"team", "web_chrome", "stt", "knowledge", "line", "course", "self_check"})

OUTWARD = ("{what} isn't available to a helper: helpers change nothing outside the agent. Write in your "
           "result exactly what should be done; the lead does it once, after the team.")
KEEP = ("{what} isn't available to a helper: memory and settings are the lead's. Put a line starting "
        "\"remember:\" in your result instead.")
REFUSED = "{what} isn't available to a helper: helpers don't start other runs, agents or the screen."
BROWSER_BUSY = ("The browser is in use by {holder} right now (one holder at a time). Use web_search or "
                "web_fetch, or do other parts of your work and try the browser later.")
LEAD_BROWSER_BUSY = ("The browser is in use by your helper {holder} until it finishes. Use web_search or "
                     "web_fetch for now, or stop that helper first.")


def _action(arguments: Any) -> str:
    if not isinstance(arguments, dict):
        return ""
    return str(arguments.get("action") or "").strip().lower()


def tool_block(name: str, arguments: Any, agent: Any) -> str:
    """Why *agent* may not make this call ("" when it may). Helpers act on
    nothing outside; the browser has one holder at a time while a team runs."""
    tool = str(name or "").strip().lower()
    found = helper_of(agent)
    if found is None:
        if tool in _BROWSER_TOOLS:
            team = active_for(agent)
            if team is not None and team.state == "running" and team.browser_holder:
                holder = team.helper(team.browser_holder)
                if holder is not None and holder.state in _LIVE_HELPER:
                    return LEAD_BROWSER_BUSY.format(holder=holder.name)
        return ""
    team, h = found
    action = _action(arguments)
    what = f"{tool} {action}".strip()
    if tool == "team":
        return "" if action in ("status", "") else (
            "Only the lead starts, steers or stops the team. Use team status to see the other helpers.")
    if tool in _HELPER_REFUSED_TOOLS:
        return REFUSED.format(what=what)
    from captain_claw.flight_deck.tool_risk import risk_for

    risk = risk_for(tool, arguments if isinstance(arguments, dict) else {})
    if tool in _BROWSER_TOOLS:
        if team.browser_holder and team.browser_holder != h.id:
            holder = team.helper(team.browser_holder)
            if holder is not None and holder.state in _LIVE_HELPER:
                return BROWSER_BUSY.format(holder=holder.name)
        if team.browser_holder != h.id:
            team.browser_holder = h.id
            _step(team, h, "note", "Took the browser")
            _emit_helper(team, h)
        return ""
    if risk in ("send", "money"):
        return OUTWARD.format(what=what)
    if tool in _HELPER_KEEP_TOOLS and risk != "read":
        return KEEP.format(what=what)
    if tool in _HELPER_ACCOUNT_TOOLS and risk != "read":
        return OUTWARD.format(what=what)
    if risk == "other" and tool not in _HELPER_OK_OTHER:
        return OUTWARD.format(what=what)
    return ""


def labelled_question(agent: Any, arguments: Any) -> Any:
    """A helper's ask_user question names the helper, so the person knows who asks."""
    found = helper_of(agent)
    if found is None or not isinstance(arguments, dict):
        return arguments
    _team, h = found
    out = dict(arguments)
    for key in ("question", "title", "text", "prompt"):
        value = out.get(key)
        if isinstance(value, str) and value.strip() and not value.startswith(f"{h.name}:"):
            out[key] = f"{h.name}: {value}"
            break
    return out


# ── Status for the tool ──────────────────────────────────────────────────

def status_text(team: Team, helper_ref: str = "", *, for_helper: Helper | None = None) -> str:
    lines = [f"Team '{team.title}' — {team.state} · {_k(team.used())} of {_k(team.budget)} tokens"]
    if team.browser_holder:
        holder = team.helper(team.browser_holder)
        if holder is not None:
            lines.append(f"Browser: held by {holder.name}")
    chosen = [team.helper(helper_ref)] if helper_ref else list(team.helpers)
    for h in chosen:
        if h is None:
            return f"No helper called {helper_ref!r}. Helpers: " + ", ".join(x.name for x in team.helpers)
        me = " (you)" if for_helper is h else ""
        lines += ["", f"{h.name}{me} — {h.state} · {h.step_count} steps · {_k(h.tokens)} tokens",
                  f"Part: {h.goal}"]
        recent = [s for s in list(h.steps)[-(8 if helper_ref else 4):] if s["kind"] != "llm"]
        for s in recent:
            lines.append(f"  · {s['text'][:200]}")
        if h.state == "done" and h.result and for_helper is None:
            text = h.result if helper_ref else h.result[:600]
            lines.append("Result: " + text + ("" if helper_ref or len(h.result) <= 600 else " …"))
        if h.error:
            lines.append(f"Error: {h.error}")
        if h.tells and for_helper is None:
            lines.append("You told it: " + " | ".join(h.tells[-3:]))
    return "\n".join(lines)
