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

from captain_claw import team_board
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


def usd_for(model: str, usage: Any) -> float | None:
    """The dollar cost of *usage* on *model*, None when its price isn't known."""
    if not model or not isinstance(usage, dict) or not usage:
        return None
    try:
        from captain_claw.flight_deck.pricing import cost_from_usage

        cost = cost_from_usage(model, usage)
    except Exception:
        return None
    return cost.get("usd") if cost.get("priced") else None


def _add_usage(into: dict, usage: Any) -> None:
    if not isinstance(usage, dict):
        return
    for key in ("prompt_tokens", "completion_tokens", "cache_read_input_tokens",
                "cache_creation_input_tokens", "total_tokens"):
        try:
            into[key] = int(into.get(key) or 0) + int(usage.get(key) or 0)
        except (TypeError, ValueError):
            continue


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
    seen_board: int = 0  # the last board entry pointed out to it as new

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
            "usd": usd_for(self.model, self.usage),
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
    board: team_board.Board = field(default_factory=team_board.Board)
    lead_tokens: int = 0       # the lead's quiet answers to helpers' asks
    lead_usage: dict = field(default_factory=dict)
    routed: set = field(default_factory=set)  # ask ids handed to the lead
    fork_len: int = 0          # how many lead messages the helpers inherited

    def used(self) -> int:
        return sum(h.tokens for h in self.helpers) + self.lead_tokens

    def usd(self) -> float | None:
        """Dollars so far ("None" when no part of it has a known price)."""
        model = self.helpers[0].model if self.helpers else ""
        parts = [usd_for(h.model, h.usage) for h in self.helpers]
        parts.append(usd_for(model, self.lead_usage) if self.lead_usage else None)
        priced = [p for p in parts if p is not None]
        return round(sum(priced), 4) if priced else None

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
            "lead_tokens": self.lead_tokens,
            "usd": self.usd(),
            "helpers": [h.snapshot(steps=steps) for h in self.helpers],
            "board": self.board.snapshot(),
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


def _unavailable(agent: Any) -> tuple[str, str]:
    """(code, why in words to the agent) — ("", "") when *agent* may team up."""
    if is_helper(agent):
        return "helper", "You're one of the helpers of a team: helpers don't team up. Do your part yourself."
    server = _SERVER
    if server is None:
        return "no_chat", "Team up works in your web chat, which isn't running here."
    settings = read_settings()
    if settings["mode"] == "off":
        return "off", ("Team up is off for you. Do the task yourself; your owner can turn it on in "
                       "Flight Deck → Agent settings.")
    if getattr(agent, "_public_scoped", False) or getattr(agent, "_speaker_principal", None) is not None:
        return "not_here", "Team up isn't available in this conversation. Do the task yourself."
    try:
        from captain_claw import speaker as _speaker

        if _speaker.current() is not None or _speaker.is_speaker_agent(agent):
            return "shared", "Team up is off on a shared agent. Do the task yourself."
    except Exception:
        pass
    lane = lane_of(agent, server)
    if not lane or _lane_refused(server, lane):
        return "lane", "Team up works only in your owner's web chat. Do the task yourself."
    try:
        channel = agent._turn_channel()
    except Exception:
        channel = ""
    if channel and channel not in ("web", "flight_deck", "fd", "kiosk", "app"):
        return "channel", f"Team up works in the web chat, not on {channel}. Do the task yourself."
    if getattr(agent, "_tenant_hidden", False):
        return "not_here", "Team up isn't available in this conversation. Do the task yourself."
    if _local_single_slot(agent):
        return "local", ("Team up is off while you run on a model served from this machine: helpers would "
                         "take turns on it and finish later than you alone. Do the task yourself.")
    live = active_for(agent)
    if live is not None and live.state in ("running", "finishing"):
        return "busy", (f"A team is already working on '{live.title}'. Check it with team status, or stop it "
                        "with team stop, before starting another.")
    return "", ""


def why_unavailable(agent: Any) -> str:
    """"" when *agent* may team up on this turn, else why not (in words)."""
    return _unavailable(agent)[1]


_PERSON_REFUSALS = {
    "helper": "A helper can't team up.",
    "no_chat": "Team up works in the web chat, which isn't running here.",
    "off": "Team up is off for this agent. Turn it on in Flight Deck → Agent settings → Team up.",
    "not_here": "Team up isn't available in this conversation.",
    "shared": "Team up is off on a shared agent.",
    "lane": "Team up works only in your own chat with the agent.",
    "channel": "Team up works in the web chat, not here.",
    "local": ("Team up is off while this agent runs on a model served from this machine: helpers would "
              "take turns on it and finish later than the agent alone."),
}


def refusal_for_person(agent: Any) -> str:
    """Why `/team` can't start here, said to the person ("" when it can)."""
    code, why = _unavailable(agent)
    if not code:
        return ""
    if code == "busy":
        live = active_for(agent)
        title = live.title if live is not None else "another task"
        return f"A team is already working on '{title}'. Wait for it, or stop it in the Team panel."
    return _PERSON_REFUSALS.get(code, why)


def is_helper_session(session: Any) -> bool:
    """A helper's own session (kept out of the owner's session lists)."""
    meta = getattr(session, "metadata", None)
    return isinstance(meta, dict) and isinstance(meta.get("team_helper"), dict)


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
    """What a chat that just (re)connected to *lane* needs to rebuild its cards:
    the teams in memory, and the records of earlier ones (after a restart)."""
    snaps = [t.snapshot(steps=steps) for t in teams_for_lane(lane)]
    known = {s["id"] for s in snaps}
    for record in records_for_lane(lane):
        if record.get("id") not in known:
            snaps.append(record_snapshot(record, lane))
    snaps.sort(key=lambda s: s.get("created_at") or s.get("started_at") or 0)
    return [{"type": "team", "event": "snapshot", "team_id": s["id"], "team": s} for s in snaps]


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
    # `/team <task>`: the person asked for the team, so it starts without a card.
    forced = bool(getattr(agent, "_team_forced", False))
    if team.mode == "on" or forced:
        agent._team_forced = False  # type: ignore[attr-defined]
        error = await start(team, by="person" if forced else "agent")
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
    team.fork_len = len(getattr(getattr(lead, "session", None), "messages", None) or [])
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
        "If your part calls for one, write in your result exactly what should be done.",
        f"- Files you make go under {folder}/ so they don't collide with the others'.",
        "- The browser is shared: only one helper holds it at a time. Prefer web_search and web_fetch.",
        "- The team board connects you with the others while you work:",
        "  · team post — share a finding, figure or decision another part may need (short, concrete). "
        "Start it with \"remember:\" for something worth keeping in memory: the lead decides after the team.",
        "  · team ask — when you need something another part owns: to a helper by name, to anyone, or to "
        "the lead (you, the agent the user talks to). It doesn't block; keep working.",
        "  · team answer — answer an ask addressed to you (it shows at your next step).",
        "  · team read / search / map — what the others found, who covers what. Check before researching "
        "something another part may already have.",
        "  · team inbox (wait) or team wait — block only when your part truly depends on an answer, another "
        "helper's result or post.",
        "- Messages from the lead and the board's news appear while you work. Follow the lead's.",
        "- Ask the user (ask_user) only if you truly can't go on without them.",
        "- When you're done, reply with your result. It goes to the lead, not straight to the user: be "
        "complete and concrete (facts, numbers, sources, file paths), no greeting, no offer of more help.",
    ]
    return "\n".join(lines)


HELPER_NOTE_CHARS = 6000


def helper_note(team: Team, h: Helper) -> str:
    """The live "New for you" block on each of a helper's calls: the lead's
    messages, asks addressed to it, answers to its own asks, and pointers to
    new posts by the others."""
    parts: list[str] = []
    if h.tells:
        parts.append("[From your lead — added while you work. Follow these.]\n"
                     + "\n".join(f"- {t}" for t in h.tells[-TELLS_KEPT:]))
    board = team.board
    asked = board.open_asks_for(h.id)
    if asked:
        parts.append("[Asked of you on the team board — answer each with team answer (ask_id, text) when you "
                     "can; keep it short. Skip one you can't help with.]\n"
                     + "\n".join(f"- #{e['id']} {e['who_name']} asks"
                                  f"{' anyone' if e.get('to') == team_board.ANYONE else ' you'}: {e['text'][:700]}"
                                  for e in asked[-6:]))
    mine: list[str] = []
    for ask in reversed(board.asks_by(h.id)):
        answers = board.answers_to(ask["id"])
        if answers:
            got = " | ".join(f"{a['who_name']}: {a['text'][:900]}" for a in answers[-2:])
            mine.append(f"- #{ask['id']} you asked {ask.get('to_name') or ask.get('to')}: {ask['text'][:200]} → {got}")
        else:
            mine.append(f"- #{ask['id']} you asked {ask.get('to_name') or ask.get('to')}: {ask['text'][:200]} — "
                        "no answer yet")
    if mine:
        parts.append("[Your asks on the team board]\n" + "\n".join(mine[:8]))
    fresh = [e for e in board.entries
             if e["id"] > h.seen_board and e["who"] != h.id and e["kind"] in ("post", "remember")]
    if fresh:
        rows = [f"- #{e['id']} {e['who_name']}: {e.get('title') or e['text'][:120]}" for e in fresh[-5:]]
        more = len(fresh) - len(rows)
        parts.append("[New on the team board — team read or team search to see them]\n" + "\n".join(rows)
                     + (f"\n(and {more} more)" if more > 0 else ""))
        h.seen_board = fresh[-1]["id"]
    if h.wrap_up:
        parts.append("[Team budget reached — stop researching now. Write your final result with what you "
                     "have, and say what is left undone.]")
    note = "\n\n".join(parts)
    return note if len(note) <= HELPER_NOTE_CHARS else note[:HELPER_NOTE_CHARS].rstrip() + "\n…"


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
    board = team.board
    to_lead = [e for e in board.entries if e["kind"] == "ask" and e.get("to") == team_board.LEAD]
    for ask in to_lead[-2:]:
        answers = board.answers_to(ask["id"])
        got = answers[-1]["text"][:200] if answers else "(answering…)"
        lines.append(f"{ask['who_name']} asked you: {ask['text'][:160]} — your answer: {got}")
    if board.entries:
        posts = sum(1 for e in board.entries if e["kind"] in ("post", "remember"))
        open_asks = sum(1 for e in board.entries if e["kind"] == "ask" and not e.get("answered"))
        lines.append(f"Board: {posts} posts, {open_asks} open asks (team read to see it).")
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
        _route_open_asks(team, h)


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


# ── The board (team_board.py): helpers talking while they work ───────────

LEAD_NAME = "Lead"
LEAD_ANSWER_TIMEOUT = 90.0
LEAD_ANSWER_TOKENS = 700


def _author(h: Helper | None) -> tuple[str, str]:
    return (h.id, h.name) if h is not None else (team_board.LEAD, LEAD_NAME)


def _board_add(team: Team, kind: str, h: Helper | None, text: str, **kw: Any) -> dict[str, Any]:
    who, who_name = _author(h)
    entry = team.board.add(kind, who, who_name, text, **kw)
    _emit(team, "board", entry=entry)
    return entry


def _not_running(team: Team) -> str:
    return "" if team.state == "running" else f"The team is {team.state}; the board is closed."


def board_post(team: Team, h: Helper, text: str, title: str = "") -> str:
    """A finding, decision or draft for the others; "remember: …" is a note for memory."""
    closed = _not_running(team)
    if closed:
        return closed
    body = str(text or "").strip()
    if not body:
        return "Give the text to post."
    if team.board.posts_by(h.id) >= team_board.MAX_POSTS_PER_AUTHOR:
        return "You've posted a lot already. Put the rest in your final result."
    if team_board.is_remember(body):
        note = team_board.strip_remember(body)
        if not note:
            return "Say what to remember after \"remember:\"."
        entry = _board_add(team, "remember", h, note, title=title)
        _step(team, h, "remember", f"Noted to remember: {note[:160]}")
        return (f"Noted for memory (#{entry['id']}). The lead decides after the team; don't repeat it in "
                "your result.")
    entry = _board_add(team, "post", h, body, title=title)
    _step(team, h, "post", f"Posted {('«' + entry['title'] + '»') if entry['title'] else ''} {body[:140]}".strip())
    return f"Posted #{entry['id']}. The others see it at their next step."


def _resolve_to(team: Team, h: Helper | None, to: str) -> tuple[str, str, Helper | None, str]:
    """(to id, to name, the helper or None, error)."""
    ref = str(to or "").strip().lower()
    if ref in ("", "anyone", "any", "all", "everyone", "team"):
        return team_board.ANYONE, "anyone", None, ""
    if ref in ("lead", "the lead", "agent", "you", "boss"):
        return team_board.LEAD, LEAD_NAME, None, ""
    target = team.helper(ref)
    if target is None:
        names = ", ".join(x.name for x in team.helpers)
        return "", "", None, f"No helper called {to!r}. Ask one of: {names}, anyone, or lead."
    if h is not None and target is h:
        return "", "", None, "That's you. Ask another helper, anyone, or the lead."
    return target.id, target.name, target, ""


async def board_ask(team: Team, h: Helper, to: str, text: str) -> str:
    """Ask a helper, anyone or the lead. Doesn't block."""
    closed = _not_running(team)
    if closed:
        return closed
    body = str(text or "").strip()
    if not body:
        return "Say what you need."
    to_id, to_name, target, error = _resolve_to(team, h, to)
    if error:
        return error
    same = team.board.same_open_ask(h.id, to_id, body)
    if same is not None:
        return f"You already asked that (#{same['id']}); its answer shows at a later step."
    if team.board.asks >= team_board.MAX_ASKS:
        return ("The team has used all its asks. Decide yourself — say what you assumed in your result — "
                "or use team read / search.")
    entry = _board_add(team, "ask", h, body, to=to_id, to_name=to_name)
    _step(team, h, "ask", f"Asked {to_name}: {body[:160]}")
    others_live = [x for x in team.helpers if x is not h and x.state in ("working", "waiting")]
    if (to_id == team_board.LEAD or (target is not None and target.state not in _LIVE_HELPER)
            or (to_id == team_board.ANYONE and not others_live)):
        _spawn_lead_answer(team, entry)
        who = "The lead answers"
    else:
        who = f"{to_name} sees it at its next step"
        if target is not None:
            _step(team, target, "ask", f"{h.name} asks: {body[:160]}")
    return (f"Asked {to_name} (#{entry['id']}). {who}; keep working — the answer shows at a later step, or "
            "use team inbox with wait to block for it.")


def board_answer(team: Team, h: Helper | None, ask_id: Any, text: str) -> str:
    """Answer an ask (a helper: one addressed to it or to anyone)."""
    closed = _not_running(team)
    if closed:
        return closed
    try:
        ask = team.board.get(int(str(ask_id).lstrip("#")))
    except (TypeError, ValueError):
        ask = None
    if ask is None or ask["kind"] != "ask":
        return f"No ask #{ask_id} on the board. team read kind=ask lists them."
    if h is not None:
        if ask["who"] == h.id:
            return "That's your own ask."
        if ask.get("to") not in (h.id, team_board.ANYONE):
            return f"Ask #{ask['id']} is for {ask.get('to_name')}, not you."
    body = str(text or "").strip()
    if not body:
        return "Give the answer."
    entry = _board_add(team, "answer", h, body, ask_id=ask["id"], to=ask["who"], to_name=ask["who_name"])
    who = h.name if h is not None else LEAD_NAME
    if h is not None:
        _step(team, h, "answer", f"Answered {ask['who_name']}: {body[:160]}")
    asker = team.helper(ask["who"]) if ask["who"] != team_board.LEAD else None
    if asker is not None:
        _step(team, asker, "answer", f"{who} answered: {body[:160]}")
    return f"Answered #{ask['id']} (#{entry['id']})."


def _spawn_lead_answer(team: Team, ask: dict[str, Any]) -> None:
    if ask["id"] in team.routed:
        return
    team.routed.add(ask["id"])

    async def _run() -> None:
        try:
            await _lead_answers(team, ask)
        except Exception as exc:  # noqa: BLE001 — the asker gets a plain fallback instead
            log.warning("Team up: lead answer failed", team=team.id, ask=ask.get("id"), error=str(exc))
            if team.state == "running":
                board_answer(team, None, ask["id"],
                             "I couldn't answer that just now. Decide yourself and say what you assumed in "
                             "your result, or ask the user with ask_user if it truly needs them.")

    asyncio.create_task(_run(), name=f"team-{team.id}-lead-answer-{ask.get('id')}")


def _lead_context(team: Team, ask: dict[str, Any]) -> str:
    """What the lead knows when it answers: the conversation (most of all what
    the user said after the team started), the team and the board."""
    messages = list(getattr(getattr(team.lead, "session", None), "messages", None) or [])
    since = [m for m in messages[team.fork_len:] if isinstance(m, dict)
             and m.get("role") in ("user", "assistant") and str(m.get("content") or "").strip()]
    before = [m for m in messages[:team.fork_len] if isinstance(m, dict)
              and m.get("role") in ("user", "assistant") and str(m.get("content") or "").strip()][-6:]

    def _rows(ms: list[dict]) -> str:
        return "\n".join(f"{m['role']}: {str(m.get('content') or '')[:800]}" for m in ms) or "(nothing)"

    lines = [
        f"The task you split: {team.title}",
        "",
        "Conversation with the user before the team started (most recent):",
        _rows(before),
        "",
        "What the user and you said since the team started:",
        _rows(since[-12:]),
        "",
        "Your helpers:",
    ]
    for x in team.helpers:
        row = f"- {x.name} ({x.state}): {x.goal}"
        if x.state == "done" and x.result:
            row += f"\n  Result: {x.result[:1500]}"
        lines.append(row)
    lines += ["", "The team board:", team_board.digest(team.board, limit=15, chars=400), ""]
    target = team.helper(ask.get("to", "")) if ask.get("to") not in (team_board.LEAD, team_board.ANYONE) else None
    if target is not None:
        lines.append(f"{ask['who_name']} asked {target.name}, who has finished ({target.state}); answer for it "
                     "from its result.")
    lines.append(f"{ask['who_name']} asks: {ask['text']}")
    return "\n".join(lines)


LEAD_ANSWER_SYSTEM = (
    "You are the lead of a Team up: the agent the user is talking to. Your helpers are copies of you, each "
    "working on one part of the user's task in the background. One of them asks you something. Answer in a "
    "few sentences from what you know: the conversation with the user, the helpers' results and the team "
    "board. If it needs a decision only the user can make and you don't know it, say so plainly and tell the "
    "helper to go with the most reasonable choice and say so in its result, or to ask the user with ask_user. "
    "Never make up facts. Reply with the answer only."
)


async def _lead_answers(team: Team, ask: dict[str, Any]) -> None:
    """The lead answers an ask quietly: one call on its own model, no tools,
    nothing in the chat (the panel shows the answer on the board)."""
    from captain_claw.llm import Message

    provider = getattr(team.lead, "provider", None)
    if provider is None:
        raise RuntimeError("the lead has no model")
    try:
        provider = copy.copy(provider)
    except Exception:
        pass
    messages = [Message(role="system", content=LEAD_ANSWER_SYSTEM),
                Message(role="user", content=_lead_context(team, ask))]
    response = await asyncio.wait_for(
        provider.complete(messages, tools=None, max_tokens=LEAD_ANSWER_TOKENS), timeout=LEAD_ANSWER_TIMEOUT)
    usage = getattr(response, "usage", None) or {}
    team.lead_tokens += turn_tokens(usage)
    _add_usage(team.lead_usage, usage)
    answer = str(getattr(response, "content", "") or "").strip()
    if not answer:
        answer = ("I don't know that. Decide yourself and say what you assumed in your result, or ask the user "
                  "with ask_user if it truly needs them.")
    if team.state == "running":
        board_answer(team, None, ask["id"], answer)
        _check_budget(team)


def _route_open_asks(team: Team, h: Helper) -> None:
    """A helper finished: the asks still waiting on it (or on anyone, when no
    one else works) are answered by the lead from what it knows."""
    if team.state != "running":
        return
    live = [x for x in team.helpers if x.state in ("working", "waiting")]
    for e in list(team.board.entries):
        if e["kind"] != "ask" or e.get("answered") or e["id"] in team.routed:
            continue
        if e.get("to") == h.id or (e.get("to") == team_board.ANYONE and not live):
            _spawn_lead_answer(team, e)


async def board_inbox(team: Team, h: Helper, wait: Any = 0) -> str:
    """The answers to this helper's asks; with *wait*, block a while for one."""
    try:
        seconds = max(0, min(team_board.INBOX_WAIT_MAX, int(wait or 0)))
    except (TypeError, ValueError):
        seconds = 0
    deadline = time.monotonic() + seconds
    while True:
        asks = team.board.asks_by(h.id)
        if not asks:
            return "You have no asks on the board."
        waiting = [a for a in asks if not team.board.answers_to(a["id"])]
        if not waiting or time.monotonic() >= deadline or team.state != "running":
            break
        await team.board.changed(deadline - time.monotonic())
    rows = []
    for a in asks:
        answers = team.board.answers_to(a["id"])
        head = f"#{a['id']} to {a.get('to_name') or a.get('to')}: {a['text'][:200]}"
        rows.append(head + ("".join(f"\n  → {x['who_name']}: {x['text']}" for x in answers) if answers
                            else "\n  (no answer yet)"))
    return "\n".join(rows)


async def board_wait(team: Team, h: Helper, helper_ref: str = "", query: str = "", wait: Any = 0) -> str:
    """Block until another helper finishes, or a post matching *query* appears."""
    try:
        seconds = max(5, min(team_board.WAIT_MAX, int(wait or team_board.WAIT_MAX)))
    except (TypeError, ValueError):
        seconds = team_board.WAIT_MAX
    target: Helper | None = None
    if helper_ref:
        target = team.helper(helper_ref)
        if target is None:
            return f"No helper called {helper_ref!r}. Helpers: " + ", ".join(x.name for x in team.helpers)
        if target is h:
            return "You can't wait for yourself."
    elif not str(query or "").strip():
        return "Give a helper to wait for, or a query for the post you need."
    start_id = team.board.next_id
    deadline = time.monotonic() + seconds
    _step(team, h, "note", f"Waiting for {target.name if target else repr(query)}")
    while True:
        if target is not None and target.state not in _LIVE_HELPER:
            result = target.result or target.error or "(no result)"
            return f"{target.name} is {target.state}. Its result:\n{result[:6000]}"
        if query:
            hits = [e for e in team.board.search(query, exclude_who=h.id, limit=5) if e["kind"] in ("post", "remember")]
            fresh = [e for e in hits if e["id"] >= start_id] or hits
            if fresh:
                return "Found on the board:\n" + "\n".join(team_board.fmt_entry(e) for e in fresh)
        if team.state != "running" or h.state != "working" or time.monotonic() >= deadline:
            break
        await team.board.changed(min(5.0, max(0.1, deadline - time.monotonic())))
    return ("Not there yet — go on with what you can, and say in your result what you assumed. The board now:\n"
            + team_board.digest(team.board, limit=10))


def board_read(team: Team, helper_ref: str = "", kind: str = "", limit: Any = 15) -> str:
    who = ""
    if helper_ref:
        ref = str(helper_ref).strip().lower()
        if ref == "lead":
            who = team_board.LEAD
        else:
            target = team.helper(ref)
            if target is None:
                return f"No helper called {helper_ref!r}."
            who = target.id
    kinds = str(kind or "").strip().lower()
    if kinds and kinds not in team_board.KINDS:
        return f"kind is one of {', '.join(team_board.KINDS)}."
    try:
        n = max(1, min(40, int(limit or 15)))
    except (TypeError, ValueError):
        n = 15
    rows = team.board.recent(who=who, kind=kinds, limit=n)
    if not rows:
        return "Nothing on the board yet." if not team.board.entries else "Nothing matches."
    return "\n".join(team_board.fmt_entry(e) for e in rows)


def board_search(team: Team, query: str, *, for_helper: Helper | None = None) -> str:
    hits = team.board.search(query, exclude_who=for_helper.id if for_helper else "")
    if not hits:
        return f"Nothing on the board matches {query!r}."
    return "\n".join(team_board.fmt_entry(e) for e in hits)


def board_map(team: Team, *, for_helper: Helper | None = None) -> str:
    """Who covers what, how far each is, what they made, and what's open."""
    lines = [f"Team '{team.title}' — {team.state}"]
    for x in team.helpers:
        me = " (you)" if x is for_helper else ""
        lines.append(f"- {x.name}{me} — {x.state} · {x.step_count} steps: {x.goal}")
        last = x.last_step()
        if last and x.state in _LIVE_HELPER:
            lines.append(f"  now: {last[:160]}")
        if x.files:
            lines.append("  files: " + ", ".join(x.files[-6:]))
        if x.state == "done" and x.result and x is not for_helper:
            lines.append(f"  result: {x.result[:300]}" + (" …" if len(x.result) > 300 else ""))
    open_asks = [e for e in team.board.entries if e["kind"] == "ask" and not e.get("answered")]
    if open_asks:
        lines.append("Open asks:")
        lines += [f"  {team_board.fmt_entry(e, full=False, chars=200)}" for e in open_asks[-6:]]
    posts = [e for e in team.board.entries if e["kind"] in ("post", "remember")]
    if posts:
        lines.append("Latest posts:")
        lines += [f"  #{e['id']} {e['who_name']}: {e.get('title') or e['text'][:120]}" for e in posts[-8:]]
    return "\n".join(lines)


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
    board = team.board
    notes = board.remember_notes()
    if notes:
        lines += ["", "## Worth remembering — the helpers' notes. Keep the ones that will matter later (your "
                  "memory tools); skip the rest. Don't list them to the user unless they ask."]
        lines += [f"- {e['who_name']}: {e['text'][:600]}" for e in notes[-12:]]
    if board.entries:
        posts = sum(1 for e in board.entries if e["kind"] == "post")
        asks = [e for e in board.entries if e["kind"] == "ask"]
        unanswered = [e for e in asks if not e.get("answered")]
        lines += ["", f"Board: {posts} posts, {len(asks)} asks ({len(unanswered)} unanswered)."]
        for e in unanswered[-4:]:
            lines.append(f"- Unanswered: {e['who_name']} asked {e.get('to_name') or e.get('to')}: {e['text'][:300]}")
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

RECORD_RESULT_CHARS = 2000
RECORD_BOARD_ENTRIES = 40
RECORD_BOARD_CHARS = 500


def _save_record(team: Team) -> None:
    """The team on the lead's session: the cards come back from it after a
    restart, and later turns can look up what each helper found."""
    session = getattr(team.lead, "session", None)
    if session is None or str(getattr(session, "id", "")) != team.lead_session_id:
        return
    meta = getattr(session, "metadata", None)
    if not isinstance(meta, dict):
        return
    record = {
        "id": team.id, "title": team.title, "lane": team.lane, "mode": team.mode, "state": team.state,
        "note": team.note, "created_at": team.created_at, "started_at": team.started_at,
        "ended_at": team.ended_at, "budget": team.budget, "used": team.used(),
        "lead_tokens": team.lead_tokens, "usd": team.usd(), "fork_len": team.fork_len,
        "delivered": team.delivered,
        "helpers": [{"id": h.id, "name": h.name, "goal": h.goal, "produces": h.produces, "state": h.state,
                     "session_id": h.session_id, "tokens": h.tokens, "usage": dict(h.usage),
                     "model": h.model, "step_count": h.step_count, "tool_count": h.tool_count,
                     "started_at": h.started_at, "ended_at": h.ended_at, "stopped_by": h.stopped_by,
                     "error": h.error[:400], "result": h.result[:RECORD_RESULT_CHARS],
                     "files": h.files[-12:]} for h in team.helpers],
        "board": [dict(e, text=str(e.get("text") or "")[:RECORD_BOARD_CHARS])
                  for e in team.board.entries[-RECORD_BOARD_ENTRIES:]],
    }
    runs = [r for r in meta.get("team_runs", []) if isinstance(r, dict) and r.get("id") != team.id]
    meta["team_runs"] = (runs + [record])[-RECORDS_KEPT:]
    # A turn on the lead saves the session as it ends; with none running
    # (a team stopped from the panel) the record is saved here.
    server = _SERVER
    if server is not None and not _lead_busy(server, team):
        try:
            from captain_claw.session import get_session_manager

            asyncio.get_running_loop().create_task(get_session_manager().save_session(session))
        except Exception:
            pass


def _lane_agent(lane: str) -> Any:
    server = _SERVER
    if server is None:
        return None
    if lane == getattr(server, "LANE_MAIN", "A"):
        return getattr(server, "agent", None)
    return (getattr(server, "_lane_agents", None) or {}).get(lane)


def records_of(session: Any) -> list[dict[str, Any]]:
    meta = getattr(session, "metadata", None)
    runs = meta.get("team_runs") if isinstance(meta, dict) else None
    return [r for r in (runs or []) if isinstance(r, dict) and r.get("id")]


def records_for_lane(lane: str) -> list[dict[str, Any]]:
    """The team records on the session the lane's agent has open now."""
    agent = _lane_agent(lane)
    return records_of(getattr(agent, "session", None)) if agent is not None else []


def record_snapshot(record: dict[str, Any], lane: str = "") -> dict[str, Any]:
    """A record in the shape of Team.snapshot, for the chat's cards. A team
    that was still working when the agent stopped shows as stopped."""
    cut = record.get("state") in ("proposed", "running", "finishing")
    helpers = []
    for h in record.get("helpers") or []:
        if not isinstance(h, dict):
            continue
        state = h.get("state") or "stopped"
        stopped = cut and state in _LIVE_HELPER
        helpers.append({
            "id": h.get("id", ""), "name": h.get("name", ""), "goal": h.get("goal", ""),
            "produces": h.get("produces", ""), "state": "stopped" if stopped else state,
            "session_id": h.get("session_id", ""), "step_count": int(h.get("step_count") or 0),
            "tool_count": int(h.get("tool_count") or 0), "tokens": int(h.get("tokens") or 0),
            "usage": dict(h.get("usage") or {}), "last": "", "error": h.get("error", ""), "draft": "",
            "result": h.get("result", ""), "files": list(h.get("files") or []),
            "started_at": h.get("started_at") or 0, "ended_at": h.get("ended_at") or 0,
            "wrap_up": False, "stopped_by": "restart" if stopped else h.get("stopped_by", ""),
            "model": h.get("model", ""), "usd": usd_for(h.get("model", ""), h.get("usage")),
            "steps": [],
        })
    state = record.get("state") or "done"
    if cut:
        state = "dismissed" if state == "proposed" else "stopped"
    return {
        "id": record["id"], "title": record.get("title", "Team"), "lane": record.get("lane") or lane,
        "mode": record.get("mode", "ask"), "state": state,
        "created_at": record.get("created_at") or record.get("started_at") or 0,
        "started_at": record.get("started_at") or 0, "ended_at": record.get("ended_at") or 0,
        "budget": int(record.get("budget") or 0), "used": int(record.get("used") or 0), "warned": False,
        "browser_holder": "",
        "note": "the agent restarted while the team worked" if cut else (record.get("note") or ""),
        "lead_tokens": int(record.get("lead_tokens") or 0), "usd": record.get("usd"),
        "helpers": helpers, "board": [dict(e) for e in record.get("board") or [] if isinstance(e, dict)],
        "from_record": True,
    }


# ── A helper's transcript (the panel's "Transcript") ─────────────────────

TRANSCRIPT_ITEMS = 150
TRANSCRIPT_TEXT = 6000
TRANSCRIPT_TOOL = 1500


def _text_of(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        out = []
        for part in content:
            if isinstance(part, dict):
                if part.get("type") in ("text", "input_text", "output_text"):
                    out.append(str(part.get("text") or ""))
                elif part.get("type") in ("image", "image_url", "input_image"):
                    out.append("[picture]")
            elif isinstance(part, str):
                out.append(part)
        return "\n".join(x for x in out if x)
    return "" if content is None else str(content)


def _cut(text: str, limit: int) -> tuple[str, bool]:
    text = text.strip()
    return (text, False) if len(text) <= limit else (text[:limit].rstrip() + " …", True)


def _calls_of(msg: dict[str, Any]) -> list[dict[str, str]]:
    calls = []
    for call in msg.get("tool_calls") or []:
        if not isinstance(call, dict):
            continue
        fn = call.get("function") if isinstance(call.get("function"), dict) else call
        name = str(fn.get("name") or "")
        args = fn.get("arguments") if "arguments" in fn else fn.get("input")
        if not isinstance(args, str):
            try:
                args = json.dumps(args, ensure_ascii=False)
            except (TypeError, ValueError):
                args = str(args)
        if name:
            calls.append({"name": name, "args": _cut(args or "", 400)[0]})
    return calls


def transcript_items(messages: list[Any], fork_len: int = 0) -> tuple[list[dict[str, Any]], int]:
    """A helper's own messages (after the conversation it inherited), made
    small for the panel: (items, how many there are in all)."""
    own = [m for m in list(messages or [])[max(0, int(fork_len or 0)):] if isinstance(m, dict)]
    items: list[dict[str, Any]] = []
    for i, msg in enumerate(own):
        role = str(msg.get("role") or "")
        text = _text_of(msg.get("content"))
        if role == "user":
            body, cut = _cut(text, TRANSCRIPT_TEXT)
            items.append({"role": "user", "kind": "brief" if i == 0 else "note", "text": body, "cut": cut})
        elif role == "assistant":
            body, cut = _cut(text, TRANSCRIPT_TEXT)
            calls = _calls_of(msg)
            if body or calls:
                items.append({"role": "assistant", "text": body, "cut": cut, "calls": calls})
        elif role == "tool":
            body, cut = _cut(text, TRANSCRIPT_TOOL)
            items.append({"role": "tool", "name": str(msg.get("tool_name") or msg.get("name") or ""),
                          "text": body, "cut": cut})
    total = len(items)
    return items[-TRANSCRIPT_ITEMS:], total


async def transcript(lane: str, team_id: str, helper_ref: str) -> dict[str, Any]:
    """One helper's work as the panel shows it. A team in memory reads its
    helper's live session; an older one is found through its record."""
    team = find(team_id)
    session = None
    fork_len = 0
    helper_id = ""
    if team is not None and team.lane == lane:
        h = team.helper(helper_ref)
        if h is None:
            return {"error": "No such helper."}
        helper_id, fork_len = h.id, team.fork_len
        session = getattr(h.agent, "session", None) if h.agent is not None else None
        session_id = h.session_id
    else:
        record = next((r for r in records_for_lane(lane) if r.get("id") == team_id), None)
        if record is None:
            return {"error": "That team isn't here any more."}
        ref = str(helper_ref or "").strip().lower()
        h_rec = next((x for x in record.get("helpers") or []
                      if isinstance(x, dict) and ref in (str(x.get("id", "")).lower(),
                                                         str(x.get("name", "")).lower())), None)
        if h_rec is None:
            return {"error": "No such helper."}
        helper_id, fork_len = str(h_rec.get("id") or ""), int(record.get("fork_len") or 0)
        session_id = str(h_rec.get("session_id") or "")
    if session is None and session_id:
        try:
            from captain_claw.session import get_session_manager

            session = await get_session_manager().load_session(session_id)
        except Exception as exc:  # noqa: BLE001
            log.debug("Team up: helper session not loaded", session=session_id, error=str(exc))
    if session is None:
        return {"helper_id": helper_id, "error": "This helper's transcript is gone."}
    items, total = transcript_items(getattr(session, "messages", None) or [], fork_len)
    return {"helper_id": helper_id, "messages": items, "total": total}


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
KEEP = ("{what} isn't available to a helper: memory and settings are the lead's. Post it with team post "
        "text=\"remember: …\" instead; the lead decides after the team.")
REFUSED = "{what} isn't available to a helper: helpers don't start other runs, agents or the screen."
BROWSER_BUSY = ("The browser is in use by {holder} right now (one holder at a time). Use web_search or "
                "web_fetch, or do other parts of your work and try the browser later.")
LEAD_BROWSER_BUSY = ("The browser is in use by your helper {holder} until it finishes. Use web_search or "
                     "web_fetch for now, or stop that helper first.")


# What a helper may do with the `team` tool: look, and talk on the board.
HELPER_TEAM_ACTIONS = frozenset({"status", "map", "read", "search", "post", "ask", "answer", "inbox", "wait"})


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
        return "" if action in HELPER_TEAM_ACTIONS or action == "" else (
            "Only the lead starts, steers or stops the team. Use team status or team map to see the others.")
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
    if team.board.entries:
        open_asks = sum(1 for e in team.board.entries if e["kind"] == "ask" and not e.get("answered"))
        lines.append(f"Board: {len(team.board.entries)} entries, {open_asks} open asks (team read / map).")
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
