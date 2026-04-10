"""In-memory registry of running games for the agent's web server.

Persistence:
- Each game lives at `~/.captain-claw/games/<game_id>/` with:
    meta.json     — seed, world_id, seat_kinds (rebuild instructions)
    intents.jsonl — append-only log used for replay
- The registry hydrates from disk on first access. Live State is
  reconstructed by replaying the intent log against a fresh
  `initial_state(world)`. Determinism guarantees this matches whatever
  was running before the agent restarted.
"""

from __future__ import annotations

import json
import random
import uuid
from pathlib import Path
from typing import Any

from captain_claw.games.demo_world import DEMO_WORLDS
from captain_claw.games.engine import order_intents, resolve
from captain_claw.games.intent import Intent
from captain_claw.games.log import GameLog
from captain_claw.games.seats import AgentSeat, HumanSeat, ScriptedSeat, Seat, SeatTable
from captain_claw.games.world import State, World, initial_state
from captain_claw.games.world_io import load_world
from captain_claw.logging import get_logger

log = get_logger(__name__)


GAMES_DIR = Path.home() / ".captain-claw" / "games"


def _parse_seat_kind(raw: str) -> tuple[str, str]:
    """Parse a seat assignment string like ``"agent:ionian"`` into ``(kind, cognitive_mode)``.

    Returns ``(kind, cognitive_mode)`` where cognitive_mode defaults to ``"neutra"``
    for agent seats and ``""`` for other kinds.
    """
    if ":" in raw:
        kind, mode = raw.split(":", 1)
        return kind.strip(), mode.strip() or "neutra"
    return raw.strip(), "neutra" if raw.strip() == "agent" else ""


def _build_seat(kind: str, provider: Any = None, cognitive_mode: str = "") -> Seat:
    if kind == "human":
        return HumanSeat()
    if kind == "agent":
        return AgentSeat(provider=provider, cognitive_mode=cognitive_mode or "neutra")
    return ScriptedSeat()


class GameSession:
    def __init__(
        self,
        game_id: str,
        world: World,
        state: State,
        seats: SeatTable,
        log: GameLog,
        seed: int,
    ) -> None:
        self.game_id = game_id
        self.world = world
        self.state = state
        self.seats = seats
        self.log = log
        self.seed = seed
        self.rng = random.Random(seed)
        self.dir = GAMES_DIR / game_id
        self.conversation_log: list[dict[str, Any]] = []  # [{tick, actor, actor_name, text, room, room_name, audience}]

    def record_conversations(self) -> None:
        """Capture say/talk events from the current tick into the conversation log and persist."""
        for record in self.state.public_say:
            actor = record["actor"]
            speaker = self.world.characters.get(actor)
            room = self.world.rooms.get(record.get("room", ""))
            entry: dict[str, Any] = {
                "tick": self.state.tick,
                "actor": actor,
                "actor_name": speaker.name if speaker else actor,
                "text": record["text"],
                "room": record.get("room", ""),
                "room_name": room.name if room else "",
                "audience": record.get("audience", []),
                "kind": record.get("kind", "say"),
            }
            if record.get("target"):
                entry["target"] = record["target"]
            self.conversation_log.append(entry)
            self._append_jsonl("conversations.jsonl", entry)

    def persist_thoughts(self) -> None:
        """Persist the latest agent thought entries to disk."""
        from captain_claw.games.seats import AgentSeat as _AgentSeat
        for cid in self.seats.all_chars():
            seat = self.seats.get(cid)
            if isinstance(seat, _AgentSeat) and seat.thought_log:
                # Only append the latest entry (the one just added)
                latest = seat.thought_log[-1]
                self._append_jsonl(f"thoughts_{cid}.jsonl", latest)

    def _append_jsonl(self, filename: str, record: dict[str, Any]) -> None:
        path = self.dir / filename
        with path.open("a", encoding="utf-8") as fp:
            fp.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _load_jsonl(self, filename: str) -> list[dict[str, Any]]:
        path = self.dir / filename
        if not path.exists():
            return []
        out: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as fp:
            for line in fp:
                line = line.strip()
                if line:
                    try:
                        out.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        return out

    def reassign_seats(
        self, seat_assignments: dict[str, str], provider: Any = None
    ) -> tuple[bool, str | None]:
        """Change seat assignments. Only allowed at tick 0 (before any ticks).

        Also persists the updated seat_kinds to meta.json.
        """
        if self.state.tick > 0:
            return False, "cannot reassign seats after the game has started"
        seat_kinds: dict[str, str] = {}
        new_seats = SeatTable()
        for cid in self.world.characters:
            raw = seat_assignments.get(cid, "scripted")
            kind, mode = _parse_seat_kind(raw)
            if kind not in ("scripted", "human", "agent"):
                return False, f"invalid seat kind '{kind}' for {cid}"
            seat_kinds[cid] = f"{kind}:{mode}" if mode else kind
            new_seats.assign(cid, _build_seat(kind, provider, mode))
        self.seats = new_seats
        # Update meta.json on disk
        meta_path = self.dir / "meta.json"
        if meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            meta["seat_kinds"] = seat_kinds
            meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        return True, None

    def restart(self) -> None:
        """Reset the game back to tick 0 — fresh state, cleared log."""
        self.state = initial_state(self.world)
        self.rng = random.Random(self.seed)
        self.log.truncate_to(-1)  # drop all entries
        self.conversation_log.clear()
        # Clear persisted conversations
        conv_path = self.dir / "conversations.jsonl"
        if conv_path.exists():
            conv_path.write_text("", encoding="utf-8")
        # Clear agent thought logs (memory + disk)
        for cid in self.seats.all_chars():
            seat = self.seats.get(cid)
            if hasattr(seat, 'thought_log'):
                seat.thought_log.clear()
            if hasattr(seat, '_history'):
                seat._history.clear()
            thought_path = self.dir / f"thoughts_{cid}.jsonl"
            if thought_path.exists():
                thought_path.write_text("", encoding="utf-8")

    def queue_human_intent(self, intent: Intent) -> tuple[bool, str | None]:
        """Drop a human-submitted intent into the right HumanSeat. Returns (ok, error)."""
        if intent.actor not in self.world.characters:
            return False, f"unknown character {intent.actor}"
        try:
            seat = self.seats.get(intent.actor)
        except KeyError:
            return False, f"no seat for {intent.actor}"
        if not isinstance(seat, HumanSeat):
            return False, f"{intent.actor} is not a human seat"
        seat.queue(intent)
        return True, None

    def to_summary(self) -> dict[str, Any]:
        return {
            "game_id": self.game_id,
            "world_id": self.world.id,
            "title": self.world.title,
            "summary": self.world.summary,
            "tick": self.state.tick,
            "terminal": self.state.terminal,
            "win": self.state.win,
            "seed": self.seed,
            "characters": [
                {
                    "id": c.id,
                    "name": c.name,
                    "glyph": c.glyph,
                    "description": c.description,
                    "objective": c.objective,
                    "start_room": c.start_room,
                }
                for c in self.world.characters.values()
            ],
            "rooms": [
                {"id": r.id, "name": r.name}
                for r in self.world.rooms.values()
            ],
            "seats": self.seats.to_summary(),
        }


class GameRegistry:
    """Process-local registry that hydrates from disk on first access."""

    def __init__(self, provider: Any = None) -> None:
        self._sessions: dict[str, GameSession] = {}
        self._hydrated: bool = False
        self.provider: Any = provider  # LLMProvider — set when agent boots

    def set_provider(self, provider: Any) -> None:
        """Inject the LLM provider (called once the agent is initialized)."""
        self.provider = provider
        # Patch any existing agent seats that were created without a provider
        for session in self._sessions.values():
            for cid in session.seats.all_chars():
                seat = session.seats.get(cid)
                if isinstance(seat, AgentSeat) and seat.provider is None:
                    seat.provider = provider

    # ── hydration ───────────────────────────────────────────────────

    def _ensure_hydrated(self) -> None:
        if self._hydrated:
            return
        self._hydrated = True
        if not GAMES_DIR.exists():
            return
        for game_dir in sorted(GAMES_DIR.iterdir()):
            if not game_dir.is_dir():
                continue
            try:
                self._load_one(game_dir)
            except Exception as exc:  # noqa: BLE001 — never fail to start the agent
                log.warning("Failed to hydrate game", game_dir=str(game_dir), error=str(exc))

    def _load_one(self, game_dir: Path) -> None:
        meta_path = game_dir / "meta.json"
        if not meta_path.exists():
            return
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        kind = meta.get("kind", "demo")
        seed = int(meta["seed"])
        seat_kinds = dict(meta.get("seat_kinds", {}))

        if kind == "demo":
            world_id = meta["world_id"]
            factory = DEMO_WORLDS.get(world_id)
            if factory is None:
                log.warning("Skipping game with unknown demo world", world_id=world_id)
                return
            world = factory()
        elif kind == "generated":
            world_path = game_dir / "world.json"
            if not world_path.exists():
                log.warning("Generated game missing world.json", game_dir=str(game_dir))
                return
            world = load_world(world_path)
        else:
            log.warning("Unknown game kind on disk", kind=kind, game_dir=str(game_dir))
            return

        seats = SeatTable()
        for cid in world.characters:
            raw = seat_kinds.get(cid, "scripted")
            kind, mode = _parse_seat_kind(raw)
            seats.assign(cid, _build_seat(kind, self.provider, mode))

        game_log = GameLog(game_dir / "intents.jsonl")
        state = initial_state(world)
        rng = random.Random(seed)
        for record in game_log.read_all():
            intents = [Intent.from_dict(d) for d in record["intents"]]
            state, _ = resolve(state, order_intents(intents), rng)

        session = GameSession(
            game_id=game_dir.name,
            world=world,
            state=state,
            seats=seats,
            log=game_log,
            seed=seed,
        )
        # Restore the rng to the post-replay position so future ticks
        # continue from the right place.
        session.rng = rng
        # Restore persisted conversations and agent thoughts
        session.conversation_log = session._load_jsonl("conversations.jsonl")
        for cid in world.characters:
            seat = seats.get(cid)
            if isinstance(seat, AgentSeat):
                seat.thought_log = session._load_jsonl(f"thoughts_{cid}.jsonl")
        self._sessions[game_dir.name] = session
        log.info("Game hydrated", game_id=game_dir.name, tick=state.tick)

    # ── creation ────────────────────────────────────────────────────

    def create_from_demo(
        self,
        world_id: str,
        seat_assignments: dict[str, str] | None = None,
        seed: int | None = None,
    ) -> GameSession:
        """Spawn a game from a hardcoded demo world.

        `seat_assignments` maps `character_id -> "scripted" | "human"`.
        Any character not listed defaults to scripted.
        """
        self._ensure_hydrated()
        factory = DEMO_WORLDS.get(world_id)
        if factory is None:
            raise ValueError(f"unknown demo world '{world_id}'")
        world = factory()
        state = initial_state(world)

        seat_kinds: dict[str, str] = {}
        seats = SeatTable()
        for cid in world.characters:
            raw = (seat_assignments or {}).get(cid, "scripted")
            kind, mode = _parse_seat_kind(raw)
            seat_kinds[cid] = f"{kind}:{mode}" if mode else kind
            seats.assign(cid, _build_seat(kind, self.provider, mode))

        game_id = uuid.uuid4().hex[:12]
        actual_seed = seed if seed is not None else random.SystemRandom().randint(0, 2**31 - 1)
        game_dir = GAMES_DIR / game_id
        game_dir.mkdir(parents=True, exist_ok=True)
        game_log = GameLog(game_dir / "intents.jsonl")

        # Persist seed, world id, and seat kinds for replay + restart reproducibility.
        (game_dir / "meta.json").write_text(
            json.dumps(
                {"seed": actual_seed, "world_id": world.id, "seat_kinds": seat_kinds, "kind": "demo"},
                indent=2,
            ),
            encoding="utf-8",
        )

        session = GameSession(
            game_id=game_id,
            world=world,
            state=state,
            seats=seats,
            log=game_log,
            seed=actual_seed,
        )
        self._sessions[game_id] = session
        return session

    async def create_from_spec(
        self,
        provider: Any,
        spec: Any,  # WorldSpec
        *,
        mode: str = "fast",
        seat_assignments: dict[str, str] | None = None,
        seed: int | None = None,
    ) -> "GameSession":
        """Generate a world from a WorldSpec, persist it, and register a session.

        `provider` is the agent's LLM provider; `spec` is a WorldSpec.
        """
        from captain_claw.games.generator import generate_world, persist_generated_game

        self._ensure_hydrated()
        world, transcript, report = await generate_world(provider, spec, mode=mode)

        seat_kinds: dict[str, str] = {}
        seats = SeatTable()
        for cid in world.characters:
            raw = (seat_assignments or {}).get(cid, "scripted")
            kind, mode = _parse_seat_kind(raw)
            seat_kinds[cid] = f"{kind}:{mode}" if mode else kind
            seats.assign(cid, _build_seat(kind, self.provider, mode))

        game_id = uuid.uuid4().hex[:12]
        actual_seed = seed if seed is not None else random.SystemRandom().randint(0, 2**31 - 1)
        game_dir = GAMES_DIR / game_id
        persist_generated_game(game_dir, spec, world, transcript, actual_seed, seat_kinds)

        state = initial_state(world)
        game_log = GameLog(game_dir / "intents.jsonl")
        session = GameSession(
            game_id=game_id,
            world=world,
            state=state,
            seats=seats,
            log=game_log,
            seed=actual_seed,
        )
        self._sessions[game_id] = session
        log.info(
            "Generated game created",
            game_id=game_id,
            mode=mode,
            solvable=report.solvable,
            ticks_to_win=report.ticks_to_win,
        )
        return session

    # ── lookup ──────────────────────────────────────────────────────

    def get(self, game_id: str) -> GameSession | None:
        self._ensure_hydrated()
        return self._sessions.get(game_id)

    def list(self) -> list[dict[str, Any]]:
        self._ensure_hydrated()
        return [s.to_summary() for s in self._sessions.values()]

    def delete(self, game_id: str) -> bool:
        self._ensure_hydrated()
        removed = self._sessions.pop(game_id, None) is not None
        if removed:
            game_dir = GAMES_DIR / game_id
            if game_dir.exists():
                import shutil
                shutil.rmtree(game_dir, ignore_errors=True)
        return removed

    # ── available worlds ────────────────────────────────────────────

    @staticmethod
    def available_worlds() -> list[dict[str, Any]]:
        out = []
        for wid, factory in DEMO_WORLDS.items():
            w = factory()
            out.append({
                "id": wid,
                "title": w.title,
                "summary": w.summary,
                "characters": [
                    {"id": c.id, "name": c.name, "glyph": c.glyph}
                    for c in w.characters.values()
                ],
            })
        return out


# ── module singleton ────────────────────────────────────────────────

_registry: GameRegistry | None = None


def get_registry() -> GameRegistry:
    global _registry
    if _registry is None:
        _registry = GameRegistry()
    return _registry
