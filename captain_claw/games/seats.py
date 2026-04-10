"""Seat abstraction — who controls a character.

A seat is independent of the engine. The engine only asks: "give me an
intent for character X this tick." How that intent is produced (scripted,
human, agent) lives here.

Seat kinds:
- ScriptedSeat: deterministic dummy — BFS toward win room, no LLM.
- HumanSeat: pulls from a per-character intent queue populated by the
  REST API. If empty when `submit_intent` is called, returns `wait`.
- AgentSeat: LLM-driven — renders the character's view, asks the model
  for a structured intent, falls back to `wait` on failure.

Solo mode (one agent owns multiple characters) is M3+.
"""

from __future__ import annotations

import json
import re
from typing import Any, Protocol

from captain_claw.games.intent import VERBS, Intent
from captain_claw.games.world import State
from captain_claw.logging import get_logger

_log = get_logger(__name__)


class Seat(Protocol):
    kind: str

    def submit_intent(self, state: State, char_id: str) -> Intent: ...


# ── Scripted seat ────────────────────────────────────────────────────


def _bfs_first_step(state: State, start: str, goal: str) -> str | None:
    """Return the direction to take from `start` to advance toward `goal`.

    BFS over the room graph; deterministic via sorted exit iteration.
    """
    if start == goal:
        return None
    rooms = state.world.rooms
    visited: set[str] = {start}
    # queue of (room_id, first_direction_taken_from_start)
    queue: list[tuple[str, str | None]] = [(start, None)]
    while queue:
        current, first_dir = queue.pop(0)
        for direction in sorted(rooms[current].exits.keys()):
            nxt = rooms[current].exits[direction]
            if nxt in visited:
                continue
            visited.add(nxt)
            chosen = first_dir if first_dir is not None else direction
            if nxt == goal:
                return chosen
            queue.append((nxt, chosen))
    return None


class ScriptedSeat:
    """Picks a deterministic intent based on the win condition.

    Strategy:
    1. If holding nothing and a takeable entity is in the room, take it.
    2. If win condition is `all_in_room`, BFS toward that room.
    3. Else if there is an unvisited exit, move through it.
    4. Else move through the first exit (lexicographic).
    5. Else `wait`.

    Used for M0 to prove the loop runs end-to-end and reaches a win
    state without any LLM in the seat.
    """
    kind = "scripted"

    def submit_intent(self, state: State, char_id: str) -> Intent:
        here_id = state.char_room[char_id]
        room = state.world.rooms[here_id]
        contents = state.room_entities.get(here_id, [])
        inv = state.char_inventory.get(char_id, [])

        # 1) take a takeable entity in this room
        if not inv:
            for ent_id in contents:
                ent = state.world.entities.get(ent_id)
                if ent and ent.takeable:
                    return Intent(actor=char_id, verb="take", args={"entity_id": ent_id})

        # 2) navigate toward win-condition room if applicable
        cond = state.world.win_condition or {}
        if cond.get("kind") == "all_in_room":
            goal = cond.get("room")
            if goal and goal != here_id:
                direction = _bfs_first_step(state, here_id, str(goal))
                if direction is not None:
                    return Intent(actor=char_id, verb="move", args={"direction": direction})
            # already in goal room — hold position
            return Intent(actor=char_id, verb="wait", args={})

        # 3) unvisited exit
        visited = set(state.visited.get(char_id, []))
        for direction in sorted(room.exits.keys()):
            target = room.exits[direction]
            if target not in visited:
                return Intent(actor=char_id, verb="move", args={"direction": direction})

        # 4) first exit (deterministic)
        if room.exits:
            direction = sorted(room.exits.keys())[0]
            return Intent(actor=char_id, verb="move", args={"direction": direction})

        # 5) wait
        return Intent(actor=char_id, verb="wait", args={})


# ── Human seat ───────────────────────────────────────────────────────


class HumanSeat:
    """Reads queued intents from a per-character mailbox.

    The REST layer drops a queued Intent into `pending` when a human
    submits an action through Flight Deck. If nothing is queued by the
    time the tick resolves, the seat returns `wait` so the loop never
    blocks.
    """
    kind = "human"

    def __init__(self) -> None:
        self.pending: dict[str, Intent] = {}

    def queue(self, intent: Intent) -> None:
        self.pending[intent.actor] = intent

    def submit_intent(self, state: State, char_id: str) -> Intent:
        intent = self.pending.pop(char_id, None)
        if intent is None:
            return Intent(actor=char_id, verb="wait", args={})
        return intent


# ── Agent seat (LLM-driven) ──────────────────────────────────────────


import time as _time

from captain_claw.games.game_instructions import load_game_instruction, render_game_instruction
from captain_claw.games.llm_usage import fire_and_forget_usage

_JSON_RE = re.compile(r"\{[^{}]*\}", re.DOTALL)
_REASONING_RE = re.compile(r"<reasoning>\s*(.*?)\s*</reasoning>", re.DOTALL)


def _extract_reasoning(raw: str) -> str:
    """Pull the reasoning text from the LLM response."""
    m = _REASONING_RE.search(raw or "")
    if m:
        return m.group(1).strip()
    # Fallback: everything before the first { is the reasoning
    text = (raw or "").strip()
    idx = text.find("{")
    if idx > 0:
        return text[:idx].strip()
    return ""


def _parse_agent_response(raw: str) -> dict[str, Any] | None:
    """Extract the first JSON object from an LLM response."""
    text = (raw or "").strip()
    if not text:
        return None
    # Try the whole thing first
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass
    # Find the first {...}
    m = _JSON_RE.search(text)
    if m:
        try:
            obj = json.loads(m.group(0))
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            pass
    return None


def _build_agent_prompt(view: dict[str, Any]) -> str:
    """Build a user prompt from the structured view dict."""
    char = view.get("character", {})
    cr = view.get("current_room")
    inv = view.get("inventory", [])
    heard = view.get("heard", [])
    visited = view.get("visited", [])

    lines: list[str] = []
    lines.append(f"You are {char.get('name', '?')} ({char.get('glyph', '@')})")
    lines.append(f"Objective: {char.get('objective', '(none)')}")
    lines.append(f"Tick: {view.get('tick', 0)}")
    lines.append("")

    if cr:
        lines.append(f"Current room: {cr['name']} (id: {cr['id']})")
        lines.append(cr.get("description", ""))
        if cr.get("entities"):
            lines.append("Items here:")
            for e in cr["entities"]:
                tags = []
                if e.get("takeable"):
                    tags.append("takeable")
                if e.get("examinable") and not e.get("examined"):
                    tags.append("examine")
                elif e.get("examined"):
                    tags.append("examined")
                tag = f" [{', '.join(tags)}]" if tags else ""
                lines.append(f"  - {e['name']} (id: {e['id']}){tag}")
        if cr.get("others_here"):
            lines.append("Others here:")
            for o in cr["others_here"]:
                lines.append(f"  - {o['name']} ({o['glyph']})")
        if cr.get("exits"):
            locked = cr.get("locked_exits", {})
            exit_parts = []
            for d, r in sorted(cr["exits"].items()):
                lock_tag = " 🔒 LOCKED" if locked.get(d) else ""
                exit_parts.append(f"{d} → {r}{lock_tag}")
            exits = ", ".join(exit_parts)
            lines.append(f"Exits: {exits}")
    else:
        lines.append("You are nowhere.")

    lines.append("")
    if inv:
        lines.append("Inventory: " + ", ".join(f"{e['name']} (id: {e['id']})" for e in inv))
    else:
        lines.append("Inventory: (empty)")

    if visited:
        lines.append("Visited rooms: " + ", ".join(r["name"] for r in visited))

    # ── things you have discovered ──
    discoveries = view.get("discoveries", [])
    if discoveries:
        lines.append("")
        lines.append("Things you have discovered:")
        for d in discoveries:
            if d.get("kind") == "examine_result":
                lines.append(f"  - [Examined {d.get('entity_name', d.get('entity_id', '?'))}]: {d.get('text', '')}")
            elif d.get("kind") == "use_result":
                lines.append(f"  - [Used {d.get('item_id', '?')} on {d.get('target_id', '?')}]: {d.get('text', '')}")

    if heard:
        lines.append("")
        lines.append("You just heard:")
        has_direct = False
        for h in heard:
            kind = h.get("kind", "say")
            tag = "[DIRECT]" if kind == "talk" else "[PUBLIC]"
            lines.append(f'  {tag} {h["actor_name"]}: "{h["text"]}"')
            if kind == "talk":
                has_direct = True
        lines.append("")
        lines.append("*** CONVERSATION RULE: Someone just spoke to you! ***")
        lines.append("Your action this tick MUST be a response to what was said.")
        if has_direct:
            lines.append("You received a [DIRECT] message — you MUST reply with 'talk' to that character.")
        else:
            lines.append("You may respond with 'say' (public) or 'talk' (private).")
        lines.append("You cannot move, take, use, examine, or do anything else this tick.")
        lines.append("Social obligation overrides all other plans. Respond first, act next tick.")

    lines.append("")
    if heard:
        has_direct = any(h.get("kind") == "talk" for h in heard)
        if has_direct:
            lines.append("What do you say back? You MUST use 'talk' to reply to the direct message.")
        else:
            lines.append("What do you say in response? Use 'say' or 'talk'.")
    else:
        lines.append("What do you do? First write your <reasoning>, then the JSON action.")
    lines.append("Remember: your action MUST match your reasoning. If you plan to speak, use 'say' or 'talk'.")
    return "\n".join(lines)


def _normalize_args(verb: str, args: dict[str, Any]) -> dict[str, Any]:
    """Fix common LLM mistakes in arg keys.

    Models often produce {"entity": "brass_key"} instead of {"entity_id": "brass_key"},
    or {"dir": "north"} instead of {"direction": "north"}, etc.
    """
    if not args:
        return args
    out = dict(args)

    # entity / item / target → entity_id (for take, drop, examine)
    if verb in ("take", "drop", "examine"):
        if "entity_id" not in out:
            for alt in ("entity", "item", "item_id", "id", "object", "target"):
                if alt in out:
                    out["entity_id"] = out.pop(alt)
                    break

    # direction aliases for move
    if verb == "move":
        if "direction" not in out:
            for alt in ("dir", "exit", "to", "room"):
                if alt in out:
                    out["direction"] = out.pop(alt)
                    break

    # use: item_id + target_id
    if verb == "use":
        if "item_id" not in out:
            for alt in ("item", "entity_id", "entity", "object"):
                if alt in out:
                    out["item_id"] = out.pop(alt)
                    break
        if "target_id" not in out:
            for alt in ("target", "on", "use_on"):
                if alt in out:
                    out["target_id"] = out.pop(alt)
                    break

    # give: entity_id + target_id
    if verb == "give":
        if "entity_id" not in out:
            for alt in ("item", "item_id", "entity", "object"):
                if alt in out:
                    out["entity_id"] = out.pop(alt)
                    break
        if "target_id" not in out:
            for alt in ("target", "to", "recipient", "character"):
                if alt in out:
                    out["target_id"] = out.pop(alt)
                    break

    # say: text
    if verb == "say":
        if "text" not in out:
            for alt in ("message", "msg", "content", "speech"):
                if alt in out:
                    out["text"] = out.pop(alt)
                    break

    # talk: target + text
    if verb == "talk":
        if "target" not in out:
            for alt in ("target_id", "to", "recipient", "character", "char"):
                if alt in out:
                    out["target"] = out.pop(alt)
                    break
        if "text" not in out:
            for alt in ("message", "msg", "content", "speech"):
                if alt in out:
                    out["text"] = out.pop(alt)
                    break

    return out


class AgentSeat:
    """LLM-driven seat — asks the model to pick an action each tick.

    The `decide` coroutine is called by `run_tick` (async path). If the
    model returns an unparseable or invalid action, falls back to `wait`.
    A conversation history is maintained per seat so the agent has context
    of what it did in previous ticks.

    Each agent seat may have a *cognitive mode* (e.g. ``ionian``,
    ``phrygian``) that shapes its reasoning style through prompt injection
    and parameter tuning.  ``"neutra"`` (default) adds no extra prompt.
    """
    kind = "agent"

    def __init__(self, provider: Any = None, cognitive_mode: str = "neutra") -> None:
        self.provider = provider  # LLMProvider — set at creation or injected later
        self.cognitive_mode = cognitive_mode  # cognitive mode name
        self._history: list[dict[str, str]] = []  # [{role, content}, ...]
        self._max_history = 20  # keep last N exchanges to bound context
        self.thought_log: list[dict[str, Any]] = []  # [{tick, reasoning, action}, ...]

    def submit_intent(self, state: State, char_id: str) -> Intent:
        """Sync fallback — returns wait. Real decisions go through `decide`."""
        return Intent(actor=char_id, verb="wait", args={})

    def _build_system_prompt(self) -> str:
        """Build the full system prompt, injecting cognitive mode if non-neutra."""
        base = load_game_instruction("agent_system.md")
        if self.cognitive_mode == "neutra" or not self.cognitive_mode:
            return base

        from captain_claw.cognitive_mode import get_mode
        mode = get_mode(self.cognitive_mode)
        if mode.name == "neutra":
            return base

        # Load game-adapted mode instructions (NOT the raw agentic ones).
        # Game-specific versions live in instructions/games/modes/<name>.md
        # and are written for a text-adventure character context.
        mode_body = ""
        try:
            mode_body = load_game_instruction(f"modes/{mode.name}.md")
        except FileNotFoundError:
            pass

        mode_block = render_game_instruction(
            "agent_cognitive_mode.md",
            mode_label=mode.label,
            mode_name=mode.name.upper(),
            mode_character=mode.character,
            mode_game_instructions=mode_body or "(no additional instructions)",
        )

        return f"{base}\n\n{mode_block}"

    def _get_temperature(self) -> float:
        """Get LLM temperature adjusted by cognitive mode params."""
        base_temp = 0.7
        if self.cognitive_mode and self.cognitive_mode != "neutra":
            from captain_claw.cognitive_mode import get_mode_params
            params = get_mode_params(self.cognitive_mode)
            # tempo_bias ranges 0.0–1.0; map to temperature adjustment
            # Low tempo_bias (e.g. ionian=0.3) → lower temp (more focused)
            # High tempo_bias (e.g. lydian=0.8) → higher temp (more creative)
            base_temp = 0.4 + (params.tempo_bias * 0.6)  # range: 0.4–1.0
        return base_temp

    async def decide(self, state: State, char_id: str) -> Intent:
        """Async entry point — two-step LLM call: reasoning first, then action."""
        if self.provider is None:
            _log.warning("AgentSeat has no provider, falling back to wait", char_id=char_id)
            return Intent(actor=char_id, verb="wait", args={})

        from captain_claw.games.view import project_view
        from captain_claw.llm import Message

        view = project_view(state, char_id)
        user_prompt = _build_agent_prompt(view)

        temperature = self._get_temperature()

        # ── Step 1: Get reasoning ──
        reasoning_messages = [Message(role="system", content=self._build_system_prompt())]
        for entry in self._history[-self._max_history:]:
            reasoning_messages.append(Message(role=entry["role"], content=entry["content"]))
        reasoning_messages.append(Message(role="user", content=user_prompt))

        try:
            t0 = _time.monotonic()
            resp = await self.provider.complete(reasoning_messages, temperature=temperature, max_tokens=512)
            raw = resp.content or ""
            fire_and_forget_usage(
                interaction="game_agent_decide",
                messages=reasoning_messages, response=resp,
                provider=self.provider, max_tokens=512,
                latency_ms=int((_time.monotonic() - t0) * 1000),
            )
        except Exception as exc:
            _log.warning("AgentSeat LLM call failed", char_id=char_id, error=str(exc))
            return Intent(actor=char_id, verb="wait", args={})

        reasoning = _extract_reasoning(raw)
        parsed = _parse_agent_response(raw)

        # Check if the first response already has a good (non-wait) action
        verb = str(parsed.get("verb", "wait")).lower() if parsed else "wait"
        args = parsed.get("args", {}) if parsed else {}
        if not isinstance(args, dict):
            args = {}

        # ── Step 2: If action is wait, force a separate action call ──
        if verb == "wait" and reasoning:
            _log.info("AgentSeat step-2 action extraction", char_id=char_id)
            action_prompt = render_game_instruction("agent_action_repair.md", reasoning=reasoning)
            try:
                repair_msgs = [Message(role="user", content=action_prompt)]
                t0 = _time.monotonic()
                action_resp = await self.provider.complete(
                    repair_msgs, temperature=0.2, max_tokens=128,
                )
                fire_and_forget_usage(
                    interaction="game_agent_repair",
                    messages=repair_msgs, response=action_resp,
                    provider=self.provider, max_tokens=128,
                    latency_ms=int((_time.monotonic() - t0) * 1000),
                )
                action_raw = action_resp.content or ""
                action_parsed = _parse_agent_response(action_raw)
                if action_parsed:
                    v2 = str(action_parsed.get("verb", "wait")).lower()
                    a2 = action_parsed.get("args", {})
                    if v2 in VERBS and isinstance(a2, dict):
                        verb = v2
                        args = a2
                        _log.info("AgentSeat step-2 repaired", char_id=char_id, verb=verb)
            except Exception as exc:
                _log.warning("AgentSeat step-2 failed", char_id=char_id, error=str(exc))

        if verb not in VERBS:
            verb = "wait"
            args = {}

        # Normalize common arg key mistakes from LLMs
        args = _normalize_args(verb, args)

        # Record the exchange in history (store the final action, not raw)
        action_json = json.dumps({"verb": verb, "args": args})
        history_response = f"<reasoning>\n{reasoning}\n</reasoning>\n{action_json}" if reasoning else action_json
        self._history.append({"role": "user", "content": user_prompt})
        self._history.append({"role": "assistant", "content": history_response})

        # Trim history
        if len(self._history) > self._max_history * 2:
            self._history = self._history[-self._max_history * 2:]

        action_str = verb
        if args:
            action_str += " " + " ".join(f"{v}" for v in args.values())
        thought_entry: dict[str, Any] = {
            "tick": state.tick, "reasoning": reasoning, "action": action_str,
        }
        if self.cognitive_mode and self.cognitive_mode != "neutra":
            thought_entry["cognitive_mode"] = self.cognitive_mode
        self.thought_log.append(thought_entry)

        return Intent(actor=char_id, verb=verb, args={str(k): str(v) for k, v in args.items()})


# ── Seat assignment table ────────────────────────────────────────────


class SeatTable:
    """Tracks which seat owns which character."""

    def __init__(self) -> None:
        self._by_char: dict[str, Seat] = {}

    def assign(self, char_id: str, seat: Seat) -> None:
        self._by_char[char_id] = seat

    def get(self, char_id: str) -> Seat:
        return self._by_char[char_id]

    def all_chars(self) -> list[str]:
        return list(self._by_char.keys())

    def humans(self) -> list[tuple[str, HumanSeat]]:
        return [(c, s) for c, s in self._by_char.items() if isinstance(s, HumanSeat)]

    def to_summary(self) -> list[dict[str, Any]]:
        out = []
        for c, s in self._by_char.items():
            entry: dict[str, Any] = {"character": c, "kind": s.kind}
            if isinstance(s, AgentSeat):
                entry["cognitive_mode"] = s.cognitive_mode
            out.append(entry)
        return out
