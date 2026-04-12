"""Tick loop driver.

`run_tick` collects one intent per character (from each seat), orders
them deterministically, hands them to the engine, writes the log entry,
and returns the new state plus the events.

The web layer calls this once per HTTP request to advance one tick.

Since M3 the function is async — AgentSeat.decide() makes an LLM call.
Sync seats (Scripted, Human) still work: run_tick checks for an async
`decide` coroutine before falling back to the sync `submit_intent`.
"""

from __future__ import annotations

import asyncio
import random
from typing import Any

from captain_claw.games.engine import order_intents, resolve
from captain_claw.games.intent import Intent
from captain_claw.games.log import GameLog
from captain_claw.games.seats import AgentSeat, SeatTable
from captain_claw.games.world import State
from captain_claw.logging import get_logger

_log = get_logger(__name__)


_REFLECTION_TEMPLATES: dict[str, str] = {
    "no exit": "That way seems to lead nowhere from here.",
    "no '": "I look around, but I can't find that here.",
    "cannot be taken": "It won't budge — not something I can carry.",
    "you do not have": "I reach for it, but it's not in my hands.",
    "does nothing": "I try, but nothing happens. Perhaps a different approach.",
    "is locked": "The way is barred. I'll need to find another means through.",
    "not here": "I glance around — they're nowhere to be seen.",
    "unknown character": "I'm not sure who that is.",
    "requires": "I'm not quite sure how to do that.",
}


def _template_fallback(error_text: str) -> str:
    """Pick a narrative template based on error text keywords."""
    lower = error_text.lower()
    for pattern, template in _REFLECTION_TEMPLATES.items():
        if pattern in lower:
            return template
    return f"That didn't work — {error_text.lower().rstrip('.')}"


async def _narrate_error(
    seat: AgentSeat,
    char_name: str,
    error_text: str,
    verb: str,
    args: dict[str, Any],
) -> str:
    """Ask the LLM to turn a raw engine error into a brief in-character reflection."""
    if seat.provider is None:
        return _template_fallback(error_text)

    from captain_claw.llm import Message

    action_desc = f"{verb} {' '.join(str(v) for v in args.values())}".strip()
    prompt = (
        f"You are {char_name}, a character in a text adventure game.\n"
        f"You just tried to: {action_desc}\n"
        f"But it failed: {error_text}\n\n"
        "Write ONE short sentence (15-25 words) as an in-character inner thought "
        "reacting to this failure. Be specific about what you tried and what went wrong. "
        "No quotes, no parentheses, no narration tags — just a plain sentence.\n"
        "Examples:\n"
        "- I search the shelves but the brass key is nowhere in this room.\n"
        "- The door resists my push — it must need a key I haven't found yet.\n"
        "- The mirrors here are part of the walls, not something I can interact with separately."
    )
    try:
        from captain_claw.games.llm_usage import fire_and_forget_usage
        import time as _time

        msgs = [Message(role="user", content=prompt)]
        t0 = _time.monotonic()
        resp = await seat.provider.complete(msgs, temperature=0.6, max_tokens=80)
        fire_and_forget_usage(
            interaction="game_agent_narrate_error",
            messages=msgs, response=resp,
            provider=seat.provider, max_tokens=80,
            latency_ms=int((_time.monotonic() - t0) * 1000),
        )
        raw = (resp.content or "").strip()
        # Clean common LLM wrapping artefacts
        for ch in ('"', "'", "(", ")", "*", "_", "—"):
            raw = raw.strip(ch)
        raw = raw.strip()
        # Pick first sentence if multi-sentence
        if ". " in raw:
            raw = raw.split(". ")[0] + "."
        if raw and len(raw) > 10:
            return raw
    except Exception as exc:
        _log.warning("narrate_error failed", char_name=char_name, error=str(exc))

    return _template_fallback(error_text)


async def run_tick(
    state: State,
    seats: SeatTable,
    log: GameLog,
    rng: random.Random,
) -> tuple[State, list[Intent], list[dict[str, Any]]]:
    """Advance the game by one tick. Returns (new_state, intents, events)."""
    if state.terminal:
        return state, [], []

    # Collect all agent decisions in parallel for speed
    all_chars = seats.all_chars()
    async_tasks: list[tuple[int, str, asyncio.Task]] = []
    sync_intents: dict[int, Intent] = {}

    for idx, char_id in enumerate(all_chars):
        seat = seats.get(char_id)
        if hasattr(seat, "decide"):
            task = asyncio.create_task(seat.decide(state, char_id))
            async_tasks.append((idx, char_id, task))
        else:
            sync_intents[idx] = seat.submit_intent(state, char_id)

    # Await all agent LLM calls concurrently
    if async_tasks:
        results = await asyncio.gather(
            *(t for _, _, t in async_tasks), return_exceptions=True,
        )
        for (idx, char_id, _), result in zip(async_tasks, results):
            if isinstance(result, Exception):
                _log.warning("Agent decide failed", char_id=char_id, error=str(result))
                sync_intents[idx] = Intent(actor=char_id, verb="wait", args={})
            else:
                sync_intents[idx] = result

    raw_intents = [sync_intents[i] for i in range(len(all_chars))]

    ordered = order_intents(raw_intents)
    new_state, events = resolve(state, ordered, rng)

    # ── Narrate agent errors as in-character reflections ──
    # Collect error action_results that belong to agent seats and narrate them
    narration_tasks: list[tuple[int, asyncio.Task]] = []
    intent_by_actor = {i.actor: i for i in ordered}
    for idx, result in enumerate(new_state.action_results):
        if result.get("kind") != "error":
            continue
        actor = result.get("actor", "")
        seat = seats.get(actor) if actor else None
        if not isinstance(seat, AgentSeat):
            continue
        char = state.world.characters.get(actor)
        char_name = char.name if char else actor
        intent = intent_by_actor.get(actor)
        verb = intent.verb if intent else "?"
        args = intent.args if intent else {}
        task = asyncio.create_task(
            _narrate_error(seat, char_name, result["text"], verb, args)
        )
        narration_tasks.append((idx, task))

    if narration_tasks:
        results = await asyncio.gather(*(t for _, t in narration_tasks), return_exceptions=True)
        for (idx, _task), narrated in zip(narration_tasks, results):
            if isinstance(narrated, str):
                new_state.action_results[idx]["kind"] = "reflection"
                new_state.action_results[idx]["original_error"] = new_state.action_results[idx]["text"]
                new_state.action_results[idx]["text"] = narrated

    log.append(tick=new_state.tick, intents=ordered, events=events)
    return new_state, ordered, events
