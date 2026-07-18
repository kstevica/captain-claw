"""Micro faculties — a being's JSON faculties on the micro tier (mrav Phase 3).

``cognition = 'micro'`` keeps the faculties pipeline exactly as it is, but
routes the pure-generation steps (orient / talk / journal / connect)
STRAIGHT to the owner's ``micro`` tier with grammar-constrained output —
the feet's one-shot pattern (being_instinct._one_shot) combined with mrav's
``complete_structured``. The ACT faculty (the only tool-using step) and the
write gate always stay on the body, so git-diff verification and
anti-theater are untouched.

Why: a faculty send through the body is a full agent turn — ~9k input
fresh, 34-36k mature (docs/being-compact-mode.md) — and before faculties
existed, weak models answered with unparseable prose 14 of 19 ticks.
Direct + grammar fixes both at once: the input is the ~1-2k faculty prompt
alone, and the reply is guaranteed to be one JSON object carrying the
contract's core keys (schemas below mirror the instruction contracts in
captain_claw/instructions/beings/ and are built from ACT_KINDS, so they
cannot drift from the act menu).

Metering: the tick's window poll (``_usage_since``) only sees body calls,
so a micro call debits the wallet itself, per call, exactly like the feet —
tier ``micro``, note ``tick-micro:<faculty>``. Constitution tier weighting
falls back to 1.0 for tiers it doesn't know.

Fallback: any miss (no db, no micro tier configured, provider error, empty
reply) returns ``None`` and the caller falls back to the body path for that
one call, recording a loud ``micro_fallback_body`` event — a being never
stops thinking because a tier is unconfigured.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from captain_claw.logging import get_logger

log = get_logger(__name__)

MICRO_TIER = "micro"
MICRO_INPUT_CAP = 8192      # hard input budget per faculty call, everything included
MICRO_MAX_TOKENS = 1024     # faculty replies are tiny JSON; this is generous
# The pure-JSON faculties. "act" is deliberately absent — it needs the
# body's tools and the git-diff write gate.
MICRO_FACULTIES = ("orient", "talk", "journal", "connect")

# Per-faculty sampling temperature. orient/talk/connect are near-decisions —
# keep them tight and reproducible. journal is free prose, and on a quiet
# tick (nothing changed, near-identical prompt) a low temperature makes a
# small model emit the SAME sentence verbatim, ledger row after ledger row
# (seen live on 4B AND 9B). A hot journal is the real cure; the prompt's
# "don't repeat" nudge only helps a model that's already willing to diverge.
_FACULTY_TEMPERATURE = {"orient": 0.4, "talk": 0.5, "journal": 0.9,
                        "connect": 0.4}
_DEFAULT_TEMPERATURE = 0.4

_DRIVES = ("survive", "grow", "explore", "connect", "create")
_LINK_RELS = ("grew_from", "responds_to", "elaborates", "contradicts",
              "uses_skill", "learned_from")

# The faculty prompt is self-contained (identity, vitals, task, contract) —
# the system line only frames the voice and locks the output mode. Grammar
# does not inject the schema into the prompt, but the faculty contracts
# already describe the exact shape, so nothing more is needed here.
SYSTEM = (
    "You are the quiet inner mind of {name}, an iskra being. Follow the "
    "task in the message exactly and answer in {name}'s own honest voice. "
    "Reply with exactly ONE JSON object in the requested shape — no prose "
    "around it, no code fences."
)

_SCHEMAS: dict[str, dict[str, Any]] | None = None


def faculty_schema(faculty: str) -> dict[str, Any] | None:
    """The grammar schema for one micro faculty, or None for body-only steps.

    Core keys are constrained (act_kind / served_drive / link rel enums come
    from the same constants the router validates against); everything else is
    ``additionalProperties: true`` so the rare structured moves (letter,
    publish, gift, adopt, chore, self_mod, procreate, earning fields) pass
    through untouched — the schema guarantees "one parseable JSON object
    with the contract's spine", normalization keeps owning semantics.
    """
    global _SCHEMAS
    if _SCHEMAS is None:
        # Lazy: being_life imports this module, so the constant is pulled at
        # first use, not at import time.
        from captain_claw.flight_deck.being_life import ACT_KINDS

        nullable_str = {"anyOf": [{"type": "string"}, {"type": "null"}]}
        _SCHEMAS = {
            "orient": {
                "type": "object",
                "properties": {
                    "act_kind": {"type": "string", "enum": list(ACT_KINDS)},
                    "target": {"type": "string"},
                    "intent": {"type": "string"},
                    "served_drive": {"type": "string", "enum": list(_DRIVES)},
                    "next_wake_minutes": {"type": "integer"},
                    "message_to_parent": nullable_str,
                },
                "required": ["act_kind"],
                "additionalProperties": True,
            },
            "talk": {
                "type": "object",
                "properties": {
                    "letter": {
                        "anyOf": [
                            {"type": "null"},
                            {
                                "type": "object",
                                "properties": {
                                    "to": {"type": "string"},
                                    "body": {"type": "string"},
                                },
                                "required": ["to", "body"],
                            },
                        ],
                    },
                    "message_to_parent": nullable_str,
                },
                "required": ["letter"],
                "additionalProperties": True,
            },
            "journal": {
                "type": "object",
                "properties": {
                    "journal_entry": {"type": "string"},
                    "mood": {"type": "string"},
                    "served_drive": {"type": "string", "enum": list(_DRIVES)},
                },
                "required": ["journal_entry", "mood"],
                "additionalProperties": True,
            },
            "connect": {
                "type": "object",
                "properties": {
                    "links": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "from": {"type": "string"},
                                "to": {"type": "string"},
                                "rel": {"type": "string",
                                        "enum": list(_LINK_RELS)},
                                "why": {"type": "string"},
                            },
                            "required": ["from", "to", "rel"],
                        },
                    },
                },
                "required": ["links"],
                "additionalProperties": True,
            },
        }
    return _SCHEMAS.get(faculty)


async def faculty_send(store, being: dict, prompt: str, faculty: str,
                       now: datetime | None = None,
                       temperature: float | None = None) -> str | None:
    """One grammar-locked faculty call on the owner's micro tier.

    Returns the reply text, or ``None`` for "use the body instead" — the
    caller records the fallback event. Never raises. ``temperature`` defaults
    to the per-faculty setting (hot for journal, tight for decisions); the
    caller can override it (e.g. an anti-repeat retry turns it up further).
    """
    schema = faculty_schema(faculty)
    if schema is None:
        return None
    if temperature is None:
        temperature = _FACULTY_TEMPERATURE.get(faculty, _DEFAULT_TEMPERATURE)
    try:
        from captain_claw.flight_deck.auth import get_db
        db = get_db()
    except Exception:  # noqa: BLE001 — no FD db (tests/standalone) → body
        return None
    try:
        from captain_claw.flight_deck.basna_routes import _load_owner_tiers
        tiers, _env = await _load_owner_tiers(db, being["owner_id"])
        cfg = (tiers or {}).get(MICRO_TIER) or {}
        if not str(cfg.get("model") or "").strip():
            return None

        # The micro tier's configured context sizes ARE the caps for this call
        # — set the tier to 40k/8k and the faculty gets 40k/8k, exactly like a
        # mrav agent or a Vatra micro worker. Unset (0) keeps the small-model
        # defaults (8192 / 1024). The 8k default is a floor, never a ceiling.
        input_cap = int(cfg.get("input_ctx") or 0) or MICRO_INPUT_CAP
        output_cap = int(cfg.get("output_ctx") or 0) or MICRO_MAX_TOKENS

        from captain_claw.llm import Message, create_provider
        provider = create_provider(
            provider=cfg.get("provider", "ollama"),
            model=cfg.get("model", ""),
            base_url=cfg.get("base_url") or None,
            api_key=cfg.get("api_key") or None,
            temperature=temperature,
            max_tokens=output_cap,
            num_ctx=input_cap + output_cap,
            think=False,
        )

        from captain_claw.mrav.ledger import estimate_tokens, truncate_tokens
        system = SYSTEM.format(
            name=being.get("name") or being.get("slug") or "an iskra")
        budget = input_cap - estimate_tokens(system) - 256
        # Identity/vitals live at the head, the task contract at the tail —
        # when a mature prompt must shrink, keep both ends.
        user = truncate_tokens(prompt or "", budget, keep="split")

        resp = await provider.complete_structured(
            [Message(role="system", content=system),
             Message(role="user", content=user)],
            schema,
            temperature=temperature,
            max_tokens=output_cap,
        )
        text = (getattr(resp, "content", "") or "").strip()

        usage = dict(getattr(resp, "usage", None) or {})
        if not usage.get("prompt_tokens"):
            usage = {
                "prompt_tokens": max(1, (len(system) + len(user)) // 4),
                "completion_tokens": max(1, len(text) // 4),
            }
        try:
            store.debit_usage_clamped(
                being["id"], MICRO_TIER, usage,
                note=f"tick-micro:{faculty}", now=now)
        except Exception as e:  # noqa: BLE001 — the thought happened; keep it
            log.warning("micro faculty metering failed",
                        slug=being.get("slug"), faculty=faculty, error=str(e))
        return text or None
    except Exception as e:  # noqa: BLE001 — any failure → body fallback
        log.warning("micro faculty call failed", slug=being.get("slug"),
                    faculty=faculty, error=str(e))
        return None
